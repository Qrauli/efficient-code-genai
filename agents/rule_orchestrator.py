from .base_agent import _create_dataframe_sample
from .rule_function_generator import RuleFunctionGenerator
from .rule_code_optimizer import RuleCodeOptimizer
from .rule_code_tester import RuleCodeTester
from .rule_code_reviewer import RuleCodeReviewer
from .rule_format_analyzer import RuleFormatAnalyzer
from .rule_test_case_generator import RuleTestCaseGenerator
from .rule_test_case_reviewer import RuleTestCaseReviewer
import pandas as pd
import os
from utils.context_retrieval import ContextRetriever, RetrievalSource

class RuleOrchestrator:
    def __init__(self, config, use_retrieval=None):
        self.rule_function_generator = RuleFunctionGenerator(config)
        self.code_optimizer = RuleCodeOptimizer(config)
        self.code_tester = RuleCodeTester(config)
        self.code_reviewer = RuleCodeReviewer(config)
        self.rule_format_analyzer = RuleFormatAnalyzer(config)
        self.test_case_generator = RuleTestCaseGenerator(config)
        self.test_case_reviewer = RuleTestCaseReviewer(config)
        
        self.config = config
        self.max_iterations = config.MAX_ITERATIONS
        # Initialize retrieval if enabled
        self.use_retrieval = use_retrieval if use_retrieval is not None else config.ENABLE_RETRIEVAL
        self.retriever = None
        
        if self.use_retrieval:
            self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the context retriever with a comprehensive set of resources"""
        self.retriever = ContextRetriever(self.config)
        
        try:
            # Check if we already have a populated vectorstore
            vectorstore_path = os.path.join(self.config.RETRIEVAL_STORAGE_PATH, "vectorstore")
            
            if os.path.exists(vectorstore_path) and os.listdir(vectorstore_path):
                self.retriever.initialize(force_reload=False)
            else:
                include_web_search = getattr(self.config, 'WEB_SEARCH_ENABLED', False)
                self.retriever.initialize_comprehensive_retrieval(include_web_search=include_web_search)
                
        except Exception as e:
            # Create a basic retriever as fallback
            self.retriever = ContextRetriever(self.config)
            self.retriever.initialize()
    
    def process_rule(self, rule_description, dataframe: pd.DataFrame, rule_id=None, sample_size=None, use_profiling=False, 
                     max_correction_attempts=3, max_restarts=3, test_percentage=0.5):
        """Process a rule description to generate and optimize a function for DataFrame rule evaluation
        
        Args:
            rule_description (str): Description of the rule to implement
            dataframe (pd.DataFrame): The DataFrame to evaluate the rule on
            rule_id (str, optional): Identifier for the rule
            sample_size (int, optional): Number of rows to sample for development
            use_profiling (bool, optional): Whether to use profiling during optimization (default: True)
            max_correction_attempts (int, optional): Maximum number of attempts to correct code before restarting (default: 3)
            max_restarts (int, optional): Maximum number of full workflow restarts (default: 2)
            
        Returns:
            dict: Generated code, execution history, and metadata
        """
        try:
            results_history = []
            profiling_history = []  # Add this before the optimization loop
            
            # Use a default rule_id if not provided
            function_name = f"execute_rule{rule_id}" if rule_id is not None else "execute_rule"
            
            # Initialize dataframe info once for reuse
            working_df = dataframe
            initial_sample_size = sample_size
            current_sample_size = initial_sample_size
            
            if current_sample_size and len(dataframe) > current_sample_size:
                working_df = dataframe.sample(current_sample_size, random_state=42)
            
            # Create dataframe info once for reuse
            dataframe_info = _create_dataframe_sample(working_df)
            
            # Phase 0: Analyze rule format and determine output structure - do this only once
            format_analysis_result = self.rule_format_analyzer.process(
                rule_description, 
                dataframe_info=dataframe_info
            )
            
            # Get the rule format as text directly from the result
            rule_format = format_analysis_result.get("rule_format", "")
            results_history.append({"step": "format_analysis", "result": format_analysis_result})

            # Phase 0.5: Generate test cases - only once
            test_case_result = self.test_case_generator.process(
                rule_description,
                rule_format,
                dataframe_info=dataframe_info
            )
            test_cases = test_case_result.get("test_cases", [])
            results_history.append({"step": "test_case_generation", "result": test_case_result})
            
            # Restart counter
            restart_count = 0
            
            # Start generating code with potential restarts
            while restart_count <= max_restarts:  # Allow initial run + max_restarts
                if restart_count > 0:
                    # Log the restart but don't redo format analysis or test cases
                    results_history.append({
                        "step": f"workflow_restart_{restart_count}",
                        "result": {
                            "reason": "Multiple correction attempts failed to fix the code",
                            "previous_attempts": correction_attempt_count if 'correction_attempt_count' in locals() else 0
                        }
                    })
                
                # Add test case explanation to the generator input if available
                test_case_guidance = ""
                if test_cases:
                    # Include information about the first test case in the guidance
                    first_test_case = test_cases[0] if test_cases else {}
                    test_case_guidance = f"""
# Test Case Information
Here is a test case that should pass with your implementation:

Sample DataFrame:
```python
{first_test_case.get('dataframe', {})}
```

Expected Output:
- Support: {first_test_case.get('expected_output', {}).get('support')}
- Confidence: {first_test_case.get('expected_output', {}).get('confidence')}
- Satisfactions: {first_test_case.get('expected_output', {}).get('satisfactions_str')}
- Violations: {first_test_case.get('expected_output', {}).get('violations_str')}

Explanation:
{first_test_case.get('explanation', '')}
"""

                # Phase 1: Generate initial function with retrieval context, format specifications and test case
                generator_context = ""
                
                generator_result = self.rule_function_generator.process(
                    rule_description, 
                    df_sample=dataframe_info,
                    function_name=function_name,
                    context=generator_context,
                    rule_format=rule_format,
                    test_case_guidance=test_case_guidance
                )
                
                current_code = initial_code = generator_result["code"]
                results_history.append({"step": f"generation_{restart_count}", "result": generator_result})
                
                # Phase 2: Optimization loop
                iterations = 0
                last_working_code = None
                running_code = None
                should_continue = True
                test_case_result = None
                profiling_result = None
                # Counter for code correction attempts
                correction_attempt_count = 0
                
                while iterations < self.max_iterations and should_continue:
                    # Step 1: Test with test cases (now mandatory)
                    test_case_result = self.code_tester.test_function_with_testcases(
                        current_code, 
                        test_cases, 
                        function_name=function_name
                    )
                    results_history.append({"step": f"refinement_{restart_count}_{iterations}_test_case", "result": test_case_result})

                    # If the code runs without exceptions (i.e., no error in any test result), and last_working_code is None,
                    # set running_code to current_code (even if the output is wrong)
                    if running_code is None:
                        test_results = test_case_result.get("test_results", [])
                        # Check if all test cases executed without raising an exception (i.e., 'error' is None or empty)
                        all_executed = all(
                            ("error" not in tr or not tr["error"].startswith("EXCEPTION:")) 
                            for tr in test_results
                        )
                        if all_executed:
                            running_code = current_code

                    # On final restart, check if we should consider partial success
                    test_results = test_case_result.get("test_results", [])
                    passed_tests = sum(1 for tr in test_results if tr.get("success", False))
                    total_tests = len(test_results)
                    success_rate = passed_tests / total_tests if total_tests > 0 else 0
                    # Consider it a partial success if at least half of tests pass
                    if restart_count == max_restarts:
                        test_percentage = 0.5
                    partial_success = success_rate >= test_percentage
                    test_case_result["success"] = partial_success
                    
                    # Collect warnings from test case execution
                    warnings = test_case_result.get("warnings", [])
                    
                    # If any test case fails, first review the code and test cases
                    if not test_case_result.get("success", False):
                        correction_attempt_count += 1

                        # Extract relevant test results to pass to reviewer
                        test_results = test_case_result.get("test_results", [])
                        
                        # Review the code and test cases to determine the issue
                        review_result = self.test_case_reviewer.process(
                            rule_description=rule_description,
                            code=current_code,
                            test_results=test_results,
                            rule_format=rule_format,
                            dataframe_info=dataframe_info
                        )
                        results_history.append({"step": f"refinement_{restart_count}_{iterations}_test_review", "result": review_result})
                        
                        # Extract the analysis from the review result
                        analysis = review_result.get("analysis", {})
                        
                        # Check if there are issues with the code or test cases
                        code_fix_approach = analysis.get("code_fix_approach")
                        corrected_test_cases = analysis.get("corrected_test_cases")
                        
                        # Determine if we should fix the code, the test cases, or both
                        fix_code = analysis.get("fix_code")
                        fix_test_cases = analysis.get("fix_test_cases")
                        
                        # Case 1: If the test cases need correction
                        if fix_test_cases and corrected_test_cases:
                            
                            for correction in corrected_test_cases:
                                idx = correction.get("test_case_index", -1)
                                if 0 <= idx < len(test_cases):
                                    # Update the expected output with corrected values
                                    test_cases[idx]["expected_output"].update(correction.get("corrected_values", {}))
                                    # Add explanation of correction to the test case
                                    test_cases[idx]["explanation"] = correction.get('explanation', 'Test case values were corrected.')
                            
                            # Re-run test case validation with corrected test cases
                            test_case_result = self.code_tester.test_function_with_testcases(
                                current_code, 
                                test_cases, 
                                function_name=function_name
                            )
                            results_history.append({"step": f"refinement_{restart_count}_{iterations}_test_case_corrected", "result": test_case_result})
                        
                        # Case 2: If the code needs to be fixed
                        if fix_code and not test_case_result.get("success", False):
                            
                            # Construct detailed guidance for the code corrector
                            correction_guidance = ""
                            if code_fix_approach:
                                correction_guidance = "The test case review identified the following code problems and approaches to fix them:\n\n"
                                correction_guidance += f"\nRecommended approach: {code_fix_approach}"
                            
                            # Extract failing tests
                            test_results = test_case_result.get("test_results", [])
                            failing_tests = [tr for tr in test_results if not tr.get("success", False)]
                            
                            corrected_result = self.code_tester.correct_code(
                                code=current_code,
                                problem_description=f"DataFrame rule evaluation function for: {rule_description}\n\n{correction_guidance}",
                                test_results=failing_tests,
                                dataframe_info=dataframe_info,
                                function_name=function_name,
                                rule_format=rule_format
                            )
                            results_history.append({"step": f"refinement_{restart_count}_{iterations}_correction_{correction_attempt_count}", "result": corrected_result})
                            
                            # Update code and retry test case in next iteration
                            current_code = corrected_result.get("corrected_code", current_code)
                            
                        # Check if we need to restart the workflow due to too many correction attempts
                        if correction_attempt_count >= max_correction_attempts and last_working_code is None:
                            restart_count += 1
                            break
                       
                        iterations += 1
                        continue
                    
                    # Reset correction attempts counter when a test passes
                    if test_case_result.get("success", True):
                        correction_attempt_count = 0
                        
                    # Step 2: Run profiling if enabled
                    if use_profiling:
                        profiling_result = self.rule_function_generator.execute_and_profile_rule(current_code, working_df, function_name=function_name)
                        profiling_history.append((current_code, profiling_result))
                        
                        if profiling_result.get("success", False):
                            profiling_history.append((current_code, profiling_result))
                        test_log = profiling_result.copy()
                        if "function_results" in test_log: 
                            del test_log["function_results"]
                        if "profile_data" in test_log: 
                            del test_log["profile_data"]
                        results_history.append({"step": f"refinement_{restart_count}_{iterations}_profiling", "result": test_log})
                        
                        # If profiling fails (not just timeout), trigger code correction
                        if not profiling_result.get("success", True):
                            correction_attempt_count += 1

                            # Prepare a "profiling failure" test result for correction
                            profiling_failure_test = {
                                "test_case_name": "Profiling Run",
                                "error": profiling_result.get("error", "Profiling failed"),
                                "success": False
                            }
                            # Use the same code correction logic as for test failures
                            corrected_result = self.code_tester.correct_code(
                                code=current_code,
                                problem_description=f"Profiling failed for DataFrame rule evaluation function for: {rule_description}",
                                test_results=[profiling_failure_test],
                                dataframe_info=dataframe_info,
                                function_name=function_name,
                                rule_format=rule_format
                            )
                            results_history.append({"step": f"refinement_{restart_count}_{iterations}_profiling_correction_{correction_attempt_count}", "result": corrected_result})

                            # Update code and retry profiling in next iteration
                            current_code = corrected_result.get("corrected_code", current_code)

                            # If too many correction attempts, break to restart
                            if correction_attempt_count >= max_correction_attempts + 1 and last_working_code is None:
                                restart_count += 1
                                break

                            iterations += 1
                            continue

                    last_working_code = current_code
                    
                    # Step 3: Code review to determine if optimization should continue
                    # Prepare profiling data if available
                    profiling_data = None
                    if use_profiling and profiling_result:
                        line_profiling = self._format_line_profiling([{"test_case": "rule_evaluation", "profile_data": profiling_result.get("profile_data", {})}], function_name)
                        profiling_data = {
                            "overall_metrics": {
                                "test": {
                                    "execution_time": profiling_result.get("execution_time"),
                                    "function_results": profiling_result.get("function_results", {})
                                }
                            },
                            "line_profiling": line_profiling
                        }
                    
                    # Set up review input - with or without profiling data
                    review_input = {
                        "code": current_code,
                        "problem_description": f"DataFrame rule evaluation function for: {rule_description}",
                        "dataframe_info": dataframe_info,
                        "rule_format": rule_format
                    }
                    
                    # Add profiling data if available
                    if profiling_data:
                        review_input["test_result"] = profiling_result
                        review_input["profiling_data"] = profiling_data

                    review_result = self.code_reviewer.process(review_input)
                    results_history.append({"step": f"refinement_{restart_count}_{iterations}_review", "result": review_result})
                    
                    # Check if we should terminate optimization based on review
                    should_continue = review_result.get("continue_optimization", True)
                    
                    if not should_continue:
                        # Reviewer suggests terminating optimization
                        results_history.append({
                            "step": f"refinement_{restart_count}_{iterations}_termination",
                            "result": {
                                "reason": "Reviewer recommended termination",
                                "optimization_potential": review_result.get("optimization_potential"),
                                "recommendations": review_result.get("improvement_recommendations")
                            }
                        })
                        break
                    
                    # Step 4: Optimize code based on profiling and reviewer feedback
                    optimization_context = None
                    if self.use_retrieval:
                        # Get specific query based on review feedback
                        optimization_query = f"pandas dataframe optimization techniques for: {rule_description}"
                        if review_result.get("improvement_recommendations", []):
                            # Use first 2 recommendations to guide retrieval
                            recommendations = review_result.get("improvement_recommendations", [])[:2]
                            optimization_query += f" to improve {', '.join(recommendations)}"
                        
                        optimization_context = self._get_relevant_context(
                            query=optimization_query,
                            source_types=["web"]
                        )
                    
                    # Prepare optimizer input - with or without profiling data
                    optimizer_input = {
                        "code": current_code,
                        "problem_description": f"Optimize the DataFrame rule evaluation function for: {rule_description}",
                        "dataframe_info": dataframe_info,
                        "phase": "optimization",
                        "review_feedback": review_result.get("improvement_recommendations", []),
                        "retrieval_context": optimization_context,
                        "rule_format": rule_format,
                        "warnings": warnings  # Include warnings from test execution
                    }
                    
                    # Add profiling data if available
                    if profiling_data:
                        optimizer_input["profiling_data"] = profiling_data
                    
                    optimizer_result = self.code_optimizer.process(optimizer_input)
                    log_optimizer_result = optimizer_result.copy()
                    if "profiling_data" in log_optimizer_result: 
                        del log_optimizer_result["profiling_data"]
                    results_history.append({"step": f"refinement_{restart_count}_{iterations}_optimization", "result": log_optimizer_result})
                    
                    # Update and prepare for next iteration
                    current_code = optimizer_result.get("optimized_code", current_code)
                    iterations += 1
                
                # If we completed optimization or have working code, we can exit the restart loop
                if last_working_code is not None or (test_case_result and test_case_result.get("success", False)):
                    break
                
            # Run test case validation after optimization to ensure the optimized code still works
            test_case_result = self.code_tester.test_function_with_testcases(
                current_code, 
                test_cases, 
                function_name=function_name
            )
            results_history.append({"step": f"final_validation", "result": test_case_result})
            
            # If test case fails after optimization, revert to previous code
            if not test_case_result.get("success", False) and last_working_code:
                current_code = last_working_code
                test_case_result["success"] = True
                results_history.append({"step": f"final_reverted_code", "result": test_case_result})
            elif not test_case_result.get("success", False) and running_code:
                current_code = running_code
                test_case_result["success"] = True
                results_history.append({"step": f"final_reverted_code_running", "result": test_case_result})
            
            # Select the fastest code based on profiling history
            fastest_code = current_code  # Default to latest
            min_time = float('inf')
            for code, prof in profiling_history:
                if prof and not prof.get("timed_out", False):
                    exec_time = prof.get("measured_execution_time")
                    if exec_time is not None and exec_time < min_time:
                        min_time = exec_time
                        fastest_code = code

            # Store the successful code for future retrieval if enabled
            if self.use_retrieval and test_case_result and test_case_result.get("success", False):
                self.retriever.add_generated_code(
                    code=fastest_code,
                    metadata={
                        "description": f"Optimized function for rule: {rule_description}",
                        "tags": ["rule_evaluation", "dataframe", "optimized"]
                    }
                )
            
            return {
                "code": fastest_code,
                "summary": "Rule function generation completed successfully." if (test_case_result and test_case_result.get("success", False)) else "Rule function generation failed. See error details in the execution history.",
                "initial_code": initial_code if 'initial_code' in locals() else None,
                "results_history": results_history,
                "function_name": function_name,
                "restart_count": restart_count,
                "success": test_case_result and test_case_result.get("success", False)
            }
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            # Log the error but also return a helpful response
            print(f"Error in process_rule: {str(e)}\n{error_traceback}")
            
            return {
                "code": "",  # Empty code since generation failed
                "summary": f"Rule function generation failed with error: {str(e)}",
                "error": str(e),
                "traceback": error_traceback,
                "results_history": results_history if 'results_history' in locals() else [],
                "function_name": function_name if 'function_name' in locals() else f"execute_rule{rule_id}" if rule_id is not None else "execute_rule",
                "success": False
            }
    
    def _extract_rule_metrics(self, test_result):
        """Extract comprehensive performance metrics from rule function test result"""
        metrics = {}
        
        if test_result and test_result.get("success", False):
            # Check if it's a test case result
            if "comparison" in test_result:
                # Extract function metrics from test case comparison
                comparison = test_result.get("comparison", {})
                metrics.update({
                    "support": comparison.get("support", {}).get("actual"),
                    "confidence": comparison.get("confidence", {}).get("actual"),
                    "execution_time": test_result.get("execution_time")
                })
            else:
                # Extract functional metrics from function_results (profiling result)
                function_results = test_result.get("function_results", {})
                metrics.update({
                    "support": function_results.get("support"),
                    "confidence": function_results.get("confidence"),
                    "row_indexes_count": function_results.get("row_indexes_count")
                })
                
                # Extract performance metrics from profiling data
                profile_data = test_result.get("profile_data", {})
                files = profile_data.get("files", {})
                
                if files:
                    # Get metrics from the first file (main script)
                    for file_path, file_data in files.items():
                        metrics.update({
                            "execution_time": file_data.get("total_cpu_seconds", 0),
                            "cpu_percent": file_data.get("n_cpu_percent_python", 0) + file_data.get("n_cpu_percent_c", 0),
                            "memory_mb": file_data.get("max_mb", 0)
                        })
                        
                        # Count the number of high-CPU lines (potential bottlenecks)
                        lines = file_data.get("lines", [])
                        high_cpu_lines = sum(1 for ln in lines if ln.get("cpu_percent", 0) > 5)
                        metrics["high_cpu_lines"] = high_cpu_lines
                        
                        # Find the max CPU percentage for any line
                        if lines:
                            max_cpu_pct = max((ln.get("cpu_percent", 0) for ln in lines), default=0)
                            metrics["max_line_cpu_percent"] = max_cpu_pct
                        
                        # Only need one file's metrics
                        break
        
        return metrics
    
    def _get_relevant_context(self, query, source_types=None, top_k=None):
        """Get relevant context from the retriever"""
        if not self.use_retrieval or not self.retriever:
            return None
            
        try:             
            # Get relevant context
            context_results = self.retriever.retrieve(
                query=query,
                filter_sources=source_types,
                top_k=top_k
            )
            
            if not context_results:
                return None
                
            # Format context for inclusion in prompts
            formatted_context = "## Relevant Context\n\n"
            
            for i, result in enumerate(context_results):
                formatted_context += f"### {i+1}. {result['source']} ({result['source_type']})\n"
                formatted_context += result['content'].strip() + "\n\n"
            
            return formatted_context
        except Exception as e:
            # Log error but don't stop the process
            print(f"Error retrieving context: {str(e)}")
            return None
        
    def _format_line_profiling(self, line_profiling, function_name="execute_rule"):
        """Format Scalene line profiling data focusing primarily on hotspots"""
        formatted = ""
        for profile in line_profiling:
            formatted += f"Test: {profile.get('test_case', 'Unknown')}\n"
            
            profile_data = profile.get('profile_data', {})

            if not profile_data:
                formatted += "No profile data available\n\n"
                continue
            
            total_cpu_seconds = profile_data.get('elapsed_time_sec', 0)
            formatted += f"Total CPU seconds: {total_cpu_seconds:.2f}\n"
            
            # Extract file-level metrics
            files = profile_data.get('files', {})
            for file_path, file_data in files.items():
                formatted += f"File: {os.path.basename(file_path)}\n"
                
                # Get line-level metrics as array of line objects
                lines_array = file_data.get('lines', [])
                
                # Create a mapping from line numbers to line data for easier access
                lines_map = {line_data.get('lineno'): line_data for line_data in lines_array if 'lineno' in line_data}
                
                # Sort line numbers
                line_numbers = sorted(lines_map.keys())
                if not line_numbers:
                    formatted += "No line information available\n\n"
                    continue
                    
                # Better function code identification
                # Find actual function definition line (where the function_name is defined)
                function_start_line = None
                function_end_line = None
                
                for line_num in line_numbers:
                    line_content = lines_map[line_num].get('line', '')
                    if f"def {function_name}" in line_content:
                        function_start_line = line_num
                        break
                
                # If we can't find the function definition, look for common boundaries
                if function_start_line is None:
                    # Find first non-import line
                    for line_num in line_numbers:
                        line_content = lines_map[line_num].get('line', '')
                        if line_content.strip() and not (
                            line_content.strip().startswith("import ") or 
                            line_content.strip().startswith("from ") or
                            line_content.strip().startswith("#")
                        ):
                            function_start_line = line_num
                            break
                
                # Find the function end line
                if function_start_line is not None:
                    for line_num in line_numbers:
                        if line_num > function_start_line:
                            line_content = lines_map[line_num].get('line', '')
                            if "# Execute the function" in line_content or "# Test" in line_content:
                                function_end_line = line_num - 1
                                break
                
                # If we still don't have bounds, use reasonable defaults
                if function_start_line is None:
                    function_start_line = min(line_numbers)
                if function_end_line is None:
                    function_end_line = max(line_numbers)
                
                # Get function lines only
                function_lines = [line_num for line_num in line_numbers 
                                if function_start_line <= line_num <= function_end_line]
                
                # Skip this file if no function lines
                if not function_lines:
                    formatted += "No function code identified\n\n"
                    continue
                    
                formatted += f"Function code profiling (lines {function_start_line}-{function_end_line}):\n"
                
                # Focus primarily on hotspots
                formatted += "\nPerformance Hotspots:\n"
                
                # Find the top 7 lines by CPU usage within function
                top_cpu_lines = sorted(
                    [(line_num, lines_map[line_num].get('n_cpu_percent_python', 0) + 
                    lines_map[line_num].get('n_cpu_percent_c', 0)) 
                    for line_num in function_lines 
                    if lines_map[line_num].get('n_cpu_percent_python', 0) + 
                        lines_map[line_num].get('n_cpu_percent_c', 0) > 0],
                    key=lambda x: x[1],
                    reverse=True
                )[:7]
                
                if top_cpu_lines:
                    formatted += "Top CPU usage lines:\n"
                    for line_num, cpu_pct in top_cpu_lines:
                        line_content = lines_map[line_num].get('line', '').rstrip()
                        cpu_seconds = (cpu_pct / 100.0) * total_cpu_seconds
                        formatted += f"Line {line_num}: {cpu_pct:.1f}% ({cpu_seconds:.4f}s) - {line_content}\n"                    
                else:
                    formatted += "No significant CPU hotspots identified\n"
                
                # Add memory hotspots within function
                top_mem_lines = sorted(
                    [(line_num, lines_map[line_num].get('n_avg_mb', 0)) 
                    for line_num in function_lines
                    if lines_map[line_num].get('n_avg_mb', 0) > 0],
                    key=lambda x: x[1],
                    reverse=True
                )[:7]
                
                if top_mem_lines:
                    formatted += "\nTop memory usage lines:\n"
                    for line_num, mem_mb in top_mem_lines:
                        line_content = lines_map[line_num].get('line', '').rstrip()
                        formatted += f"Line {line_num}: {mem_mb:.2f} MB - {line_content}\n"
                else:
                    formatted += "No significant memory hotspots identified\n"
            
            formatted += "\n"
        
        return formatted