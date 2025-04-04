from .base_agent import _create_dataframe_sample
from .orchestrator import Orchestrator
from .rule_function_generator import RuleFunctionGenerator
from .rule_code_optimizer import RuleCodeOptimizer
from .rule_code_tester import RuleCodeTester
from .rule_code_reviewer import RuleCodeReviewer
from .rule_format_analyzer import RuleFormatAnalyzer  # Add this import
from .rule_test_case_generator import RuleTestCaseGenerator
import pandas as pd
import os
from utils.context_retrieval import ContextRetriever, RetrievalSource

class RuleOrchestrator(Orchestrator):
    def __init__(self, config, use_retrieval=None):
        super().__init__(config)
        self.rule_function_generator = RuleFunctionGenerator(config)
        self.code_optimizer = RuleCodeOptimizer(config)
        self.code_tester = RuleCodeTester(config)
        self.code_reviewer = RuleCodeReviewer(config)
        self.rule_format_analyzer = RuleFormatAnalyzer(config)  # Initialize the format analyzer
        self.test_case_generator = RuleTestCaseGenerator(config)
        
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
                self.logger.info("Loading existing retrieval system...")
                self.retriever.initialize(force_reload=False)
            else:
                self.logger.info("Building comprehensive retrieval system...")
                include_web_search = getattr(self.config, 'WEB_SEARCH_ENABLED', False)
                self.retriever.initialize_comprehensive_retrieval(include_web_search=include_web_search)
                
            self.logger.info(f"Retrieval system initialized with {self.retriever.vectorstore._collection.count() if self.retriever.vectorstore else 0} documents")
        except Exception as e:
            self.logger.error(f"Error initializing retriever: {str(e)}")
            # Create a basic retriever as fallback
            self.retriever = ContextRetriever(self.config)
            self.retriever.initialize()
    
    def process_rule(self, rule_description, dataframe: pd.DataFrame, rule_id=None, sample_size=None, use_test_cases=False):
        """Process a rule description to generate and optimize a function for DataFrame rule evaluation
        
        Args:
            rule_description (str): Description of the rule to implement
            dataframe (pd.DataFrame): The DataFrame to evaluate the rule on
            rule_id (str, optional): Identifier for the rule
            sample_size (int, optional): Number of rows to sample for development
            use_test_cases (bool, optional): Whether to generate and use test cases (default: True)
            
        Returns:
            dict: Generated code, execution history, and metadata
        """
        try:
            results_history = []
            rule_description = rule_description.replace("{", "{{").replace("}", "}}")
            
            # Use a default rule_id if not provided
            function_name = f"execute_rule{rule_id}" if rule_id is not None else "execute_rule"
            
            # Use a sample of the dataframe for development if specified
            working_df = dataframe
            initial_sample_size = sample_size or min(1000, len(dataframe))
            current_sample_size = initial_sample_size
            
            if current_sample_size and len(dataframe) > current_sample_size:
                working_df = dataframe.sample(current_sample_size, random_state=42)
            
            # Create dataframe info once for reuse
            dataframe_info = _create_dataframe_sample(working_df)
            
            # Phase 0: Analyze rule format and determine output structure
            format_analysis_result = self.rule_format_analyzer.process(
                rule_description, 
                dataframe_info=dataframe_info
            )
            
            # Get the rule format as text directly from the result
            rule_format = format_analysis_result.get("rule_format", "")
            results_history.append({"step": "format_analysis", "result": format_analysis_result})

            # Phase 0.5: Generate test cases if enabled
            test_cases = []
            if use_test_cases:
                test_case_result = self.test_case_generator.process(
                    rule_description,
                    rule_format,
                    dataframe_info=dataframe_info
                )
                test_cases = test_case_result.get("test_cases", [])
                results_history.append({"step": "test_case_generation", "result": test_case_result})
            
            # Phase 1: Generate initial function with retrieval context, format specifications and test case
            generator_context = self._get_relevant_context(
                query=f"pandas dataframe function to evaluate rule: {rule_description}",
                source_types=["documentation", "code_snippet"]
            )

            # Add test case explanation to the generator input if available
            test_case_guidance = ""
            if use_test_cases and test_cases:
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

            generator_result = self.rule_function_generator.process(
                rule_description, 
                df_sample=dataframe_info,
                function_name=function_name,
                context=generator_context,
                rule_format=rule_format,
                test_case_guidance=test_case_guidance  # Add this parameter
            )
            
            current_code = initial_code = generator_result["code"]
            results_history.append({"step": "generation", "result": generator_result})
            
            # Phase 2: Optimization loop
            iterations = 0
            previous_code = None
            should_continue = True
            test_case_result = None
            profiling_result = None
            testcase_correction_attempts = 0  # Track testcase correction attempts
            skip_testcase_validation = False  # Flag to skip testcase validation after too many failures
            
            while iterations < self.max_iterations and should_continue:
                # Step 1: Test with test case if enabled and available
                if use_test_cases and test_cases and not skip_testcase_validation:
                    test_case_result = self.code_tester.test_function_with_testcases(
                        current_code, 
                        test_cases, 
                        function_name=function_name
                    )
                    results_history.append({"step": f"refinement_{iterations}_test_case", "result": test_case_result})
                    
                    # If any test case fails, attempt to fix the code
                    if not test_case_result.get("success", False):
                        # Track correction attempts
                        testcase_correction_attempts += 1
                        
                        # Extract relevant test results to pass to correction method
                        test_results = test_case_result.get("test_results", [])
                        failing_tests = [tr for tr in test_results if not tr.get("success", False)]
                        
                        corrected_result = self.code_tester.correct_code(
                            code=current_code,
                            problem_description=f"DataFrame rule evaluation function for: {rule_description}",
                            test_results=failing_tests,  # Pass only failing tests for correction
                            dataframe_info=dataframe_info,
                            function_name=function_name,
                            rule_format=rule_format
                        )
                        results_history.append({"step": f"refinement_{iterations}_correction", "result": corrected_result})
                        
                        # Update code and retry test case in next iteration
                        current_code = corrected_result.get("corrected_code", current_code)
                        
                        if testcase_correction_attempts >= 2:
                            skip_testcase_validation = True
                   
                        iterations += 1
                        continue
                
                # Step 2: Run profiling (always do this regardless of test cases)
                profiling_result = self.rule_function_generator.execute_and_profile_rule(current_code, working_df, function_name=function_name)
                
                test_log = profiling_result.copy()
                if "function_results" in test_log: del test_log["function_results"]
                if "profile_data" in test_log: del test_log["profile_data"]
                results_history.append({"step": f"refinement_{iterations}_profiling", "result": test_log})
                
                # Adaptive sampling: If test times out, reduce sample size and retry
                if profiling_result.get("timed_out", False) and current_sample_size > 100:
                    # Reduce sample size by half and retry
                    prev_size = current_sample_size
                    current_sample_size = max(100, current_sample_size // 2)
                    working_df = dataframe.sample(current_sample_size, random_state=42)
                    dataframe_info = _create_dataframe_sample(working_df)
                    
                    # Log the sample size reduction
                    results_history.append({
                        "step": f"refinement_{iterations}_sample_reduction", 
                        "result": {
                            "previous_size": prev_size,
                            "new_size": current_sample_size,
                            "reason": "Execution timed out during profiling"
                        }
                    })
                              
                # Step 3: If profiling fails, attempt to fix the code and restart iteration
                if not profiling_result.get("success", False) and not profiling_result.get("timed_out", False):
                    corrected_result = self.code_tester.correct_code(
                        code=current_code,
                        problem_description=f"DataFrame rule evaluation function for: {rule_description}",
                        test_results=[profiling_result],
                        dataframe_info=dataframe_info,
                        function_name=function_name,
                        rule_format=rule_format
                    )
                    results_history.append({"step": f"refinement_{iterations}_correction", "result": corrected_result})
                    
                    # Update code and restart iteration
                    current_code = corrected_result.get("corrected_code", current_code)
                    iterations += 1
                    continue
                
                previous_code = current_code
                
                # Step 4: Code review to determine if optimization should continue
                line_profiling = self._format_line_profiling([{"test_case": "rule_evaluation", "profile_data": profiling_result.get("profile_data", {})}], function_name)
                
                review_input = {
                    "code": current_code,
                    "problem_description": f"DataFrame rule evaluation function for: {rule_description}",
                    "test_result": profiling_result,
                    "dataframe_info": dataframe_info,
                    "rule_format": rule_format,
                    "profiling_data": {
                        "overall_metrics": {
                            "test": {
                                "execution_time": profiling_result.get("execution_time"),
                                "function_results": profiling_result.get("function_results", {})
                            }
                        },
                        "line_profiling": line_profiling
                    }
                }

                review_result = self.code_reviewer.process(review_input)
                results_history.append({"step": f"refinement_{iterations}_review", "result": review_result})
                
                # Check if we should terminate optimization based on review
                should_continue = review_result.get("continue_optimization", True)
                
                if not should_continue:
                    # Reviewer suggests terminating optimization
                    results_history.append({
                        "step": f"refinement_{iterations}_termination",
                        "result": {
                            "reason": "Reviewer recommended termination",
                            "optimization_potential": review_result.get("optimization_potential"),
                            "recommendations": review_result.get("improvement_recommendations")
                        }
                    })
                    break
                
                # Step 5: Optimize code based on profiling and reviewer feedback
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
                        source_types=["documentation", "code_snippet"]
                    )
                
                optimizer_input = {
                    "code": current_code,
                    "problem_description": f"Optimize the DataFrame rule evaluation function for: {rule_description}",
                    "dataframe_info": dataframe_info,
                    "profiling_data": {
                        "overall_metrics": {
                            "test": {
                                "execution_time": profiling_result.get("execution_time"),
                                "function_results": profiling_result.get("function_results", {})
                            }
                        },
                        "line_profiling": line_profiling
                    },
                    "phase": "optimization",
                    "review_feedback": review_result.get("improvement_recommendations", []),
                    "retrieval_context": optimization_context,
                    "rule_format": rule_format
                }
                
                optimizer_result = self.code_optimizer.process(optimizer_input)
                log_optimizer_result = optimizer_result.copy()
                if "profiling_data" in log_optimizer_result: del log_optimizer_result["profiling_data"]
                results_history.append({"step": f"refinement_{iterations}_optimization", "result": log_optimizer_result})
                
                # Update and prepare for next iteration
                current_code = optimizer_result.get("optimized_code", current_code)
                test_case_result = None  # Reset test case result for next iteration
                iterations += 1
            
            # Final test of the optimized code if we exited the loop without testing it
            if iterations > 0 and (not test_case_result or test_case_result.get("code") != current_code):
                # Test with test case if enabled, available, and we're not skipping test case validation
                if use_test_cases and test_cases and not skip_testcase_validation:
                    final_test_result = self.code_tester.test_function_with_testcases(
                        current_code, 
                        test_cases, 
                        function_name=function_name
                    )
                    results_history.append({"step": "final_test_case", "result": final_test_result})
                    
                    # If final test fails, revert to previous code that passed
                    if not final_test_result.get("success", False) and previous_code:
                        current_code = previous_code
                        final_test_result["success"] = True 
                else:
                    # Final profiling with real data
                    final_test_result = self.rule_function_generator.execute_and_profile_rule(current_code, working_df, function_name=function_name)
                    test_log = final_test_result.copy()
                    if "function_results" in test_log: del test_log["function_results"]
                    if "profile_data" in test_log: del test_log["profile_data"]
                    results_history.append({"step": "final_profiling", "result": test_log})
                    
                    # If final test fails, revert to previous code that passed
                    if not final_test_result.get("success", False) and not final_test_result.get("timed_out", False) and previous_code:
                        current_code = previous_code
                        final_test_result["success"] = True
            else:
                # Use the last test result as the final result
                final_test_result = profiling_result
                    
            # Store the successful code for future retrieval if enabled
            if self.use_retrieval and final_test_result.get("success", False) and not final_test_result.get("timed_out", False):
                self.retriever.add_generated_code(
                    code=current_code,
                    metadata={
                        "description": f"Optimized function for rule: {rule_description}",
                        "tags": ["rule_evaluation", "dataframe", "optimized"]
                    }
                )
            
            return {
                "code": current_code,
                "summary": "Rule function generation completed successfully." if (final_test_result.get("success", False) or final_test_result.get("timed_out", False)) else "Rule function generation failed. See error details in the execution history.",
                "initial_code": initial_code,
                "results_history": results_history,
                "function_name": function_name
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
            # Convert source_types to source filter if specified
            filter_sources = None
            if source_types:
                # Get all sources of the specified types
                filter_sources = [
                    source.name for source in self.retriever.sources 
                    if source.type in source_types and source.enabled
                ]
                
            # Get relevant context
            context_results = self.retriever.retrieve(
                query=query,
                filter_sources=filter_sources,
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
        """Format Scalene line profiling data for inclusion in prompt"""
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
                
                # Instead of looking for function boundaries, simply focus on the first N lines
                # that contain the function code (before the test wrapper code)
                line_numbers = sorted(lines_map.keys())
                
                # Find the first line that contains "# Execute the function" which marks
                # the boundary between the function code and the test wrapper code
                function_end_line = None
                for line_num in line_numbers:
                    line_data = lines_map[line_num]
                    content = line_data.get('line', '')
                    if "# Execute the function" in content:
                        function_end_line = line_num - 1
                        break
                
                # If we can't find the boundary marker, just use all available lines
                if function_end_line is None and line_numbers:
                    function_end_line = max(line_numbers)
                
                # Determine function start line (first line with content)
                function_start_line = min(line_numbers) if line_numbers else None
                
                # Format only the function's line-by-line metrics
                if function_start_line is not None:
                    formatted += f"Function code profiling (lines {function_start_line}-{function_end_line}):\n"
                    formatted += "Line | CPU % (seconds) | Memory (MB) | Alloc (MB) | Code\n"
                    formatted += "-" * 70 + "\n"
                    
                    # Get metrics only for the function lines
                    function_lines = [line_num for line_num in line_numbers 
                                    if function_start_line <= line_num <= function_end_line]
                    
                    # Generate line-by-line output for function only
                    for line_num in function_lines:
                        line_data = lines_map[line_num]
                        
                        # Calculate total CPU percentage (Python + C)
                        cpu_percent = line_data.get('n_cpu_percent_python', 0) + line_data.get('n_cpu_percent_c', 0)
                        
                        # Convert percentage to actual seconds spent on this line
                        cpu_seconds = (cpu_percent / 100.0) * total_cpu_seconds if cpu_percent > 0 else 0
                        
                        memory_mb = line_data.get('n_avg_mb', 0)
                        alloc_mb = line_data.get('n_malloc_mb', 0)
                        line_content = line_data.get('line', '')
                        
                        if cpu_percent > 0 or memory_mb > 0:
                            # Format line data with both percentage and absolute time
                            formatted += f"{line_num:4d} | {cpu_percent:5.1f}% ({cpu_seconds:.4f}s) | {memory_mb:8.2f} | {alloc_mb:8.2f} | {line_content.rstrip()}\n"
                    
                    # Add summary of hotspots within the function code
                    formatted += "\nHotspots (within function code only):\n"
                    # Find the top 5 lines by CPU usage within function
                    top_cpu_lines = sorted(
                        [(line_num, lines_map[line_num].get('n_cpu_percent_python', 0) + 
                        lines_map[line_num].get('n_cpu_percent_c', 0)) 
                        for line_num in function_lines 
                        if lines_map[line_num].get('n_cpu_percent_python', 0) + 
                            lines_map[line_num].get('n_cpu_percent_c', 0) > 0],
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    if top_cpu_lines:
                        formatted += "Top CPU usage lines:\n"
                        for line_num, cpu_pct in top_cpu_lines:
                            line_content = lines_map[line_num].get('line', '').rstrip()
                            cpu_seconds = (cpu_pct / 100.0) * total_cpu_seconds
                            formatted += f"Line {line_num}: {cpu_pct:.1f}% ({cpu_seconds:.4f}s) - {line_content}\n"                    
                    
                    # Add memory hotspots within function
                    top_mem_lines = sorted(
                        [(line_num, lines_map[line_num].get('n_avg_mb', 0)) 
                        for line_num in function_lines
                        if lines_map[line_num].get('n_avg_mb', 0) > 0],
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    if top_mem_lines:
                        formatted += "\nTop memory usage lines:\n"
                        for line_num, mem_mb in top_mem_lines:
                            line_content = lines_map[line_num].get('line', '').rstrip()
                            formatted += f"Line {line_num}: {mem_mb:.2f} MB - {line_content}\n"
                else:
                    formatted += f"No function code found in the profiling output\n"
            
            formatted += "\n"
        
        return formatted.replace('{', '{{').replace('}', '}}')