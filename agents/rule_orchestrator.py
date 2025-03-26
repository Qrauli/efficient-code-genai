from .base_agent import _create_dataframe_sample
from .orchestrator import Orchestrator
from .rule_function_generator import RuleFunctionGenerator
from .rule_code_optimizer import RuleCodeOptimizer
from .rule_code_tester import RuleCodeTester
from .rule_code_reviewer import RuleCodeReviewer
from .rule_format_analyzer import RuleFormatAnalyzer  # Add this import
import pandas as pd

# Add imports
from utils.context_retrieval import ContextRetriever, RetrievalSource

class RuleOrchestrator(Orchestrator):
    def __init__(self, config, use_retrieval=None):
        super().__init__(config)
        self.rule_function_generator = RuleFunctionGenerator(config)
        self.code_optimizer = RuleCodeOptimizer(config)
        self.code_tester = RuleCodeTester(config)
        self.code_reviewer = RuleCodeReviewer(config)
        self.rule_format_analyzer = RuleFormatAnalyzer(config)  # Initialize the format analyzer
        
        # Initialize retrieval if enabled
        self.use_retrieval = use_retrieval if use_retrieval is not None else config.ENABLE_RETRIEVAL
        self.retriever = None
        
        if self.use_retrieval:
            self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the context retriever with default sources"""
        self.retriever = ContextRetriever(self.config)
        
        # Add default sources from config
        for source_config in self.config.DEFAULT_RETRIEVAL_SOURCES:
            source = RetrievalSource(**source_config)
            self.retriever.add_source(source)
            
        # Initialize the retriever
        self.retriever.initialize()
    
    def process_rule(self, rule_description, dataframe: pd.DataFrame, rule_id=None, sample_size=None):
        """Process a rule description to generate and optimize a function for DataFrame rule evaluation"""
        results_history = []
        
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
        
        # Phase 1: Generate initial function with retrieval context and format specifications
        generator_context = self._get_relevant_context(
            query=f"pandas dataframe function to evaluate rule: {rule_description}",
            source_types=["documentation", "code_snippet"]
        )
        
        generator_result = self.rule_function_generator.process(
            rule_description, 
            df_sample=dataframe_info,
            function_name=function_name,
            context=generator_context,
            rule_format=rule_format
        )
        
        current_code = generator_result["code"]
        results_history.append({"step": "generation", "result": generator_result})
        
        # Phase 2: Test function execution and collect performance metrics in one step
        test_result = self.rule_function_generator.test_function(current_code, working_df, function_name=function_name)
        
        # Adaptive sampling: If test times out, reduce sample size and retry
        if test_result.get("timed_out", False) and current_sample_size > 100:
            # Reduce sample size by half and retry
            current_sample_size = max(100, current_sample_size // 2)
            working_df = dataframe.sample(current_sample_size, random_state=42)
            dataframe_info = _create_dataframe_sample(working_df)
            
            # Log the sample size reduction
            results_history.append({
                "step": "sample_size_reduction", 
                "result": {
                    "previous_size": initial_sample_size,
                    "new_size": current_sample_size,
                    "reason": "Execution timed out with larger sample"
                }
            })
            
            # Retry with smaller sample
            test_result = self.rule_function_generator.test_function(current_code, working_df, function_name=function_name)
        
        test_log = test_result.copy()
        if "function_results" in test_log: del test_log["function_results"]
        if "profile_data" in test_log: del test_log["profile_data"]
        results_history.append({"step": "initial_testing", "result": test_log})
        
        # Phase 3: Optimization loop
        iterations = 0
        previous_code = None
        should_continue = True
        
        while iterations < self.max_iterations and should_continue:
            # Skip if initial testing failed
            if not test_result.get("success", False):
                # Attempt to fix the code first
                corrected_result = self.code_tester.correct_code(
                    code=current_code,
                    problem_description=f"DataFrame rule evaluation function for: {rule_description}",
                    test_results=[test_result],
                    dataframe_info=dataframe_info,
                    function_name=function_name,
                    rule_format=rule_format  # Pass the rule format
                )
                results_history.append({"step": f"refinement_{iterations}_correction", "result": corrected_result})
                
                # Update code and test again
                current_code = corrected_result.get("corrected_code", current_code)
                test_result = self.rule_function_generator.test_function(current_code, working_df, function_name=function_name)
                test_log = test_result.copy()
                if "function_results" in test_log: del test_log["function_results"]
                if "profile_data" in test_log: del test_log["profile_data"]
                results_history.append({"step": f"refinement_{iterations}_retest", "result": test_log})
                
                iterations += 1
                continue
            
            # Add code review step to determine if we should continue optimization
            review_input = {
                "code": current_code,
                "previous_code": previous_code,
                "problem_description": f"DataFrame rule evaluation function for: {rule_description}",
                "test_result": test_result,
                "dataframe_info": dataframe_info,
                "rule_format": rule_format  # Pass the rule format
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
            
            # Add retrieval of relevant optimization techniques
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
            
            # Perform optimization with profiling information and retrieval context
            optimizer_input = {
                "code": current_code,
                "problem_description": f"Optimize the DataFrame rule evaluation function for: {rule_description}",
                "dataframe_info": dataframe_info,
                "profiling_data": {
                    "overall_metrics": {
                        "test": {
                            "execution_time": test_result.get("execution_time"),
                            "function_results": test_result.get("function_results", {})
                        }
                    },
                    "line_profiling": [{"test_case": "rule_evaluation", "profile_data": test_result.get("profile_data", {})}]
                },
                "phase": "optimization",
                "review_feedback": review_result.get("improvement_recommendations", []),
                "retrieval_context": optimization_context,
                "rule_format": rule_format  # Pass the rule format
            }
            
            optimizer_result = self.code_optimizer.process(optimizer_input)
            log_optimizer_result = optimizer_result.copy()
            if "profiling_data" in log_optimizer_result: del log_optimizer_result["profiling_data"]
            results_history.append({"step": f"refinement_{iterations}_optimization", "result": log_optimizer_result})
            
            # Update and retest the optimized code
            previous_code = current_code
            current_code = optimizer_result.get("optimized_code", current_code)
            
            # Test the updated code
            test_result = self.rule_function_generator.test_function(current_code, working_df, function_name=function_name)
            test_log = test_result.copy()
            if "function_results" in test_log: del test_log["function_results"]
            if "profile_data" in test_log: del test_log["profile_data"]
            results_history.append({"step": f"refinement_{iterations}_validation", "result": test_log})
            
            iterations += 1
        
        # Generate final summary
        final_test_result = test_result
        if not final_test_result.get("success", False):
            # If the last test failed, use the last successful test result
            for history_item in reversed(results_history):
                if "result" in history_item and history_item["result"].get("success", False):
                    final_test_result = history_item["result"]
                    current_code = previous_code
                    break
        
        summary = self._generate_rule_summary(current_code, results_history, final_test_result)
        
        # Store the successful code for future retrieval if enabled
        if self.use_retrieval and final_test_result.get("success", False):
            metrics = self._extract_rule_metrics(final_test_result)
            self.retriever.add_generated_code(
                code=current_code,
                metadata={
                    "description": f"Optimized function for rule: {rule_description}",
                    "tags": ["rule_evaluation", "dataframe", "optimized"],
                    "execution_time": metrics.get("execution_time"),
                    "memory_usage": metrics.get("memory_mb"),
                    "support": metrics.get("support"),
                    "confidence": metrics.get("confidence")
                }
            )
        
        return {
            "code": current_code,
            "summary": summary,
            "results_history": results_history,
            "final_metrics": self._extract_rule_metrics(final_test_result),
            "function_name": function_name
        }
    
    def _extract_rule_metrics(self, test_result):
        """Extract comprehensive performance metrics from rule function test result"""
        metrics = {}
        
        if test_result and test_result.get("success", False):
            # Extract functional metrics from function_results
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
    
    def _should_terminate_rule_optimization(self, previous_code, current_code, previous_metrics, current_metrics):
        """Determine if optimization should be terminated based on performance metrics"""
        # If we don't have previous metrics, continue optimization
        if previous_code is None or previous_metrics is None:
            return False
        
        # Code hasn't changed significantly (possible convergence)
        if previous_code == current_code:
            return True
        
        # Check if execution time improved significantly
        if previous_metrics.get("execution_time") and current_metrics.get("execution_time"):
            time_improvement = previous_metrics["execution_time"] - current_metrics["execution_time"]
            relative_improvement = time_improvement / previous_metrics["execution_time"] if previous_metrics["execution_time"] > 0 else 0
            
            # If execution time improved by less than 5%, consider terminating
            if relative_improvement < 0.05:
                # Unless memory usage also improved significantly
                memory_previous = previous_metrics.get("memory_mb", 0)
                memory_current = current_metrics.get("memory_mb", 0)
                memory_improvement = (memory_previous - memory_current) / memory_previous if memory_previous > 0 else 0
                
                if memory_improvement < 0.1:
                    return True
        
        # Continue optimization
        return False
    
    def _generate_rule_summary(self, code, results_history, final_test_result):
        """Generate a summary of the rule function generation and optimization process"""
        if not final_test_result.get("success", False):
            return "Rule function generation failed. See error details in the execution history."
        
        # Extract initial and final metrics if available
        initial_metrics = None
        final_metrics = None
        
        for entry in results_history:
            if "initial_testing" in entry["step"] and entry["result"].get("success", False):
                initial_metrics = {
                    "execution_time": entry["result"].get("execution_time"),
                    "memory_usage": entry["result"].get("memory_usage")
                }
            
            if "full_dataset_validation" in entry["step"] or "refinement_" in entry["step"] and "_testing" in entry["step"]:
                if entry["result"].get("success", False):
                    final_metrics = {
                        "execution_time": entry["result"].get("execution_time"),
                        "memory_usage": entry["result"].get("memory_usage")
                    }
        
        # Use the final test result if we didn't capture the metrics above
        if not final_metrics:
            final_metrics = {
                "execution_time": final_test_result.get("execution_time"),
                "memory_usage": final_test_result.get("memory_usage")
            }
        
        # Calculate improvements if we have both metrics
        if initial_metrics and final_metrics:
            time_improvement = 0
            memory_improvement = 0
            
            # if initial_metrics["execution_time"] > 0:
            #    time_improvement = ((initial_metrics["execution_time"] - final_metrics["execution_time"]) / 
            #                      initial_metrics["execution_time"] * 100)
            
            # if initial_metrics["memory_usage"] > 0:
            #    memory_improvement = ((initial_metrics["memory_usage"] - final_metrics["memory_usage"]) / 
            #                        initial_metrics["memory_usage"] * 100)
                
            # Get rule evaluation metrics
            function_results = final_test_result.get("function_results", {})
            support = function_results.get("support", 0)
            confidence = function_results.get("confidence", 0)
            is_violations = function_results.get("is_violations", False)
            row_indexes_count = function_results.get("row_indexes_count", 0)
            
            return f"""Rule function generation completed successfully.
Support: {support:.4f}
Confidence: {confidence:.4f}
Returned {'violation' if is_violations else 'satisfying'} indexes: {row_indexes_count} rows
Performance: {time_improvement:.2f}% execution time improvement, {memory_improvement:.2f}% memory usage improvement"""
        
        # If we don't have both metrics, just return a simple summary
        function_results = final_test_result.get("function_results", {})
        support = function_results.get("support", "N/A")
        confidence = function_results.get("confidence", "N/A")
        is_violations = function_results.get("is_violations", False)
        row_indexes_count = function_results.get("row_indexes_count", 0)
        
        return f"""Rule function generation completed successfully.
Support: {support}
Confidence: {confidence}
Returned {'violation' if is_violations else 'satisfying'} indexes: {row_indexes_count} rows"""
    
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