from .orchestrator import Orchestrator
from .rule_function_generator import RuleFunctionGenerator
from .code_optimizer import CodeOptimizer
import pandas as pd

class RuleOrchestrator(Orchestrator):
    def __init__(self, config):
        super().__init__(config)
        self.rule_function_generator = RuleFunctionGenerator(config)
        self.code_optimizer = CodeOptimizer(config)
    
    def process_rule(self, rule_description, dataframe: pd.DataFrame, sample_size=None):
        """Process a rule description to generate and optimize a function for DataFrame rule evaluation"""
        results_history = []
        
        # Use a sample of the dataframe for development if specified
        working_df = dataframe
        if sample_size and len(dataframe) > sample_size:
            working_df = dataframe.sample(sample_size, random_state=42)
        
        # Phase 1: Generate initial function
        generator_result = self.rule_function_generator.process(rule_description, example_dataframe=working_df)
        current_code = generator_result["code"]
        results_history.append({"step": "generation", "result": generator_result})
        
        # Phase 2: Test function execution and collect performance metrics in one step
        test_result = self.rule_function_generator.test_function(current_code, working_df)
        results_history.append({"step": "initial_testing", "result": test_result})
        
        # Phase 3: Optimization loop
        iterations = 0
        previous_code = None
        previous_metrics = None
        
        while iterations < self.max_iterations:
            # Skip if initial testing failed
            if not test_result.get("success", False):
                # Attempt to fix the code first
                corrected_result = self.code_tester.correct_code(
                    code=current_code,
                    problem_description=f"DataFrame rule evaluation function for: {rule_description}",
                    test_results=[test_result]
                )
                results_history.append({"step": f"refinement_{iterations}_correction", "result": corrected_result})
                
                # Update code and test again
                current_code = corrected_result.get("corrected_code", current_code)
                test_result = self.rule_function_generator.test_function(current_code, working_df)
                results_history.append({"step": f"refinement_{iterations}_retest", "result": test_result})
                
                iterations += 1
                continue
                
            # Extract current metrics
            # current_metrics = self._extract_rule_metrics(test_result)
            
            # Check if we should terminate optimization
            # if self._should_terminate_rule_optimization(previous_code, current_code, previous_metrics, current_metrics):
            #    break
            
            # Perform optimization with profiling information from the test_result
            optimizer_input = {
                "code": current_code,
                "problem_description": f"Optimize the DataFrame rule evaluation function for: {rule_description}",
                "profiling_data": {
                    "overall_metrics": {
                        "test": {
                            "execution_time": test_result.get("execution_time"),
                            "function_results": test_result.get("function_results", {})
                        }
                    },
                    "line_profiling": [{"test_case": "rule_evaluation", "profile_data": test_result.get("profile_data", {})}]
                },
                "phase": "optimization"
            }
            
            optimizer_result = self.code_optimizer.process(optimizer_input)
            results_history.append({"step": f"refinement_{iterations}_optimization", "result": optimizer_result})
            
            # Update and retest the optimized code
            previous_code = current_code
            # previous_metrics = current_metrics
            current_code = optimizer_result.get("optimized_code", current_code)
            
            # Test the updated code
            test_result = self.rule_function_generator.test_function(current_code, working_df)
            results_history.append({"step": f"refinement_{iterations}_validation", "result": test_result})
            
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
        
        return {
            "code": current_code,
            "summary": summary,
            "results_history": results_history,
            "final_metrics": self._extract_rule_metrics(final_test_result)
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
            support = function_results.get("support", "N/A")
            confidence = function_results.get("confidence", "N/A")
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