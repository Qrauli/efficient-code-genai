from agents.base_agent import BaseAgent
from .code_generator import CodeGenerator
from .code_optimizer import CodeOptimizer
from .code_tester import CodeTester
import tempfile
import os
import subprocess
import json
import sys
sys.path.append("..")
from utils.code_execution import execute_code, indent_code

class Orchestrator:
    def __init__(self, config):
        # Initialize specialized agents
        self.code_generator = CodeGenerator(config)
        self.code_optimizer = CodeOptimizer(config)
        self.code_tester = CodeTester(config)
        
        self.config = config
        self.max_iterations = config.MAX_ITERATIONS
    
    def process(self, problem_description, existing_test_cases=None):
        """Orchestrate the entire code generation workflow with integrated refinement phase
        
        Args:
            problem_description (str): Description of the problem to solve
            existing_test_cases (list, optional): List of pre-defined test cases
        """
        results_history = []
        
        # Phase 1: Generate or process test cases first (to avoid overfitting)
        if existing_test_cases:
            # Use existing test cases but maybe generate additional ones
            test_cases = existing_test_cases
            results_history.append({"step": "test_loading", "result": {"test_results": test_cases}})
            
            # Optionally generate additional test cases for more comprehensive testing
            if self.config.GENERATE_ADDITIONAL_TESTS:
                tester_result = self.code_tester.generate_additional_tests(problem_description, existing_test_cases)
                # Merge existing and additional test cases, avoiding duplicates
                new_tests = tester_result.get("additional_tests", [])
                if new_tests:
                    test_cases.extend(new_tests)
                    results_history.append({"step": "test_enhancement", "result": tester_result})
        else:
            # Generate test cases based only on problem description
            tester_result = self.code_tester.generate_tests(problem_description)
            test_cases = tester_result.get("test_results", [])
            results_history.append({"step": "test_generation", "result": tester_result})
        
        # Phase 2: Initial code generation - now aware of test cases
        generator_result = self.code_generator.process(problem_description, test_cases)
        current_code = generator_result["code"]
        results_history.append({"step": "generation", "result": generator_result})
        
        # Phase 3: Integrated refinement loop (correctness & performance)
        iterations = 0
        previous_code = None
        previous_metrics = None
        
        while iterations < self.max_iterations:
            # Step 1: Test for correctness
            correctness_result = self._evaluate_correctness(current_code, test_cases, problem_description)
            results_history.append({"step": f"refinement_{iterations}_correctness", "result": correctness_result})
            
            # If code doesn't pass all tests, fix correctness issues first
            if not correctness_result["all_tests_passed"]:
                optimizer_input = {
                    "code": current_code,
                    "problem_description": problem_description,
                    "test_results": correctness_result["test_results"],
                    "phase": "correctness"
                }
                optimizer_result = self.code_optimizer.process(optimizer_input)
                results_history.append({"step": f"refinement_{iterations}_fix", "result": optimizer_result})
                current_code = optimizer_result["optimized_code"]
                
                # Skip performance optimization for this iteration, as we need to verify correctness first
                iterations += 1
                continue
            
            # Step 2: Only optimize performance if code is correct
            # Run profiling
            profiling_result = self._run_profiling(current_code, test_cases)
            results_history.append({"step": f"refinement_{iterations}_profiling", "result": profiling_result})
            
            # Store metrics for comparison
            current_metrics = self._extract_metrics(profiling_result)
            
            # Check termination conditions
            if self._should_terminate(previous_code, current_code, previous_metrics, current_metrics):
                break
            
            # Optimize for performance
            optimizer_input = {
                "code": current_code,
                "problem_description": problem_description,
                "test_results": test_cases,
                "profiling_data": profiling_result,
                "phase": "optimization"
            }
            optimizer_result = self.code_optimizer.process(optimizer_input)
            results_history.append({"step": f"refinement_{iterations}_optimization", "result": optimizer_result})
            
            # Update the code and metrics for next iteration
            previous_code = current_code
            previous_metrics = current_metrics
            current_code = optimizer_result["optimized_code"]
            
            iterations += 1
        
        # Final evaluation with profiling
        final_profiling = self._run_profiling(current_code, test_cases)
        results_history.append({"step": "final_evaluation", "result": final_profiling})
        
        # Final correctness check
        final_correctness = self._evaluate_correctness(current_code, test_cases, problem_description)
        results_history.append({"step": "final_correctness", "result": final_correctness})
        
        # Generate summary
        final_result = self._generate_summary(current_code, results_history)
        
        return {
            "final_code": current_code,
            "execution_history": results_history,
            "summary": final_result,
            "all_tests_pass": final_correctness["all_tests_passed"],
            "metadata": {
                "total_iterations": iterations
            }
        }
    
    def _evaluate_correctness(self, code, test_cases, problem_description):
        """Run tests to check code correctness"""
        all_tests_passed = True
        updated_test_results = []
        
        for test_case in test_cases:
            # Create a test wrapper that calls the function and compares the result
            function_call = test_case.get("function_call", "")
            expected_output = test_case.get("expected_output", "")
            indented_code = indent_code(code)
            
            test_wrapper = f"""
import json
try:
    # First execute the original code to define the function
{indented_code}
                
    # Then execute the test call and capture the result
    result = {function_call}
    print(json.dumps({{"actual_output": str(result)}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
            
            # Execute with existing code_execution utility
            result = execute_code(test_wrapper, self.config)
            
            # Parse the output to get the actual result
            success = False
            try:
                output_data = json.loads(result["output"])
                if "error" not in output_data:
                    actual_output = output_data.get("actual_output", "")
                    success = str(actual_output) == str(expected_output)
                else:
                    all_tests_passed = False
            except:
                all_tests_passed = False
                
            updated_test_results.append({
                "test_case": test_case["name"],
                "execution_time": result["execution_time"],
                "memory_usage": result["memory_usage"],
                "output": result["output"],
                "success": success
            })
            
            if not success:
                all_tests_passed = False
        
        return {
            "all_tests_passed": all_tests_passed,
            "test_results": updated_test_results
        }
    
    def _run_profiling(self, code, test_cases):
        """Run detailed line-by-line profiling"""
        profiling_results = {"line_profiling": [], "memory_profiling": [], "overall_metrics": {}}
        
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(code.encode())
        
        try:
            # Line profiling
            for test_case in test_cases:
                function_call = test_case.get("function_call", "")
                if not function_call:
                    continue
                
                # Create a wrapper script for line_profiler
                wrapper_code = f"""
import line_profiler
import sys

# Load the code with functions
with open('{temp_file_path}', 'r') as f:
    code = f.read()
namespace = dict()
exec(code, namespace)

# Extract main function name from the function call
func_name = '{function_call}'.split('(')[0].strip()
func = namespace.get(func_name)

# Set up profiler
profile = line_profiler.LineProfiler(func)
profile.runcall({function_call})
profile.print_stats()
                """
                
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as prof_file:
                    prof_file_path = prof_file.name
                    prof_file.write(wrapper_code.encode())
                
                # Run line profiler
                try:
                    result = subprocess.run(
                        [sys.executable, prof_file_path], 
                        capture_output=True, 
                        text=True,
                        timeout=self.config.TIMEOUT_SECONDS
                    )
                    profiling_results["line_profiling"].append({
                        "test_case": test_case["name"],
                        "profile_output": result.stdout
                    })
                except Exception as e:
                    profiling_results["line_profiling"].append({
                        "test_case": test_case["name"],
                        "error": str(e)
                    })
                
                # Clean up wrapper
                os.unlink(prof_file_path)
            
            # Memory profiling (using memory-profiler if available)
            # Similar approach as line profiling, but with memory_profiler
            
            # Overall execution metrics (using our existing code_execution utility)
            for test_case in test_cases:
                function_call = test_case.get("function_call", "")
                test_wrapper = f"""
                # First execute the original code to define the function
            {code.replace(chr(10), chr(10) + '    ')}
                
                # Then execute the test call
                {function_call}
                """
                
                result = execute_code(test_wrapper, self.config)
                profiling_results["overall_metrics"][test_case["name"]] = {
                    "execution_time": result["execution_time"],
                    "memory_usage": result["memory_usage"]
                }
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        return profiling_results
    
    def _should_terminate(self, previous_code, current_code, previous_metrics, current_metrics):
        """Decide whether to terminate the optimization process"""
        # If code hasn't changed significantly
        if previous_code == current_code:
            return True
        
        # If we don't have previous metrics to compare
        if not previous_metrics or not current_metrics:
            return False
        
        # Check for minimal performance improvement
        time_improvement = 0
        if ("avg_execution_time" in previous_metrics and 
            "avg_execution_time" in current_metrics and 
            previous_metrics["avg_execution_time"] > 0):
            
            time_improvement = ((previous_metrics["avg_execution_time"] - current_metrics["avg_execution_time"]) / 
                            previous_metrics["avg_execution_time"]) * 100
        
        # If improvement is less than threshold (e.g., 5%), terminate
        if time_improvement < 5:
            return True
        
        return False
    
    def _extract_metrics(self, profiling_result):
        """Extract performance metrics from profiling result for comparison"""
        metrics = {}
        
        if profiling_result and "overall_metrics" in profiling_result:
            overall_metrics = profiling_result["overall_metrics"]
            
            if overall_metrics:
                execution_times = []
                memory_usages = []
                
                for test_name, test_metrics in overall_metrics.items():
                    execution_time = test_metrics.get("execution_time")
                    memory_usage = test_metrics.get("memory_usage")
                    
                    if execution_time is not None:
                        execution_times.append(execution_time)
                    if memory_usage is not None:
                        memory_usages.append(memory_usage)
                
                if execution_times:
                    metrics["avg_execution_time"] = sum(execution_times) / len(execution_times)
                if memory_usages:
                    metrics["max_memory_usage"] = max(memory_usages)
        
        return metrics
    
    def _generate_summary(self, final_code, results_history):
        """Generate a summary of the refinement process"""
        # Extract initial and final metrics
        initial_metrics = None
        final_metrics = None
        
        for entry in results_history:
            if "test_generation" in entry["step"]:
                # Get initial metrics
                if "test_results" in entry["result"]:
                    test_results = entry["result"]["test_results"]
                    if test_results:
                        execution_times = [tr.get("execution_time", 0) for tr in test_results if "execution_time" in tr]
                        memory_usages = [tr.get("memory_usage", 0) for tr in test_results if "memory_usage" in tr]
                        if execution_times and memory_usages:
                            initial_metrics = {
                                "avg_execution_time": sum(execution_times) / len(execution_times),
                                "max_memory_usage": max(memory_usages)
                            }
            
            if "final_evaluation" in entry["step"]:
                # Get final metrics
                if "overall_metrics" in entry["result"]:
                    metrics = entry["result"]["overall_metrics"]
                    if metrics:
                        execution_times = [m.get("execution_time", 0) for m in metrics.values()]
                        memory_usages = [m.get("memory_usage", 0) for m in metrics.values()]
                        if execution_times and memory_usages:
                            final_metrics = {
                                "avg_execution_time": sum(execution_times) / len(execution_times),
                                "max_memory_usage": max(memory_usages)
                            }
        
        # Calculate improvements
        if initial_metrics and final_metrics:
            time_improvement = ((initial_metrics["avg_execution_time"] - final_metrics["avg_execution_time"]) / 
                               initial_metrics["avg_execution_time"] * 100) if initial_metrics["avg_execution_time"] > 0 else 0
            memory_improvement = ((initial_metrics["max_memory_usage"] - final_metrics["max_memory_usage"]) / 
                                 initial_metrics["max_memory_usage"] * 100) if initial_metrics["max_memory_usage"] > 0 else 0
                                 
            return f"Code refinement completed with {time_improvement:.2f}% execution time improvement and {memory_improvement:.2f}% memory usage improvement."
        
        return "Code refinement completed successfully."