from .base_agent import BaseAgent
import re
import os

class CodeOptimizer(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert code optimization agent specialized in improving both correctness and efficiency of code, 
        particularly for data science tasks dealing with large datasets. 
        Your goal is to analyze code and suggest improvements based on test results, error messages, and profiling data."""
        super().__init__(config, "CodeOptimizer", system_prompt)
    
    def process(self, input_data):
        """Optimize the given code for correctness or efficiency based on phase"""
        code = input_data["code"]
        problem_description = input_data.get("problem_description", "")
        test_results = input_data.get("test_results", [])
        phase = input_data.get("phase", "optimization")  # 'correctness' or 'optimization'
        profiling_data = input_data.get("profiling_data", {})
        
        return self._optimize_performance(code, problem_description, test_results, profiling_data)
    
    def _optimize_performance(self, code, problem_description, test_results, profiling_data):
        """Optimize code for performance based on profiling data"""
        template = """
        Problem Description: {problem_description}
        
        Current Code:
        ```python
        {code}
        ```
        
        Line-by-Line Profiling:
        {line_profiling}
        
        Overall Performance Metrics:
        {overall_metrics}
        
        Provide an optimized version of this code that improves efficiency while maintaining functionality.
        Focus specifically on the lines identified as bottlenecks in the profiling data.
        
        Optimization strategies to consider:
        1. Algorithmic improvements for hot spots identified in profiling
        2. More efficient data structures for frequently accessed data
        3. Vectorization opportunities using NumPy/Pandas
        4. Reducing unnecessary computations or memory allocations
        5. Avoiding repeated calculations by caching results
        
        Return your answer in the following format:
        
        OPTIMIZED_CODE:
        ```python
        # Your optimized code here
        ```
        
        OPTIMIZATION_RATIONALE:
        Briefly explain the key optimizations you made and why they improve performance.
        """
        # Format profiling data for template
        line_profiling = self._format_line_profiling(profiling_data.get("line_profiling", []))
        # print(line_profiling)
        overall_metrics = self._format_overall_metrics(profiling_data.get("overall_metrics", {}))
        parser = self._extract_code_parser()
        
        chain = self._create_chain(template=template)
        
        result = chain.invoke({
            "problem_description": problem_description,
            "code": code,
            "line_profiling": line_profiling,
            "overall_metrics": overall_metrics
        }
        )
        
        return {
            "original_code": code,
            "optimized_code": result['code'],
            "phase": "optimization",
            "profiling_data": profiling_data,
            "metadata": {
                "agent": self.name
            }
        }
    
    def _format_test_results(self, test_results):
        """Format test results for inclusion in prompt"""
        formatted = ""
        for i, test in enumerate(test_results):
            formatted += f"Test {i+1}: {test.get('test_case', 'Unnamed test')}\n"
            formatted += f"Execution Time: {test.get('execution_time', 'N/A')}\n"
            formatted += f"Memory Usage: {test.get('memory_usage', 'N/A')} MB\n"
            formatted += f"Output: {test.get('output', 'N/A')}\n"
            formatted += f"Success: {test.get('success', False)}\n"
            if not test.get('success', False):
                formatted += f"Error: {test.get('error', 'Unknown error')}\n"
            formatted += "\n"
        return formatted
    
    def _format_line_profiling(self, line_profiling):
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
                
                # Find the evaluate_rule function definition line
                function_start_line = None
                function_end_line = None
                
                # First pass: find function boundaries
                line_numbers = sorted(lines_map.keys())
                for line_num in line_numbers:
                    line_data = lines_map[line_num]
                    content = line_data.get('line', '')
                    
                    # Look for function definition
                    if content.startswith("def evaluate_rule(") or "def evaluate_rule(" in content:
                        function_start_line = line_num
                    # Once we found start, look for lines outside the function indentation
                    elif function_start_line is not None and function_end_line is None:
                        # Check if this line is outside function (no indentation)
                        if content and not content.startswith(" ") and not content.startswith("\t"):
                            function_end_line = line_num - 1
                            break
                
                # Handle case where function continues until end of file
                if function_start_line is not None and function_end_line is None:
                    function_end_line = max(line_numbers) if line_numbers else 0
                
                # Format only the function's line-by-line metrics
                if function_start_line is not None:
                    formatted += f"Function definition found at lines {function_start_line}-{function_end_line}\n"
                    formatted += "Line-by-line profiling (function only):\n"
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
                    
                    # Add summary of hotspots within the function only
                    formatted += "\nHotspots (within function only):\n"
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
                    formatted += "Function 'evaluate_rule' not found in the profiling output\n"
            
            formatted += "\n"
        
        return formatted
    
    def _format_overall_metrics(self, overall_metrics):
        """Format overall metrics for inclusion in prompt"""
        formatted = ""
        for test_name, metrics in overall_metrics.items():
            formatted += f"Test: {test_name}\n"
            formatted += f"Execution Time: {metrics.get('execution_time', 'N/A')} seconds\n"
            formatted += f"Memory Usage: {metrics.get('memory_usage', 'N/A')} MB\n\n"
        return formatted