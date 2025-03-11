from .base_agent import BaseAgent
import re

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
        
        if phase == "correctness":
            return self._fix_correctness(code, problem_description, test_results)
        else:
            return self._optimize_performance(code, problem_description, test_results, profiling_data)
    
    def _fix_correctness(self, code, problem_description, test_results):
        """Fix code to make it pass all tests"""
        # Extract failing tests and their error messages
        failing_tests = [tr for tr in test_results if not tr.get("success", False)]
        
        template = """
        Problem Description: {problem_description}
        
        Current Code:
        ```python
        {code}
        ```
        
        Failing Tests:
        {failing_tests}
        
        Fix the code to make all tests pass. Focus only on correctness for now, not optimization.
        The goal is to make the code work correctly according to the requirements.
        
        Return your answer in the following format:
        
        Your response should ONLY contain the python code and nothing else.
        ALWAYS wrap your code in ```python and ``` markers.
        """
        
        parser = self._extract_code_parser()
        
        chain = self._create_chain(
            template=template       
        ) 
        result = chain.invoke({
            "problem_description": problem_description,
            "code": code,
            "failing_tests": self._format_test_results(failing_tests)
        }
        )
        
        return {
            "original_code": code,
            "optimized_code": result['code'],
            "phase": "correctness",
            "metadata": {
                "agent": self.name
            }
        }
    
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
        """Format line profiling data for inclusion in prompt"""
        formatted = ""
        for profile in line_profiling:
            formatted += f"Test: {profile.get('test_case', 'Unknown')}\n"
            if "error" in profile:
                formatted += f"Error: {profile['error']}\n"
            else:
                formatted += f"Profile Output:\n{profile.get('profile_output', '')}\n"
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