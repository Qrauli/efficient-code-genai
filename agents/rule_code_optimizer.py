from .base_agent import BaseAgent, common_mistakes_prompt
import re
import os

class RuleCodeOptimizer(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert code optimization agent specialized in improving both correctness and efficiency of DataFrame rule evaluation code.
        Your goal is to analyze and optimize code that processes large datasets (potentially millions of rows)."""
        super().__init__(config, "CodeOptimizer", system_prompt)
    
    def process(self, input_data):
        """Optimize the given code for correctness or efficiency based on phase"""
        code = input_data["code"]
        problem_description = input_data.get("problem_description", "")
        test_results = input_data.get("test_results", [])
        profiling_data = input_data.get("profiling_data", {})
        dataframe_info = input_data.get("dataframe_info", "")
        review_feedback = input_data.get("review_feedback", [])
        rule_format = input_data.get("rule_format", None)
        
        return self._optimize_performance(code, problem_description, profiling_data, dataframe_info, review_feedback, rule_format)
    
    def _optimize_performance(self, code, problem_description, profiling_data, dataframe_info, 
                              review_feedback, rule_format=None, retrieval_context=""):
        """Optimize code for performance based on profiling data"""
        
        # Format profiling data for template
        line_profiling = profiling_data.get("line_profiling", [])
        # Format reviewer feedback if available
        if review_feedback:
            review_feedback_text = "Reviewer suggested these improvements:\n" + "\n".join(f"- {rec.replace('{', '{{').replace('}', '}}')}" for rec in review_feedback)
        else:
            review_feedback_text = "No specific improvement recommendations from reviewer."
            
        # Format the rule format information if available
        format_guidance = ""
        if rule_format:
            # Handle rule_format as text instead of dictionary
            format_guidance = f"""
# Required Output Format
The optimized code must maintain this exact output structure:

{rule_format}
"""
        escaped_format_guidance = format_guidance.replace("{", "{{").replace("}", "}}")
        template = f"""
# Task
Optimize the efficiency of the provided rule evaluation function to improve performance while maintaining its behavior. 
Look at the bottleneck operations identified in the profiling data and address the reviewer recommendations given.

# Rule Description : {problem_description}

# DataFrame Information:
{dataframe_info}

# Current Code:
```python
{code.replace('{', '{{').replace('}', '}}')}
```

# Line-by-Line Profiling:
{line_profiling}

# Reviewer Recommendations:
{review_feedback_text}

{retrieval_context.replace('{', '{{').replace('}', '}}')}

{escaped_format_guidance}

Return your answer in the following format:

OPTIMIZED_CODE:
```python
# Your optimized code here
```
"""
        chain = self._create_chain(template=template)
        result = chain.invoke({
            "problem_description": problem_description,
            "line_profiling": line_profiling,
            "dataframe_info": dataframe_info,
            "review_feedback": review_feedback_text,
            "retrieval_context": retrieval_context or "",
            "format_guidance": format_guidance
        })
        
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
    
