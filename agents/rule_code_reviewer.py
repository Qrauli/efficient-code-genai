from .base_agent import BaseAgent
import re

class RuleCodeReviewer(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert code reviewing agent specialized in analyzing DataFrame rule evaluation functions.
        Your goal is to thoroughly assess code quality, identify potential improvements, and determine if further optimization is worthwhile.
        
        You understand that rule evaluation requires:
        1. Compute Support: The proportion of rows where the body of the rule is satisfied. If the rule has no body, support is 1.
        2. Compute Confidence: The proportion of rows where both the body and head of the rule are satisfied, out of the rows where the body is satisfied.
        3. Efficiently identifying violating or satisfying rows
        4. Returning a tuple of (support, confidence, satisfying_indexes, violation_indexes)
        
        You evaluate code based on:
        1. Correctness - Does the function correctly implement the rule logic?
        2. Efficiency - Is the implementation using optimal pandas/numpy techniques?
        3. Readability - Is the code well-structured and appropriately commented?
        4. Robustness - Does it handle edge cases like missing values properly?
        
        Based on your analysis, you'll recommend whether to:
        - Terminate optimization (if code is already well-optimized or further improvements would be minimal)
        - Continue optimization (suggesting specific improvement areas)
        """
        super().__init__(config, "CodeReviewer", system_prompt)
    
    def process(self, input_data):
        """Review the code and determine if further optimization is needed"""
        code = input_data["code"]
        problem_description = input_data.get("problem_description", "")
        previous_code = input_data.get("previous_code")
        test_result = input_data.get("test_result", {})
        dataframe_info = input_data.get("dataframe_info", "")
        
        return self._review_code(code, previous_code, problem_description, test_result, dataframe_info)
    
    def _review_code(self, code, previous_code, problem_description, test_result, dataframe_info):
        """Review the code and provide recommendations for improvement"""
        template = """
# Problem Description
{problem_description}

# DataFrame Information
{dataframe_info}

# Current Code
```python
{code}
```

# Previous Version (if applicable)
```python
{previous_code}
```

# Performance Metrics
- Execution Time: {execution_time}
- Memory Usage: {memory_usage}
- Function Results: {function_results}

Thoroughly review this DataFrame rule evaluation function and answer the following questions:

1. Is the code correct? Does it properly implement the rule logic?
2. Is the code efficiently implemented? Is it using optimal pandas/numpy techniques?
3. Are there any performance bottlenecks or inefficient operations that could be improved?
4. Is the code clear, well-structured, and appropriately commented?
5. Does it handle edge cases like missing values properly?

Based on your analysis, provide:
1. A list of specific improvement recommendations (if any)
2. Your assessment of whether further optimization would yield significant benefits
3. A final recommendation: "CONTINUE OPTIMIZATION" or "TERMINATE OPTIMIZATION"

Format your response as follows:

CODE_ANALYSIS:
[Your detailed code analysis here]

IMPROVEMENT_RECOMMENDATIONS:
- [Recommendation 1]
- [Recommendation 2]
- etc.

OPTIMIZATION_POTENTIAL: [High/Medium/Low/None]

FINAL_RECOMMENDATION: [CONTINUE OPTIMIZATION/TERMINATE OPTIMIZATION]

REASONING:
[Your reasoning for the final recommendation]
"""
        
        # Extract metrics from test result
        execution_time = test_result.get("execution_time", "N/A")
        memory_usage = test_result.get("memory_usage", "N/A")
        function_results = test_result.get("function_results", {})
        
        # For previous_code, handle the case where it might be None
        prev_code_display = previous_code if previous_code else "No previous version available"
        
        chain = self._create_chain(template=template)
        
        result = chain.invoke({
            "problem_description": problem_description,
            "dataframe_info": dataframe_info,
            "code": code,
            "previous_code": prev_code_display,
            "execution_time": execution_time,
            "memory_usage": memory_usage,
            "function_results": function_results
        })
        
        # Parse the review result to extract structured information
        review_text = result.get('code', result) if isinstance(result, dict) else result
        
        # Extract recommendations and final decision
        analysis = self._extract_review_sections(review_text)
        
        return {
            "code": code,
            "review": review_text,
            "continue_optimization": analysis.get("final_recommendation") == "CONTINUE OPTIMIZATION",
            "optimization_potential": analysis.get("optimization_potential", "Unknown"),
            "improvement_recommendations": analysis.get("improvement_recommendations", []),
            "metadata": {
                "agent": self.name
            }
        }
    
    def _extract_review_sections(self, review_text):
        """Extract structured data from the review text"""
        analysis = {}
        
        # Extract code analysis
        code_analysis_match = re.search(r'CODE_ANALYSIS:\s*(.*?)(?:\n\s*IMPROVEMENT_RECOMMENDATIONS:|\Z)', review_text, re.DOTALL)
        if code_analysis_match:
            analysis["code_analysis"] = code_analysis_match.group(1).strip()
        
        # Extract improvement recommendations
        recommendations_match = re.search(r'IMPROVEMENT_RECOMMENDATIONS:\s*(.*?)(?:\n\s*OPTIMIZATION_POTENTIAL:|\Z)', review_text, re.DOTALL)
        if recommendations_match:
            recommendations_text = recommendations_match.group(1).strip()
            # Split by bullet points
            recommendations = [rec.strip()[2:].strip() for rec in recommendations_text.split('\n-') if rec.strip()]
            if recommendations and not recommendations[0].startswith('-'):
                recommendations[0] = recommendations[0][1:].strip() if recommendations[0].startswith('-') else recommendations[0]
            analysis["improvement_recommendations"] = recommendations
        
        # Extract optimization potential
        potential_match = re.search(r'OPTIMIZATION_POTENTIAL:\s*(High|Medium|Low|None)', review_text)
        if potential_match:
            analysis["optimization_potential"] = potential_match.group(1)
        
        # Extract final recommendation
        recommendation_match = re.search(r'FINAL_RECOMMENDATION:\s*(CONTINUE OPTIMIZATION|TERMINATE OPTIMIZATION)', review_text)
        if recommendation_match:
            analysis["final_recommendation"] = recommendation_match.group(1)
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?:\Z)', review_text, re.DOTALL)
        if reasoning_match:
            analysis["reasoning"] = reasoning_match.group(1).strip()
        
        return analysis