from .base_agent import BaseAgent, common_improvement_recommendations, common_mistakes_prompt
import re

class RuleCodeReviewer(BaseAgent):
    def __init__(self, config):
        system_prompt = """
You are an expert code reviewing agent specialized in analyzing DataFrame rule evaluation functions on their efficiency.
Your goal is to identify potential improvements and determine if further optimization is worthwhile.
        
Keep in mind that the dataframes can be large and performance is critical. 
Focus on improving time complexity, memory usage, and overall performance. Execution time is the most critical factor even if it means sacrificing memory usage.
        
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
        profiling_data = input_data.get("profiling_data", {})
        
        return self._review_code(code, previous_code, problem_description, test_result, dataframe_info, profiling_data)
    
    def _review_code(self, code, previous_code, problem_description, test_result, dataframe_info, profiling_data):
        
        # Extract metrics from test result
        line_profiling = profiling_data.get("line_profiling", [])
        
        # prev_code = previous_code.replace('{', '{{').replace('}', '}}') if previous_code else "No previous version available"
        """
        # Previous Version (if applicable)
        ```python
        {previous_code}
        ```
        """
        
        """Review the code and provide recommendations for improvement"""
        template = """
# Problem Description
Thoroughly review this DataFrame rule evaluation function and answer if the code is efficiently implemented? Is it using optimal pandas/numpy techniques? Are there any performance bottlenecks or inefficient operations that could be improved?

{problem_description}

# DataFrame Information
{dataframe_info}

# Current Code
```python
{code}
```

# Line-by-Line Profiling:
{line_profiling}

Based on your analysis, provide:
1. A list of specific improvement recommendations (if any, most important first)
2. Your assessment of whether further optimization would yield significant benefits
3. A final recommendation: "CONTINUE OPTIMIZATION" or "TERMINATE OPTIMIZATION"

{common_improvement_recommendations}

Format your response as follows:

CODE_ANALYSIS:
[Your detailed code analysis here]

IMPROVEMENT_RECOMMENDATIONS:
- [Recommendation 1]
- [Recommendation 2]
- etc.

OPTIMIZATION_POTENTIAL: [High/Medium/Low/None]

FINAL_RECOMMENDATION: [CONTINUE OPTIMIZATION/TERMINATE OPTIMIZATION]
"""
        
        
        chain = self._create_chain(
            template=template, 
            parse_with_prompt=False,
            run_name="RuleCodeReviewer"
        )
        result = chain.invoke({
            "problem_description": problem_description,
            "code": code,
            "dataframe_info": dataframe_info,
            "line_profiling": line_profiling,
            "common_improvement_recommendations": common_improvement_recommendations()
        })
        
        # Parse the review result to extract structured information
        review_text = result.get('content', result) if isinstance(result, dict) else result
        
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
            # Split by bullet points and properly handle each recommendation
            raw_recommendations = recommendations_text.split('\n-')
            recommendations = []
            
            for i, rec in enumerate(raw_recommendations):
                if not rec.strip():
                    continue
                    
                # Handle the first item which might not start with a dash
                if i == 0 and not rec.strip().startswith('-'):
                    recommendations.append(rec.strip())
                else:
                    recommendations.append(rec.strip())
            
            # Clean up any leading dashes
            recommendations = [rec[1:].strip() if rec.startswith('-') else rec for rec in recommendations]
            # Filter out empty recommendations
            recommendations = [rec for rec in recommendations if rec]
            
            analysis["improvement_recommendations"] = recommendations
        
        # Extract optimization potential
        potential_match = re.search(r'OPTIMIZATION_POTENTIAL:\s*(High|Medium|Low|None)', review_text)
        if potential_match:
            analysis["optimization_potential"] = potential_match.group(1)
        
        # Extract final recommendation
        recommendation_match = re.search(r'FINAL_RECOMMENDATION:\s*(CONTINUE OPTIMIZATION|TERMINATE OPTIMIZATION)', review_text)
        if recommendation_match:
            analysis["final_recommendation"] = recommendation_match.group(1)
        
        return analysis