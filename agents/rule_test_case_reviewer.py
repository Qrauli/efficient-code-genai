from .base_agent import BaseAgent
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Define Pydantic models for structured output validation
class CorrectedTestCaseValues(BaseModel):
    support: Optional[float] = None
    confidence: Optional[float] = None
    satisfactions_indexes: Optional[List[int]] = None
    violations_indexes: Optional[List[int]] = None
    satisfactions_str: Optional[str] = None
    violations_str: Optional[str] = None

class CorrectedTestCase(BaseModel):
    test_case_index: int
    corrected_values: CorrectedTestCaseValues
    explanation: str

class ReviewAnalysisOutput(BaseModel):
    code_fix_approach: Optional[str] = Field(None, description="Detailed description of code problems and approach to fixing the code if needed")
    corrected_test_cases: Optional[List[CorrectedTestCase]] = Field(None, description="List of test cases with corrected values if test cases were incorrect")
    fix_code: bool = Field(False, description="Indicates if the code needs fixing")
    fix_test_cases: bool = Field(False, description="Indicates if the test cases need fixing")

class RuleTestCaseReviewer(BaseAgent):
    def __init__(self, config):
        system_prompt = """
You are an expert reviewer that analyzes the correctness of code implementations and test cases 
for data quality rules. You identify why implementations fail test cases, and determine whether the issue is in 
the code or in the test case itself.
"""
        super().__init__(config, "RuleTestCaseReviewer", system_prompt)
    
    def process(self, rule_description, code, test_results, rule_format, dataframe_info=None):
        """Review failing test cases and assess whether the implementation or test cases are incorrect
        
        Args:
            rule_description (str): Description of the rule being implemented
            code (str): Current implementation of the rule function
            test_results (list): Results from test case runs that failed
            rule_format (str): Format specification for the rule output
            dataframe_info (str, optional): DataFrame sample information
            
        Returns:
            dict: Analysis of failure causes and recommendations for fixing either code or test cases
        """
        template = """
# Rule Description
{rule_description}

# Rule Format Specification
{rule_format}

# Current Code Implementation
```python
{code}
```

# Failed Test Case Results
{test_results}

# Task
Carefully analyze the provided rule implementation, test cases, and test failures. Your job is to:

1. Determine whether the failure is due to issues in the code implementation or in the test cases themselves
2. Provide detailed analysis of the specific issues found
3. Recommend how to fix the problems
4. If the test case appears to be incorrect, provide corrected expected values

## Analysis Steps:
1. Identify what the rule is supposed to check and what its outputs should be
2. Review the implementation logic for correctness
3. Manually trace through each test case with the provided code to find discrepancies
4. Check whether the expected outputs in the test cases match the rule requirements
5. Determine if there are edge cases the test cases or implementation aren't handling properly

## Focus on:
- Correct calculation of support and confidence values
- Accurate identification of satisfactions and violations
- Proper structure of the output dictionaries
- Edge cases involving empty values, special conditions, or corner cases

Please provide your analysis in the following structured format:

```json
{{
    "code_fix_approach": "Detailed description of code problems and approach to fixing the code if needed",
    "corrected_test_cases": [
        {{
            "test_case_index": 0,
            "explanation": "Consise and clear explanation of why these values are expected detailing why certain rows are included in satisfactions or violations according to the rule",
            "corrected_values": {{
                "support": 0.X,
                "confidence": 0.Y,
                "satisfactions_str": "string representation of the satisfactions structure",
                "violations_str": "string representation of the violations structure"
            }}
        }},
        ...
    ],
    "fix_code": true|false // Indicates if the code needs fixing
    "fix_test_cases": true|false // Indicates if the test cases need fixing
}}
```

IMPORTANT: 
- Be thorough in your analysis - trace through the code execution step by step
- If you determine test cases are incorrect, always provide corrected values with detailed calculations
- The primary issue location should be your overall assessment of where the main problem lies
- Don't guess - analyze the code execution carefully to identify actual issues
- Only include "code_fix_approach" if you found actual problems in the code
- Only include "corrected_test_cases" if you found actual problems in the test cases
- If either "code_fix_approach" or "corrected_test_cases" would be empty, omit that key entirely from your response
- At least one of "fix_code" or "fix_test_cases" must be true since the test cases are failing
- The indexes present in satisfactions_str and satisfactions_indexes should match, and the same for violations_str and violations_indexes
"""
        
        # Format the test results for inclusion in the template
        formatted_test_results = self._format_test_results(test_results)
        
        # Create a JSON output parser
        json_parser = JsonOutputParser()
        
        chain = self._create_chain(
            template=template,
            parser=json_parser,
            run_name="RuleTestCaseReviewer"
        )
        
        result = chain.invoke({
            "rule_description": rule_description,
            "rule_format": rule_format,
            "code": code,
            "test_results": formatted_test_results,
            "dataframe_info": dataframe_info or ""
        })
        
        # Clean up the result structure to remove empty arrays
        cleaned_result = {}
        
        # Only include non-empty fields
        if result and "code_fix_approach" in result and result["code_fix_approach"]:
            cleaned_result["code_fix_approach"] = result["code_fix_approach"]
        
        if result and "corrected_test_cases" in result and result["corrected_test_cases"]:
            cleaned_result["corrected_test_cases"] = result["corrected_test_cases"]
            
        if result and "fix_code" in result:
            cleaned_result["fix_code"] = result["fix_code"]
            
        if result and "fix_test_cases" in result:
            cleaned_result["fix_test_cases"] = result["fix_test_cases"]
        
        return {
            "analysis": cleaned_result,
            "metadata": {
                "agent": self.name,
                "rule_description": rule_description
            }
        }
    
    def _format_test_results(self, test_results):
        """Format test results for inclusion in template
        
        Args:
            test_results (list): List of test result objects
            
        Returns:
            str: Formatted test results
        """
        if not isinstance(test_results, list):
            test_results = [test_results]
            
        formatted = ""
        for i, result in enumerate(test_results): 
            if result.get("success", False):
                continue  # Skip successful test cases

            formatted += f"## test case index: {result.get('test_case_index')}\n\n"
            """
            # Add test dataframe
            test_data = result.get("test_data", {})
            if test_data:
                formatted += "### Test DataFrame:\n```\n"
                import pandas as pd
                import json
                try:
                    # Try to format as a DataFrame
                    df_str = str(pd.DataFrame(test_data))
                    formatted += df_str
                except:
                    # If that fails, just use the raw data
                    formatted += json.dumps(test_data, indent=2)
                formatted += "\n```\n\n"
            
            # Add expected output
            if "comparison" in result:
                comparison = result.get("comparison", {})
                formatted += "### Expected Output:\n"
                formatted += f"- Support: {comparison.get('support', {}).get('expected')}\n"
                formatted += f"- Confidence: {comparison.get('confidence', {}).get('expected')}\n"
                formatted += f"- Satisfactions Indexes: {comparison.get('satisfactions', {}).get('expected_indexes')}\n"
                formatted += f"- Violations Indexes: {comparison.get('violations', {}).get('expected_indexes')}\n\n"
                
                # Add actual output
                formatted += "### Actual Output:\n"
                formatted += f"- Support: {comparison.get('support', {}).get('actual')}\n"
                formatted += f"- Confidence: {comparison.get('confidence', {}).get('actual')}\n"
                formatted += f"- Satisfactions Indexes: {comparison.get('satisfactions', {}).get('actual_indexes')}\n"
                formatted += f"- Violations Indexes: {comparison.get('violations', {}).get('actual_indexes')}\n\n"
                
                # Add comparison results
                formatted += "### Comparison Results:\n"
                formatted += f"- Support Match: {comparison.get('support', {}).get('match')}\n"
                formatted += f"- Confidence Match: {comparison.get('confidence', {}).get('match')}\n"
                formatted += f"- Satisfactions Match: {comparison.get('satisfactions', {}).get('match')}\n"
                formatted += f"- Violations Match: {comparison.get('violations', {}).get('match')}\n\n"
            """
            # Add error message if present
            if result.get("error"):
                formatted += "### Error Message:\n"
                formatted += result.get("error", "No specific error message") + "\n\n"
            
            # Add execution time
            # formatted += f"### Execution Time: {result.get('execution_time', 'N/A')} seconds\n\n"
            
            # Add separator between test cases
            formatted += "-" * 10 + "\n\n"
            
        return formatted