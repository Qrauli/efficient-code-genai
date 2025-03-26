from typing import List, TypedDict, Any
from .base_agent import BaseAgent, common_mistakes_prompt

import sys
sys.path.append("..")
from utils.code_execution import execute_code, indent_code
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

class TestCase(TypedDict):
    name: str
    function_call: str
    expected_output: Any
    description: str
    
class RuleCodeTester(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert code correction agent specialized in fixing the correctness and efficiency of DataFrame rule evaluation functions.
        Your goal is to fix bugs and validate that rule evaluation code returns the expected output format."""
        super().__init__(config, "CodeTester", system_prompt)
    
    def process(self, input_data):
        """Test the given code for correctness and performance"""
        code = input_data["code"]
        problem_description = input_data.get("problem_description", "")
        
        # Check if test cases are provided, otherwise generate them
        if "test_cases" in input_data and input_data["test_cases"]:
            test_cases = input_data["test_cases"]
        else:
            # Generate test cases with both problem description AND code
            test_cases = self._generate_test_cases(problem_description, code)
        
        # Execute code with test cases
        test_results = self._execute_test_cases(code, test_cases)
        
        # Analyze test results
        analysis = self._analyze_test_results(code, test_results, problem_description)
        
        return {
            "code": code,
            "test_results": test_results,
            "analysis": analysis,
            "metadata": {
                "agent": self.name
            }
        }
        
    def correct_code(self, code, problem_description, test_results, dataframe_info, function_name="execute_rule", rule_format=None):
        """Fix code to make it pass all tests"""
        # Extract failing tests and their error messages
        failing_tests = [tr.get("error", "") for tr in test_results if not tr.get("success", False)]
        
        # Format the rule format information if available
        format_guidance = ""
        if rule_format:
            # Handle rule_format as text instead of dictionary
            format_guidance = f"""
# Required Output Format
When fixing the code, ensure it maintains this exact return structure:

{rule_format}
"""
        escaped_format_guidance = format_guidance.replace("{", "{{").replace("}", "}}")
        template = f"""
Problem Description: {problem_description}

Current Code:
```python
{code.replace('{', '{{').replace('}', '}}')}
```

Errors:
{failing_tests}

DataFrame Structure:
{dataframe_info}

# Task
Fix the code to make all tests pass. Focus only on correctness for now, not optimization.
The goal is to make the code work correctly according to the requirements.

{escaped_format_guidance}

Common Issues to Check:
1. Incorrect column names or DataFrame access patterns
2. Issues handling missing values (NaN)
3. Type errors (mixing numeric/string operations)
4. Incorrect logical conditions
5. Improper handling of the return format 

Your response should ONLY contain the python code and nothing else.
ALWAYS wrap your code in ```python and ``` markers.

{common_mistakes_prompt()}
"""
        chain = self._create_chain(
            template=template       
        ) 
        result = chain.invoke({
            "problem_description": problem_description,
            "failing_tests": failing_tests,
            "dataframe_info": dataframe_info
        })
        
        return {
            "original_code": code,
            "corrected_code": result['code'],
            "metadata": {
                "agent": self.name,
                "action": "code_correction"
            }
        }