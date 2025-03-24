from typing import List, TypedDict, Any
from .base_agent import BaseAgent

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
        system_prompt = """You are an expert code testing agent specialized in verifying the correctness and efficiency of DataFrame rule evaluation functions.
        Your goal is to identify bugs and validate that rule evaluation code returns the expected output format (support, confidence, row_indexes, is_violations)."""
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
        
    def correct_code(self, code, problem_description, test_results, dataframe_info):
        """Fix code to make it pass all tests"""
        # Extract failing tests and their error messages
        failing_tests = [tr.get("error", "") for tr in test_results if not tr.get("success", False)]
        template = """
Problem Description: {problem_description}

Current Code:
```python
{code}
```

Errors:
{failing_tests}

DataFrame Structure:
{dataframe_info}

Fix the code to make all tests pass. Focus only on correctness for now, not optimization.
The goal is to make the code work correctly according to the requirements.

Common Issues to Check:
1. Incorrect column names or DataFrame access patterns
2. Issues handling missing values (NaN)
3. Type errors (mixing numeric/string operations)
4. Incorrect logical conditions
5. Improper handling of the return format (support, confidence, row_indexes, is_violations)

Your response should ONLY contain the python code and nothing else.
ALWAYS wrap your code in ```python and ``` markers.
"""
                
        chain = self._create_chain(
            template=template       
        ) 
        result = chain.invoke({
            "problem_description": problem_description,
            "code": code,
            "failing_tests": failing_tests,
            "dataframe_info": dataframe_info
        }
        )
        
        return {
            "original_code": code,
            "corrected_code": result['code'],
            "metadata": {
                "agent": self.name,
                "action": "code_correction"
            }
        }