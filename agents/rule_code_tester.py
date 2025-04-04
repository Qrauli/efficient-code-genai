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
        Your goal is to fix bugs and validate that rule evaluation code returns the expected output format. 
        If you are provided with a failing test case, you should modify the code to ensure it passes all tests."""
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
        # test_results can now be a list of test results or a single test result
        if not isinstance(test_results, list):
            test_results = [test_results]
            
        # Combine error messages from all failing tests
        failing_tests_errors = []
        for test_result in test_results:
            test_name = test_result.get("test_case_name", "Unknown test")
            error_msg = test_result.get("error", "")
            if error_msg:
                failing_tests_errors.append(f"--- {test_name} ---\n{error_msg}")
        
        failing_tests = failing_tests_errors
        failing_tests = [failing_test.replace("{", "{{").replace("}", "}}") for failing_test in failing_tests]
        
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
6. Make sure that if the rule is unconditional, the support is 1.0 and that every row in the DataFrame is either a satisfaction or a violation.

Your response should ONLY contain the python code and nothing else.
ALWAYS wrap your code in ```python and ``` markers.

{common_mistakes_prompt()}
"""
        chain = self._create_chain(
            template=template       
        ) 
        
        result = chain.invoke({
            "problem_description": problem_description,
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
        
    def test_function_with_testcase(self, function_code, test_case, function_name="execute_rule"):
        """Test the function code with a specified test case using exec()
        
        Args:
            function_code (str): The function code to test
            test_case (dict): Test case with dataframe and expected output
            function_name (str): Name of the function to call
            
        Returns:
            dict: Test results including success status and comparison
        """
        import pandas as pd
        import numpy as np
        import io
        from contextlib import redirect_stdout, redirect_stderr
        import traceback
        import time
        import ast
        
        # Create test dataframe from the test case
        test_df_data = test_case.get("dataframe", [])
        expected_output = test_case.get("expected_output", {})
        test_explanation = test_case.get("explanation", "")  # Get the test case explanation
        
        # Skip if no test data is provided
        if not test_df_data:
            return {
                "success": False,
                "error": "No test data provided in test case",
                "execution_time": None
            }
        
        # Create a namespace for execution
        namespace = {
            "pd": pd,
            "np": np,
            "test_df": pd.DataFrame(test_df_data)
        }
        
        # Prepare stdout/stderr capture
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        start_time = time.time()
        success = False
        actual_output = None
        error = None
        
        try:
            # First, execute the function definition code
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(function_code, namespace)
                
                # Then execute the function with the test dataframe
                exec(f"result = {function_name}(test_df)", namespace)
                
                # Get the result
                actual_output = namespace.get("result")
                
            # Check if we got the expected structure for a dictionary return
            if isinstance(actual_output, dict) and all(k in actual_output for k in ['support', 'confidence', 'satisfactions', 'violations']):
                support = actual_output.get('support')
                confidence = actual_output.get('confidence')
                satisfactions = actual_output.get('satisfactions')
                violations = actual_output.get('violations')
                
                # Compare with expected values
                support_match = abs(support - expected_output.get("support", 0)) < 0.001
                confidence_match = abs(confidence - expected_output.get("confidence", 0)) < 0.001
                
                # Extract and compare indexes from satisfactions and violations
                actual_satisfaction_indexes = extract_indexes(satisfactions)
                actual_violation_indexes = extract_indexes(violations)
                
                # Convert expected satisfactions and violations from string to Python data structure if needed
                expected_satisfactions = expected_output.get("satisfactions_indexes")
                expected_violations = expected_output.get("violations_indexes")
                
                # Need to evaluate strings if they're provided that way
                if isinstance(expected_satisfactions, str):
                    try:
                        expected_satisfactions = eval(expected_satisfactions)
                    except:
                        expected_satisfactions = {}
                
                if isinstance(expected_violations, str):
                    try:
                        expected_violations = eval(expected_violations)
                    except:
                        expected_violations = {}
                
                expected_satisfaction_indexes = extract_indexes(expected_satisfactions)
                expected_violation_indexes = extract_indexes(expected_violations)
                
                # Compare extracted indexes
                satisfactions_match = actual_satisfaction_indexes == expected_satisfaction_indexes
                violations_match = actual_violation_indexes == expected_violation_indexes
                
                # Set success based on value matches
                # Include satisfaction and violation index checks in success criteria
                success = support_match and confidence_match
                if expected_satisfaction_indexes or expected_violation_indexes:
                    # Only include structure checks if expected indexes are provided
                    success = satisfactions_match and violations_match # and success
                
                # Create detailed comparison for debugging
                comparison = {
                    "support": {
                        "actual": support,
                        "expected": expected_output.get("support"),
                        "match": support_match
                    },
                    "confidence": {
                        "actual": confidence,
                        "expected": expected_output.get("confidence"),
                        "match": confidence_match
                    },
                    "satisfactions": {
                        "actual_indexes": sorted(list(actual_satisfaction_indexes)),
                        "expected_indexes": sorted(list(expected_satisfaction_indexes)),
                        "match": satisfactions_match
                    },
                    "violations": {
                        "actual_indexes": sorted(list(actual_violation_indexes)),
                        "expected_indexes": sorted(list(expected_violation_indexes)),
                        "match": violations_match
                    }
                }
                
                # Generate error message for mismatched values
                error_message = None
                if not success:
                    error_message = "Test case values don't match expected output:\n"
                    if not support_match:
                        error_message += f"- Support: Expected {expected_output.get('support')}, got {support}\n"
                    if not confidence_match:
                        error_message += f"- Confidence: Expected {expected_output.get('confidence')}, got {confidence}\n"
                    if not satisfactions_match:
                        error_message += f"- Satisfaction indexes: Expected {sorted(list(expected_satisfaction_indexes))}, extracted from satisfactions output {sorted(list(actual_satisfaction_indexes))}\n"
                    if not violations_match:
                        error_message += f"- Violation indexes: Expected {sorted(list(expected_violation_indexes))}, extracted from violations output {sorted(list(actual_violation_indexes))}\n"
                
                    # Include the test DataFrame in the error message
                    error_message += "\nTest DataFrame used:\n"
                    error_message += str(pd.DataFrame(test_df_data).head(10))
                    if len(test_df_data) > 10:
                        error_message += f"\n... (total rows: {len(test_df_data)})"
                    
                    # Include the test explanation if available
                    if test_explanation:
                        error_message += "\nTest Case Explanation:\n"
                        error_message += test_explanation
                
                return {
                    "success": success,
                    "execution_time": time.time() - start_time,
                    "comparison": comparison,
                    "actual_output": {
                        "support": support,
                        "confidence": confidence,
                        "satisfaction_indexes": sorted(list(actual_satisfaction_indexes)),
                        "violation_indexes": sorted(list(actual_violation_indexes))
                    },
                    "error": error_message,
                    "test_data": test_df_data 
                }           
            else:
                error_msg = f"Exception: Function did not return expected dictionary format with required keys. Got: {type(actual_output)}"
                
                # Include the test explanation if available
                if test_explanation:
                    error_msg += "\nTest Case Explanation:\n"
                    error_msg += test_explanation
                    
                return {
                    "success": False,
                    "execution_time": time.time() - start_time,
                    "error": error_msg
                }
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
                
            return {
                "success": False,
                "execution_time": time.time() - start_time,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "error": error_msg
            }

    def test_function_with_testcases(self, function_code, test_cases, function_name="execute_rule"):
        """Test the function code with multiple test cases
        
        Args:
            function_code (str): The function code to test
            test_cases (list): List of test cases with dataframes and expected outputs
            function_name (str): Name of the function to call
            
        Returns:
            dict: Aggregated test results including success status and individual test results
        """
        # Check if test_cases is a list or a single test case
        if isinstance(test_cases, dict) and "dataframe" in test_cases:
            # For backward compatibility - convert single test case to a list
            test_cases = [test_cases]
        
        # Run each test case individually
        results = []
        all_passed = True
        
        for i, test_case in enumerate(test_cases):
            result = self.test_function_with_testcase(
                function_code=function_code,
                test_case=test_case,
                function_name=function_name
            )
            
            # Add test case name or index to result
            result["test_case_name"] = test_case.get("name", f"Test Case {i+1}")
            results.append(result)
            
            # Update overall success flag
            if not result.get("success", False):
                all_passed = False
        
        return {
            "success": all_passed,
            "test_results": results,
            "num_tests": len(results),
            "num_passed": sum(1 for r in results if r.get("success", False)),
            "code": function_code
        }

def extract_indexes(structure):
    """
    Recursively extract all row indexes from a nested structure of dictionaries, lists, sets, and tuples.
    
    Args:
        structure: The nested structure to extract indexes from (dict, list, set, tuple, or primitive)
        
    Returns:
        set: A set of all indexes found in the structure
    """
    indexes = set()
    
    if isinstance(structure, dict):
        # Extract indexes from both keys and values
        for k, v in structure.items():
            # If the key is a tuple containing indexes, process it
            if isinstance(k, tuple):
                # Skip column names in tuple keys like (('column_name', value), ...)
                for item in k:
                    if isinstance(item, tuple) and len(item) == 2:
                        continue  # Skip column name tuples
                    elif isinstance(item, (int, float)) and not isinstance(item, bool):
                        indexes.add(int(item))
            elif isinstance(k, (int, float)) and not isinstance(k, bool):
                indexes.add(int(k))
            
            # Recursively process values
            indexes.update(extract_indexes(v))
    
    elif isinstance(structure, (list, set, tuple)):
        for item in structure:
            if isinstance(item, (dict, list, set, tuple)):
                indexes.update(extract_indexes(item))
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                indexes.add(int(item))
    
    elif isinstance(structure, (int, float)) and not isinstance(structure, bool):
        indexes.add(int(structure))
    
    return indexes