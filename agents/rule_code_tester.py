from typing import List, TypedDict, Any
from .base_agent import BaseAgent, common_mistakes_prompt, common_generation_prompt

import sys
sys.path.append("..")
from utils.code_execution import execute_code, indent_code
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
import concurrent.futures
import pandas as pd
import numpy as np
import io
from contextlib import redirect_stdout, redirect_stderr
import traceback
import time
import ast
import warnings
        

class TestCase(TypedDict):
    name: str
    function_call: str
    expected_output: Any
    description: str
    
class RuleCodeTester(BaseAgent):
    def __init__(self, config):
        system_prompt = """
You are an expert code correction agent specialized in fixing the correctness and efficiency of DataFrame rule evaluation functions.
Your goal is to fix bugs and validate that rule evaluation code returns the expected output format. 
If you are provided with a failing test case, you should modify the code to ensure it passes all tests.
"""
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
        
        # Format the rule format information if available
        format_guidance = ""
        if rule_format:
            # Handle rule_format as text instead of dictionary
            format_guidance = f"""
# Required Output Format
When fixing the code, ensure it maintains this exact return structure:

{rule_format}
"""
        is_multi_df = dataframe_info is not None and "--- DataFrame:" in dataframe_info
        if is_multi_df:
            format_guidance += """
IMPORTANT:
For numeric ID comparisons ALWAYS convert columns using `astype(int).astype(str)` instead of just `astype(str)` ONLY if the datatypes don't match, e.g int vs float, see dataframe info. Otherwise there might be issues with the formatting of the string.
"""            

        template = """
Problem Description: {problem_description}

# Task
Fix the code to make all tests pass. Focus only on correctness for now, not optimization.
The goal is to make the code work correctly according to the requirements.

Current Code:
```python
{code}
```

Errors:
{failing_tests}

DataFrame Structure:
{dataframe_info}

{format_guidance}

Common Issues to Check:
1. Incorrect column names or DataFrame access patterns
2. Issues handling missing values (NaN)
3. Type errors (mixing numeric/string operations)
4. Incorrect logical conditions
5. Improper handling of the return format 
6. Make sure that if the rule is unconditional, the support is 1.0 and that every row in the DataFrame is either a satisfaction or a violation.

{common_generation_prompt}

Your response should ONLY contain the python code and nothing else.
ALWAYS wrap your code in ```python and ``` markers.
"""
        chain = self._create_chain(
            template=template,
            run_name="CodeCorrection"    
        ) 
        
        result = chain.invoke({
            "format_guidance": format_guidance,
            "problem_description": problem_description,
            "failing_tests": "\n".join(failing_tests),
            "dataframe_info": dataframe_info,
            "code": code,
            "common_mistakes": common_mistakes_prompt(),
            "common_generation_prompt": common_generation_prompt()
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
        
        # Setup warning capture
        warning_messages = []
        def warning_collector(message, category, filename, lineno, file=None, line=None):
            warning_messages.append(f"{category.__name__}: {message}")
        
        # Store original warning filter and showwarning function
        original_filters = warnings.filters.copy()
        original_showwarning = warnings.showwarning
        
        try:
            # Configure warning capture
            warnings.resetwarnings()
            warnings.simplefilter('always')  # Show all warnings
            warnings.showwarning = warning_collector
            
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
                support_match = abs(support - expected_output.get("support", 0)) < 0.01
                confidence_match = abs(confidence - expected_output.get("confidence", 0)) < 0.01
                
                # Extract and compare indexes from satisfactions and violations
                actual_satisfaction_indexes = sorted(list(extract_indexes(satisfactions)))
                actual_violation_indexes = sorted(list(extract_indexes(violations)))
                
                # Convert expected satisfactions and violations from string to Python data structure if needed
                expected_satisfactions = expected_output.get("satisfactions_str")
                expected_violations = expected_output.get("violations_str")
                
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
                
                expected_satisfaction_indexes = sorted(list(extract_indexes(expected_satisfactions)))
                expected_violation_indexes = sorted(list(extract_indexes(expected_violations)))
                
                # Compare extracted indexes
                satisfactions_match = actual_satisfaction_indexes == expected_satisfaction_indexes
                violations_match = actual_violation_indexes == expected_violation_indexes
                
                # Set success based on value matches
                # Include satisfaction and violation index checks in success criteria
                # success = support_match and confidence_match
                # if expected_satisfaction_indexes or expected_violation_indexes:
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
                        "actual_indexes": actual_satisfaction_indexes,
                        "expected_indexes": expected_satisfaction_indexes,
                        "match": satisfactions_match
                    },
                    "violations": {
                        "actual_indexes": actual_violation_indexes,
                        "expected_indexes": expected_violation_indexes,
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
                        error_message += f"- Satisfaction indexes: Expected {expected_satisfaction_indexes}, extracted from satisfactions output {actual_satisfaction_indexes}\n"
                        error_message += f"- Satisfaction structure: Expected {expected_satisfactions}, got {satisfactions}\n"
                    if not violations_match:
                        error_message += f"- Violation indexes: Expected {expected_violation_indexes}, extracted from violations output {actual_violation_indexes}\n"
                        error_message += f"- Violation structure: Expected {expected_violations}, got {violations}\n"
                        
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
                    "error": error_message,
                    "test_data": test_df_data,
                    "warnings": warning_messages  # Include captured warnings in the result
                }           
            else:
                error_msg = f"EXCEPTION: Function did not return expected dictionary format with required keys. Got: {type(actual_output)}"
                
                # Include the test explanation if available
                if test_explanation:
                    error_msg += "\nTest Case Explanation:\n"
                    error_msg += test_explanation
                    
                return {
                    "success": False,
                    "execution_time": time.time() - start_time,
                    "error": error_msg,
                    "warnings": warning_messages  # Include captured warnings in the result
                }
        except Exception as e:
            error_msg = f"EXCEPTION: {str(e)}\n{traceback.format_exc()}"
                
            return {
                "success": False,
                "execution_time": time.time() - start_time,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "error": error_msg,
                "warnings": warning_messages  # Include captured warnings in the result
            }
        finally:
            # Restore original warning behavior
            warnings.filters = original_filters
            warnings.showwarning = original_showwarning

    @staticmethod
    def _run_test_case_in_subprocess(args):
        """
        Standalone function to run a single test case in a subprocess.
        Args:
            args: tuple of (function_code, test_case, function_name)
        Returns:
            dict: test result
        """

        function_code, test_case, function_name = args

        test_df_data = test_case.get("dataframe", [])
        expected_output = test_case.get("expected_output", {})
        test_explanation = test_case.get("explanation", "")

        if not test_df_data:
            return {
                "success": False,
                "error": "No test data provided in test case",
                "execution_time": None
            }

        # Determine if it's a multi-dataframe test case
        is_multi_df = isinstance(test_df_data, dict) and all(isinstance(v, dict) for v in test_df_data.values())

    

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        start_time = time.time()
        success = False
        actual_output = None
        error = None

        warning_messages = []
        def warning_collector(message, category, filename, lineno, file=None, line=None):
            warning_messages.append(f"{category.__name__}: {message}")

        original_filters = warnings.filters.copy()
        original_showwarning = warnings.showwarning

        try:
            warnings.resetwarnings()
            warnings.simplefilter('always')
            warnings.showwarning = warning_collector
            
            if is_multi_df:
            # Create a dictionary of DataFrames
                test_data_input = {name: pd.DataFrame(df_data) for name, df_data in test_df_data.items()}
                exec_arg_name = "test_data_input_dict" # Use a distinct name for the dict
            else:
                # Create a single DataFrame
                test_data_input = pd.DataFrame(test_df_data)
                exec_arg_name = "test_data_input_df" # Use a distinct name for the single df

            namespace = {
                "pd": pd,
                "np": np,
                exec_arg_name: test_data_input # Add the prepared data to the namespace
            }

            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(function_code, namespace)
                # Pass the correct data structure (single df or dict) to the function
                exec(f"result = {function_name}({exec_arg_name})", namespace)
                actual_output = namespace.get("result")

            if isinstance(actual_output, dict) and all(k in actual_output for k in ['support', 'confidence', 'satisfactions', 'violations']):
                support = actual_output.get('support')
                confidence = actual_output.get('confidence')
                satisfactions = actual_output.get('satisfactions')
                violations = actual_output.get('violations')

                support_match = abs(support - expected_output.get("support", 0)) < 0.01
                confidence_match = abs(confidence - expected_output.get("confidence", 0)) < 0.01

                actual_satisfaction_indexes = sorted(list(extract_indexes(satisfactions)))
                actual_violation_indexes = sorted(list(extract_indexes(violations)))

                expected_satisfactions = expected_output.get("satisfactions_str")
                expected_violations = expected_output.get("violations_str")

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

                expected_satisfaction_indexes = sorted(list(extract_indexes(expected_satisfactions)))
                expected_violation_indexes = sorted(list(extract_indexes(expected_violations)))

                satisfactions_match = actual_satisfaction_indexes == expected_satisfaction_indexes
                violations_match = actual_violation_indexes == expected_violation_indexes

                success = satisfactions_match and violations_match

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
                        "actual_indexes": actual_satisfaction_indexes,
                        "expected_indexes": expected_satisfaction_indexes,
                        "match": satisfactions_match
                    },
                    "violations": {
                        "actual_indexes": actual_violation_indexes,
                        "expected_indexes": expected_violation_indexes,
                        "match": violations_match
                    }
                }

                error_message = ""
                if not success:
                    error_message = "Test case values don't match expected output:\n"
                    if not support_match:
                        error_message += f"- Support: Expected {expected_output.get('support')}, got {support}\n"
                    if not confidence_match:
                        error_message += f"- Confidence: Expected {expected_output.get('confidence')}, got {confidence}\n"
                    if not satisfactions_match:
                        error_message += f"- Satisfaction indexes: Expected {expected_satisfaction_indexes}, extracted from satisfactions output {actual_satisfaction_indexes}\n"
                        error_message += f"- Satisfaction structure: Expected {expected_satisfactions}, got {satisfactions}\n"
                    if not violations_match:
                        error_message += f"- Violation indexes: Expected {expected_violation_indexes}, extracted from violations output {actual_violation_indexes}\n"
                        error_message += f"- Violation structure: Expected {expected_violations}, got {violations}\n"

                    error_message += "\nTest DataFrames used:\n"
                    if is_multi_df:
                        for name, df_dict in test_df_data.items():
                            df_str = str(pd.DataFrame(df_dict).head(10))
                            error_message += f"--- {name} ---\n{df_str}\n"
                            if len(df_dict.get(next(iter(df_dict), ''), [])) > 10:
                                error_message += f"... (total rows: {len(df_dict.get(next(iter(df_dict), ''), []))})\n"
                    else:
                        df_str = str(pd.DataFrame(test_df_data).head(10))
                        error_message += f"{df_str}\n"
                        if len(test_df_data.get(next(iter(test_df_data), ''), [])) > 10:
                             error_message += f"... (total rows: {len(test_df_data.get(next(iter(test_df_data), ''), []))})\n"

                    if test_explanation:
                        error_message += "\nTest Case Explanation:\n"
                        error_message += test_explanation

                return {
                    "success": success,
                    "execution_time": time.time() - start_time,
                    "comparison": comparison,
                    "error": error_message,
                    "test_data": test_df_data,
                    "warnings": warning_messages
                }
            else:
                error_msg = f"EXCEPTION: Function did not return expected dictionary format with required keys. Got: {type(actual_output)}"
                if test_explanation:
                    error_msg += "\nTest Case Explanation:\n"
                    error_msg += test_explanation
                return {
                    "success": False,
                    "execution_time": time.time() - start_time,
                    "error": error_msg,
                    "warnings": warning_messages
                }

        except Exception as e:
            error_msg = f"EXCEPTION: {str(e)}\n{traceback.format_exc()}"
            return {
                "success": False,
                "execution_time": time.time() - start_time,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "error": error_msg,
                "warnings": warning_messages
            }
        finally:
            warnings.filters = original_filters
            warnings.showwarning = original_showwarning

    def test_function_with_testcases(self, function_code, test_cases, function_name="execute_rule"):
        """Test the function code with multiple test cases in parallel (using separate processes).
        Args:
            function_code (str): The function code to test
            test_cases (list): List of test cases with dataframes and expected outputs
            function_name (str): Name of the function to call
        Returns:
            dict: Aggregated test results including success status and individual test results
        """
        if isinstance(test_cases, dict) and "dataframe" in test_cases:
            test_cases = [test_cases]

        results = []
        all_passed = True
        all_warnings = []

        # Prepare arguments for each test case
        args_list = [
            (function_code, test_case, function_name)
            for test_case in test_cases
        ]

        # Use ProcessPoolExecutor for true isolation
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_index = {
                executor.submit(RuleCodeTester._run_test_case_in_subprocess, args): i
                for i, args in enumerate(args_list)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                result = future.result()
                # Add test case name or index to result
                result["test_case_name"] = test_cases[i].get("name", f"Test Case {i+1}")
                result["test_case_index"] = i  # <-- Add this line
                results.append((i, result))
                if "warnings" in result and result["warnings"]:
                    all_warnings.extend(result["warnings"])
                if not result.get("success", False):
                    all_passed = False

        # Sort results by original test case order
        results_sorted = [r for _, r in sorted(results, key=lambda x: x[0])]

        # Remove duplicates from warnings while preserving order
        unique_warnings = []
        for warning in all_warnings:
            if warning not in unique_warnings:
                unique_warnings.append(warning)

        return {
            "success": all_passed,
            "test_results": results_sorted,
            "num_tests": len(results_sorted),
            "num_passed": sum(1 for r in results_sorted if r.get("success", False)),
            "code": function_code,
            "warnings": unique_warnings
        }

def extract_indexes(structure):
    """
    Recursively extract all row indexes from a nested structure of dictionaries, lists, sets, and tuples.
    Extracts numbers from leaf lists and single numeric leaf values.
    
    Args:
        structure: The nested structure to extract indexes from (dict, list, set, tuple, or primitive)
        
    Returns:
        set: A set of all indexes found in the structure
    """
    indexes = set()
    
    if isinstance(structure, dict):
        # Only process values in dictionaries
        for v in structure.values():
            indexes.update(extract_indexes(v))
    
    elif isinstance(structure, (list, set, tuple)):
        # For lists, check if it's a leaf list (contains only primitives)
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in structure):
            # If it's a leaf list with numbers, extract those numbers
            for item in structure:
                indexes.add(int(item))
        else:
            # If not a leaf list, recursively process its elements
            for item in structure:
                indexes.update(extract_indexes(item))
    
    # Handle the case where the leaf is a single number
    elif isinstance(structure, (int, float)) and not isinstance(structure, bool):
        indexes.add(int(structure))
    
    return indexes