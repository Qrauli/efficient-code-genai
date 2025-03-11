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
    
class CodeTester(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert code testing agent specialized in verifying the correctness and efficiency of code.
        Your goal is to generate comprehensive test cases, identify bugs, and validate that code meets requirements, especially for data science tasks."""
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
        
    def generate_tests(self, problem_description):
        """Generate test cases based only on problem description (before code generation)"""
        test_cases = self._generate_test_cases_from_description(problem_description)
        
        return {
            "test_results": test_cases,
            "metadata": {
                "agent": self.name,
                "source": "problem_description_only"
            }
        }
        
    def generate_additional_tests(self, problem_description, existing_test_cases):
        """Generate additional test cases to complement existing ones"""
        template = """
        Problem Description: {problem_description}
        
        Existing Test Cases:
        {existing_test_cases}
        
        Generate additional test cases that cover scenarios not addressed by the existing tests.
        Focus on edge cases, performance tests, or other aspects that might be missing.
        Keep the amount of test cases under a maximum of 10 to ensure quality over quantity.
        
        Return the additional test cases in a structured JSON format with the following schema:
        [
            {{
                "name": "Test case name",
                "function_call": "The exact function call to test, e.g., 'function_name(1, 2)'",
                "expected_output": "Expected return value",
                "description": "Brief description of what this test case checks"
            }}
        ]
        
        Your response should ONLY contain the JSON array of additional test cases and nothing else.
        DO NOT use markdown code blocks or any ```json markers. Return ONLY the raw JSON array.
        """
        
        json_parser = JsonOutputParser(pydantic_object=List[TestCase])
        retry_parser = RetryWithErrorOutputParser.from_llm(self.llm, json_parser)

        chain = self._create_chain(
            template=template,
            parser=retry_parser
        )
        
        additional_tests = chain.invoke({
            "problem_description": problem_description,
            "existing_test_cases": str(existing_test_cases)
        }
        )
        
        return {
            "additional_tests": additional_tests,
            "metadata": {
                "agent": self.name,
                "source": "complementary_generation"
            }
        }
    
    def _execute_test_cases(self, code, test_cases):
        """Execute code with the given test cases"""
        test_results = []
        for test_case in test_cases:
            # Create a test wrapper that calls the function and compares the result
            function_call = test_case.get("function_call", "")
            expected_output = test_case.get("expected_output", "")
            
            # Fix: Proper code indentation with dedicated indent function
            indented_code = indent_code(code)
            
            test_wrapper = f"""
import json
try:
    # First execute the original code to define the function
{indented_code}
    
    # Then execute the test call and capture the result
    result = {function_call}
    print(json.dumps({{"actual_output": result}}, default=str))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
            result = execute_code(test_wrapper, self.config)
            
            # Parse the output to get the actual result
            import json
            success = False
            try:
                output_data = json.loads(result["output"])
                if "error" not in output_data:
                    actual_output = output_data.get("actual_output", "")
                    # Improved comparison that handles different data types
                    if isinstance(expected_output, list) and isinstance(actual_output, list):
                        success = sorted(expected_output) == sorted(actual_output)
                    else:
                        success = str(actual_output) == str(expected_output)
            except:
                pass
                
            test_results.append({
                "test_case": test_case,
                "execution_time": result["execution_time"],
                "memory_usage": result["memory_usage"],
                "output": result["output"],
                "success": success
            })
            
        return test_results
    
    def _generate_test_cases(self, problem_description, code):
        """Generate test cases based on the problem description and the code"""
        template = """
        Problem Description: {problem_description}
        
        Code to test:
        ```python
        {code}
        ```
        
        Generate a comprehensive set of test cases for this code. 
        Keep the amount of test cases under a maximum of 10 to ensure quality over quantity.

        Return the test cases in a structured JSON format with the following schema:
        [
            {{
                "name": "Test case name",
                "function_call": "The exact function call to test, e.g., 'function_name(1, 2)'",
                "expected_output": "Expected return value",
                "description": "Brief description of what this test case checks"
            }}
        ]
        
        Include a mix of:
        1. Basic functionality tests
        2. Edge cases (empty inputs, large inputs, etc.)
        3. Corner cases specific to this problem
        4. Performance considerations for larger inputs
        
        IMPORTANT GUIDELINES:
        - For very large expected outputs (like arrays with thousands of elements), use a small representative sample or a more concise description
        - DO NOT use ellipsis (...) or other shorthand notation in your expected outputs, as this will cause parsing errors
        - For large array outputs, consider using specific test values rather than massive arrays
        - Your response should ONLY contain valid JSON - every value must be properly formatted JSON
        - DO NOT use markdown code blocks or any ```json markers. Return ONLY the raw JSON array.
  
        """
        json_parser = JsonOutputParser(pydantic_object=List[TestCase])
        retry_parser = RetryWithErrorOutputParser.from_llm(self.llm, json_parser)

        chain = self._create_chain(
            template=template,
            parser=retry_parser
        )
        
        test_cases = chain.invoke({
            "problem_description": problem_description,
            "code": code
        }
        )
        
        if not test_cases:
            test_cases = [
                {"name": "Default test", "function_call": "", "expected_output": "", "description": "Default test case"}
            ]
        
        return test_cases
        
    def _generate_test_cases_from_description(self, problem_description):
        """Generate test cases based only on the problem description (before code is written)"""
        template = """
        Problem Description: {problem_description}
        
        Generate a comprehensive set of test cases for a solution to this problem BEFORE seeing any implementation.
        This will help ensure the code meets requirements without being biased by a specific implementation.
        Keep the amount of test cases under a maximum of 10 to ensure quality over quantity.
        
        Return the test cases in a structured JSON format with the following schema:
        [
            {{
                "name": "Test case name",
                "function_call": "The exact function call to test, e.g., 'function_name(1, 2)'",
                "expected_output": "Expected return value",
                "description": "Brief description of what this test case checks"
            }}
        ]
        
        Include a mix of:
        1. Basic functionality tests
        2. Edge cases (empty inputs, large inputs, etc.)
        3. Corner cases specific to this problem
        4. Performance considerations for larger inputs
        
        IMPORTANT GUIDELINES:
        - Make reasonable assumptions about the function name and signature based on the problem description
        - For very large expected outputs (like arrays with thousands of elements), use a small representative sample or a more concise description
        - DO NOT use ellipsis (...) or other shorthand notation in your expected outputs, as this will cause parsing errors
        - For large array outputs, consider using specific test values rather than massive arrays
        - Your response should ONLY contain valid JSON - every value must be properly formatted JSON
        - DO NOT use markdown code blocks or any ```json markers. Return ONLY the raw JSON array.
        """
        
        json_parser = JsonOutputParser(pydantic_object=List[TestCase])
        retry_parser = RetryWithErrorOutputParser.from_llm(self.llm, json_parser)

        chain = self._create_chain(
            template=template,
            parser=retry_parser
        )
        
        test_cases = chain.invoke({
            "problem_description":problem_description
        }
        )
        
        if not test_cases:
            test_cases = [
                {"name": "Default test", "function_call": "", "expected_output": "", "description": "Default test case"}
            ]
        
        return test_cases
    
    def _analyze_test_results(self, code, test_results, problem_description):
        """Analyze test results and provide feedback"""
        template = """
        Problem Description: {problem_description}
        
        Code:
        ```python
        {code}
        ```
        
        Test Results:
        {test_results}
        
        Analyze these test results and provide:
        1. Issues identified (correctness, performance)
        2. Suggestions for improvement
        3. Overall assessment of the code quality
        """
        
        chain = self._create_chain(
            template=template
        )
        
        result = chain.invoke({
            "problem_description": problem_description,
            "code": code,
            "test_results": str(test_results)
        }
        )
        
        return result

