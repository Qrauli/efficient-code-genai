from .base_agent import BaseAgent, common_improvement_recommendations, common_mistakes_prompt, common_generation_prompt
import pandas as pd
import numpy as np
import sys
sys.path.append("..")

from utils.code_execution import profile_with_scalene

class RuleFunctionGenerator(BaseAgent):
    def __init__(self, config):
        system_prompt = """
You are an expert code generation agent specialized in creating efficient functions to evaluate rules on pandas DataFrames.
Your goal is to generate high-quality, optimized code that evaluates rule satisfaction, calculates metrics like support and confidence, and efficiently identifies satisfying or violating rows. 
Your primary task is to produce correct and efficient code, so focus on correctness for now, but you should also consider the performance of the code you produce.
"""
        super().__init__(config, "RuleFunctionGenerator", system_prompt)
    
    def process(self, rule_description, df_sample=None, function_name="execute_rule", context=None, rule_format=None, test_cases=None):
        """Generate a function that evaluates the given rule on a pandas DataFrame
        
        Args:
            rule_description (str): Description of the rule to implement
            df_sample (str): DataFrame sample information
            function_name (str): Name of the function to generate (default: execute_rule)
            context (str, optional): Retrieved context information
            rule_format (str, optional): Format specification for rule output structure
            
        Returns:
            dict: Generated code and metadata
        """ 
        is_multi_df = df_sample is not None and "--- DataFrame:" in df_sample

        task_description = f"Generate a Python function named `{function_name}` that takes a pandas DataFrame as input and evaluates the rule described above."
        
        if is_multi_df:
            task_description = f"""
Generate a Python function named `{function_name}` that takes a dictionary of pandas DataFrames as input and evaluates the rule described above.
- The input will be a dictionary where keys are DataFrame names (as shown in the sample) and values are the corresponding pandas DataFrames.
- Access specific DataFrames using dictionary syntax, e.g., `dfs['df_name1']`.
"""

        # Format the rule format information if available
        format_guidance = ""
        if rule_format:
            # Handle rule_format as text instead of dictionary
            format_guidance = f"""
# Output Format Specification
Based on the rule analysis, implement the following output structure:

{rule_format}
"""

        # Add test case explanation to the generator input if available
        test_case_guidance = ""
        if test_cases:
            first_test_case = test_cases[0] if test_cases else {}
            # Adapt test case display for multi-df if necessary
            dataframe_display = first_test_case.get('dataframe', {})
            if is_multi_df and isinstance(dataframe_display, dict):
                 # Format dict of dicts nicely
                 dataframe_display_str = "{\n"
                 for name, df_dict in dataframe_display.items():
                     dataframe_display_str += f"    '{name}': {df_dict},\n"
                 dataframe_display_str += "}"
            else:
                 dataframe_display_str = str(dataframe_display) # Keep as string for single df

            test_case_guidance = f"""
# Test Case Information
Here is a test case that should pass with your implementation:

Sample DataFrame(s):
```python
{dataframe_display_str}
```

Expected Output:
- Support: {first_test_case.get('expected_output', {}).get('support')}
- Confidence: {first_test_case.get('expected_output', {}).get('confidence')}
- Satisfactions: {first_test_case.get('expected_output', {}).get('satisfactions_str')}
- Violations: {first_test_case.get('expected_output', {}).get('violations_str')}

Explanation:
{first_test_case.get('explanation', '')}
"""

        template = """
# Rule Description
{rule_description}

# Task
{task_description}

- The function should return a dictionary with the following keys:
    - `support`: the support value   
    - `confidence`: the confidence value
    - `satisfactions`: presentations of units that satisfy the rule
    - `violations`: presentations of units that violate the rule

# DataFrame Sample
{df_sample}

{context}

{format_guidance}
{test_case_guidance}

Prioritize:
- Vectorized operations over loops
- Pandas native functions over custom implementations
- Memory efficiency for large DataFrames
- Clear, readable code with appropriate comments

IMPORTANT:
- Your response should ONLY contain the Python function wrapped in ```python and ``` markers.
- Make sure that your code uses the exact column names specified in the DataFrame sample not the rule description. For example, if a rule mentions 'AreaCode' but the DataFrame sample shows the column as 'AreaCode(String)', you MUST use 'AreaCode(String)' in your code.
- Note that the code will be used for tables with multiple millions of rows. Ensure that the code is efficient and uses as little memory as possible. Try to avoid copying and use in-place operations if possible.
- Do not assume that the DataFrame sample is exhaustive. Your function should work with any DataFrame that has the same structure.
- In the generated code, please delete unused intermediate variables to free memory before returning the results. Use `del` to delete variables and `gc.collect()` to free memory.
- In the generated code, you should first limit the input DataFrame to the columns used, eliminating unused columns, and use this simplified DataFrame for the remaining operations.
- Be careful when you write the code of generating the inner dictionaries in the violations, do not replace the existing entries already contained in the inner dictionary. 
- Ensure that each key used in the dictionaries for satisfactions or violations of group validation rules, including keys in both outer and inner dictionaries, always includes the column name when a column value is part of the key. For example, valid keys can be ("A", 100) or (("A", 100), ("B", 200)), but not (100) or (100, 200).
- Make sure that if the rule is unconditional, the support is 1.0 and that every row in the DataFrame is either a satisfaction or a violation.

{common_generation_prompt}

{common_improvement_recommendations}

{common_mistakes_prompt}
"""
        
        parser = self._extract_code_parser()
        chain = self._create_chain(
            template=template,
            run_name="RuleFunctionGenerator"
        )
        
        result = chain.invoke({
            "rule_description": rule_description,
            "df_sample": df_sample,
            "format_guidance": format_guidance,
            "test_case_guidance": test_case_guidance or "",
            "context": context or "",
            "function_name": function_name,
            "task_description": task_description,
            "common_mistakes_prompt": common_mistakes_prompt(),
            "common_improvement_recommendations": common_improvement_recommendations(is_multi_dataframe=is_multi_df),
            "common_generation_prompt": common_generation_prompt()
        })
        
        return {
            "code": result['code'],
            "metadata": {
                "agent": self.name,
                "rule_description": rule_description,
                "function_name": function_name
            }
        }
    
    def execute_and_profile_rule(self, function_code, data, function_name="execute_rule"):
        """Test the function code against data (single DataFrame or dict of DataFrames) while also collecting performance metrics"""
        
        import tempfile
        import os
        import pandas as pd # Ensure pandas is imported here
        
        # Determine if the input is a single DataFrame or a dictionary of DataFrames
        is_multi_df = isinstance(data, dict)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_file_path = temp_file.name
            # Pickle the data structure (single df or dict of dfs)
            pd.to_pickle(data, temp_file_path)
            
        escaped_path = temp_file_path.replace('\\', '\\\\')

        # Create a test wrapper that includes both function definition and execution
        # Adjust the loading and function call based on whether it's multi-df or not
        test_wrapper_code = f"""
{function_code}

# Execute the function
import traceback
import pandas as pd
import json

try:
    # Load the pickled data (could be a single DataFrame or a dict)
    loaded_data = pd.read_pickle("{escaped_path}")
        
    # Run the function and capture results, passing the loaded data structure
    result = {function_name}(loaded_data)
    print("SUCCESS")
except Exception as e:
    print("ERROR:", str(e))
    print("TRACEBACK:", traceback.format_exc())
"""
        
        try:
            # Run the wrapped function code with profiling
            profile_result = profile_with_scalene(test_wrapper_code, self.config)
            
            # Parse the output to extract function results
            function_results = {}
            success = profile_result.get("timed_out", False)
            error_message = None
            traceback_info = None
            collecting_traceback = False
            
            output_lines = profile_result.get("output", "").split('\n')
            for i, line in enumerate(output_lines):
                if line.startswith("SUCCESS"):
                    success = True
                elif line.startswith("ERROR:"):
                    error_message = line.replace("ERROR:", "").strip()
                    success = False
                elif line.startswith("TRACEBACK:"):
                    # Start collecting traceback
                    collecting_traceback = True
                    traceback_info = line.replace("TRACEBACK:", "").strip() + "\n"
                elif collecting_traceback:
                    # Continue collecting traceback lines until we hit another marker or end
                    if line.startswith("FUNCTION_RESULT:") or line.startswith("ERROR:"):
                        collecting_traceback = False
                        # Process this line again as it's a new marker
                        i -= 1
                        continue
                    traceback_info += line + "\n"
            
            # Combine error message with traceback if available
            if error_message and traceback_info:
                error_message = f"{error_message}\n{traceback_info}"
            elif traceback_info:
                error_message = traceback_info
            
            # Extract execution time from Scalene profile data if available
            execution_time = None
            profile_data = profile_result.get("profile_data", {})
            if profile_data and "files" in profile_data:
                # Just get the first file's metrics
                for file_data in profile_data.get("files", {}).values():
                    execution_time = file_data.get("total_cpu_seconds", None)
                    break
                
            # Combine everything into a single comprehensive result
            return {
                "success": success,
                "function_results": function_results,
                "error": error_message or profile_result.get("error"),
                "profile_data": profile_data,
                "execution_time": execution_time,
                "timed_out": profile_result.get("timed_out", False)
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

