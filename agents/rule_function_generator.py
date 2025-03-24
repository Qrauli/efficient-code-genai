from .base_agent import BaseAgent
import pandas as pd
import numpy as np
import sys
sys.path.append("..")

from utils.code_execution import profile_with_scalene

class RuleFunctionGenerator(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert code generation agent specialized in creating efficient functions to evaluate rules on pandas DataFrames.
        Your goal is to generate high-quality, optimized code that evaluates rule satisfaction, calculates metrics like support and confidence, and efficiently identifies satisfying or violating rows. 
        Focus on vectorized operations and pandas-native functions for best performance."""
        super().__init__(config, "RuleFunctionGenerator", system_prompt)
    
    def process(self, rule_description, df_sample=None, context=None):
        """Generate a function that evaluates the given rule on a pandas DataFrame
        
        Args:
            rule_description (str): Description of the rule to implement
            df_sample (str): DataFrame sample information
            context (str, optional): Retrieved context information
            
        Returns:
            dict: Generated code and metadata
        """ 
        
        template = f"""
# Rule Description
{rule_description}

# DataFrame Sample
{df_sample}

{context or ""}

# Task
Generate a Python function named `evaluate_rule` that takes a pandas DataFrame as input and evaluates the rule described above.

- The function should compute:
       - Support: The proportion of rows where the body of the rule is satisfied. If the rule has no body, support is 1.
         Support = (Number of rows where the entire rule is satisfied) / (Total number of rows in the dataset) 
       
       - Confidence: The proportion of rows where both the body and head of the rule are satisfied, out of the rows where the body is satisfied.
         Confidence = (Number of rows where the entire rule is satisfied) / (Number of rows where the body of the rule is satisfied)
    
    - The function should return:
        - `support`: the support value   
        - `confidence`: the confidence value.
        - `satisfying_indexes`, `violation_indexes` : presentations of units that satisfy or violate the rule. This presentation should utilize the index of the dataframe to represent rows. 

Prioritize:
- Vectorized operations over loops
- Pandas native functions over custom implementations
- Memory efficiency for large DataFrames
- Clear, readable code with appropriate comments

IMPORTANT:
- Your response should ONLY contain the Python function wrapped in ```python and ``` markers.
- Make sure that your code uses the exact column names specified in the DataFrame sample not the rule description. For example, if a rule mentions 'AreaCode' but the DataFrame sample shows the column as 'AreaCode(String)', you MUST use 'AreaCode(String)' in your code.
- Note that the code will be used for tables with multiple millions of rows. Ensure that the code is efficient and uses as little memory as possible.
- Do not assume that the DataFrame sample is exhaustive. Your function should work with any DataFrame that has the same structure.
"""
        
        parser = self._extract_code_parser()
        chain = self._create_chain(
            template=template
        )
        
        result = chain.invoke({
            "rule_description": rule_description,
            "df_sample": df_sample
        })
        
        return {
            "code": result['code'],
            "metadata": {
                "agent": self.name,
                "rule_description": rule_description
            }
        }
    

    
    def test_function(self, function_code, dataframe):
        """Test the function code against a dataframe while also collecting performance metrics"""
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_file_path = temp_file.name
            dataframe.to_pickle(temp_file_path)
            
        escaped_path = temp_file_path.replace('\\', '\\\\')

        # Create a test wrapper that includes both function definition and execution
        test_wrapper_code = f"""
{function_code}

import pandas as pd
import json
import traceback

# Execute the function
try:
    # Create a DataFrame copy to avoid side effects
    df = pd.read_pickle("{escaped_path}")
        
    # Run the function and capture results
    result = evaluate_rule(df)
    
    # Extract results
    if isinstance(result, tuple) and len(result) >= 4:
        support, confidence, satisfying_indexes, violation_indexes = result        
        # Create a proper JSON object and print it using json.dumps
        result_dict = {{
            "support": support,
            "confidence": confidence, 
            "satisfying_indexes": satisfying_indexes,
            "violation_indexes": violation_indexes
        }}
        print("FUNCTION_RESULT: " + json.dumps(result_dict))
        success = True
    else:
        success = False
        print("ERROR: Function did not return the expected tuple format")
except Exception as e:
    print("ERROR:", str(e))
    print("TRACEBACK:", traceback.format_exc())
"""
        
        try:
            # Run the wrapped function code with profiling
            profile_result = profile_with_scalene(test_wrapper_code, self.config)
            
            # Parse the output to extract function results
            function_results = {}
            success = False
            error_message = None
            traceback_info = None
            collecting_traceback = False
            
            output_lines = profile_result.get("output", "").split('\n')
            for i, line in enumerate(output_lines):
                if line.startswith("FUNCTION_RESULT:"):
                    try:
                        import json
                        result_str = line.replace("FUNCTION_RESULT:", "").strip()
                        function_results = json.loads(result_str)
                        success = bool(function_results)  # Success if we got any results
                    except Exception as e:
                        error_message = f"Failed to parse function results: {str(e)}"
                elif line.startswith("ERROR:"):
                    error_message = line.replace("ERROR:", "").strip()
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
                "success": success and profile_result.get("success", False),
                "function_results": function_results,
                "error": error_message or profile_result.get("error"),
                "profile_data": profile_data,
                "execution_time": execution_time,
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)