from .base_agent import BaseAgent
import pandas as pd
import numpy as np
import sys
sys.path.append("..")

from utils.code_execution import profile_with_scalene

class RuleFunctionGenerator(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert code generation agent specialized in creating efficient functions to evaluate rules on pandas DataFrames.
        Your goal is to generate high-quality, optimiTake a look at my code generation workflow.zed code that evaluates rule satisfaction, calculates metrics like support and confidence,
        and efficiently identifies satisfying or violating rows. Focus on vectorized operations and pandas-native functions for best performance."""
        super().__init__(config, "RuleFunctionGenerator", system_prompt)
    
    def process(self, rule_description, example_dataframe=None, dataframe_schema=None):
        """Generate a function that evaluates the given rule on a pandas DataFrame
        
        Args:
            rule_description (str): Description of the rule to implement
            example_dataframe (pd.DataFrame, optional): Example DataFrame for context
            dataframe_schema (dict, optional): Schema description if example_dataframe not provided
            
        Returns:
            dict: Generated code and metadata
        """
        # Prepare dataframe information for the prompt
        if example_dataframe is not None:
            df_info = self._extract_dataframe_info(example_dataframe)
            # Create a sample representation of the data
            df_sample = self._create_dataframe_sample(example_dataframe)
        elif dataframe_schema is not None:
            df_info = dataframe_schema
            df_sample = "DataFrame sample not available"
        else:
            df_info = "DataFrame schema not provided"
            df_sample = "DataFrame sample not available"
            
        template = """
        # Rule Description
        {rule_description}
        
        # DataFrame Information
        {df_info}
        
        # DataFrame Sample
        {df_sample}
        
        # Task
        Generate a Python function named `evaluate_rule` that takes a pandas DataFrame as input and evaluates the rule described above.
        
        The function should:
        1. Compute Support: The proportion of rows where the body of the rule is satisfied. If the rule has no body, support is 1.
        2. Compute Confidence: The proportion of rows where both the body and head of the rule are satisfied, out of the rows where the body is satisfied.
        3. Return a tuple containing:
           - support (float): The calculated support value
           - confidence (float): The calculated confidence value
           - row_indexes (set): Either the indexes of violating rows or satisfying rows (but not both)
           - is_violations (bool): True if row_indexes contains violation indexes, False if it contains satisfying indexes
        
        The function should decide which indexes to return based on efficiency - if the confidence is high (≥ 0.9), 
        return violation indexes as they'll be fewer. Otherwise, return satisfying indexes.
        
        Prioritize:
        - Vectorized operations over loops
        - Pandas native functions over custom implementations
        - Memory efficiency for large DataFrames
        - Clear, readable code with appropriate comments
        
        Your response should ONLY contain the Python function wrapped in ```python and ``` markers.
        """
        
        parser = self._extract_code_parser()
        chain = self._create_chain(
            template=template
        )
        
        result = chain.invoke({
            "rule_description": rule_description,
            "df_info": df_info,
            "df_sample": df_sample
        })
        
        return {
            "code": result['code'],
            "metadata": {
                "agent": self.name,
                "rule_description": rule_description
            }
        }
    
    def _extract_dataframe_info(self, df):
        """Extract relevant information from example DataFrame"""
        info = []
        
        # Add shape information
        info.append(f"DataFrame Shape: {df.shape}")
        
        # Add column names and types
        info.append("Columns:")
        for col in df.columns:
            dtype = df[col].dtype
            sample = str(df[col].iloc[0]) if len(df) > 0 else "N/A"
            unique_count = df[col].nunique()
            info.append(f"  - {col} (type: {dtype}, unique values: {unique_count}, sample: {sample})")
        
        # Add basic stats for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info.append("Numeric Column Statistics:")
            for col in numeric_cols[:5]:  # Limit to first 5 to keep prompt size reasonable
                info.append(f"  - {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}")
        
        return "\n".join(info)
    
    def _create_dataframe_sample(self, df):
        """Create a representative sample of the DataFrame for the prompt"""
        # Determine sample size - balance between informativeness and prompt size
        sample_size = min(5, len(df))
        sample_df = df.head(sample_size)
        
        # Format as a pretty printed table
        formatted_sample = "DataFrame Sample (first few rows):\n"
        formatted_sample += sample_df.to_string()
        
        # Also include a code representation for clarity
        formatted_sample += "\n\nSample as code:\n"
        formatted_sample += f"df = pd.DataFrame(\n{sample_df.to_dict()}\n)"
        
        return formatted_sample
    
    def test_function(self, function_code, dataframe, expected_outcomes=None):
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
        support, confidence, row_indexes, is_violations = result
        row_indexes_count = len(row_indexes) if hasattr(row_indexes, '__len__') else 0
        
        # Create a proper JSON object and print it using json.dumps
        result_dict = {{
            "support": support,
            "confidence": confidence, 
            "is_violations": is_violations,
            "row_indexes_count": row_indexes_count
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