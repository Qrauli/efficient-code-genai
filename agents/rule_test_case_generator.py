from .base_agent import BaseAgent
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class ExpectedOutput(BaseModel):
    support: float
    confidence: float
    satisfactions_str: str
    violations_str: str

class TestCase(BaseModel):
    name: str
    dataframe: Union[Dict[str, List[Any]], Dict[str, Dict[str, List[Any]]]] # Allow single or multi-df structure
    explanation: str
    expected_output: ExpectedOutput
    
class RuleTestCaseGenerator(BaseAgent):
    def __init__(self, config):
        system_prompt = """
You are an expert test case generator for data quality rules. Your job is to create precise, 
accurate test cases that validate the implementation of data quality rules on pandas DataFrames.
"""
        super().__init__(config, "RuleTestCaseGenerator", system_prompt)
    
    def process(self, rule_description, rule_format, dataframe_info=None, num_test_cases=3):
        """Generate multiple test cases for a rule description based on the format specification
        
        Args:
            rule_description (str): Description of the rule to implement
            rule_format (str): Format specification for the rule output
            dataframe_info (str, optional): DataFrame sample information to help understand the data structure
            num_test_cases (int, optional): Number of test cases to generate (default: 3)
            
        Returns:
            dict: Dictionary containing multiple test cases with sample dataframes and expected outputs
        """
        
        # Check if the rule involves multiple DataFrames based on the info format
        is_multi_df = dataframe_info is not None and "--- DataFrame:" in dataframe_info
        
        multi_df_instructions = ""
        if is_multi_df:
            multi_df_instructions = """
## Multi-DataFrame Test Case Instructions:

The rule involves multiple DataFrames. Pay close attention to the following:
- Represent the test data in the `dataframe` field as a dictionary where keys are the DataFrame names (as shown in `dataframe_info`) and values are dictionaries representing each DataFrame (columns as keys, lists of values).
- In the explanation, clearly state which DataFrame acts as the primary source for row indices in `satisfactions` and `violations`, according to the `rule_format` specification.
- Ensure the test data across DataFrames is coherent and allows for testing the interactions described in the rule.
- Accurately calculate support and confidence based on the *primary* DataFrame, as defined in the `rule_format`.
- Indexes in 'satisfactions_str' and 'violations_str' should refer to rows in the primary DataFrame unless explicitly stated otherwise by the rule format.
"""                         
        
        # Determine the correct dataframe JSON structure example for the prompt
        dataframe_json_structure_example_single = """
    "dataframe": {
      "column1": [column1_values],
      "column2": [column2_values],
      ...
    }""" # Indentation matters
        dataframe_json_structure_example_multi = """
    "dataframe": {
      "df_name1": {
         "columnA": [values_A],
         ...
      },
      "df_name2": {
         "columnB": [values_B],
         ...
      },
      ...
    }"""   
        
        dataframe_json_structure_example = dataframe_json_structure_example_multi if is_multi_df else dataframe_json_structure_example_single

        
        template = """
# Rule Description
{rule_description}

# Rule Format Specification
{rule_format}

{dataframe_info}

# Task
Create {num_test_cases} diverse and precise test case/s for validating this data quality rule. Each test case should include:

1. A small sample DataFrame or set of DataFrames (3-5 rows per DataFrame is sufficient). Format the `dataframe` field according to the requirements below.
2. The expected output values (support, confidence, satisfactions_indexes, violations_indexes)
3. The complete expected satisfactions and violations structures as Python string representations
4. A detailed explanation of why these values are expected (row-by-row breakdown)

{multi_df_instructions}

# Additional Requirements
- Make sure that the test cases help the llm to understand the rule and its implementation
- Give a detailed explanation of the expected values, including how they were calculated and why certain rows are included in satisfactions or violations
    
Test case/s should:
- Use minimal but sufficient data to validate both satisfying and violating scenarios
- Use the exact column names from the DataFrame info
- Include rows that both satisfy and violate the rule
- Have precisely calculated expected values
- Should include all rows in either satisfactions or violations, but not both, unless the rule is conditional

## Rule Examples with Test Cases

### Example 1: Single-Row Rule
**Rule Description**: If Column_A is greater than 10, then Column_B must be less than 5.

**Sample Test Case**:
```json
{{
  "dataframe": {{
    "Column_A": [15, 5, 20, 8, 12],
    "Column_B": [2, 10, 7, 3, 4]
  }},
  "expected_output": {{
    "support": 0.6,
    "confidence": 0.67,
    "satisfactions_str": "{{0, 4}}",
    "violations_str": "{{2}}"
  }}
}}
```

**Explanation**:
- Rows 0, 2, and 4 have Column_A > 10 (the rule body applies)
- Among those, rows 0 and 4 satisfy the rule (Column_B < 5)
- Row 2 violates the rule (Column_A > 10, but Column_B = 7, which is not < 5)
- Support = 3/5 = 0.6 (3 rows involved in the rule)
- Confidence = 2/3 = 0.67 (2 satisfying rows out of 3 rows where the body applies)

### Example 2: Multi-Row Functional Dependency Rule
**Rule Description**: For all rows with the same Department, the Manager should be the same.

**Sample Test Case**:
```json
{{
  "dataframe": {{
    "Department": ["Sales", "Engineering", "Sales", "Engineering", "Marketing"],
    "Manager": ["Smith", "Johnson", "Smith", "Wilson", "Davis"]
  }},
  "expected_output": {{
    "support": 1,
    "confidence": 0.66,
    "satisfactions_str": "{{(('Department', 'Sales')): {{('Manager', 'Smith'): [0, 2]}}, (('Department', 'Marketing')): {{('Manager', 'Davis'): [4]}}}}",
    "violations_str": "{{(('Department', 'Engineering')): {{('Manager', 'Johnson'): [1], ('Manager', 'Wilson'): [3]}}}}"
  }}
}}
```

**Explanation**:
- We have 2 groups where the rule applies (Sales and Engineering departments each appear multiple times)
- Sales group (rows 0, 2) has consistent Manager (Smith), so it satisfies the rule
- Engineering group (rows 1, 3) has different Managers (Johnson, Wilson), so it violates the rule
- Marketing only appears once, so it is satisfiying by default
- Support = 1 (5 out of 5 rows are involved in the rule - either in satisfactions or violations)
- Confidence = 0.66 (2 satisfying group out of 3 total groups)

### Example 3: Uniqueness Constraint Rule
**Rule Description**: The combination of ProductID and StoreID should be unique.

**Sample Test Case**:
```json
{{
  "dataframe": {{
    "ProductID": [101, 102, 101, 103, 102],
    "StoreID": [1, 2, 2, 1, 2]
  }},
  "expected_output": {{
    "support": 1,
    "confidence": 0.75,
    "satisfactions_str": "{{(('ProductID', 101), ('StoreID', 1)): {{'unique_record': 0}}, (('ProductID', 101), ('StoreID', 2)): {{'unique_record': 2}}, (('ProductID', 103), ('StoreID', 1)): {{'unique_record': 3}}}}",
    "violations_str": "{{(('ProductID', 102), ('StoreID', 2)): {{'duplicates': [1, 4]}}}}"
  }}
}}
```

**Explanation**:
- The combinations (101,1), (102,2), (101,2), (103,1) should all be unique
- But (102,2) appears twice (rows 1 and 4), violating the uniqueness constraint
- Rows 0, 2, 3 contain unique combinations
- Rows 1 and 4 are involved in the violation (same combination)
- Support = 1 (5 out of 5 rows are involved in the rule - either in satisfactions or violations)
- Confidence = 0.75 (unique combinations / total unique combinations = 3/4)

I want you to return {num_test_cases} test cases in a structured JSON format that can be easily parsed:

```json
[
  {{
    "name": "Test Case 1",
{dataframe_json_structure_example},
    "explanation": "Consise and clear explanation of why these values are expected detailing why certain rows are included in satisfactions or violations according to the rule",
    "expected_output": {{
      "support": 0.X,
      "confidence": 0.Y,
      "satisfactions_str": "string representation of the satisfactions structure",
      "violations_str": "string representation of the violations structure"
    }}
  }},
  {{
    "name": "Test Case 2",
    "dataframe": {{...}},
    "explanation": "...",
    "expected_output": {{...}}
  }},
  ...
]
```

IMPORTANT: 
- Double check that each test case is accurate and the expected values are calculated correctly
- Verify that the satisfaction and violation indexes are mutually exclusive (a row cannot be both)
- Make sure the dataframe uses the exact column names from the DataFrame info
- Keep the test cases simple - the goal is to validate the rule implementation, not to stress test it
- When calculating support and confidence, follow the exact formulas specified in the rule format
- For multi-row rules, it's fundamentally impossible to represent a violation with a single row index, since the rule is evaluating relationships between multiple rows. A violation always involves a group of rows that collectively fail to satisfy the rule's constraints.
- Multi-row rules often seem conditional since they work on groups of rows, but single-row groups are also possible and should be present in either the satisfactions or violations.
- For satisfactions_str and violations_str, provide Python-syntax string representations that could be evaluated with eval() to recreate the actual data structure
- Include only the columns needed to test the rule (don't include unnecessary columns)
- Make sure the JSON contains no numeric operators when representing the expected output values e.g. 0.5 instead of 1/2
- Calculate support and confidence values precisely according to the formula in the rule format
- For satisfactions_indexes and violations_indexes, provide the exact row indexes that should be present in the output
- For satisfactions and violations, provide the string representation of the Python structure as it would appear in the output
- The indexes present in satisfactions_str and satisfactions_indexes should match, and the same for violations_str and violations_indexes
"""
        
        # Create a JSON output parser
        json_parser = JsonOutputParser(pydantic_object=List[TestCase])
        
        chain = self._create_chain(
            template=template,
            parser=json_parser,
            run_name="RuleTestCaseGenerator"
        )
        
        result = chain.invoke({
            "rule_description": rule_description,
            "rule_format": rule_format,
            "dataframe_info": dataframe_info or "",
            "num_test_cases": num_test_cases,
            "multi_df_instructions": multi_df_instructions, 
            "dataframe_json_structure_example": dataframe_json_structure_example
        })
                
        return {
            "test_cases": result,
            "metadata": {
                "agent": self.name,
                "rule_description": rule_description,
                "num_test_cases": num_test_cases
            }
        }