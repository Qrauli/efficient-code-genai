from .base_agent import BaseAgent
from langchain_core.output_parsers import JsonOutputParser

class RuleTestCaseGenerator(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert test case generator for data quality rules. Your job is to create precise, 
        accurate test cases that validate the implementation of data quality rules on pandas DataFrames."""
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
        template = """
# Rule Description
{rule_description}

# Rule Format Specification
{rule_format}

{dataframe_info}

# Task
Create {num_test_cases} diverse and precise test case/s for validating this data quality rule. Each test case should include:

1. A small sample DataFrame (3-5 rows is sufficient)
2. The expected output values (support, confidence, satisfactions_indexes, violations_indexes)
3. The complete expected satisfactions and violations structures as Python string representations
4. A detailed explanation of why these values are expected (row-by-row breakdown)

# Additional Requirements
- Make sure that the test cases help the llm to understand the rule and its implementation
- Give a detailed explanation of the expected values, including how they were calculated and why certain rows are included in satisfactions or violations
    
Test case/s should:
- Use minimal but sufficient data to validate both satisfying and violating scenarios
- Use the exact column names from the DataFrame info
- Include rows that both satisfy and violate the rule
- Have precisely calculated expected values

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
    "support": 0.4,
    "confidence": 0.67,
    "satisfactions_indexes": [0, 4],
    "violations_indexes": [2],
    "satisfactions_str": "{{0, 4}}",
    "violations_str": "{{2}}"
  }}
}}
```

**Explanation**:
- Rows 0, 2, and 4 have Column_A > 10 (the rule body applies)
- Among those, rows 0 and 4 satisfy the rule (Column_B < 5)
- Row 2 violates the rule (Column_A > 10, but Column_B = 7, which is not < 5)
- Support = 2/5 = 0.4 (2 satisfying rows out of 5 total rows)
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
    "support": 0.8,
    "confidence": 0.5,
    "satisfactions_indexes": [0, 2],
    "violations_indexes": [1, 3],
    "satisfactions_str": "{{(('Department', 'Sales')): {{('Manager', 'Smith'): [0, 2]}}}}",
    "violations_str": "{{(('Department', 'Engineering')): {{('Manager', 'Johnson'): [1], ('Manager', 'Wilson'): [3]}}}}"
  }}
}}
```

**Explanation**:
- We have 2 groups where the rule applies (Sales and Engineering departments each appear multiple times)
- Sales group (rows 0, 2) has consistent Manager (Smith), so it satisfies the rule
- Engineering group (rows 1, 3) has different Managers (Johnson, Wilson), so it violates the rule
- Marketing only appears once, so it's not relevant for this rule
- Support = 0.8 (4 out of 5 rows are involved in the rule - either in satisfactions or violations)
- Confidence = 0.5 (1 satisfying group out of 2 total groups)

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
    "support": 0.6,
    "confidence": 0.67,
    "satisfactions_indexes": [0, 2, 3],
    "violations_indexes": [1, 4],
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
- Support = 0.6 (unique combinations / total rows = 3/5)
- Confidence = 0.75 (unique combinations / total unique combinations = 3/4)

I want you to return {num_test_cases} test cases in a structured JSON format that can be easily parsed:

```json
{{
  "test_cases": [
    {{
      "name": "Test Case 1: [Brief description of what aspect this tests]",
      "dataframe": {{
        "column1": [column1_values],
        "column2": [column2_values],
        ...
      }},
      "explanation": "Detailed explanation of why these values are expected",
      "expected_output": {{
        "support": 0.X,
        "confidence": 0.Y,
        "satisfactions_indexes": [list of row indexes that satisfy the rule e.g. [0, 1]],
        "violations_indexes": [list of row indexes that violate the rule e.g. [2, 3]],
        "satisfactions_str": "string representation of the satisfactions structure",
        "violations_str": "string representation of the violations structure"
      }}
    }},
    {{
      "name": "Test Case 2: [Brief description of what aspect this tests]",
      "dataframe": {{...}},
      "explanation": "...",
      "expected_output": {{...}}
    }},
    ...
  ]
}}
```

IMPORTANT: 
- Double check that each test case is accurate and the expected values are calculated correctly
- Verify that the satisfaction and violation indexes are mutually exclusive (a row cannot be both)
- Make sure the dataframe uses the exact column names from the DataFrame info
- Keep the test cases simple - the goal is to validate the rule implementation, not to stress test it
- When calculating support and confidence, follow the exact formulas specified in the rule format
- For multi-row rules, it's fundamentally impossible to represent a violation with a single row index, since the rule is evaluating relationships between multiple rows. A violation always involves a group of rows that collectively fail to satisfy the rule's constraints.
- For satisfactions_str and violations_str, provide Python-syntax string representations that could be evaluated with eval() to recreate the actual data structure
- Include only the columns needed to test the rule (don't include unnecessary columns)
- Calculate support and confidence values precisely according to the formula in the rule format
- For satisfactions_indexes and violations_indexes, provide the exact row indexes that should be present in the output
- For satisfactions and violations, provide the string representation of the Python structure as it would appear in the output

"""
        
        # Create a JSON output parser
        json_parser = JsonOutputParser()
        
        chain = self._create_chain(
            template=template,
            parser=json_parser
        )
        
        result = chain.invoke({
            "rule_description": rule_description,
            "rule_format": rule_format,
            "dataframe_info": dataframe_info or "",
            "num_test_cases": num_test_cases
        })
        
        return {
            "test_cases": result.get("test_cases", []),
            "metadata": {
                "agent": self.name,
                "rule_description": rule_description,
                "num_test_cases": num_test_cases
            }
        }