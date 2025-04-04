from .base_agent import BaseAgent
from langchain_core.output_parsers import JsonOutputParser

class RuleFormatAnalyzer(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert rule analysis agent that determines the appropriate structure and format 
        for implementing data quality rules. Your job is to analyze a rule description and determine what type of rule it is
        and the appropriate format for the output."""
        super().__init__(config, "RuleFormatAnalyzer", system_prompt)
    
    def process(self, rule_description, dataframe_info=None):
        """Analyze a rule description to determine the appropriate format for the output
        
        Args:
            rule_description (str): Description of the rule to implement
            dataframe_info (str, optional): DataFrame sample information to help understand the data structure
            
        Returns:
            dict: Rule type and format specification
        """
        template = """
# Rule Description
{rule_description}

{dataframe_info}

# Task
Analyze the provided rule and determine:
1. Whether it is a single-row rule or a multi-row rule
2. If it's a multi-row rule, whether it's a group-validation rule
3. The exact structure needed for the rule's output, including:
   - How support and confidence should be calculated
   - How satisfactions and violations should be structured
   - What specific keys should be used in the dictionaries

## Fundamental Concepts for Rule Evaluation:

### Rule Components:
- **Body (Antecedent)**: The condition part of a rule that must be true for the rule to be applicable. This is the "if" part of the rule. Quite often, there is no body in the rule and the rule is unconditional: e.g., "All values in column A must be unique".
- **Head (Consequent)**: The expected outcome if the body is true. What the rule is asserting should be true.

## Reference Information on Rule Types:

### Single-Row Rule:
A rule that evaluates each row independently, ensuring that values within the same row satisfy certain conditions or constraints.
These rules do not depend on relationships with other rows.

Example: "If ProductCategory is 'Electronics', then Price must be > 100"
- Body: ProductCategory is 'Electronics'
- Head: Price > 100

### Multi-Row Rule:
A rule that evaluates relationships, dependencies, or patterns across multiple rows in the dataset.
These rules involve comparisons, groupings, or validations involving more than one row.

Example: "For each CustomerID, all Orders must have the same ShippingAddress"
- Body: Rows sharing the same CustomerID
- Head: All ShippingAddress values must be identical within the group

### Group-Validation Rule:
A multi-row rule that involves grouping rows based on specific column values and evaluating whether certain criteria are satisfied within each group.
Examples include functional dependencies, unique key constraints, outlier detection, and aggregation constraints.

### Support and Confidence Calculation:

#### Support:
Support measures how many rows in the dataset are involved in the rule, relative to the total number of rows in the dataset.
- For single-row rules: Support = (Number of rows where the entire rule (body AND head) is satisfied) / (Total number of rows)
- For rules without a body (unconditional constraints): Support = 1.0
- For multi-row rules: 
    - Support = (Number of unique rows involved in satisfactions and violations) / (Total number of rows in the dataset)
        Note that number of groups means number of group_keys
    - Steps to Compute Support:
        - Extract all row indexes that appear in both violations and satisfactions.
        - Count the number of unique row indexes.
        - Divide this count by the total number of rows in the dataset.
        
#### Confidence:
Confidence measures how often the rule is satisfied when applied to the dataset.
- For single-row rules: Confidence = (Number of rows where the entire rule (body AND head) is satisfied) / (Number of rows where only the body of the rule is satisfied)
    - If the body is never satisfied (denominator is zero), confidence is typically defined as 1.0
    - For rules without a body: Confidence = Support
- For multi-row rules: 
    - Confidence = (Number of groups in satisfactions) / (Number of groups in violations + Number of groups in satisfactions)
    - Steps to Compute Confidence:
        - Count the number of group keys in the satisfactions dictionary.
        - Count the number of group keys in the violations dictionary.
        - Apply the formula using these counts.

### Satisfactions and Violations Structure:

#### Single-Row Rules:
- **Satisfactions**: Set of row indices where the rule is fully satisfied (both body and head are true)
- **Violations**: Set of row indices where the rule is violated (body is true but head is false)

#### Multi-Row Rules (Group-Validation):
- **Satisfactions**: Dictionary with group identifiers as keys and information about satisfying groups as values
- **Violations**: Dictionary with group identifiers as keys and information about violating groups as values

A data quality checking rule that applies to multiple rows in each check should have its **violations** and **satisfactions** represented as a dictionary:

{{
    group_key: 
        a list of dictionaries  
        or  
        a single dictionary
}}

Each `group_key` represents a group of rows, i.e., set of rows grouped by one or more columns. 
Each `group_key` can be structured as:
- A single tuple if the group key is defined on a single column, for example: `("employee_id", "tech112212")`.
- A tuple of key-value pairs if the group key is defined on multiple columns: `(("team_id", "sales"), ("employee_name", "Jack"))`.

Each group_key's corresponding value represents either violations or satisfactions of the rule in that group. 
This value:
- Can be a list of dictionaries (when multiple violation representations or satisfaction representations exist in the group).  
- Or a single dictionary (when a single violation representation or a single satisfaction representation is required.)

Each dictionary inside the corresponding value of a group_key is called a violation representation or a satisfaction representation.  

Each representation dictionary consists of keys that describe the roles played by different rows (or by different row groups) in the violation or satisfaction.  
The values corresponding to these keys store the row indexes involved in the representation.

In each representation dictionary, a key's corresponding value can be:
  - A single row index (if only one row plays that role).
  - A list of row indexes (if multiple rows play that role).

By default, each distinct role in a representation forms a "participant"—a set of rows or a single row that contribute to the rule violation or satisfaction in a specific way.
    - If there are multiple roles, then the participants of different roles together explain why the rule is violated or satisfied.
    - If there is only one role, then the rows assigned to that role collectively form a participant that explains how the rule is violated or satisfied.

Within each group, all participants together explain why the rule is violated or satisfied in that group.

## Sample Output Structure Reference:

### Single-Row Rules:
For single-row rules, the output structure typically looks like:

```
return {{
    "support": support_value,  # Float between 0 and 1
    "confidence": confidence_value,  # Float between 0 and 1
    "satisfactions": {{1, 5, 9, 12}},  # Set of row indices that satisfy the rule
    "violations": {{2, 3, 7, 10}}  # Set of row indices that violate the rule
}}
```

### Multi-Row Rules (Group-Validation):
For multi-row group validation rules, the structure might look like:

```
# Example 1: Sum-to-Total Comparison Rule
return {{
    "support": support_value,
    "confidence": confidence_value,
    "satisfactions": {{
        (("project_group", "marketing"), ("project_name", "AdBoost")): {{
            "sum": [7, 14, 18],  # Rows whose amounts are summed
            "compare": 20  # Row containing the total amount
        }}
    }},
    "violations": {{
        (("project_group", "technique"), ("project_name", "DeepClean")): {{
            "sum": [1, 6, 9],
            "compare": 11
        }}
    }}
}}

# Example 2: Unique Key Constraint Rule (Unconditional)
return {{
    "support": support_value,
    "confidence": confidence_value,
    "satisfactions": {{
        (("A", "700"), ("B", "800"), ("C", "900")): {{
            "unique_record": 7
        }}
    }},
    "violations": {{
        (("A", "100"), ("B", "200"), ("C", "300")): {{
            "duplicates": [1, 2, 3]
        }}
    }}
}}

# Example 3: Functional Dependency Rule (Uncoditional)
If two rows have the same values for key columns (e.g., "department", "employee_id"),  
then they must have the same value for a dependent column (e.g., "employee_name"). 

return {{
    "support": support_value,
    "confidence": confidence_value,
    "satisfactions": {{
        (("department", "tech"), ("employee_id", "e292122")): {{
            ("employee_name", "Yiyang Qianxi"): 122
        }}
    }},
    "violations": {{
        (("department", "tech"), ("employee_id", "e12121")): {{
            ("employee_name", "Jacky Ma"): [12, 15, 19],
            ("employee_name", "Leon Ma"): [20, 22, 28]
        }}
    }}
}}

Each group key represents a unique combination of key columns (`department`, `employee_id`), where the rule ensures that the dependent column (`employee_name`) must have consistent values.
```

## Your Analysis

Based on the rule description and DataFrame sample, provide a detailed analysis of the format needed for implementing this rule.

I want you to return your analysis as a JSON object with the following structure:

```json
{{
    "support_calculation": "Detailed explanation of how support should be calculated",
    "confidence_calculation": "Detailed explanation of how confidence should be calculated",
    "satisfactions_format": "Detailed explanation of the structure for satisfactions with example output",
    "violations_format": "Detailed explanation of the structure for violations with example output"
}}
```

IMPORTANT: 
- Make sure to use the exact column names from the DataFrame sample in your explanations
- The output format explanations should be detailed and clear with specific examples tailored to this rule
- For multi-row rules, clearly specify how groups should be formed and what keys should be used
- Make sure you check if the rule is conditional or unconditional and adjust the explanations accordingly. Especially mention that if the rule is unconditional, the support is 1.0 and that every row in the DataFrame is either a satisfaction or a violation.
"""
        
        # Create a JSON output parser
        json_parser = JsonOutputParser()
        
        chain = self._create_chain(
            template=template,
            parser=json_parser
        )
        
        result = chain.invoke({
            "rule_description": rule_description,
            "dataframe_info": dataframe_info or ""
        })
        
        # Convert the output format to a text representation for backward compatibility
        rule_format = f"""# Support Calculation
{result.get('support_calculation', '')}

# Confidence Calculation
{result.get('confidence_calculation', '')}

# Satisfactions Format
{result.get('satisfactions_format', '')}

# Violations Format
{result.get('violations_format', '')}
"""
                
        return {
            "rule_format": rule_format,
            "metadata": {
                "agent": self.name,
                "rule_description": rule_description
            }
        }