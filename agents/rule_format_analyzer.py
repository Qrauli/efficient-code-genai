from .base_agent import BaseAgent
import pandas as pd

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
Analyze this rule and determine:
1. Whether it is a single-row rule or a multi-row rule
2. If it's a multi-row rule, whether it's a group-validation rule
3. The exact structure needed for the rule's output, including:
   - How support and confidence should be calculated
   - How satisfactions and violations should be structured
   - What specific keys should be used in the dictionaries

## Fundamental Concepts for Rule Evaluation:

### Rule Components:
- **Body (Antecedent)**: The condition part of a rule that must be true for the rule to be applicable.
- **Head (Consequent)**: The expected outcome if the body is true. What the rule is asserting should be true.

### Support and Confidence Calculation:

#### Support:
Support measures how frequently the rule appears in the dataset.
- For single-row rules: Support = (Number of rows where the entire rule (body AND head) is satisfied) / (Total number of rows)
- For rules without a body (unconditional constraints): Support = 1.0
- For multi-row rules: Support may need to be adjusted based on the specific rule type

#### Confidence:
Confidence measures the reliability of the rule.
- Confidence = (Number of rows where the entire rule (body AND head) is satisfied) / (Number of rows where only the body of the rule is satisfied)
- If the body is never satisfied (denominator is zero), confidence is typically defined as 1.0
- For rules without a body: Confidence = Support

### Satisfactions and Violations Structure:

#### Single-Row Rules:
- **Satisfactions**: Set of row indices where the rule is fully satisfied (both body and head are true)
- **Violations**: Set of row indices where the rule is violated (body is true but head is false)

#### Multi-Row Rules (Group-Validation):
- **Satisfactions**: Dictionary with group identifiers as keys and information about satisfying groups as values
- **Violations**: Dictionary with group identifiers as keys and information about violating groups as values

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

# Example 2: Unique Key Constraint Rule
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

# Example 3: Functional Dependency Rule
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
```

## Your Analysis

Based on the rule description and DataFrame sample, provide:
1. Rule Type Classification: Single-row, multi-row, or group-validation rule
2. Output Format Specification: Detailed structure of the output including how to format the support, confidence, satisfactions, and violations
3. Calculation Guidance: How to calculate support and confidence for this specific rule

Structure your response with clear headings:
    
# Support Calculation
[Explanation of how support should be calculated]

# Confidence Calculation
[Explanation of how confidence should be calculated]

# Satisfactions Format
[Explanation of the structure for satisfactions with examples]

# Violations Format
[Explanation of the structure for violations with examples]

IMPORTANT: Your response should focus only on the format and structure, not the implementation details or code generation.
"""
        
        chain = self._create_chain(
            template=template,
            parse_with_prompt=False
        )
        
        result = chain.invoke({
            "rule_description": rule_description,
            "dataframe_info": dataframe_info
        })
        
        return {
            "rule_format": result.get("content", {}),
            "metadata": {
                "agent": self.name,
                "rule_description": rule_description
            }
        }