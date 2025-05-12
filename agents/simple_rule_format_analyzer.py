from .base_agent import BaseAgent
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel  # Added import

# Define Pydantic model for the expected output
class RuleFormatAnalysisOutput(BaseModel):
    support_calculation: str
    confidence_calculation: str
    satisfactions_format: str
    violations_format: str

# Simplified RuleFormatAnalyzer class that just returns single-row rule format
class RuleFormatAnalyzer(BaseAgent):
    def __init__(self, config):
        system_prompt = """
You are an expert rule analysis agent that determines the appropriate structure and format for implementing data quality rules. Your job is to analyze a rule description and determine what type of rule it is and the appropriate format for the output.
"""

        super().__init__(config, "RuleFormatAnalyzer", system_prompt)
    
    def process(self, rule_description, dataframe_info=None):
        """Analyze a rule description to determine the appropriate format for the output
        
        Args:
            rule_description (str): Description of the rule to implement
            dataframe_info (str, optional): DataFrame sample information to help understand the data structure
            
        Returns:
            dict: Rule type and format specification
        """
        
        is_multi_df = dataframe_info is not None and "--- DataFrame:" in dataframe_info

        multi_df_instructions = ""
        if is_multi_df:
            multi_df_instructions = """
## Multi-DataFrame Rules Specifics:

The rule involves multiple DataFrames (as indicated by the `--- DataFrame: ---` separators in the sample info):
- Identify the **primary DataFrame** where the rule's violations or satisfactions are primarily tracked (i.e., which DataFrame's row indices will be stored in the `satisfactions` and `violations` outputs).
- Many multi-DataFrame rules involve checking references or conditions between a primary DataFrame and one or more secondary DataFrames (e.g., "Column X in DataFrame A must exist in Column Y of DataFrame B").
- Even with multiple DataFrames, the rule often behaves like a **single-row rule** concerning the *primary* DataFrame. Each row in the primary DataFrame is checked against conditions potentially involving other DataFrames.
- Clearly state in your `satisfactions_format` and `violations_format` explanations **which DataFrame's indices** are being used.
- Support and Confidence calculations should generally be based on the rows of the **primary DataFrame** that are subject to the rule's conditions.
"""

        template = """
# Rule Description
{rule_description}

{dataframe_info}

# Task
Analyze the provided rule and determine:
The exact structure needed for the rule's output, including:
   - How support and confidence should be calculated
   - How satisfactions and violations should be structured/calculated

## Fundamental Concepts for Rule Evaluation:

### Rule Components:
- **Body (Antecedent)**: The condition part of a rule that must be true for the rule to be applicable. This is the "if" part of the rule, although not every "if" statement is a rule body. Quite often, there is no body in the rule and the rule is unconditional: e.g., "All values in column A must be unique".
- **Head (Consequent)**: The expected outcome if the body is true. What the rule is asserting should be true.

{multi_df_instructions}

### Support and Confidence Calculation:

#### Support:
Support measures how many rows in the dataset are involved in the rule, relative to the total number of rows in the dataset.
- For single-row rules: Support = (Number of rows where the entire rule (body AND head) is satisfied) / (Total number of rows)
- For rules without a body (unconditional constraints): Support = 1.0
        
#### Confidence:
Confidence measures how often the rule is satisfied when applied to the dataset.
- For single-row rules: Confidence = (Number of rows where the entire rule (body AND head) is satisfied) / (Number of rows where only the body of the rule is satisfied)
    - If the body is never satisfied (denominator is zero), confidence is typically defined as 1.0
    - For rules without a body: Confidence = Support

### Satisfactions and Violations Structure:

#### Single-Row Rules:
- **Satisfactions**: Set of row indices where the rule is fully satisfied (both body and head are true)
- **Violations**: Set of row indices where the rule is violated (body is true but head is false)

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

## Your Analysis

Based on the rule description and DataFrame sample, provide a detailed analysis of the format needed for implementing this rule.

I want you to return your analysis as a JSON object with the following structure:

```json
{{
    "support_calculation": "Detailed explanation of how support should be calculated",
    "confidence_calculation": "Detailed explanation of how confidence should be calculated",
    "satisfactions_format": "Detailed explanation of the structure for satisfactions",
    "violations_format": "Detailed explanation of the structure for violations"
}}
```

IMPORTANT: 
- If the rule description already describes the output format try to follow it as closely as possible.
- Make sure to use the exact column names from the DataFrame sample in your explanations
- The output format explanations should be detailed and clear with specific examples tailored to this rule
- Make sure you check if the rule is conditional or unconditional and adjust the explanations accordingly. Especially mention that if the rule is unconditional, the support is 1.0 and that every row in the DataFrame is either a satisfaction or a violation. 
- Conditional rules should be mentioned as well, in which case only the rows that satisfy the body of the rule should be present in the satisfactions or violations structures. Rows that do not satisfy the body of the rule should not be present in either structure.
- Some rules are formulated in a way that seem more complex than they are. Normally these rules can be reduced to a simpler form. For example, "If rows in question all have the same value in State, then Phone determines AreaCode." can be reduced to "If rows in question all have the same value in State and Phone, then AreaCode should be the same for all rows.". Try to reduce the rule to its simplest form and rely on this simpler form for your analysis.
"""
        
        # Create a JSON output parser with the Pydantic model
        json_parser = JsonOutputParser(pydantic_object=RuleFormatAnalysisOutput)
        
        chain = self._create_chain(
            template=template,
            parser=json_parser,  # Pass the configured parser
            run_name="RuleFormatAnalyzer"
        )
        
        result = chain.invoke({
            "rule_description": rule_description,
            "dataframe_info": dataframe_info or "",
            "multi_df_instructions": multi_df_instructions
        })
        
        # Convert the output format to a text representation for backward compatibility
        # Access attributes directly from the Pydantic object
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