from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import re
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.output_parsers import RegexParser
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

class BaseAgent(ABC):
    def __init__(self, config, name, system_prompt):
        self.config = config
        self.name = name
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.AGENT_TEMPERATURE,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        
    @abstractmethod
    def process(self, input_data):
        """Process the input and return the result"""
        pass
    
    def _create_chain(self, template, parser: BaseOutputParser = StrOutputParser(), parse_with_prompt=True):
        """Create an LLM chain with the given template and input variables"""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            ("human", template)
        ])
        
        # Extract the content from AIMessage before passing to parser
        def extract_content(chain_output):
            completion_content = chain_output["completion"].content
            
            # For JSON parsers, clean the output first
            if isinstance(parser, JsonOutputParser) or isinstance(parser, RetryWithErrorOutputParser):
                completion_content = clean_json_output(completion_content)
            elif not parse_with_prompt:
                return {"content": completion_content}
            else:
                extracted = extract_code(completion_content)
                if extracted:
                    return {"code": extracted}
        
            return parser.parse(completion_content)
    
        chain = prompt | self.llm
        main_chain = RunnableParallel(
            completion=chain, prompt_value=prompt
        ) | RunnableLambda(extract_content)
        return main_chain

    def _extract_code_parser(self):
        """Create a code block parser for LLM responses"""
        # Create a more robust regex pattern that can handle multiline code
        regex_parser = RegexParser(
        regex=r"```(?:python)?\n([\s\S]*?)\n```", 
            output_keys=["code"],
            default_output_key="code"
        )
        
        # Wrap with RetryWithErrorOutputParser to handle parsing failures
        return RetryWithErrorOutputParser.from_llm(
            self.llm,
            regex_parser,
            max_retries=2
        )

def _create_dataframe_sample(df):
    """Create a representative sample of the DataFrame for the prompt"""
    # Determine sample size - balance between informativeness and prompt size
    sample_size = min(3, len(df))
    sample_df = df.head(sample_size)
    
    # Format as a pretty printed table
    formatted_sample = "DataFrame Sample (returned by df.head(sample_size)):\n"
    formatted_sample += sample_df.to_string()
    
    # Add explicit column information with types
    formatted_sample += "\n\nDataFrame Column Names (MUST USE THESE EXACT NAMES):\n"
    for col in df.columns:
        formatted_sample += f"- {col}\n"
    
    return formatted_sample

def clean_json_output(text):
    """Remove code block markers from the output before JSON parsing"""
    # First check if the text contains markdown code block markers
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    
    # If no code blocks, try to find JSON-like content starting with [ or {
    # This helps when the model returns raw JSON without code blocks
    json_pattern = r"(\[[\s\S]*\]|\{[\s\S]*\})"
    match = re.search(json_pattern, text)
    if match:
        return match.group(1).strip()
    
    # If all else fails, return the original text
    return text

def extract_code(text):
    """Extract code from LLM response, handling markdown code blocks properly"""
    # First try to find code in markdown code blocks
    pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    
    if match:
        # Return just the code content without the language identifier
        return match.group(1).strip()
    
    # If no code blocks found, return the original text
    return text

def common_mistakes_prompt():
    """Prompt for common mistakes in code debugging tasks"""
    return """
Common Mistakes to Avoid When You Generate Code:
1. Failing to Prepare Data for Regular Expressions: Not converting data to the correct format (e.g., strings) before applying regular expressions can lead to errors or unintended matches.
2. Ignoring Missing Values: Failing to account for NaN values in the dataset can lead to unexpected results. Always handle missing values using fillna() or dropna() appropriately.
3. Using Chained Indexing: Using chained indexing (e.g., df[df['column'] > 0]['column2'] = value) can lead to SettingWithCopy warnings and incorrect assignments. Use .loc[] instead.
4. Iterating Inefficiently: Using .iterrows() or .apply() for row-wise operations instead of vectorized operations can significantly reduce performance.
5. Allowing Data Type Mismatches: Not ensuring columns have consistent data types before operations can cause errors. Use df.astype() to cast types where needed.
6. Creating Unnecessary Copies: Creating unnecessary copies of the dataframe wastes memory. Work on slices or views whenever possible.
7. Mismanaging Index Operations: Resetting or setting indexes carelessly can disrupt the integrity of the dataframe. Always verify the dataframe after index operations.
8. Assuming Column Order is Fixed: Relying on column order instead of column names can break the code if the order changes unexpectedly.
9. Overwriting Critical Data: Overwriting important variables inadvertently can cause loss of critical data. Use meaningful variable names to track changes.
10. Ignoring Memory Usage: Processing large datasets without monitoring memory usage can lead to crashes. Use df.info() and optimize operations for memory efficiency.
11. Recalculating Intermediate Results: Recomputing the same results multiple times instead of storing them in temporary variables wastes resources.
12. Hardcoding Values: Hardcoding column names, thresholds, or parameters reduces flexibility. Use variables or configuration files instead.
13. Ignoring Performance Warnings: Overlooking warnings or errors in the console may result in performance or correctness issues.
14. Skipping Code Documentation: Failing to add comments for non-trivial operations makes the code harder to understand and maintain.
15. Neglecting Thorough Testing: Not testing the code with edge cases like empty dataframes, extreme values, or unexpected structures can result in undetected bugs.
16. Overlooking Duplicate Entries: Ignoring duplicate rows or entries can affect data quality checks. Use df.duplicated() to identify and handle duplicates.
17. Using Inconsistent Column Name Case: Referencing column names inconsistently with case sensitivity can cause KeyErrors in datasets with varying conventions.
18. Failing to Validate Output: Not verifying that data quality rules are applied correctly can result in undetected errors. Check flagged rows and passing rows.
19. Altering Indices Unintentionally: Changing indices during intermediate steps without tracking the original index can cause alignment issues.
20. Misusing Inplace Operations: Using inplace=True carelessly can overwrite data unintentionally. Avoid inplace operations unless absolutely necessary.
"""