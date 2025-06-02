from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import re
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.output_parsers import RegexParser
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
from typing import Union, Dict

class BaseAgent(ABC):
    def __init__(self, config, name, system_prompt):
        self.config = config
        self.name = name
        self.system_prompt = system_prompt
        # self.llm = ChatOpenAI(
        #     model=config.LLM_MODEL,
        #     temperature=config.AGENT_TEMPERATURE,
        #     api_key=config.LLM_API_KEY,
        #     base_url=config.LLM_BASE_URL,
        # )
        self.llm = ChatVertexAI(
            model=config.LLM_MODEL,
            temperature=config.AGENT_TEMPERATURE,
            thinking_budget=0
        )
        
    @abstractmethod
    def process(self, input_data):
        """Process the input and return the result"""
        pass
    
    def _create_chain(self, template, parser: BaseOutputParser = StrOutputParser(), parse_with_prompt=True, run_name="Sequence"):
        """Create an LLM chain with the given template and input variables"""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            ("human", template)
        ])
        
        # Extract the content from AIMessage before passing to parser
        def extract_content(chain_output):
            completion_content = chain_output["completion"].content
            prompt_value = chain_output["prompt_value"]
            # For JSON parsers, clean the output first
            if isinstance(parser, JsonOutputParser):
                completion_content = clean_json_output(completion_content)
                parser_retry = RetryWithErrorOutputParser.from_llm(
                    self.llm,
                    parser,
                    max_retries=3
                )
                return parser_retry.parse_with_prompt(completion_content, prompt_value=prompt_value)
            elif not parse_with_prompt:
                return {"content": completion_content}
            else:
                extracted = extract_code(completion_content)
                if extracted:
                    return {"code": extracted}
        
            return parser.parse(completion_content)
    
        chain = prompt | self.llm
        main_chain = (RunnableParallel(
            completion=chain, prompt_value=prompt
        ) | RunnableLambda(extract_content)).with_config({"run_name": run_name})
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
            max_retries=3
        )

def _create_dataframe_sample(df_input: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
    """Create a representative sample of the DataFrame(s) for the prompt"""
    formatted_sample = ""

    if isinstance(df_input, dict):
        # Handle multiple DataFrames
        for name, df in df_input.items():
            formatted_sample += f"--- DataFrame: {name} ---\n"
            # Determine sample size - balance between informativeness and prompt size
            sample_size = min(3, len(df))
            sample_df = df.head(sample_size)

            # Format as a pretty printed table
            formatted_sample += "DataFrame Sample (returned by df.head(sample_size)):\n"
            formatted_sample += sample_df.to_string()

            # Add number of distinct values and dtype per column
            formatted_sample += "\n\nColumn Details (Distinct Values, Dtype):\n"
            distinct_counts = df.nunique(dropna=False)
            dtypes = df.dtypes
            for col in df.columns:
                formatted_sample += f"- {col}: {distinct_counts[col]}, {dtypes[col]}\n"
            formatted_sample += "\n" # Add a separator between DataFrames
    else:
        # Handle single DataFrame (existing logic)
        df = df_input
        # Determine sample size - balance between informativeness and prompt size
        sample_size = min(3, len(df))
        sample_df = df.head(sample_size)

        # Format as a pretty printed table
        formatted_sample += "DataFrame Sample (returned by df.head(sample_size)):\n"
        formatted_sample += sample_df.to_string()

        # Add number of distinct values and dtype per column
        formatted_sample += "\n\nColumn Details (Distinct Values, Dtype):\n"
        distinct_counts = df.nunique(dropna=False)
        dtypes = df.dtypes
        for col in df.columns:
            formatted_sample += f"- {col}: {distinct_counts[col]}, {dtypes[col]}\n"

    return formatted_sample.strip() # Remove trailing newline if any

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

def common_improvement_recommendations(is_multi_dataframe=False):
    """Common improvement recommendations for code optimization"""   
        
    prompt = """
Common Improvement Recommendations:
1. Iterating Inefficiently: Using .iterrows() or .apply() for row-wise operations instead of vectorized operations can significantly reduce performance. If vectorized operations are not possible (which is the fastest option by far), consider using map()/applymap(), .apply(), .itertuples(), or .iterrows() in that order of preference.
2. Creating Unnecessary Copies: Creating unnecessary copies of the dataframe wastes memory. Work on slices or views whenever possible.
3. Misusing Inplace Operations: Using inplace=True carelessly can overwrite data unintentionally. Avoid inplace operations unless absolutely necessary.
4. Use Built-in Pandas and NumPy Functions that have implemented C like 'sum()', 'mean()', or 'max()' when needed/possible.
5. Use vectorized operations that can apply to entire DataFrames and Series including mathematical operations, comparisons, and logic to create a boolean mask to select multiple rows from your data set.
6. Avoid lambda functions in groupby() and apply() methods. Instead, use built-in functions or vectorized operations whenever possible.
7. Try to combine multiple operations into a single pass through the data. For example, instead of grouping multiple times on different columns, group once and aggregate all necessary columns in that single pass.
8. Try filtering DataFrames with boolean masks for better performance.
9. Avoid manual unique combination tracking and manual index collection; let pandas handle with/during grouping.
10. Avoid filtering the DataFrame in loops and nested grouping; instead, process groups directly
"""

# 7. Pandas has optimized operations based on indices, allowing for faster lookup or merging tables based on indices. Single lookups using indices outperform other methods with great margin. When retrieving a single value, using .at[] is faster than using .loc[]. Setting indexes on columns used for grouping might speed up groupby operations.

    if is_multi_dataframe:
        prompt += """
IMPORTANT:
For numeric ID comparisons ALWAYS convert columns using `astype(int).astype(str)` instead of just `astype(str)` ONLY if the datatypes don't match, e.g int vs float, see dataframe info. Otherwise there might be issues with the formatting of the string.
"""
    return prompt
    

def common_mistakes_prompt():
    """Prompt for common mistakes in code debugging tasks"""
    return """
Common Mistakes to Avoid When You Generate Code:
1. Failing to Prepare Data for Regular Expressions: Not converting data to the correct format (e.g., strings) before applying regular expressions can lead to errors or unintended matches.
2. Ignoring Missing Values: Failing to account for NaN values in the dataset can lead to unexpected results. Always handle missing values using fillna() or dropna() appropriately.
3. Using Chained Indexing: Using chained indexing (e.g., df[df['column'] > 0]['column2'] = value) can lead to SettingWithCopy warnings and incorrect assignments. Use .loc[] instead.
4. Allowing Data Type Mismatches: Not ensuring columns have consistent data types before operations can cause errors. Use df.astype() to cast types where needed.
5. Mismanaging Index Operations: Resetting or setting indexes carelessly can disrupt the integrity of the dataframe. Always verify the dataframe after index operations.
6. Assuming Column Order is Fixed: Relying on column order instead of column names can break the code if the order changes unexpectedly.
7. Overwriting Critical Data: Overwriting important variables inadvertently can cause loss of critical data. Use meaningful variable names to track changes.
8. Recalculating Intermediate Results: Recomputing the same results multiple times instead of storing them in temporary variables wastes resources.
9. Hardcoding Values: Hardcoding column names, thresholds, or parameters reduces flexibility. Use variables or configuration files instead.
10. Skipping Code Documentation: Failing to add comments for non-trivial operations makes the code harder to understand and maintain.
11. Overlooking Duplicate Entries: Ignoring duplicate rows or entries can affect data quality checks. Use df.duplicated() to identify and handle duplicates.
12. Using Inconsistent Column Name Case: Referencing column names inconsistently with case sensitivity can cause KeyErrors in datasets with varying conventions.
13. Altering Indices Unintentionally: Changing indices during intermediate steps without tracking the original index can cause alignment issues.
14. If the rule is unconditional every row in the DataFrame is either a satisfaction or a violation.
15. Make sure that you don't overwrite the existing entries already contained in the inner dictionaries of the satisfactions or violations.
16. Keep the keys used in the dictionaries for satisfactions or violations of group validation rules, including keys in both outer and inner dictionaries, contextually relevant based on the column names.
17. Do not use libraries other than pandas and numpy that are not available in the standard library.
18. Do not write unfinished code, make sure that the code is complete and also don't use auxiliary functions that are not defined in the code.
19. Do not define variables and constants outside the main function.
20. Only reference varibles after they are defined/assigned.
21. Some columns in the DataFrame might be of type string or represented as strings, but they are not actually strings, so convert them to the appropriate type before using them in the code.
"""

def common_generation_prompt():
    """Common generation prompt for code generation tasks"""
    return """
Keep in mind the following when generating code:
1. IMPORTANT: If the rule is not about checking None values, please exclude all rows where any of the relevant columns is None, using df.dropna(subset=relevant_columns), before analysis. None values should not be considered as satisfactions or violations except when explicitly stated in the rule.
"""