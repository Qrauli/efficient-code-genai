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
    
    def _create_chain(self, template, parser: BaseOutputParser = StrOutputParser()):
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
            else:
                extracted = extract_code(completion_content)
                if extracted:
                    return {"code": extracted}
        
            return parser.parse_with_prompt(completion=completion_content, prompt_value=chain_output["prompt_value"])
    
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