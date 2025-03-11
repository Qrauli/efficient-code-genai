import sys
sys.path.append("..")

from .base_agent import BaseAgent
from retrieval.retriever import Retriever

class CodeGenerator(BaseAgent):
    def __init__(self, config):
        system_prompt = """You are an expert code generation agent specialized in writing efficient code, particularly for data science tasks. 
        Your goal is to generate high-quality, optimized code based on the given requirements and relevant examples.
        ALWAYS wrap your code in ```python and ``` markers."""
        super().__init__(config, "CodeGenerator", system_prompt)
        self.retriever = Retriever(config)
        
    def process(self, problem_description, test_cases=None):
        """Generate initial code based on problem description and optional test cases"""
        # Retrieve relevant code examples
        relevant_examples = self.retriever.retrieve(problem_description)
        
        # Create a prompt with problem description, test cases, and retrieved examples
        if test_cases:
            template = """
            Requirements: {problem_description}
            
            Test Cases (your code must satisfy these):
            {test_cases}
            
            Relevant examples:
            {relevant_examples}
            
            Generate efficient and high-quality code that solves the given requirements and passes all the provided test cases.
            Include only the code that is necessary to meet the requirements and no test code.
            Focus particularly on performance for data science tasks.
            
            Your response should ONLY contain the python code and nothing else.
            """
            
            parser = self._extract_code_parser()
            chain = self._create_chain(
                template=template
            )
            
            result = chain.invoke({
                "problem_description": problem_description,
                "test_cases": self._format_test_cases(test_cases),
                "relevant_examples": relevant_examples
            }
            )
        else:
            # Use original template without test cases
            template = """ 
            Requirements: {problem_description}
            
            Relevant examples:
            {relevant_examples}
            
            Generate efficient and high-quality code that solves the given requirements. 
            Focus particularly on performance for data science tasks.
            IMPORTANT: Wrap your complete solution in ```python and ``` markers.
            """
            
            chain = self._create_chain(
                template=template
            )
            
            result = chain.invoke({
                "problem_description": problem_description,
                "relevant_examples": relevant_examples
            }
            )
        
        return {
            "code": result['code'],
            "metadata": {
                "agent": self.name,
                "retrieved_examples": relevant_examples,
                "test_cases_provided": test_cases is not None
            }
        }
        
    def _format_test_cases(self, test_cases):
        """Format test cases for inclusion in prompt"""
        formatted = ""
        for i, test in enumerate(test_cases):
            formatted += f"Test {i+1}: {test.get('name', 'Unnamed test')}\n"
            formatted += f"Function Call: {test.get('function_call', '')}\n"
            formatted += f"Expected Output: {test.get('expected_output', '')}\n"
            formatted += f"Description: {test.get('description', '')}\n\n"
        return formatted
