import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
import traceback
import time
import multiprocessing
from pathlib import Path

# Add the parent directory to sys.path to import from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from agents.rule_orchestrator import RuleOrchestrator
from agents.base_agent import clean_json_output, extract_code
from single_prompt_generation import RuleObject, safe_format_prompt, summarize_dataset, generate_code_prompt

from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser

# --- Helper Functions ---

def convert_to_json_serializable(item):
    """Recursively converts an item to a JSON serializable format."""
    if isinstance(item, dict):
        return {convert_to_json_serializable(k): convert_to_json_serializable(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_to_json_serializable(i) for i in item]
    elif isinstance(item, set):
        try:
            return sorted([convert_to_json_serializable(i) for i in item])
        except TypeError:
            return sorted([str(i) for i in item])
    elif pd.isna(item):
        return None
    elif isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, np.floating):
        return float(item)
    elif isinstance(item, np.bool_):
        return bool(item)
    elif isinstance(item, pd.Timestamp):
        return item.isoformat()
    elif isinstance(item, (str, int, float, bool, type(None))):
        return item
    else:
        return str(item)

def code_execution_worker(code: str, function_name: str, result_queue: multiprocessing.Queue):
    """Worker function to be run in a separate process for safe execution."""
    try:
        namespace = {}
        exec(code, namespace)
        
        if function_name not in namespace:
            raise NameError(f"Function '{function_name}' not found in generated code.")
            
        dataset_generator_func = namespace[function_name]
        df = dataset_generator_func()
        result_queue.put(df)
    except Exception as e:
        # Put the full traceback string into the queue
        result_queue.put(traceback.format_exc())

# --- Core Classes ---

class DemonstrationGenerator:
    """
    An agent for generating datasets and rules from a knowledge base.
    """
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatVertexAI(
            model="gemini-2.5-pro", 
            temperature=config.AGENT_TEMPERATURE
        )

    def _call_llm(self, system_prompt: str, human_template: str, inputs: dict) -> str:
        """A reusable method to call the LLM."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("human", human_template)
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(inputs)

    def generate_dataset_creation_code(self, kb_content: str, rule_type: str, error_context: str = None) -> str:
        """Generates Python code to create a synthetic dataset, with optional error correction."""
        system_prompt = (
            "You are a data science expert specializing in creating synthetic datasets for testing data quality rules. "
            "You will be given a knowledge base about a type of rule. Your task is to generate Python code that uses "
            "pandas and the Faker library to create a DataFrame that is suitable for testing rules of this type. "
            "The generated code must be self-contained in a single function."
        )
        
        human_template = """
        Based on the provided knowledge base about '{rule_type}' rules, generate Python code for a function named `create_dataset()`.

        **Knowledge Base:**
        ```json
        {kb_content}
        ```
        {error_prompt}
        **Requirements for the `create_dataset()` function:**
        1.  It must use `pandas` and `Faker`.
        2.  It should generate a DataFrame with at least 100,000 rows.
        3.  The columns should be relevant to the rule type described in the knowledge base.
        4.  Crucially, the data must contain a mix of both **satisfying** and **violating** examples for rules of this type.
        5.  The function should take no arguments and return the generated pandas DataFrame.
        6.  The code should be complete, self-contained, and ready to be executed. Include all necessary imports.

        Your response MUST ONLY contain the Python code, wrapped in ```python and ``` markers.
        """
        
        error_prompt_str = ""
        if error_context:
            error_prompt_str = f"""
        **IMPORTANT: The previously generated code failed with the following error. Please analyze the traceback and provide a corrected version of the code.**
        ```
        {error_context}
        ```
        """
        
        response = self._call_llm(
            system_prompt, 
            human_template, 
            {
                "kb_content": kb_content, 
                "rule_type": rule_type,
                "error_prompt": error_prompt_str
            }
        )
        return extract_code(response)

    def generate_specific_rule(self, kb_content: str, rule_type: str, dataset_summary: str) -> dict:
        """Generates a specific rule instance based on the KB and a dataset summary."""
        system_prompt = (
            "You are an expert in data quality assurance. Your task is to define a single, specific, and testable data quality "
            "rule based on a general knowledge base and a concrete dataset schema. You must output only a valid JSON object."
        )

        human_template = """
        Given the knowledge base on '{rule_type}' rules and the following dataset summary, create one specific and representative rule that can be tested on this data.

        **Knowledge Base:**
        ```json
        {kb_content}
        ```

        **Dataset Summary and Sample:**
        ```json
        {dataset_summary}
        ```

        **Task:**
        Create a single rule instance that fits the rule type and is applicable to the dataset columns.

        Respond with ONLY a single JSON object with the following keys:
        - "rule_id": A short, descriptive ID for the rule (e.g., "zip_determines_city").
        - "rule": A clear, natural language description of the rule.
        - "explanation": A brief explanation of the rule's logic and purpose.
        - "rule_type": The general type of the rule (e.g., "Functional Dependency").
        - "relevant_columns": A list of column names from the dataset involved in this rule.
        """

        response = self._call_llm(
            system_prompt,
            human_template,
            {"kb_content": kb_content, "rule_type": rule_type, "dataset_summary": dataset_summary}
        )
        cleaned_response = clean_json_output(response)
        return json.loads(cleaned_response)

# --- Workflow Functions ---

def execute_generated_code(generator: DemonstrationGenerator, kb_content: str, rule_type: str, output_path: str, max_retries: int = 3, timeout_seconds: int = 300) -> pd.DataFrame:
    """Generates and executes Python code to create a dataset, with retries and timeout."""
    last_error = None
    for attempt in range(max_retries):
        print(f"\n[STEP 1/5] Generating and executing dataset code (Attempt {attempt+1}/{max_retries})...")
        
        # Step 1.1: Generate Code (with error context on retries)
        dataset_code = generator.generate_dataset_creation_code(kb_content, rule_type, last_error)
        if not dataset_code:
            last_error = "LLM returned an empty response."
            print(f"   > ERROR: {last_error} Retrying...")
            continue

        dataset_code_path = Path(output_path).parent / "generate_dataset.py"
        with open(dataset_code_path, 'w', encoding='utf-8') as f:
            f.write(dataset_code)
        print(f"   > Dataset generation code saved to {dataset_code_path}")

        # Step 1.2: Execute Code with Timeout
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=code_execution_worker, args=(dataset_code, "create_dataset", result_queue))
        
        try:
            process.start()
            result = result_queue.get(timeout=timeout_seconds)
            process.join()

            if isinstance(result, str): # Worker sent a traceback string
                raise RuntimeError(result)
            
            # Success!
            print("   > Successfully executed generated function 'create_dataset'.")
            print(f"   > Saving generated dataset to {output_path}...")
            result.to_csv(output_path, index=False)
            return result
        
        except multiprocessing.queues.Empty:
            last_error = f"Execution timed out after {timeout_seconds} seconds."
            print(f"   > ERROR: {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"   > ERROR: Failed to execute the generated dataset code. Full error below.")
            print(last_error)
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
    
    print(f"FATAL: Failed to generate and execute dataset code after {max_retries} attempts.")
    raise RuntimeError("Could not create dataset.")

def run_single_prompt_generation(rule: dict, dataset: pd.DataFrame, output_folder: Path, config: Config) -> str:
    """Generates the initial 'baseline' code using the single-prompt approach."""
    dataset_summary = summarize_dataset(dataset, n_samples=3)
    prompt_template = generate_code_prompt()
    llm = ChatVertexAI(model="gemini-2.5-pro", temperature=config.AGENT_TEMPERATURE)
    
    rule_obj = RuleObject(rule)
    formatted_prompt = safe_format_prompt(prompt_template, rule_obj, dataset_summary, "execute_rule").replace('{', '{{').replace('}', '}}')
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert Python programmer specializing in data quality assurance and pandas operations."),
        ("human", formatted_prompt)
    ])
    chain = chat_prompt | llm | StrOutputParser()
    
    response = chain.invoke({})
    code = extract_code(response)
    if not code:
        raise ValueError("Failed to extract code from the single-prompt LLM response.")
        
    code_file_path = output_folder / "rule_code.py"
    with open(code_file_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"   > Baseline code saved to {code_file_path}")
    return code

def run_multi_agent_improvement(rule: dict, dataset: pd.DataFrame, start_code: str, output_folder: Path, config: Config) -> bool:
    """Runs the full multi-agent workflow to improve the baseline code."""
    orchestrator = RuleOrchestrator(config)
    result = orchestrator.process_rule(
        rule_description=rule['rule'], dataframes=dataset, start_code=start_code,
        use_profiling=True, use_test_case_generation=True, use_test_case_review=True,
        use_code_correction=True, use_code_review=True, use_code_optimization=True,
        max_correction_attempts=5, max_restarts=3, test_percentage=1, fast_code_selection=True
    )

    history_filename = "multi_agent_history.json" if result.get('success') else "multi_agent_failure_history.json"
    with open(output_folder / history_filename, 'w', encoding='utf-8') as f:
        json.dump(result.get('results_history', []), f, indent=2, default=str)

    if result.get('success', False):
        final_code = result.get('code', '')
        code_file_path = output_folder / "rule_code_multi.py"
        print(f"   > Multi-agent workflow SUCCEEDED. Saving final code to {code_file_path}")
        with open(code_file_path, 'w', encoding='utf-8') as f:
            f.write(final_code)
    else:
        print("   > Multi-agent workflow FAILED. See history for details.")
        final_code = result.get('code', '')
        if final_code:
            with open(output_folder / "rule_code_multi_failed.py", 'w', encoding='utf-8') as f:
                f.write(final_code)
    
    return result.get('success', False)

def execute_final_code(code: str, dataset: pd.DataFrame, output_path: Path):
    """Executes the final, optimized code, times it, and saves the results."""
    print("   > Executing final optimized code...")
    try:
        namespace = {'pd': pd, 'np': np}
        exec(code, namespace)
        
        rule_func = namespace.get("execute_rule")
        if not rule_func:
            raise NameError("Function 'execute_rule' not found in the final code.")
            
        start_time = time.time()
        result = rule_func(dataset)
        execution_time = time.time() - start_time
        
        print(f"   > Final code executed in {execution_time:.4f} seconds.")
        
        output_data = {
            "execution_time_seconds": execution_time,
            "results": convert_to_json_serializable(result)
        }
        
        result_file_path = output_path / "final_execution_results.json"
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"   > Final execution results saved to {result_file_path}")

    except Exception:
        print("   > ERROR: Failed to execute the final optimized code. Full error below.")
        print(traceback.format_exc())

def process_kb_file(kb_path: Path, output_path: Path, config: Config) -> bool:
    """Runs the full generation and improvement pipeline for a single KB file."""
    generator = DemonstrationGenerator(config)

    with open(kb_path, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)
    
    rule_type = list(kb_data.keys())[0]
    kb_content = json.dumps(kb_data[rule_type], indent=2)

    # Step 1: Generate and Execute Dataset Code (with retries)
    dataset_csv_path = output_path / "synthetic_dataset.csv"
    dataset = execute_generated_code(generator, kb_content, rule_type, str(dataset_csv_path))

    # Step 2: Generate Specific Rule
    print("\n[STEP 2/5] Generating a specific rule for the new dataset...")
    dataset_summary = summarize_dataset(dataset, n_samples=5)
    specific_rule = generator.generate_specific_rule(kb_content, rule_type, dataset_summary)
    rule_json_path = output_path / "rule.json"
    with open(rule_json_path, 'w', encoding='utf-8') as f:
        json.dump(specific_rule, f, indent=4)
    print(f"   > Specific rule saved to {rule_json_path}")
    
    # Step 3: Single-Prompt Code Generation
    print("\n[STEP 3/5] Generating initial baseline code (single-prompt)...")
    baseline_code = run_single_prompt_generation(specific_rule, dataset, output_path, config)
        
    # Step 4: Multi-Agent Improvement
    print("\n[STEP 4/5] Improving baseline code with the multi-agent workflow...")
    success = run_multi_agent_improvement(specific_rule, dataset, baseline_code, output_path, config)
    
    # Step 5: Execute Final Code
    print("\n[STEP 5/5] Executing and timing the final improved code...")
    if success:
        final_code_path = output_path / "rule_code_multi.py"
        if final_code_path.exists():
            with open(final_code_path, 'r', encoding='utf-8') as f:
                final_code = f.read()
            execute_final_code(final_code, dataset, output_path)
    else:
        print("   > Multi-agent workflow failed. Skipping final execution.")
        
    return success

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate full demonstration packages for data quality rule types.")
    parser.add_argument("input_path", type=str, help="Path to a single JSON KB file or a directory containing them.")
    parser.add_argument("output_dir", type=str, help="Path to the main directory where all artifacts will be saved.")
    args = parser.parse_args()

    config = Config()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: Input path not found at {input_path}")
        sys.exit(1)

    kb_files = []
    if input_path.is_dir():
        kb_files.extend(sorted(input_path.glob("*.json")))
    elif input_path.is_file() and input_path.suffix == '.json':
        kb_files.append(input_path)
    else:
        print(f"ERROR: Input path must be a .json file or a directory.")
        sys.exit(1)

    if not kb_files:
        print(f"No .json knowledge base files found in {input_path}.")
        return

    print(f"Found {len(kb_files)} KB file(s) to process.")
    successful_kbs = []
    failed_kbs = []

    for kb_path in kb_files:
        kb_name = kb_path.stem
        rule_output_path = output_dir / kb_name
        rule_output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print(f"PROCESSING KB: {kb_path.name}")
        print(f"OUTPUT FOLDER: {rule_output_path}")
        print("="*80)

        try:
            success = process_kb_file(kb_path, rule_output_path, config)
            if success:
                successful_kbs.append(kb_path.name)
            else:
                failed_kbs.append(kb_path.name)
        except Exception as e:
            print(f"\nCRITICAL ERROR processing {kb_path.name}. Skipping to next file.")
            print(traceback.format_exc())
            failed_kbs.append(kb_path.name)

    print("\n" + "="*50)
    print("      ALL DEMONSTRATION GENERATION JOBS COMPLETE")
    print("="*50)
    print(f"Total Processed: {len(kb_files)}")
    print(f"✅ Successful: {len(successful_kbs)}")
    print(f"❌ Failed: {len(failed_kbs)}")
    if failed_kbs:
        print("\nFailed KBs:")
        for name in failed_kbs:
            print(f"  - {name}")
    print("="*50)

if __name__ == "__main__":
    # Set start method for multiprocessing to avoid issues on some platforms
    multiprocessing.set_start_method("spawn", force=True)
    main()