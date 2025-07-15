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
import html

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

def safe_read(file_path: Path) -> str:
    """Safely reads content from a file path, returning a default message if not found or unreadable."""
    if not file_path or not file_path.exists():
        return "File not found."
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

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
        print(f"   > Generating and executing dataset code (Attempt {attempt+1}/{max_retries})...")
        
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

# --- HTML Report Generation (Overhauled) ---

def _render_simple_list_view(indices: list, df: pd.DataFrame, max_rows: int = 20) -> str:
    """Renders a simple list/set of row indices by showing their data."""
    if not indices:
        return "<p>None found.</p>"
    
    sample_indices = indices[:max_rows]
    data_sample = df.loc[sample_indices]
    
    table_html = data_sample.to_html(classes='dataframe', border=0, index=True)
    
    # Wrap table in a scrollable container
    final_html = f'<div class="table-container">{table_html}</div>'

    if len(indices) > max_rows:
        final_html += f"<p><i>... and {len(indices) - max_rows:,} more.</i></p>"
        
    return final_html

def _render_group_based_view(groups: dict, df: pd.DataFrame, max_groups: int = 10, max_rows_per_role: int = 5) -> str:
    """Renders a complex group-based dictionary of violations/satisfactions."""
    if not groups:
        return "<p>None found.</p>"

    def format_key(key):
        try:
            # Handle stringified tuple keys from JSON
            if isinstance(key, str) and key.startswith('('):
                key = eval(key)
            if isinstance(key, tuple):
                return ", ".join([f"<code>{k}</code> = {html.escape(str(v))}" for k, v in key])
            return html.escape(str(key))
        except:
            return html.escape(str(key))

    rows = []
    for i, (group_key, representations) in enumerate(groups.items()):
        if i >= max_groups:
            break

        group_str = format_key(group_key)
        
        # Standardize: a group's representations can be a single dict or a list of dicts.
        if not isinstance(representations, list):
            representations = [representations]

        details_html = '<div class="representation-block">'
        for rep in representations:
            details_html += '<div class="representation">'
            if not isinstance(rep, dict):
                details_html += f"<p>Invalid representation format: {html.escape(str(rep))}</p>"
                continue

            for role, indices in rep.items():
                # Standardize: indices can be a single int or a list.
                if not isinstance(indices, list):
                    indices = [indices]
                
                details_html += f"<h5>Role: <code>{html.escape(role)}</code> ({len(indices)} rows)</h5>"
                
                if indices:
                    sample_indices = indices[:max_rows_per_role]
                    data_sample = df.loc[sample_indices]
                    sample_table_html = data_sample.to_html(classes='dataframe dataframe-small', border=0, index=True)
                    # Wrap inner table in scrollable container
                    details_html += f'<div class="table-container">{sample_table_html}</div>'
                    if len(indices) > max_rows_per_role:
                        details_html += f"<p><i>... and {len(indices) - max_rows_per_role:,} more.</i></p>"
                else:
                    details_html += "<p>No rows for this role.</p>"
            details_html += '</div>'
        details_html += '</div>'

        rows.append(f"<tr><td>{group_str}</td><td>{details_html}</td></tr>")
    
    header = '<thead><tr><th>Group</th><th>Participants</th></tr></thead>'
    table_html = f'<table class="dataframe">{header}<tbody>{"".join(rows)}</tbody></table>'
    
    # Wrap the main group table in a scrollable container as well
    final_html = f'<div class="table-container">{table_html}</div>'
    
    if len(groups) > max_groups:
        final_html += f"<p><i>... and {len(groups) - max_groups:,} more groups.</i></p>"
        
    return final_html

def _render_results_details(data: dict, df: pd.DataFrame, data_type_name: str) -> str:
    """Dispatcher to render violation/satisfaction data based on its structure."""
    if not data:
        return f"<p>No {data_type_name} found.</p>"
    
    # Case 1: Single-Row Rule (output is a list/set of indices)
    if isinstance(data, list):
        return _render_simple_list_view(data, df)
    
    # Case 2: Multi-Row Rule (output is a dictionary of groups)
    if isinstance(data, dict):
        return _render_group_based_view(data, df)
        
    return f"<p>Unrecognized format for {data_type_name}: {html.escape(str(type(data)))}</p>"


def generate_html_report(output_folder: Path):
    """Generates a self-contained HTML report summarizing the demonstration artifacts."""
    print(f"   > Reading artifacts from {output_folder} to generate report...")

    # --- Read all artifacts safely ---
    rule_path = output_folder / "rule.json"
    rule_data = json.loads(safe_read(rule_path)) if rule_path.exists() else {}

    dataset_path = output_folder / "synthetic_dataset.csv"
    df, df_head_html, df_describe_html, dataset_shape, dataset_cols_html = None, "Not available", "Not available", ("?", "?"), "Not available"
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        dataset_shape = df.shape
        df_head_html = df.head(10).to_html(classes='dataframe', border=0, index=False)
        df_describe_html = df.describe(include='all').to_html(classes='dataframe', border=0)
        dataset_cols_html = pd.DataFrame(df.columns, columns=['Column Name']).to_html(classes='dataframe', border=0, index=False)

    results_path = output_folder / "final_execution_results.json"
    violations_html, satisfactions_html = "<p>Execution results not found.</p>", ""
    num_violations, num_satisfactions, support, confidence, exec_time = 0, 0, "N/A", "N/A", "N/A"

    if results_path.exists():
        results_data = json.loads(safe_read(results_path))
        exec_time = f"{results_data.get('execution_time_seconds', 0):.4f}s"
        rule_results = results_data.get('results', {})
        
        support_val = rule_results.get('support', 0)
        confidence_val = rule_results.get('confidence', 0)
        support = f"{support_val:.2%}" if isinstance(support_val, (int, float)) else "N/A"
        confidence = f"{confidence_val:.2%}" if isinstance(confidence_val, (int, float)) else "N/A"
        
        violations = rule_results.get('violations', {})
        satisfactions = rule_results.get('satisfactions', {})

        num_violations = len(violations)
        num_satisfactions = len(satisfactions)

        if df is not None:
            violations_html = _render_results_details(violations, df, "violations")
            satisfactions_html = _render_results_details(satisfactions, df, "satisfactions")
        else:
            violations_html = "<p>Dataset file not found, cannot display row data.</p>"
            satisfactions_html = "<p>Dataset file not found, cannot display row data.</p>"


    dataset_code = safe_read(output_folder / "generate_dataset.py")
    baseline_code = safe_read(output_folder / "rule_code.py")
    final_code_path = output_folder / "rule_code_multi.py"
    final_code = safe_read(final_code_path)
    
    history_path = output_folder / "multi_agent_history.json"
    if not history_path.exists():
        history_path = output_folder / "multi_agent_failure_history.json"
    history_json = safe_read(history_path)
    multi_agent_success = "Success" if "multi_agent_history.json" in str(history_path) else "Failure"

    # --- HTML Template ---
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Demonstration Report: {html.escape(rule_data.get('rule_id', 'Unknown Rule'))}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; }}
            h1, h2, h3, h4, h5 {{ color: #1a1a1a; }}
            h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            details {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 1em; overflow: hidden; }}
            summary {{ font-weight: bold; font-size: 1.1em; padding: 1em; cursor: pointer; display: block; background: #f7f7f7; }}
            details[open] > summary {{ border-bottom: 1px solid #ddd; }}
            .content {{ padding: 1em; }}
            pre {{ background: #2d2d2d; color: #f2f2f2; padding: 1em; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }}
            code {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; background-color: #eef; padding: 2px 4px; border-radius: 3px; font-size: 0.9em; }}
            pre code {{ background: transparent; color: inherit; padding: 0; border-radius: 0; font-size: 1em; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1em; margin-bottom: 2em; }}
            .summary-card {{ background: #fff; padding: 1.5em; border-radius: 8px; border: 1px solid #ddd; text-align: center; }}
            .summary-card h3 {{ margin-top: 0; color: #555; font-size: 1em; text-transform: uppercase; }}
            .summary-card p {{ font-size: 1.6em; font-weight: bold; margin: 0; color: #007bff; word-wrap: break-word; line-height: 1.2; }}
            .dataframe {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }}
            .dataframe th, .dataframe td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; vertical-align: top; }}
            .dataframe th {{ background-color: #f2f2f2; }}
            .dataframe-small {{ font-size: 0.85em; }}
            .representation-block {{ display: flex; flex-direction: column; gap: 1em; }}
            .representation {{ border: 1px solid #eee; padding: 0.5em 1em; border-radius: 4px; }}
            .table-container {{ overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>Demonstration Report for <code>{html.escape(rule_data.get('rule_id', 'N/A'))}</code></h1>
        <h2>Overview</h2>
        <div class="summary-grid">
            <div class="summary-card"><h3>Rule Type</h3><p>{html.escape(rule_data.get('rule_type', 'N/A'))}</p></div>
            <div class="summary-card"><h3>Dataset Shape</h3><p>{dataset_shape[0]:,} x {dataset_shape[1]}</p></div>
            <div class="summary-card"><h3>Violations</h3><p style="color: #dc3545;">{num_violations:,}</p></div>
            <div class="summary-card"><h3>Satisfactions</h3><p style="color: #28a745;">{num_satisfactions:,}</p></div>
            <div class="summary-card"><h3>Support</h3><p>{support}</p></div>
            <div class="summary-card"><h3>Confidence</h3><p>{confidence}</p></div>
            <div class="summary-card"><h3>Exec Time</h3><p>{exec_time}</p></div>
            <div class="summary-card"><h3>Agent Workflow</h3><p>{multi_agent_success}</p></div>
        </div>
        <details>
            <summary>Rule Definition</summary>
            <div class="content">
                <h3>{html.escape(rule_data.get('rule_id', 'N/A'))}</h3>
                <p><strong>Rule:</strong> {html.escape(rule_data.get('rule', 'N/A'))}</p>
                <p><strong>Explanation:</strong> {html.escape(rule_data.get('explanation', 'N/A'))}</p>
                <p><strong>Relevant Columns:</strong> {html.escape(', '.join(rule_data.get('relevant_columns', [])))}</p>
                <h4>Full JSON:</h4><pre><code>{html.escape(json.dumps(rule_data, indent=2))}</code></pre>
            </div>
        </details>
        <details>
            <summary>Generated Dataset</summary>
            <div class="content">
                <p>A synthetic dataset was generated to test this rule. <a href="synthetic_dataset.csv" target="_blank" rel="noopener noreferrer">Download full CSV</a>.</p>
                <h3>Dataset Schema (Columns)</h3><div class="table-container">{dataset_cols_html}</div>
                <h3>Data Sample (First 10 Rows)</h3><div class="table-container">{df_head_html}</div>
                <h3>Data Statistics (.describe())</h3><div class="table-container">{df_describe_html}</div>
            </div>
        </details>
        <details>
            <summary>Execution Results</summary>
            <div class="content">
                <p>The final code was executed on the full dataset. <a href="final_execution_results.json" target="_blank" rel="noopener noreferrer">Download full results JSON</a>.</p>
                <h3>Summary</h3>
                <ul>
                    <li><strong>Violations:</strong> {num_violations:,}</li>
                    <li><strong>Satisfactions:</strong> {num_satisfactions:,}</li>
                    <li><strong>Support:</strong> {support}</li>
                    <li><strong>Confidence:</strong> {confidence}</li>
                    <li><strong>Execution Time:</strong> {exec_time}</li>
                </ul>
                <h3>Violations (Sample)</h3>{violations_html}
                <h3>Satisfactions (Sample)</h3>{satisfactions_html}
            </div>
        </details>
        <details>
            <summary>Generated Code</summary>
            <div class="content">
                <h3>Final Improved Code (<code>rule_code_multi.py</code>)</h3><pre><code>{html.escape(final_code)}</code></pre>
                <h3>Baseline Code (<code>rule_code.py</code>)</h3><pre><code>{html.escape(baseline_code)}</code></pre>
                <h3>Dataset Generation Code (<code>generate_dataset.py</code>)</h3><pre><code>{html.escape(dataset_code)}</code></pre>
            </div>
        </details>
        <details>
            <summary>Multi-Agent Workflow History</summary>
            <div class="content">
                <p>This is the JSON log of the multi-agent refinement process. <a href="{history_path.name}" target="_blank" rel="noopener noreferrer">Download full history</a>.</p>
                <pre><code>{html.escape(history_json)}</code></pre>
            </div>
        </details>
    </body>
    </html>
    """
    
    # --- Write the report ---
    report_path = output_folder / "demonstration_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"   > HTML report saved to {report_path}")


def process_kb_file(kb_path: Path, output_path: Path, config: Config) -> bool:
    """Runs the full generation and improvement pipeline for a single KB file."""
    generator = DemonstrationGenerator(config)

    with open(kb_path, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)
    
    rule_type = list(kb_data.keys())[0]
    kb_content = json.dumps(kb_data[rule_type], indent=2)

    # --- Step 1: Generate and Execute Dataset Code ---
    dataset_csv_path = output_path / "synthetic_dataset.csv"
    if dataset_csv_path.exists():
        print(f"\n[STEP 1/6] Found existing dataset. Loading from {dataset_csv_path}...")
        dataset = pd.read_csv(dataset_csv_path)
    else:
        print("\n[STEP 1/6] Generating new dataset...")
        dataset = execute_generated_code(generator, kb_content, rule_type, str(dataset_csv_path))

    # --- Step 2: Generate Specific Rule ---
    rule_json_path = output_path / "rule.json"
    if rule_json_path.exists():
        print(f"\n[STEP 2/6] Found existing rule definition. Loading from {rule_json_path}...")
        with open(rule_json_path, 'r', encoding='utf-8') as f:
            specific_rule = json.load(f)
    else:
        print("\n[STEP 2/6] Generating a specific rule...")
        dataset_summary = summarize_dataset(dataset, n_samples=5)
        specific_rule = generator.generate_specific_rule(kb_content, rule_type, dataset_summary)
        with open(rule_json_path, 'w', encoding='utf-8') as f:
            json.dump(specific_rule, f, indent=4)
        print(f"   > Specific rule saved to {rule_json_path}")
    
    # --- Step 3: Single-Prompt Code Generation ---
    baseline_code_path = output_path / "rule_code.py"
    if baseline_code_path.exists():
        print(f"\n[STEP 3/6] Found existing baseline code. Loading from {baseline_code_path}...")
        with open(baseline_code_path, 'r', encoding='utf-8') as f:
            baseline_code = f.read()
    else:
        print("\n[STEP 3/6] Generating initial baseline code...")
        baseline_code = run_single_prompt_generation(specific_rule, dataset, output_path, config)
        
    # --- Step 4: Multi-Agent Improvement ---
    final_code_path = output_path / "rule_code_multi.py"
    failed_code_path = output_path / "rule_code_multi_failed.py"
    multi_agent_success = False
    if final_code_path.exists():
        print(f"\n[STEP 4/6] Found existing successful multi-agent code. Skipping improvement.")
        multi_agent_success = True
    elif failed_code_path.exists():
        print(f"\n[STEP 4/6] Found existing failed multi-agent code. Skipping improvement.")
    else:
        print("\n[STEP 4/6] Improving baseline code with the multi-agent workflow...")
        multi_agent_success = run_multi_agent_improvement(specific_rule, dataset, baseline_code, output_path, config)
    
    # --- Step 5: Execute Final Code ---
    final_results_path = output_path / "final_execution_results.json"
    if final_results_path.exists():
        print(f"\n[STEP 5/6] Found existing final execution results. Skipping execution.")
    elif multi_agent_success:
        print("\n[STEP 5/6] Executing and timing the final improved code...")
        if final_code_path.exists():
            with open(final_code_path, 'r', encoding='utf-8') as f:
                final_code = f.read()
            execute_final_code(final_code, dataset, output_path)
    else:
        print("\n[STEP 5/6] Multi-agent workflow was not successful. Skipping final execution.")
        
    # --- Step 6: Generate HTML Report ---
    report_path = output_path / "demonstration_report.html"
    if report_path.exists():
        print(f"\n[STEP 6/6] HTML report already exists. Skipping generation.")
    else:
        if rule_json_path.exists() and dataset_csv_path.exists():
            print("\n[STEP 6/6] Generating HTML report...")
            generate_html_report(output_path)
        else:
            print("\n[STEP 6/6] Cannot generate report: Missing key files (rule, dataset).")

    return multi_agent_success

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