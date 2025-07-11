import os
import json
import pandas as pd
from pathlib import Path
import sys
import re
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import threading
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
import traceback
from typing import List, Tuple
import numpy as np
# Add the parent directory to sys.path to import from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from rc_prompt_template import generate_code_prompt, summarize_dataset

class RuleObject:
    """Simple class to hold rule data with attribute access"""
    def __init__(self, rule_dict):
        self.rule = rule_dict.get('rule', '')
        self.explanation = rule_dict.get('explanation', '')
        self.rule_id = rule_dict.get('rule_id', '')
        self.rule_type = rule_dict.get('rule_type', '')
        self.relevant_columns = rule_dict.get('relevant_columns', [])
        
def convert_to_json_serializable(item):
    """Converts an item to a JSON serializable format."""
    if isinstance(item, dict):
        return {convert_to_json_serializable(k): convert_to_json_serializable(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_to_json_serializable(i) for i in item]
    elif isinstance(item, set):
        # Sort for consistent output, especially for comparison if needed later
        # Ensure elements are comparable or convert to string for sorting key
        try:
            return sorted([convert_to_json_serializable(i) for i in item])
        except TypeError: # Handle unorderable types if they occur
            return sorted([convert_to_json_serializable(i) for i in item], key=str)
    elif pd.isna(item):  # Handles np.nan, pd.NaT, etc. Must come before float/int checks for np.nan
        return None
    elif isinstance(item, (np.integer, np.int64)):
        return int(item.item())
    elif isinstance(item, (np.floating, np.float64)):
        return float(item.item())
    elif isinstance(item, np.bool_):
        return bool(item.item())
    elif isinstance(item, pd.Timestamp):
        return item.isoformat()
    elif isinstance(item, (str, int, float, bool, type(None))):
        return item
    else:
        # Fallback for other types: convert to string
        return str(item)

def safe_format_prompt(template, rule_obj, sample, fun_name):
    """Safely format the prompt template by replacing placeholders manually"""
    
    # Replace the specific placeholders we know about
    formatted = template.replace("{rule.rule}", rule_obj.rule)
    formatted = formatted.replace("{rule.explanation}", rule_obj.explanation)
    formatted = formatted.replace("{sample}", sample)
    formatted = formatted.replace("{fun_name}", fun_name)
    
    return formatted

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

def load_dataset(dataset_name, data_folder):
    """Load dataset from the data folder"""
    dataset_path = os.path.join(data_folder, f"{dataset_name.lower()}.csv")
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    else:
        print(f"Warning: Dataset {dataset_path} not found")
        return None

def has_existing_code(rule_file_path):
    """Check if rule_code.py already exists for this rule"""
    code_file_path = os.path.join(os.path.dirname(rule_file_path), "rule_code.py")
    return os.path.exists(code_file_path)

def process_rule_file(rule_file_path, dataset, config, thread_id=None):
    """Process a single rule file and generate code"""
    
    rule_id = "unknown"
    try:
        # Load rule information
        with open(rule_file_path, 'r') as f:
            rule_data = json.load(f)
        
        rule_id = rule_data.get('rule_id', 'unknown')
        
        # Create LLM instance - each thread gets its own instance
        llm = ChatVertexAI(
            model="gemini-2.5-pro",
            temperature=config.AGENT_TEMPERATURE
        )
        
        # Generate dataset summary
        dataset_summary = summarize_dataset(dataset, n_samples=3)
        
        # Create the prompt template
        prompt_template = generate_code_prompt()
        
        rule_obj = RuleObject(rule_data)

        formatted_prompt = safe_format_prompt(
            prompt_template,
            rule_obj,
            dataset_summary,
            "execute_rule"
        ).replace('{', '{{').replace('}', '}}')
        
        # Create the chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an expert Python programmer specializing in data quality assurance and pandas operations."),
            ("human", formatted_prompt)
        ])
        
        # Create the chain
        chain = chat_prompt | llm | StrOutputParser()
        
        # Generate the response
        response = chain.invoke({})
        
        # Extract code from response
        extracted_code = extract_code(response)
        
        if extracted_code:
            # Save the generated code
            code_file_path = os.path.join(os.path.dirname(rule_file_path), "rule_code.py")
            with open(code_file_path, 'w') as f:
                f.write(extracted_code)
            
            print(f"✓ [Thread {thread_id}] Generated code for rule {rule_id} in {os.path.dirname(rule_file_path)}")
            return True, rule_id, None
        else:
            error_msg = f"Failed to extract code from response"
            print(f"✗ [Thread {thread_id}] {error_msg} for rule {rule_id}")
            # Save the raw response for debugging
            debug_file_path = os.path.join(os.path.dirname(rule_file_path), "raw_response.txt")
            with open(debug_file_path, 'w') as f:
                f.write(response)
            return False, rule_id, error_msg
            
    except Exception as e:
        error_msg = f"Error processing rule: {str(e)}"
        print(f"✗ [Thread {thread_id}] {error_msg} for rule {rule_id}")
        print(f"  Full traceback: {traceback.format_exc()}")
        return False, rule_id, error_msg

def find_rule_files(rules_folder):
    """Find all rule.json files in the rules folder structure"""
    rule_files = []
    
    for root, dirs, files in os.walk(rules_folder):
        if "rule.json" in files:
            rule_file_path = os.path.join(root, "rule.json")
            
            # Skip if code already exists
            if has_existing_code(rule_file_path):
                print(f"⏭️  Skipping {rule_file_path} - rule_code.py already exists")
                continue
            
            # Extract dataset name from path
            path_parts = Path(root).parts
            rules_folder_parts = Path(rules_folder).parts
            relative_parts = path_parts[len(rules_folder_parts):]
            
            if relative_parts:
                dataset_name = relative_parts[0]
                rule_files.append((rule_file_path, dataset_name))
    
    return rule_files

def process_rules_batch(batch_data: List[Tuple], datasets: dict, config: Config, max_workers: int = 30):
    """Process a batch of rules concurrently"""
    
    def worker(rule_data):
        rule_file_path, dataset_name, thread_id = rule_data
        dataset = datasets.get(dataset_name)
        if dataset is None:
            return False, f"unknown_{dataset_name}", f"Dataset {dataset_name} not found"
        
        return process_rule_file(rule_file_path, dataset, config, thread_id)
    
    # Add thread IDs to the batch data
    batch_with_ids = [(rule_file_path, dataset_name, i) for i, (rule_file_path, dataset_name) in enumerate(batch_data)]
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, rule_data) for rule_data in batch_with_ids]
        
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout per rule
                results.append(result)
            except Exception as e:
                print(f"✗ Worker thread failed: {str(e)}")
                results.append((False, "unknown", str(e)))
    
    return results

def main():
    """Main function to process all rules"""
    
    # Initialize configuration
    config = Config()
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    rules_folder = os.path.join(current_dir, "rules_and_programs")
    data_folder = os.path.join(current_dir, "")
    
    print(f"Looking for rules in: {rules_folder}")
    print(f"Looking for datasets in: {data_folder}")
    
    if not os.path.exists(rules_folder):
        print(f"Error: Rules folder not found at {rules_folder}")
        return
    
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found at {data_folder}")
        return
    
    # Find all rule files (excluding those with existing code)
    rule_files = find_rule_files(rules_folder)
    
    if not rule_files:
        print("No new rule files found to process!")
        return
    
    print(f"Found {len(rule_files)} new rule files to process")
    
    # Load datasets once
    datasets = {}
    unique_datasets = set(dataset_name for _, dataset_name in rule_files)
    
    print(f"Loading {len(unique_datasets)} unique datasets...")
    for dataset_name in unique_datasets:
        dataset = load_dataset(dataset_name, data_folder)
        if dataset is not None:
            datasets[dataset_name] = dataset
            print(f"✓ Loaded dataset: {dataset_name}")
        else:
            print(f"✗ Failed to load dataset: {dataset_name}")
    
    # Filter out rules for missing datasets
    valid_rule_files = [(rule_file, dataset_name) for rule_file, dataset_name in rule_files 
                       if dataset_name in datasets]
    
    if len(valid_rule_files) != len(rule_files):
        print(f"Filtered out {len(rule_files) - len(valid_rule_files)} rules due to missing datasets")
    
    if not valid_rule_files:
        print("No valid rule files to process!")
        return
    
    print(f"\nProcessing {len(valid_rule_files)} rules concurrently (max 30 at once)...")
    
    # Process rules in batches of 30
    batch_size = 30
    total_success = 0
    total_failure = 0
    
    for i in range(0, len(valid_rule_files), batch_size):
        batch = valid_rule_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(valid_rule_files) + batch_size - 1) // batch_size
        
        print(f"\n--- Processing Batch {batch_num}/{total_batches} ({len(batch)} rules) ---")
        
        results = process_rules_batch(batch, datasets, config, max_workers=min(30, len(batch)))
        
        batch_success = sum(1 for success, _, _ in results if success)
        batch_failure = len(results) - batch_success
        
        total_success += batch_success
        total_failure += batch_failure
        
        print(f"Batch {batch_num} completed: {batch_success} success, {batch_failure} failed")
    
    print(f"\n=== Final Summary ===")
    print(f"Successfully processed: {total_success}")
    print(f"Failed to process: {total_failure}")
    print(f"Total rules processed: {len(valid_rule_files)}")
    print(f"Success rate: {total_success / len(valid_rule_files) * 100:.1f}%")

if __name__ == "__main__":
    main()