import os
import json
import pandas as pd
from pathlib import Path
import sys
import re
import asyncio
import aiofiles
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import traceback
from typing import List, Tuple
import numpy as np
import multiprocessing as mp

# Add the parent directory to sys.path to import from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from agents.rule_orchestrator import RuleOrchestrator

class RuleObject:
    """Simple class to hold rule data with attribute access"""
    def __init__(self, rule_dict):
        self.rule = rule_dict.get('rule', '')
        self.explanation = rule_dict.get('explanation', '')
        self.rule_id = rule_dict.get('rule_id', '')
        self.rule_type = rule_dict.get('rule_type', '')
        self.relevant_columns = rule_dict.get('relevant_columns', [])

def load_dataset(dataset_name, data_folder):
    """Load dataset from the data folder"""
    dataset_path = os.path.join(data_folder, f"{dataset_name.lower()}.csv")
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    else:
        print(f"Warning: Dataset {dataset_path} not found")
        return None

def has_existing_single_prompt_code(rule_file_path):
    """Check if rule_code.py already exists for this rule"""
    code_file_path = os.path.join(os.path.dirname(rule_file_path), "rule_code.py")
    return os.path.exists(code_file_path)

def has_existing_multi_agent_code(rule_file_path):
    """Check if rule_code_multi.py already exists for this rule"""
    code_file_path = os.path.join(os.path.dirname(rule_file_path), "rule_code_multi.py")
    return os.path.exists(code_file_path)

def read_single_prompt_code(rule_file_path):
    """Read the existing single prompt generated code"""
    code_file_path = os.path.join(os.path.dirname(rule_file_path), "rule_code.py")
    try:
        with open(code_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading single prompt code from {code_file_path}: {str(e)}")
        return None

def process_single_rule_worker(rule_data_tuple):
    """Worker function for processing a single rule - runs in separate process"""
    rule_file_path, dataset_data, config_dict, worker_id = rule_data_tuple
    
    try:
        # Reconstruct config object from dictionary
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        # Recreate dataset from data
        dataset = pd.DataFrame(dataset_data)
        
        # Load rule information
        with open(rule_file_path, 'r') as f:
            rule_data = json.load(f)
        
        rule_id = rule_data.get('rule_id', 'unknown')
        rule_description = rule_data.get('rule', '')
        
        # Read the existing single prompt code
        start_code = read_single_prompt_code(rule_file_path)
        if start_code is None:
            error_msg = f"Could not read single prompt code for rule {rule_id}"
            print(f"✗ [Worker {worker_id}] {error_msg}")
            return False, rule_id, error_msg
        
        print(f"⚡ [Worker {worker_id}] Starting multi-agent improvement for rule {rule_id}")
        
        # Create orchestrator instance in this process
        orchestrator = RuleOrchestrator(config)
        
        # Use the multi-agent orchestrator to improve the code
        result = orchestrator.process_rule(
            rule_description=rule_description,
            dataframes=dataset,
            use_profiling=True,
            use_test_case_generation=True,
            use_test_case_review=True,
            use_code_correction=True,
            use_code_review=True,
            use_code_optimization=True,
            max_correction_attempts=5,
            max_restarts=3,
            test_percentage=1,
            start_code=start_code,  # Pass the single prompt code as starting point
            fast_code_selection=True
        )
        
        if result.get('success', False):
            # Save the improved code
            improved_code = result.get('code', '')
            code_file_path = os.path.join(os.path.dirname(rule_file_path), "rule_code_multi.py")
            
            with open(code_file_path, 'w', encoding='utf-8') as f:
                f.write(improved_code)
            
            # Save the execution history for debugging
            history_file_path = os.path.join(os.path.dirname(rule_file_path), "multi_agent_history.json")
            with open(history_file_path, 'w', encoding='utf-8') as f:
                json.dump(result.get('results_history', []), f, indent=2, default=str)
            
            print(f"✓ [Worker {worker_id}] Improved code for rule {rule_id} in {os.path.dirname(rule_file_path)}")
            return True, rule_id, None
        else:
            error_msg = f"Multi-agent workflow failed: {result.get('summary', 'Unknown error')}"
            print(f"✗ [Worker {worker_id}] {error_msg} for rule {rule_id}")
            
            # Save the failure history for debugging
            history_file_path = os.path.join(os.path.dirname(rule_file_path), "multi_agent_failure_history.json")
            with open(history_file_path, 'w', encoding='utf-8') as f:
                json.dump(result.get('results_history', []), f, indent=2, default=str)
            
            return False, rule_id, error_msg
            
    except Exception as e:
        rule_id = "unknown"
        try:
            with open(rule_file_path, 'r') as f:
                rule_data = json.load(f)
            rule_id = rule_data.get('rule_id', 'unknown')
        except:
            pass
            
        error_msg = f"Error processing rule with multi-agent: {str(e)}"
        print(f"✗ [Worker {worker_id}] {error_msg} for rule {rule_id}")
        print(f"  Full traceback: {traceback.format_exc()}")
        return False, rule_id, error_msg

def find_rule_files_with_single_prompt_code(rules_folder):
    """Find all rule.json files that have existing single prompt code but no multi-agent code"""
    rule_files = []
    
    for root, dirs, files in os.walk(rules_folder):
        if "rule.json" in files:
            rule_file_path = os.path.join(root, "rule.json")
            
            # Check if single prompt code exists
            if not has_existing_single_prompt_code(rule_file_path):
                print(f"⏭️  Skipping {rule_file_path} - no single prompt code found")
                continue
            
            # Skip if multi-agent code already exists
            if has_existing_multi_agent_code(rule_file_path):
                print(f"⏭️  Skipping {rule_file_path} - rule_code_multi.py already exists")
                continue
            
            # Extract dataset name from path
            path_parts = Path(root).parts
            rules_folder_parts = Path(rules_folder).parts
            relative_parts = path_parts[len(rules_folder_parts):]
            
            if relative_parts:
                dataset_name = relative_parts[0]
                rule_files.append((rule_file_path, dataset_name))
    
    return rule_files

def process_rules_sequential(valid_rule_files: List[Tuple], datasets: dict, config: Config):
    """Process rules sequentially to avoid gRPC threading issues"""
    results = []
    
    for i, (rule_file_path, dataset_name) in enumerate(valid_rule_files):
        print(f"\n--- Processing Rule {i+1}/{len(valid_rule_files)} ---")
        
        dataset = datasets.get(dataset_name)
        if dataset is None:
            print(f"✗ Dataset {dataset_name} not found")
            results.append((False, f"unknown_{dataset_name}", f"Dataset {dataset_name} not found"))
            continue
        
        # Create a fresh orchestrator for each rule
        orchestrator = RuleOrchestrator(config)
        
        try:
            # Load rule information
            with open(rule_file_path, 'r') as f:
                rule_data = json.load(f)
            
            rule_id = rule_data.get('rule_id', 'unknown')
            rule_description = rule_data.get('rule', '')
            
            # Read the existing single prompt code
            start_code = read_single_prompt_code(rule_file_path)
            if start_code is None:
                error_msg = f"Could not read single prompt code for rule {rule_id}"
                print(f"✗ {error_msg}")
                results.append((False, rule_id, error_msg))
                continue
            
            print(f"⚡ Starting multi-agent improvement for rule {rule_id}")
            
            # Use the multi-agent orchestrator to improve the code
            result = orchestrator.process_rule(
                rule_description=rule_description,
                dataframes=dataset,
                use_profiling=True,
                use_test_case_generation=True,
                use_test_case_review=True,
                use_code_correction=True,
                use_code_review=True,
                use_code_optimization=True,
                max_correction_attempts=5,
                max_restarts=3,
                test_percentage=1,
                start_code=start_code,
                fast_code_selection=True
            )
            
            if result.get('success', False):
                # Save the improved code
                improved_code = result.get('code', '')
                code_file_path = os.path.join(os.path.dirname(rule_file_path), "rule_code_multi.py")
                
                with open(code_file_path, 'w', encoding='utf-8') as f:
                    f.write(improved_code)
                
                # Save the execution history for debugging
                history_file_path = os.path.join(os.path.dirname(rule_file_path), "multi_agent_history.json")
                with open(history_file_path, 'w', encoding='utf-8') as f:
                    json.dump(result.get('results_history', []), f, indent=2, default=str)
                
                print(f"✓ Improved code for rule {rule_id}")
                results.append((True, rule_id, None))
            else:
                error_msg = f"Multi-agent workflow failed: {result.get('summary', 'Unknown error')}"
                print(f"✗ {error_msg} for rule {rule_id}")
                
                # Save the failure history for debugging
                history_file_path = os.path.join(os.path.dirname(rule_file_path), "multi_agent_failure_history.json")
                with open(history_file_path, 'w', encoding='utf-8') as f:
                    json.dump(result.get('results_history', []), f, indent=2, default=str)
                
                results.append((False, rule_id, error_msg))
                
        except Exception as e:
            rule_id = "unknown"
            try:
                with open(rule_file_path, 'r') as f:
                    rule_data = json.load(f)
                rule_id = rule_data.get('rule_id', 'unknown')
            except:
                pass
                
            error_msg = f"Error processing rule with multi-agent: {str(e)}"
            print(f"✗ {error_msg} for rule {rule_id}")
            print(f"  Full traceback: {traceback.format_exc()}")
            results.append((False, rule_id, error_msg))
    
    return results

def main():
    """Main function to improve all existing single prompt codes using multi-agent workflow"""
    
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
    
    # Find all rule files that have single prompt code but no multi-agent code
    rule_files = find_rule_files_with_single_prompt_code(rules_folder)
    
    if not rule_files:
        print("No rule files found with single prompt code to improve!")
        return
    
    print(f"Found {len(rule_files)} rule files with single prompt code to improve")
    
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
    
    print(f"\nProcessing {len(valid_rule_files)} rules sequentially to avoid gRPC threading issues...")
    
    # Process rules sequentially to avoid gRPC issues
    results = process_rules_sequential(valid_rule_files, datasets, config)
    
    # Calculate results
    total_success = sum(1 for success, _, _ in results if success)
    total_failure = len(results) - total_success
    
    print(f"\n=== Final Summary ===")
    print(f"Successfully improved: {total_success}")
    print(f"Failed to improve: {total_failure}")
    print(f"Total rules processed: {len(valid_rule_files)}")
    print(f"Success rate: {total_success / len(valid_rule_files) * 100:.1f}%")

if __name__ == "__main__":
    main()