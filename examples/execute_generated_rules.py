import os
import json
import pandas as pd
import traceback
import time
import gc
import psutil
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
import sys
import importlib.util
import subprocess
from typing import Dict, Any, Optional

# Add the parent directory to sys.path to import from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_to_json_serializable(item):
    """Converts an item to a JSON serializable format."""
    if isinstance(item, dict):
        return {convert_to_json_serializable(k): convert_to_json_serializable(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_to_json_serializable(i) for i in item]
    elif isinstance(item, set):
        try:
            return sorted([convert_to_json_serializable(i) for i in item])
        except TypeError:
            return sorted([convert_to_json_serializable(i) for i in item], key=str)
    elif pd.isna(item):
        return None
    elif hasattr(item, 'item'):  # numpy types
        try:
            return item.item()
        except (ValueError, TypeError):
            return str(item)
    elif isinstance(item, pd.Timestamp):
        return item.isoformat()
    elif isinstance(item, (str, int, float, bool, type(None))):
        return item
    else:
        return str(item)

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return -1

def execute_single_rule(rule_info: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single rule in an isolated process"""
    rule_file_path, dataset_path, rule_id = rule_info['rule_file_path'], rule_info['dataset_path'], rule_info['rule_id']
    
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        dataset_size = len(df)
        
        # Get initial memory
        initial_memory = get_memory_usage()
        
        # Load and execute the rule code
        code_file_path = os.path.join(os.path.dirname(rule_file_path), "rule_code_multi.py")
        
        if not os.path.exists(code_file_path):
            return {
                'rule_id': rule_id,
                'error': f'Code file not found: {code_file_path}',
                'time_cost': -1,
                'memory_usage_mb': -1,
                'code_type': 'single_reasoning',
                'dataset_size': dataset_size
            }
        
        # Load the Python module
        spec = importlib.util.spec_from_file_location("rule_module", code_file_path)
        rule_module = importlib.util.module_from_spec(spec)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Execute the module to define the function
            spec.loader.exec_module(rule_module)
            
            # Get the execute_rule function
            if hasattr(rule_module, 'execute_rule'):
                execute_func = rule_module.execute_rule
            else:
                return {
                    'rule_id': rule_id,
                    'error': 'Function execute_rule not found in code',
                    'time_cost': -1,
                    'memory_usage_mb': -1,
                    'code_type': 'single_reasoning',
                    'dataset_size': dataset_size
                }
            
            # Execute the rule
            result = execute_func(df)
            
            # End timing
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Get final memory and calculate usage
            final_memory = get_memory_usage()
            memory_usage = final_memory - initial_memory if initial_memory > 0 and final_memory > 0 else -1
            
            # Extract results from the function output
            if isinstance(result, dict):
                # Convert to JSON serializable format
                json_result = convert_to_json_serializable(result)
                
                # Add metadata
                json_result['rule_id'] = rule_id
                json_result['time_cost'] = execution_time
                json_result['memory_usage_mb'] = memory_usage
                json_result['code_type'] = 'single_reasoning'
                json_result['dataset_size'] = dataset_size
                
                return json_result
            else:
                return {
                    'rule_id': rule_id,
                    'error': f'Function returned unexpected type: {type(result)}',
                    'time_cost': execution_time,
                    'memory_usage_mb': memory_usage,
                    'code_type': 'single_reasoning',
                    'dataset_size': dataset_size
                }
                
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            final_memory = get_memory_usage()
            memory_usage = final_memory - initial_memory if initial_memory > 0 and final_memory > 0 else -1
            
            return {
                'rule_id': rule_id,
                'error': f'Error: {str(e)}',
                'time_cost': execution_time if execution_time < 300 else -1,
                'memory_usage_mb': memory_usage,
                'code_type': 'single_reasoning',
                'dataset_size': dataset_size
            }
            
    except Exception as e:
        return {
            'rule_id': rule_id,
            'error': f'Setup error: {str(e)}',
            'time_cost': -1,
            'memory_usage_mb': -1,
            'code_type': 'single_reasoning',
            'dataset_size': -1
        }
    finally:
        # Clean up
        gc.collect()

def has_execution_result(rule_folder_path: str) -> bool:
    """Check if execution result already exists"""
    result_file_path = os.path.join(rule_folder_path, "result.json")
    return os.path.exists(result_file_path)

def save_execution_result(rule_folder_path: str, result: Dict[str, Any]):
    """Save execution result to JSON file"""
    result_file_path = os.path.join(rule_folder_path, "result.json")
    try:
        with open(result_file_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"✓ Saved execution result for rule {result.get('rule_id', 'unknown')}")
    except Exception as e:
        print(f"✗ Failed to save result for rule {result.get('rule_id', 'unknown')}: {e}")

def find_rules_to_execute(rules_folder: str, data_folder: str):
    """Find all rules that need execution"""
    rules_to_execute = []
    
    for root, dirs, files in os.walk(rules_folder):
        if "rule.json" in files and "rule_code_multi.py" in files:
            rule_folder_path = root
            
            # Skip if result already exists
            if has_execution_result(rule_folder_path):
                continue
            
            # Extract dataset name from path
            path_parts = Path(root).parts
            rules_folder_parts = Path(rules_folder).parts
            relative_parts = path_parts[len(rules_folder_parts):]
            
            if relative_parts:
                dataset_name = relative_parts[0]
                
                # Find dataset file
                dataset_path = os.path.join(data_folder, f"{dataset_name.lower()}.csv")
                if not os.path.exists(dataset_path):
                    print(f"⏭️  Skipping rule in {rule_folder_path} - dataset {dataset_name} not found")
                    continue
                
                # Load rule to get rule_id
                rule_file_path = os.path.join(rule_folder_path, "rule.json")
                try:
                    with open(rule_file_path, 'r') as f:
                        rule_data = json.load(f)
                    rule_id = rule_data.get('rule_id', 'unknown')
                except:
                    print(f"⏭️  Skipping rule in {rule_folder_path} - could not read rule.json")
                    continue
                
                rules_to_execute.append({
                    'rule_file_path': rule_file_path,
                    'rule_folder_path': rule_folder_path,
                    'dataset_path': dataset_path,
                    'rule_id': rule_id,
                    'dataset_name': dataset_name
                })
    
    return rules_to_execute

def execute_rules_batch(rules_to_execute: list, max_workers: int = 3, timeout_seconds: int = 300):
    """Execute rules in batches with timeout handling"""
    
    print(f"Executing {len(rules_to_execute)} rules with {max_workers} workers (timeout: {timeout_seconds}s)")
    
    completed_count = 0
    failed_count = 0
    timeout_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_rule = {
            executor.submit(execute_single_rule, rule_info): rule_info 
            for rule_info in rules_to_execute
        }
        
        # Process completed jobs
        for future in as_completed(future_to_rule, timeout=timeout_seconds + 10):
            rule_info = future_to_rule[future]
            rule_id = rule_info['rule_id']
            rule_folder_path = rule_info['rule_folder_path']
            
            try:
                # Get result with timeout
                result = future.result(timeout=timeout_seconds)
                
                if 'error' in result:
                    print(f"✗ Rule {rule_id} failed: {result['error']}")
                    failed_count += 1
                else:
                    print(f"✓ Rule {rule_id} executed successfully in {result.get('time_cost', 'unknown')}s")
                    completed_count += 1
                
                # Save result regardless of success/failure
                save_execution_result(rule_folder_path, result)
                
            except TimeoutError:
                print(f"⏱️  Rule {rule_id} timed out after {timeout_seconds}s")
                timeout_count += 1
                
                # Create timeout result
                timeout_result = {
                    'rule_id': rule_id,
                    'error': f'time out ({timeout_seconds}s)',
                    'time_cost': f'time out ({timeout_seconds}s)',
                    'memory_usage_mb': -1,
                    'code_type': 'single_reasoning',
                    'dataset_size': -1
                }
                save_execution_result(rule_folder_path, timeout_result)
                
            except Exception as e:
                print(f"✗ Rule {rule_id} execution error: {str(e)}")
                failed_count += 1
                
                # Create error result
                error_result = {
                    'rule_id': rule_id,
                    'error': f'Execution error: {str(e)}',
                    'time_cost': -1,
                    'memory_usage_mb': -1,
                    'code_type': 'single_reasoning',
                    'dataset_size': -1
                }
                save_execution_result(rule_folder_path, error_result)
    
    print(f"\n=== Execution Summary ===")
    print(f"Successfully executed: {completed_count}")
    print(f"Failed executions: {failed_count}")
    print(f"Timed out: {timeout_count}")
    print(f"Total processed: {completed_count + failed_count + timeout_count}")

def main():
    """Main function to execute all generated rule codes"""
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rules_folder = os.path.join(current_dir, "rules_and_programs")
    data_folder = os.path.join(current_dir, "")  # Assumes datasets are in the examples folder
    
    print(f"Looking for rules in: {rules_folder}")
    print(f"Looking for datasets in: {data_folder}")
    
    if not os.path.exists(rules_folder):
        print(f"Error: Rules folder not found at {rules_folder}")
        return
    
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found at {data_folder}")
        return
    
    # Find rules that need execution
    rules_to_execute = find_rules_to_execute(rules_folder, data_folder)
    
    if not rules_to_execute:
        print("No rules found that need execution!")
        return
    
    print(f"Found {len(rules_to_execute)} rules to execute")
    
    # Group by dataset for better resource management
    rules_by_dataset = {}
    for rule_info in rules_to_execute:
        dataset_name = rule_info['dataset_name']
        if dataset_name not in rules_by_dataset:
            rules_by_dataset[dataset_name] = []
        rules_by_dataset[dataset_name].append(rule_info)
    
    print(f"Rules grouped by {len(rules_by_dataset)} datasets:")
    for dataset_name, rules in rules_by_dataset.items():
        print(f"  {dataset_name}: {len(rules)} rules")
    
    # Execute rules dataset by dataset
    for dataset_name, dataset_rules in rules_by_dataset.items():
        print(f"\n--- Executing rules for dataset: {dataset_name} ---")
        execute_rules_batch(dataset_rules, max_workers=3, timeout_seconds=300)

if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    if sys.platform.startswith('win'):
        multiprocessing.set_start_method('spawn', force=True)
    
    main()