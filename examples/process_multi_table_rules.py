# --- START OF FILE process_multi_table_rules.py ---

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import io
from contextlib import redirect_stdout, redirect_stderr

# Add parent directory to path to import RuleOrchestrator and Config
# This assumes the script is in the 'examples' directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rule_orchestrator import RuleOrchestrator
from config import Config

# --- Helper function: extract_indexes (copied from your run_code.py) ---
def extract_indexes(structure):
    """
    Recursively extract all row indexes from a nested structure of dictionaries, lists, sets, and tuples.
    Extracts numbers from leaf lists and single numeric leaf values.
    
    Args:
        structure: The nested structure to extract indexes from (dict, list, set, tuple, or primitive)
        
    Returns:
        set: A set of all indexes found in the structure
    """
    indexes = set()
    
    if isinstance(structure, dict):
        # Only process values in dictionaries
        for v in structure.values():
            indexes.update(extract_indexes(v))
    
    elif isinstance(structure, (list, set, tuple)):
        # For lists, check if it's a leaf list (contains only primitives)
        is_leaf_list_of_numbers = True
        if not structure: # Handle empty list/tuple/set
            is_leaf_list_of_numbers = False
        else:
            for item in structure:
                if not (isinstance(item, (int, float)) and not isinstance(item, bool)):
                    is_leaf_list_of_numbers = False
                    break
        
        if is_leaf_list_of_numbers:
            # If it's a leaf list with numbers, extract those numbers
            for item in structure:
                if isinstance(item, (int, float)) and not isinstance(item, bool): # Ensure it's a number
                    indexes.add(int(item))
        else:
            # If not a leaf list, recursively process its elements
            for item in structure:
                indexes.update(extract_indexes(item))
    
    # Handle the case where the leaf is a single number
    elif isinstance(structure, (int, float)) and not isinstance(structure, bool):
        indexes.add(int(structure))
    
    return indexes

# --- Helper function to convert results to JSON serializable format (from your rule_function_example.py) ---
def convert_to_serializable(obj):
    """Convert obj to a JSON serializable object."""
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return sorted(list(obj)) # Convert sets to sorted lists for consistent JSON output
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj) # Tuples become lists in JSON
    elif pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)): # Handle pandas NaT or NaN
        return None
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)): # Handle pandas Timestamp/Timedelta
        return str(obj)
    else:
        return obj

def main():
    # --- Configuration ---
    data_path = os.path.dirname(os.path.abspath(__file__)) 
    rules_file_path = os.path.join(data_path, "mutli-table-rules-ext.json")
    output_results_path = os.path.join(data_path, "multi_table_rule_evaluation_results_2.5.json")

    app_config = Config()
    orchestrator = RuleOrchestrator(app_config, use_retrieval=False)

    # --- Load Rules ---
    try:
        with open(rules_file_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Rules file not found at {rules_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {rules_file_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading rules: {e}")
        return

    loaded_dataframes_cache = {}

    def get_dataframe(table_name_str):
        if table_name_str not in loaded_dataframes_cache:
            possible_filenames = [f"{table_name_str}.csv", table_name_str]
            df_loaded = None
            loaded_filename = None
            for fname in possible_filenames:
                csv_file_path_abs = os.path.join(data_path, fname)
                try:
                    print(f"Attempting to load DataFrame: {csv_file_path_abs}")
                    df_loaded = pd.read_csv(csv_file_path_abs)
                    loaded_dataframes_cache[table_name_str] = df_loaded
                    loaded_filename = csv_file_path_abs
                    print(f"Successfully loaded {loaded_filename}")
                    break 
                except FileNotFoundError:
                    print(f"Info: CSV file '{csv_file_path_abs}' not found.")
                except Exception as e_load:
                    print(f"Error loading CSV {csv_file_path_abs}: {e_load}")
            if df_loaded is None:
                print(f"Error: Could not load any CSV for table '{table_name_str}'. Searched for: {possible_filenames}")
                loaded_dataframes_cache[table_name_str] = None 
                return None
        elif loaded_dataframes_cache[table_name_str] is None:
             print(f"Info: DataFrame for table '{table_name_str}' was previously unloadable.")
             return None
        return loaded_dataframes_cache[table_name_str]

    all_rule_results = []
    rule_id_counter = 0 

    # --- Variables for Summary Statistics ---
    total_execution_time = 0
    num_rules_with_execution_time = 0
    num_rules_with_injection = 0
    num_rules_where_injected_id_found = 0

    for rule_info in rules_data:
        rule_id_counter += 1
        print(f"\n--- Processing Rule {rule_id_counter} ---")
        print(f"Rule Definition: {rule_info.get('rule_definition')}")
        print(f"Natural Language: {rule_info.get('natural_language_explanation')[:100]}...")

        rule_description = rule_info.get("natural_language_explanation") + " " + rule_info.get("rule_definition")
        table1_name = rule_info.get("table1")
        table2_name = rule_info.get("table2")
        expected_violation_rate_raw = rule_info.get("violation_rate")
        expected_violation_rate = None
        if isinstance(expected_violation_rate_raw, (int, float)):
            expected_violation_rate = float(expected_violation_rate_raw)

        current_rule_result = {
            "rule_id": rule_id_counter,
            "rule_info": { # Store a copy of rule_info, excluding violation_data_generator for brevity if needed
                k: v for k, v in rule_info.items() if k != "violation_data_generator"
            }, 
            "status": "pending",
            "expected_violation_rate_from_json": expected_violation_rate, # Original rate from JSON
            "was_data_injected": False,
            "injected_record_ids": [], # Changed from expected_violating_injected_id
            "is_expected_violating_id_found": False,
            "num_violations_expected_if_injection_works": None
        }

        if not rule_description or not table1_name or not table2_name:
            print("Skipping rule due to missing critical information (description, table1, or table2).")
            current_rule_result["status"] = "skipped_missing_info"
            current_rule_result["error_message"] = "Missing rule_description, table1, or table2 name."
            all_rule_results.append(current_rule_result)
            continue

        df1 = get_dataframe(table1_name)
        df2 = get_dataframe(table2_name)

        if df1 is None or df2 is None:
            error_msg = f"Could not load DataFrames. df1 ('{table1_name}') loaded: {df1 is not None}, df2 ('{table2_name}') loaded: {df2 is not None}."
            print(error_msg)
            current_rule_result["status"] = "skipped_dataframe_load_failed"
            current_rule_result["error_message"] = error_msg
            all_rule_results.append(current_rule_result)
            continue
        
        df1_copy = df1.copy()
        df2_copy = df2.copy()

        current_rule_dataframes = {
            table1_name: df1_copy,
            table2_name: df2_copy
        }

        # --- Data Injection ---
        violation_generator_instructions = rule_info.get("violation_data_generator")
        if violation_generator_instructions:
            current_rule_result["was_data_injected"] = True
            print("  Injecting data for violation testing...")
            if not isinstance(violation_generator_instructions, list):
                violation_generator_instructions = [violation_generator_instructions] # Normalize to list

            for instruction in violation_generator_instructions:
                table_name_to_modify = instruction.get("table_to_modify")
                record_to_add_dict = instruction.get("record_to_add")

                if not table_name_to_modify or not record_to_add_dict:
                    print(f"    Warning: Invalid instruction in violation_data_generator: {instruction}")
                    continue
                
                if table_name_to_modify not in current_rule_dataframes:
                    print(f"    Warning: Table '{table_name_to_modify}' for injection not found in current rule's DataFrames. Skipping injection.")
                    continue

                target_df = current_rule_dataframes[table_name_to_modify]
                new_row_df = pd.DataFrame([record_to_add_dict])
                current_rule_dataframes[table_name_to_modify] = pd.concat([target_df, new_row_df], ignore_index=True)
                injected_id = record_to_add_dict.get("_id")
                if injected_id is not None:
                    current_rule_result["injected_record_ids"].append(injected_id)
                print(f"    Injected record into '{table_name_to_modify}' with _id: {injected_id}")

            # Removed the heuristic block for setting a single expected_violating_injected_id
            
            if expected_violation_rate == 0.0: # If original rate was 0, we expect at least 1 violation now
                 current_rule_result["num_violations_expected_if_injection_works"] = ">=1" # Changed from 1 to ">=1" for generality
            else: # If original rate was non-zero, it's harder to predict the exact new count
                 current_rule_result["num_violations_expected_if_injection_works"] = ">=1"


        # --- Generate and Optimize Code ---
        print("Invoking RuleOrchestrator to generate code...")
        orchestrator_start_time = time.time()
        orchestrator_result = orchestrator.process_rule(
            rule_description,
            dataframes=current_rule_dataframes, 
            rule_id=str(rule_id_counter),
            use_profiling=True,
            test_percentage=0.5 
        )
        orchestrator_duration = time.time() - orchestrator_start_time
        print(f"Orchestrator processing finished in {orchestrator_duration:.2f}s. Success: {orchestrator_result.get('success')}")

        current_rule_result["orchestrator_summary"] = orchestrator_result.get("summary")
        current_rule_result["orchestrator_success"] = orchestrator_result.get("success", False)
        current_rule_result["generated_code"] = orchestrator_result.get("code")
        current_rule_result["function_name"] = orchestrator_result.get("function_name")

        if not current_rule_result["orchestrator_success"] or not current_rule_result["generated_code"]:
            print(f"Orchestrator failed or did not produce code for rule: {rule_description}")
            current_rule_result["status"] = "orchestrator_failed_no_code"
            all_rule_results.append(current_rule_result)
            continue

        # --- Execute Generated Code ---
        execution_output = None
        execution_error_msg = None
        execution_time_taken = None
        num_violations = 0
        num_satisfactions = 0
        calculated_violation_rate = None
        extracted_violation_ids_from_output = set()
        
        namespace_exec = {
            "pd": pd,
            "np": np,
            "dataframes_for_rule": current_rule_dataframes 
        }

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        print(f"Executing generated function: {current_rule_result['function_name']}...")
        exec_start_time = time.time()
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(current_rule_result["generated_code"], namespace_exec)
                rule_fn = namespace_exec[current_rule_result["function_name"]]
                execution_output = rule_fn(current_rule_dataframes) 
            
            execution_time_taken = time.time() - exec_start_time
            print(f"Code execution successful. Time: {execution_time_taken:.4f}s")
            current_rule_result["status"] = "processed_execution_successful"

            if execution_output and isinstance(execution_output, dict):
                violations_data = execution_output.get('violations')
                satisfactions_data = execution_output.get('satisfactions')
                
                if violations_data is not None:
                    extracted_violation_ids_from_output = extract_indexes(violations_data)
                    num_violations = len(extracted_violation_ids_from_output)
                if satisfactions_data is not None:
                    num_satisfactions = len(extract_indexes(satisfactions_data))

                total_evaluated_for_rate = num_violations + num_satisfactions
                if total_evaluated_for_rate > 0:
                    calculated_violation_rate = num_violations / total_evaluated_for_rate
                elif num_violations == 0 and num_satisfactions == 0: 
                    calculated_violation_rate = 0.0 
                
                print(f"  Extracted Violations: {num_violations} (IDs: {list(extracted_violation_ids_from_output)[0] if extracted_violation_ids_from_output else 'None'}), Extracted Satisfactions: {num_satisfactions}")
                print(f"  Calculated Violation Rate: {calculated_violation_rate}, Expected (from JSON): {expected_violation_rate}")

                # Check if any of the injected violating IDs were found
                if current_rule_result["was_data_injected"] and current_rule_result.get("injected_record_ids"):
                    found_any_injected_id = False
                    if current_rule_result["injected_record_ids"]: # Ensure list is not empty
                        for injected_id_val in current_rule_result["injected_record_ids"]:
                            if injected_id_val is not None and (injected_id_val - 1) in extracted_violation_ids_from_output:
                                found_any_injected_id = True
                                break 
                    current_rule_result["is_expected_violating_id_found"] = found_any_injected_id
                    
                    original_injected_ids_str = str(current_rule_result.get('injected_record_ids', []))
                    print(f"    Injected data test: Original injected _ids {original_injected_ids_str}. Any corresponding index found in violations: {current_rule_result['is_expected_violating_id_found']}.")
                    
                    if not current_rule_result['is_expected_violating_id_found']:
                        expected_indexes_in_output = sorted([i_id - 1 for i_id in current_rule_result.get("injected_record_ids", []) if i_id is not None])
                        if num_violations > 0 :
                            print(f"      Warning: Violations were found ({num_violations}), but none of the injected records (original _ids: {original_injected_ids_str}, expected as indexes: {expected_indexes_in_output}) were detected in the violation output indexes: {sorted(list(extracted_violation_ids_from_output))}.")
                        elif num_violations == 0: # No violations found at all
                             print(f"      Warning: No violations found. None of the injected records (original _ids: {original_injected_ids_str}, expected as indexes: {expected_indexes_in_output}) were detected.")


            else:
                execution_error_msg = "Execution output was not a dictionary or was None."
                print(f"  Warning: {execution_error_msg}")
                current_rule_result["status"] = "processed_bad_output_format"
        
        except Exception as e_exec:
            execution_time_taken = time.time() - exec_start_time
            execution_error_msg = f"Exception during code execution: {type(e_exec).__name__}: {str(e_exec)}"
            print(f"  Error during execution: {execution_error_msg}")
            current_rule_result["status"] = "processed_execution_error"
        
        finally:
            current_rule_result["execution_time_seconds"] = execution_time_taken
            current_rule_result["execution_stdout"] = stdout_buffer.getvalue()
            current_rule_result["execution_stderr"] = stderr_buffer.getvalue()
            current_rule_result["execution_error_message"] = execution_error_msg
            current_rule_result["num_extracted_violations"] = num_violations
            current_rule_result["num_extracted_satisfactions"] = num_satisfactions
            current_rule_result["calculated_violation_rate"] = calculated_violation_rate

            if calculated_violation_rate is not None and expected_violation_rate is not None:
                current_rule_result["violation_rate_matches_json_expected"] = abs(calculated_violation_rate - expected_violation_rate) < 1e-6
            elif calculated_violation_rate is None and expected_violation_rate is None:
                 current_rule_result["violation_rate_matches_json_expected"] = True 
            elif calculated_violation_rate == 0.0 and expected_violation_rate == 0.0:
                 current_rule_result["violation_rate_matches_json_expected"] = True
            else:
                current_rule_result["violation_rate_matches_json_expected"] = False
            
            all_rule_results.append(current_rule_result)

            # --- Update Summary Statistics ---
            if execution_time_taken is not None:
                total_execution_time += execution_time_taken
                num_rules_with_execution_time += 1
            
            if current_rule_result.get("was_data_injected"):
                num_rules_with_injection += 1
                if current_rule_result.get("is_expected_violating_id_found"):
                    num_rules_where_injected_id_found += 1
            
            del df1_copy, df2_copy, current_rule_dataframes, orchestrator_result
            if 'rule_fn' in namespace_exec: del namespace_exec['rule_fn']
            del namespace_exec
            import gc
            gc.collect()
        
    print(f"\n--- Saving all rule evaluation results to {output_results_path} ---")
    try:
        # --- Calculate Summary Metrics ---
        average_execution_time = None
        if num_rules_with_execution_time > 0:
            average_execution_time = total_execution_time / num_rules_with_execution_time

        percentage_injected_id_found = None
        if num_rules_with_injection > 0:
            percentage_injected_id_found = (num_rules_where_injected_id_found / num_rules_with_injection) * 100
        
        summary_stats = {
            "total_rules_processed": rule_id_counter,
            "average_execution_time_seconds": average_execution_time,
            "num_rules_with_successful_execution": num_rules_with_execution_time,
            "num_rules_with_data_injection": num_rules_with_injection,
            "num_rules_where_injected_id_was_found": num_rules_where_injected_id_found,
            "percentage_of_injected_rules_where_id_was_found": percentage_injected_id_found
        }

        output_data = {
            "summary_statistics": summary_stats,
            "rule_results": all_rule_results
        }

        final_serializable_results = convert_to_serializable(output_data)
        with open(output_results_path, 'w', encoding='utf-8') as f:
            json.dump(final_serializable_results, f, indent=4)
        print("Results saved successfully.")
    except Exception as e_save:
        print(f"Error saving results to JSON: {e_save}")

if __name__ == "__main__":
    main()

