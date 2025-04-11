import pandas as pd
import numpy as np
import sys
import os
from langchain.globals import set_debug, set_verbose
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rule_orchestrator import RuleOrchestrator
from config import Config
import json

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
        return list(obj)
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def main():
    """Example usage of the RuleOrchestrator for generating rule evaluation functions"""
    # set_debug(True)
    # Load data from CSV
    # df = pd.read_csv("Tax.csv")
    df = pd.read_csv("Tax-title-cleaned.csv")
    
    # Initialize config and orchestrator
    config = Config()
    orchestrator = RuleOrchestrator(config, use_retrieval=False)
    
    # Example rule definition
    rule_description = """
Rule: If rows in question all have the same value in State, then the following rule applies: LName determines FName.
    """
    """
Rule: AreaCode determines City and State
    """
    
    # Process the rule (using a sample for initial development)
    result = orchestrator.process_rule(rule_description, df, use_profiling=True, test_percentage=1)
    
    import numpy as np
    import io
    from contextlib import redirect_stdout, redirect_stderr
    import time
    
    namespace = {
        "pd": pd,
        "np": np,
        "test_df": df
    }
    
    # Prepare stdout/stderr capture
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    start_time = time.time()
    
    """try:
        # First, execute the function definition code
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(result["code"], namespace)
            
            # Then execute the function with the test dataframe
            exec(f"result = execute_rule(test_df)", namespace)
            
            run_time = time.time() - start_time
            print(f"Execution time: {run_time:.4f} seconds")
    except Exception as e:
        pass"""
    
    # Print results
    print("\nRule Description:")
    print(rule_description)
    print("\nGenerated Function:")
    print(result["code"])  # Changed from "final_code" to "code"
    # print("\nRule Evaluation Results:")
    
    # Extract metrics from final_metrics
    final_metrics = result.get("final_metrics", {})
    # print(f"Support: {final_metrics.get('support')}")
    # print(f"Confidence: {final_metrics.get('confidence')}")
    print("\nPerformance Summary:")
    print(result["summary"])
    
    # Convert result to JSON serializable format
    serializable_result = convert_to_serializable(result)
    
    # Save whole result for further analysis
    with open("../results/test_run/rule_evaluation_result.json", "w") as f:
        json.dump(serializable_result, f)
        
if __name__ == "__main__":
    main()