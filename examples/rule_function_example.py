import pandas as pd
import numpy as np
import sys
import os
from langchain.globals import set_debug, set_verbose
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rule_orchestrator import RuleOrchestrator
from config import Config

def main():
    """Example usage of the RuleOrchestrator for generating rule evaluation functions"""
    # set_debug(True)
    # Load data from CSV
    df = pd.read_csv("Tax.csv")
        
    # Initialize config and orchestrator
    config = Config()
    orchestrator = RuleOrchestrator(config)
    
    # Example rule definition
    rule_description = """
    Rule: AreaCode determines City and State
    """
    
    # Process the rule (using a sample for initial development)
    result = orchestrator.process_rule(rule_description, df, sample_size=100000)
    
    # Print results
    print("\nRule Description:")
    print(rule_description)
    print("\nGenerated Function:")
    print(result["code"])  # Changed from "final_code" to "code"
    print("\nRule Evaluation Results:")
    
    # Extract metrics from final_metrics
    final_metrics = result.get("final_metrics", {})
    print(f"Support: {final_metrics.get('support')}")
    print(f"Confidence: {final_metrics.get('confidence')}")
    print("\nPerformance Summary:")
    print(result["summary"])
    
    import json
    # Save whole result for further analysis
    with open("../results/test_run/rule_evaluation_result.json", "w") as f:
        json.dump(result, f)  
        
if __name__ == "__main__":
    main()