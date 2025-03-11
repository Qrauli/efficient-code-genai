from agents.orchestrator import Orchestrator
from config import Config
import argparse
import json
import os
from langchain.globals import set_verbose, set_debug

def main():
    set_verbose(True)
    
    parser = argparse.ArgumentParser(description='Retrieval-Augmented Multi-Agent Code Generation')
    parser.add_argument('--problem', type=str, required=True, help='Path to problem description file')
    parser.add_argument('--output', type=str, default='./results', help='Output directory for results')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load problem description
    with open(args.problem, 'r') as f:
        problem_description = f.read()
    
    # Initialize orchestrator
    orchestrator = Orchestrator(config)
    
    # Generate code
    results = orchestrator.process(problem_description)
    
    # Save results
    output_path = os.path.join(args.output, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save final code separately
    code_path = os.path.join(args.output, 'final_code.py')
    with open(code_path, 'w') as f:
        f.write(results['final_code'])
    
    print(f"Code generation completed. Results saved to {args.output}")
    print(f"Final code saved to {code_path}")

if __name__ == "__main__":
    main()