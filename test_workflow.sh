#!/bin/bash

# Create test directory if it doesn't exist
mkdir -p ./data/benchmark_problems
mkdir -p ./results/test_run

# Create a simple test problem
cat > ./data/benchmark_problems/test_problem.txt << 'EOL'
# Efficient String Matching

Write a Python function `find_pattern(text, pattern)` that finds all occurrences of a pattern in a text.

Requirements:
1. The function should return a list of starting indices where the pattern is found in the text.
2. For empty pattern, return an empty list.
3. The implementation should be efficient for large texts (millions of characters).
4. The search should be case-sensitive.

Example:
- find_pattern("ABABABA", "ABA") should return [0, 2, 4]
- find_pattern("ABABABA", "ABC") should return []

For large inputs, naive string matching will be too slow. Consider using an efficient algorithm 
like Knuth-Morris-Pratt (KMP), Boyer-Moore, or Rabin-Karp.
EOL

# Set up environment variables for testing
# export LLM_API_KEY="your_api_key_here"  # Replace with your actual API key
# export LLM_MODEL="gpt-4o"
# export MAX_ITERATIONS=3  # Limit iterations for testing
# export TIMEOUT_SECONDS=30  # Shorter timeout for testing

# Run the workflow
echo "Running code generation workflow on test problem..."
python main.py --problem ./data/benchmark_problems/test_problem.txt --output ./results/test_run

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "Workflow completed successfully!"
    echo "Results saved to: ./results/test_run/"
    echo "Generated code:"
    echo "----------------------------------------"
    cat ./results/test_run/final_code.py
    echo "\n----------------------------------------"
    
    # Optional: Run a quick test on the generated code
    python -c "
import json
import sys
sys.path.insert(0, './results/test_run')
from final_code import find_pattern

# Simple test cases
test_cases = [
    {\"input\": [\"ABABABA\", \"ABA\"], \"expected\": [0, 2, 4]},
    {\"input\": [\"ABABABA\", \"ABC\"], \"expected\": []},
    {\"input\": [\"\", \"A\"], \"expected\": []}
]

# Run tests
passed = 0
for i, test in enumerate(test_cases):
    result = find_pattern(*test['input'])
    if result == test['expected']:
        print(f'✅ Test {i+1} passed')
        passed += 1
    else:
        print(f'❌ Test {i+1} failed. Expected: {test[\"expected\"]}, Got: {result}')

print(f'Passed {passed}/{len(test_cases)} tests')
"
else
    echo "Workflow failed with error code: $?"
fi

# Show execution statistics
if [ -f "./results/test_run/results.json" ]; then
    echo "Execution statistics:"
    python -c "
import json
with open('./results/test_run/results.json', 'r') as f:
    data = json.load(f)
    
# print(f'Correctness iterations: {data[\"metadata\"][\"correctness_iterations\"]}')
# print(f'Performance iterations: {data[\"metadata\"][\"performance_iterations\"]}')
print(f'Total iterations: {data[\"metadata\"][\"total_iterations\"]}')

if 'summary' in data:
    print(f'\\nPerformance summary: {data[\"summary\"]}')
"
fi