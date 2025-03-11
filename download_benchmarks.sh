#!/bin/bash

# Create the necessary directory structure if it doesn't exist
mkdir -p ./data/benchmark_problems

# Define the download URL
BENCHMARK_URL="https://github.com/huangd1999/EffiBench/raw/refs/heads/main/data/dataset.json"

# Download the benchmark dataset
echo "Downloading EffiBench dataset..."
curl -L $BENCHMARK_URL -o ./data/benchmark_problems/effibench_dataset.json

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download successful!"
    echo "Benchmark dataset saved to: ./data/benchmark_problems/effibench_dataset.json"
    
    # Count the number of problems in the dataset
    if command -v jq > /dev/null; then
        NUM_PROBLEMS=$(jq '. | length' ./data/benchmark_problems/effibench_dataset.json)
        echo "Dataset contains $NUM_PROBLEMS problems."
    else
        echo "Note: Install 'jq' for additional dataset information."
    fi
else
    echo "Error: Failed to download the benchmark dataset."
    exit 1
fi

echo "Setup complete."