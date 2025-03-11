# Efficient Code Generation with Retrieval-Augmented Multi-Agent Workflow

## Overview

This repository implements a retrieval-augmented multi-agent workflow for generating efficient and maintainable code, with a special focus on data science tasks. The system leverages Large Language Models (LLMs) through a coordinated multi-agent architecture that iteratively improves code quality, correctness, and efficiency.

## Problem Statement

Writing efficient code is challenging, especially for data science tasks processing millions of rows of data. Current LLM-based code generation approaches have limitations in consistency, efficiency, and correctness. When processing large datasets, inefficient code can significantly increase processing time with notable economic implications.

## Architecture

The system employs a multi-agent architecture with specialized components:

### Agents
- **[CodeGenerator](agents/code_generator.py)**: Generates initial code based on problem descriptions, leveraging retrieval for relevant examples
- **[CodeTester](agents/code_tester.py)**: Creates test cases and verifies code correctness
- **[CodeOptimizer](agents/code_optimizer.py)**: Improves code efficiency using profiling data and fixes correctness issues
- **[Orchestrator](agents/orchestrator.py)**: Coordinates the workflow between agents in iterative refinement cycles

### Retrieval System
- **[Retriever](retrieval/retriever.py)**: Implements hybrid retrieval (dense and sparse) to find relevant code examples
- **Document Store**: Manages the code corpus for retrieval

### Utilities
- **[Code Execution](utils/code_execution.py)**: Safely executes code with performance measurements
- **Prompt Templates**: Manages specialized prompts for different agents

## Workflow

1. **Test Generation**: Create comprehensive test cases from the problem description
2. **Initial Code Generation**: Generate initial code with retrieval augmentation
3. **Correctness Refinement**: Iteratively test and fix code until all tests pass
4. **Performance Optimization**: Profile code and optimize for efficiency
5. **Final Evaluation**: Measure performance improvements

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/efficient-code-genai.git
cd efficient-code-genai

# Install dependencies
pip install -r requirements.txt

# Create a .env file with your API keys
echo "LLM_API_KEY=your_api_key_here" > .env
```

## Usage

```bash
# Run the system on a problem
python main.py --problem ./data/benchmark_problems/your_problem.txt --output ./results
```

## Project Structure

```
__init__.py
config.py             # Configuration settings
main.py               # Entry point
README.md
requirements.txt
agents/               # Multi-agent components
    __init__.py
    base_agent.py     # Abstract base agent class
    code_generator.py # Code generation agent
    code_optimizer.py # Code optimization agent
    code_tester.py    # Test creation/verification agent
    orchestrator.py   # Workflow coordination
data/
    benchmark_problems/ # Problem descriptions
    code_corpus/        # Code examples for retrieval
evaluation/
    __init__.py
    benchmarks.py     # Benchmark management
    evaluator.py      # Evaluation framework
    metrics.py        # Performance metrics
retrieval/
    __init__.py
    document_store.py # Document management
    indexer.py        # Corpus indexing
    retriever.py      # Hybrid retrieval implementation
    vectore_store.py  # Vector database interface
utils/
    __init__.py
    code_execution.py  # Safe code execution with metrics
    prompt_templates.py # LLM prompt templates
```

## Research Context

This project is part of a master thesis investigating how retrieval-augmented multi-agent workflows can enhance the efficiency and correctness of LLM-generated code. The research explores:

1. Leveraging retrieval systems to provide relevant code snippets and context during generation
2. Designing multi-agent architectures that coordinate specialized agents for iterative code refinement
3. Measuring effectiveness of the proposed workflow compared to existing approaches

## License

[MIT License](LICENSE)