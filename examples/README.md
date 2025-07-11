# Demonstration Generator Usage Guide

## 1. Overview

The `demonstration_generator.py` script is the main entry point for an end-to-end workflow that automates the creation of a full demonstration package for a specific data quality rule type.

Starting from a high-level "Knowledge Base" (KB) JSON file describing a rule (e.g., "Functional Dependency"), the script performs the following key actions:
1.  **Generates a large, synthetic dataset** tailored to the rule type, containing both satisfying and violating examples.
2.  **Generates a specific, concrete rule** instance applicable to the new dataset.
3.  **Generates a "baseline" Python function** to check the rule using a comprehensive single-prompt approach.
4.  **Improves the baseline code** using a multi-agent system that iteratively tests, reviews, and optimizes the function for correctness and performance.
5.  **Executes the final, optimized code** and saves all generated artifacts (dataset, rule, code versions, execution results, and history) into a structured output folder.

## 2. Prerequisites

Before running the script, you need to set up your environment.

### A. Install Dependencies
Install all required Python packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### B. Configure Environment Variables
The script uses a `.env` file to manage credentials and model settings. Create a file named `.env` in the project's root directory.

**For Google Vertex AI (Default):**
Your `.env` file is not strictly necessary if you have the `gcloud` CLI configured and authenticated. The library will automatically pick up your credentials.

**For other providers (like OpenAI):**
You will need to set the API key. See Section 6 for more details.
```env
# Example for OpenAI
OPENAI_API_KEY="sk-..."
```

## 3. How to Run

The script is executed from the command line and takes two main arguments:

-   `input_path`: The path to a single KB `.json` file or a directory containing multiple KB files.
-   `output_dir`: The path to a directory where all generated artifacts will be saved.

### Syntax
```bash
python demonstration_generator.py <input_path> <output_dir>
```

### Examples
**Processing a single knowledge base file:**
```bash
python demonstration_generator.py ./knowledge_bases/functional_dependency.json ./demonstrations
```
This will create a new folder `./demonstrations/functional_dependency/` containing all the generated files for this rule.

**Processing all knowledge base files in a directory:**
```bash
python demonstration_generator.py ./knowledge_bases/ ./demonstrations
```
This will find all `.json` files in the `./knowledge_bases` directory and process them one by one, creating a corresponding subfolder for each in `./demonstrations`.

## 4. Workflow and File Connections

The `demonstration_generator.py` script orchestrates the entire process by leveraging several other key files in the project. Understanding their roles is crucial.

| File | Role in the Workflow |
| :--- | :--- |
| **`demonstration_generator.py`** | **Main Orchestrator.** This is the entry point. It drives the high-level steps: generating the dataset, generating the rule, calling the single-prompt generator, and then calling the multi-agent system. |
| **`config.py`** | **Central Configuration.** Defines settings used across the entire project, such as LLM model names, agent parameters (temperature, max iterations), and file paths. |
| **`rc_prompt_template.py`** | **Core Prompt Engineering.** Contains the master prompt template (`generate_code_prompt`) used for the initial "baseline" code generation. It's a highly detailed prompt with instructions on rule types, output formats, and metric calculations. |
| **`single_prompt_generation.py`** | **Baseline Code Generator.** This script is responsible for the *initial* code generation. It takes the detailed prompt from `rc_prompt_template.py`, combines it with the specific rule and dataset sample, and makes a single call to the LLM to produce the first version of the rule-checking function. |
| **`rule_orchestrator.py`** | **Multi-Agent System Manager.** This is the "manager" of the agentic improvement workflow. It takes the baseline code and orchestrates a team of specialized agents (tester, reviewer, optimizer) to iteratively refine it. |
| **`base_agent.py`** | **Agent Foundation.** Provides the abstract `BaseAgent` class that all other agents inherit from. It standardizes agent initialization, including how the LLM client is created. **This is a key file to modify when switching LLM providers.** |
| **`requirements.txt`** | **Project Dependencies.** Lists all the necessary Python libraries for the project to run. |

## 5. Expected Output Structure

For each processed KB file (e.g., `my_rule.json`), the script will create a dedicated output directory with the following structure:

```
<output_dir>/
└── my_rule/
    ├── synthetic_dataset.csv         # The large, generated dataset.
    ├── rule.json                     # The specific rule instance for the dataset.
    ├── generate_dataset.py           # The code used to generate the dataset.
    ├── rule_code.py                  # The initial "baseline" code from single_prompt_generation.
    ├── rule_code_multi.py            # The final, optimized code from the multi-agent system.
    ├── multi_agent_history.json      # A detailed log of the multi-agent improvement process.
    └── final_execution_results.json  # The support, confidence, and execution time of the final code.
```

## 6. Switching LLM Providers

The project is configured to use `ChatVertexAI` by default. To switch to another provider like `ChatOpenAI` or `AzureChatOpenAI`, you need to replace the LLM instantiation in **three key places**.

### Files to Modify:
1.  `base_agent.py` (in the `BaseAgent` class `__init__`)
2.  `demonstration_generator.py` (in the `DemonstrationGenerator` class `__init__`)
3.  `single_prompt_generation.py` (in the `run_single_prompt_generation` function)

Below are drop-in replacement examples for each provider.

---

### A. Using `ChatOpenAI` (OpenAI API)

**1. Update your `.env` file:**
```env
OPENAI_API_KEY="sk-..."
LLM_MODEL="gpt-4o" # or another model like "gpt-3.5-turbo"
```

**2. Modify the code:**
Replace the `self.llm = ChatVertexAI(...)` block in all three specified locations with the following:

```python
# Add this import at the top of the file
from langchain_openai import ChatOpenAI

# ... inside the class or function ...

# Replace the ChatVertexAI block with this:
self.llm = ChatOpenAI(
    model=config.LLM_MODEL,           # Reads from .env
    temperature=config.AGENT_TEMPERATURE,
    api_key=config.LLM_API_KEY,         # Reads from .env
)
```

---

### B. Using `AzureChatOpenAI` (Microsoft Azure)

**1. Update your `.env` file:**
Azure requires more configuration details.
```env
AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com/"
OPENAI_API_VERSION="2024-02-01" # Use the API version for your deployment
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="<your-deployment-name>" # The name of your model deployment
```

**2. Modify the code:**
Replace the `self.llm = ChatVertexAI(...)` block in all three specified locations with the following:

```python
# Add this import at the top of the file
from langchain_openai import AzureChatOpenAI
import os

# ... inside the class or function ...

# Replace the ChatVertexAI block with this:
self.llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=config.AGENT_TEMPERATURE,
    # The AzureChatOpenAI client automatically reads the endpoint and key 
    # from the standard environment variables.
)
```