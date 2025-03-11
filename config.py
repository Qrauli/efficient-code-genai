# config.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables from .env file

class Config:
    # LLM Configuration
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    
    # Retrieval System Configuration
    VECTOR_DB_CONNECTION = os.getenv("VECTOR_DB_CONNECTION", "chroma://localhost:8000")
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))  # Weight between sparse and dense retrieval
    
    # Agent Configuration
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
    
    # Evaluation Configuration
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "60"))
    MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "1024"))
    
    # Paths
    CODE_CORPUS_PATH = os.getenv("CODE_CORPUS_PATH", "./data/code_corpus")
    BENCHMARK_PROBLEMS_PATH = os.getenv("BENCHMARK_PROBLEMS_PATH", "./data/benchmark_problems")
    RESULTS_PATH = os.getenv("RESULTS_PATH", "./results")
    
    GENERATE_ADDITIONAL_TESTS = True  # Whether to generate additional tests when existing ones are provided