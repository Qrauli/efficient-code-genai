# config.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables from .env file

class Config:
    # LLM Configuration
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash-preview")
    
    # Retrieval System Configuration
    VECTOR_DB_CONNECTION = os.getenv("VECTOR_DB_CONNECTION", "chroma://localhost:8000")
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))  # Weight between sparse and dense retrieval
    
    # Agent Configuration
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "7"))
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
    
    # Evaluation Configuration
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "18"))
    MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "1024"))
    
    # Paths
    CODE_CORPUS_PATH = os.getenv("CODE_CORPUS_PATH", "./data/code_corpus")
    BENCHMARK_PROBLEMS_PATH = os.getenv("BENCHMARK_PROBLEMS_PATH", "./data/benchmark_problems")
    RESULTS_PATH = os.getenv("RESULTS_PATH", "./results")
    
    GENERATE_ADDITIONAL_TESTS = True  # Whether to generate additional tests when existing ones are provided
    
    # Retrieval settings
    ENABLE_RETRIEVAL = False
    RETRIEVAL_STORAGE_PATH = "../data/retrieval_store"
    RETRIEVAL_TOP_K = 5
    
    # Add to your config file
    PROFILE_INTERVAL = 3  # Scalene profile sampling interval in seconds
    ADAPTIVE_SAMPLING = True  # Enable adaptive sample size reduction
    MIN_SAMPLE_SIZE = 100  # Minimum sample size to use when reducing
    
    # Default retrieval sources
    DEFAULT_RETRIEVAL_SOURCES = [
        {
            "name": "pandas_documentation",
            "type": "web",
            "path": "https://pandas.pydata.org/docs/reference/frame.html",
            "description": "Official pandas DataFrame documentation"
        },
        {
            "name": "pandas_optimization_guide",
            "type": "web",
            "path": "https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html",
            "description": "Pandas performance optimization guide"
        },
        {
            "name": "pandas_cookbook",
            "type": "web",
            "path": "https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html",
            "description": "Pandas cookbook with efficient patterns"
        }
    ]
    
    # Add to your Config class
    WEB_SEARCH_ENABLED = True
    WEB_SEARCH_MAX_RESULTS = 10
    WEB_SEARCH_PATTERN_TYPES = ["performance", "vectorization", "memory"]
    WEB_SEARCH_TRUSTED_DOMAINS = [
        "pandas.pydata.org",
        "stackoverflow.com", 
        "github.com",
        "towardsdatascience.com",
        "medium.com",
        "kaggle.com"
    ]
    
    # Additional static retrieval sources
    ADDITIONAL_RETRIEVAL_SOURCES = [
        {
            "name": "pandas_performance_blog",
            "type": "web",
            "path": "https://pythonspeed.com/articles/pandas-vectorization/",
            "description": "Comprehensive guide on pandas vectorization"
        },
        {
            "name": "pandas_scaling_blog",
            "type": "web",
            "path": "https://towardsdatascience.com/how-to-make-your-pandas-loop-71-803-times-faster-805030df4f06",
            "description": "Techniques for scaling pandas operations"
        },
        {
            "name": "stackoverflow_pandas_performance",
            "type": "web",
            "path": "https://stackoverflow.com/questions/23361964/why-does-pandas-iterrows-have-such-poor-performance",
            "description": "StackOverflow discussion on pandas iteration performance"
        },
        {
            "name": "pandas_memory_optimization",
            "type": "web",
            "path": "https://www.dataquest.io/blog/pandas-big-data/",
            "description": "Memory optimization for pandas with large datasets"
        }
    ]