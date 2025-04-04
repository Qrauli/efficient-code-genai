from setuptools import setup, find_packages

setup(
    name="efficient-code-genai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-core>=0.1.0",
        "openai>=1.0.0",
        "psutil>=5.9.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "regex>=2023.0.0",
        "pydantic>=2.0.0",
        "jupyter>=1.0.0",
        "python-dotenv>=1.0.0",
        "langchain-community>=0.3.19",
        "rank_bm25>=0.2.2",
        "scalene",
        "langchain-google-vertexai"
    ],
    author="Lucas",
    author_email="lucas.gugler@gmail.com",
    description="Utilities for efficient code/rule generation with generative AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/efficient-code-genai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)