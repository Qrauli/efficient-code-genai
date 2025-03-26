from setuptools import setup, find_packages

setup(
    name="efficient-code-genai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "tiktoken",
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
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