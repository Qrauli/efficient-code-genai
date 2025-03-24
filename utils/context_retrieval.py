import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
import logging

@dataclass
class RetrievalSource:
    """Configuration for a retrieval source"""
    name: str
    type: str  # "web", "documentation", "code_snippet"
    path: str  # URL or file path
    description: str
    enabled: bool = True
    
class ContextRetriever:
    """Retrieves relevant context from various sources for code generation"""
    
    def __init__(self, config, sources: List[RetrievalSource] = None):
        self.config = config
        self.sources = sources or []
        self.embeddings = VertexAIEmbeddings()
        self.vectorstore = None
        self.retriever = None
        self.initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.config.RETRIEVAL_STORAGE_PATH, exist_ok=True)
        
    def add_source(self, source: RetrievalSource):
        """Add a new source to the retriever"""
        self.sources.append(source)
        self.initialized = False  # Need to reinitialize when sources change
        
    def initialize(self, force_reload: bool = False):
        """Initialize or update the vector database with all sources"""
        if self.initialized and not force_reload:
            return
            
        try:
            # Check if vectorstore already exists
            vectorstore_path = os.path.join(self.config.RETRIEVAL_STORAGE_PATH, "vectorstore")
            
            if os.path.exists(vectorstore_path) and not force_reload:
                self.vectorstore = Chroma(
                    persist_directory=vectorstore_path,
                    embedding_function=self.embeddings
                )
                self.logger.info(f"Loaded existing vectorstore with {self.vectorstore._collection.count()} documents")
            else:
                self.logger.info("Creating new vectorstore...")
                # Create text splitters for different content types
                code_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                doc_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
                )
                
                # Process each source
                all_documents = []
                
                for source in self.sources:
                    if not source.enabled:
                        continue
                        
                    self.logger.info(f"Processing source: {source.name} ({source.type})")
                    
                    try:
                        if source.type == "web":
                            loader = WebBaseLoader(source.path)
                            documents = loader.load()
                            
                            # Use appropriate splitter
                            if "documentation" in source.description.lower():
                                splitter = doc_splitter
                            else:
                                splitter = code_splitter
                                
                            split_docs = splitter.split_documents(documents)
                            
                        elif source.type == "documentation":
                            # For local documentation files
                            if os.path.isdir(source.path):
                                loader = DirectoryLoader(
                                    source.path, 
                                    glob="**/*.{md,txt,py,ipynb}",
                                    loader_cls=TextLoader
                                )
                            else:
                                loader = TextLoader(source.path)
                                
                            documents = loader.load()
                            split_docs = doc_splitter.split_documents(documents)
                            
                        elif source.type == "code_snippet":
                            # For code snippet collections
                            if os.path.isdir(source.path):
                                loader = DirectoryLoader(
                                    source.path,
                                    glob="**/*.py",
                                    loader_cls=TextLoader
                                )
                            else:
                                loader = TextLoader(source.path)
                                
                            documents = loader.load()
                            split_docs = code_splitter.split_documents(documents)
                            
                        # Add source metadata
                        for doc in split_docs:
                            doc.metadata["source"] = source.name
                            doc.metadata["source_type"] = source.type
                            doc.metadata["description"] = source.description
                            
                        all_documents.extend(split_docs)
                        self.logger.info(f"Added {len(split_docs)} chunks from {source.name}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing source {source.name}: {str(e)}")
                
                # Create the vectorstore
                if all_documents:
                    self.vectorstore = Chroma.from_documents(
                        documents=all_documents,
                        embedding=self.embeddings,
                        persist_directory=vectorstore_path
                    )
                    self.vectorstore.persist()
                    self.logger.info(f"Created vectorstore with {len(all_documents)} documents")
                else:
                    self.logger.warning("No documents loaded. Vectorstore not created.")
                    return
            
            # Initialize retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.RETRIEVAL_TOP_K}
            )
            self.initialized = True
            
        except Exception as e:
            self.logger.error(f"Error initializing retriever: {str(e)}")
            raise
    
    def retrieve(self, query: str, filter_sources: List[str] = None, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context based on the query"""
        if not self.initialized:
            self.initialize()
            
        if not self.retriever:
            self.logger.warning("Retriever not initialized. No context retrieved.")
            return []
            
        search_kwargs = {"k": top_k or self.config.RETRIEVAL_TOP_K}
        
        # Add filter if specific sources are requested
        if filter_sources:
            search_kwargs["filter"] = {"source": {"$in": filter_sources}}
            
        try:
            docs = self.retriever.get_relevant_documents(query)
            
            # Format results
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "source_type": doc.metadata.get("source_type", "Unknown"),
                    "description": doc.metadata.get("description", ""),
                    "relevance_score": doc.metadata.get("score", 1.0)
                })
                
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def add_generated_code(self, code: str, metadata: Dict[str, Any]):
        """Add successfully generated and optimized code to the knowledge base"""
        if not self.initialized:
            self.initialize()
            
        try:
            # Create storage dir for code snippets if needed
            snippets_dir = os.path.join(self.config.RETRIEVAL_STORAGE_PATH, "generated_snippets")
            os.makedirs(snippets_dir, exist_ok=True)
            
            # Save the code to a file
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snippet_{timestamp}.py"
            filepath = os.path.join(snippets_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(f"# {metadata.get('description', 'Generated code snippet')}\n")
                f.write(f"# Tags: {', '.join(metadata.get('tags', []))}\n")
                f.write(f"# Metrics: Time={metadata.get('execution_time', 'N/A')}s, Memory={metadata.get('memory_usage', 'N/A')}MB\n\n")
                f.write(code)
            
            # Create a new source for this snippet
            snippet_source = RetrievalSource(
                name=f"snippet_{timestamp}",
                type="code_snippet",
                path=filepath,
                description=metadata.get("description", "Generated code snippet")
            )
            
            # Add to sources and reinitialize
            self.add_source(snippet_source)
            self.initialize(force_reload=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding generated code: {str(e)}")
            return False