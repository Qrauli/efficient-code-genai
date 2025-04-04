from typing import List, Dict, Any, Optional
import os
import time
import logging
from urllib.parse import urlparse
from datetime import datetime
import random
from dataclasses import dataclass
from googlesearch import search

from langchain_community.document_loaders import WebBaseLoader
from .context_retrieval import RetrievalSource, ContextRetriever

class WebSearchIntegration:
    """Integrates web search results into the ContextRetriever knowledge base"""
    
    def __init__(self, retriever: ContextRetriever):
        """Initialize with a ContextRetriever instance"""
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)
        
        # Create a cache directory for web search results
        self.cache_dir = os.path.join(self.retriever.config.RETRIEVAL_STORAGE_PATH, "web_search_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Domains to prioritize for quality pandas content
        self.trusted_domains = [
            "pandas.pydata.org",
            "stackoverflow.com",
            "github.com",
            "towardsdatascience.com",
            "medium.com",
            "kaggle.com",
            "datacamp.com",
            "pythonspeed.com",
            "docs.python.org",
            "realpython.com",
            "blog.jetbrains.com",
            "datascientist.com"
        ]
    
    def search_and_add_sources(self, search_query: str, num_results: int = 5, 
                               source_type: str = "web") -> List[RetrievalSource]:
        """Search for web content and add it to the retrieval system
        
        Args:
            search_query (str): The search query
            num_results (int): Number of search results to process
            source_type (str): Type of source to create (defaults to "web")
            
        Returns:
            List[RetrievalSource]: List of sources added to the retriever
        """
        # Ensure the query is pandas-focused
        if "pandas" not in search_query.lower():
            search_query = f"pandas {search_query}"
        
        self.logger.info(f"Searching for: {search_query}")
        
        # Get search results
        try:
            # Add random pauses to avoid blocking
            search_results = []
            for url in search(search_query, num_results=num_results * 2):
                search_results.append(url)
                time.sleep(random.uniform(1.0, 3.0))  # Random delay between requests
                
                if len(search_results) >= num_results * 2:
                    break
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []
        
        # Filter and prioritize search results
        filtered_results = self._filter_and_prioritize_results(search_results)
        top_results = filtered_results[:num_results]
        
        # Create sources from search results
        added_sources = []
        for url in top_results:
            try:
                # Create a source name based on the domain and path
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                path = parsed_url.path.strip('/')
                if not path:
                    source_name = domain
                else:
                    # Use the last path component as part of the name
                    path_parts = path.split('/')
                    source_name = f"{domain}_{path_parts[-1]}"
                
                # Replace invalid characters
                source_name = source_name.replace('.', '_').replace('-', '_')
                
                # Add timestamp to ensure uniqueness
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                source_name = f"search_{source_name}_{timestamp}"
                
                # Create source
                source = RetrievalSource(
                    name=source_name,
                    type=source_type,
                    path=url,
                    description=f"Search result for: {search_query}"
                )
                
                # Add to retriever
                self.retriever.add_source(source)
                added_sources.append(source)
                
                self.logger.info(f"Added source: {source.name} - {url}")
                
                # Small delay between processing sources
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error adding source {url}: {str(e)}")
        
        # Reinitialize the retriever with new sources
        if added_sources:
            self.retriever.initialize(force_reload=True)
            
        return added_sources
    
    def _filter_and_prioritize_results(self, urls: List[str]) -> List[str]:
        """Filter and prioritize search results based on relevance and trusted domains"""
        scored_urls = []
        
        for url in urls:
            # Skip non-http URLs
            if not url.startswith('http'):
                continue
                
            # Parse the URL
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                
                # Calculate a relevance score
                score = 0
                
                # Prioritize trusted domains
                if domain in self.trusted_domains:
                    score += 10
                    
                # Favor documentation sites
                if "doc" in domain or "docs" in domain or "documentation" in domain:
                    score += 5
                    
                # Favor guides and tutorials
                path = parsed_url.path.lower()
                if any(keyword in path for keyword in ["guide", "tutorial", "howto", "best-practices"]):
                    score += 3
                    
                # Favor pandas-specific content
                if "pandas" in domain or "pandas" in path:
                    score += 5
                    
                # Penalize unrelated content
                # if any(keyword in domain for keyword in ["shop", "store", "buy", "product"]):
                #     score -= 10
                
                scored_urls.append((url, score))
            except Exception:
                # Skip URLs that can't be parsed
                continue
        
        # Sort by score (descending) and return just the URLs
        return [url for url, score in sorted(scored_urls, key=lambda x: x[1], reverse=True)]
    
    def search_for_pandas_patterns(self, pattern_type: str = "performance") -> List[RetrievalSource]:
        """Search for common pandas patterns and add them to the knowledge base
        
        Args:
            pattern_type (str): Type of patterns to search for ("performance", "vectorization", etc.)
            
        Returns:
            List[RetrievalSource]: List of sources added to the retriever
        """
        # Define search queries based on pattern type
        search_queries = {
            "performance": [
                "pandas dataframe performance optimization techniques",
                "efficient pandas operations for large datasets",
                "pandas vectorization vs apply performance",
                "pandas query vs loc performance comparison"
            ],
            "vectorization": [
                "pandas vectorization techniques examples",
                "numpy vectorization with pandas dataframes",
                "replace pandas loops with vectorized operations examples",
                "pandas apply vs vectorization benchmark"
            ],
            "memory": [
                "pandas memory optimization techniques",
                "reduce pandas dataframe memory usage",
                "pandas categorical dtype memory benefits",
                "pandas efficient memory management large datasets"
            ],
            "general": [
                "pandas advanced data processing techniques",
                "pandas efficient data transformation patterns",
                "pandas dataframe best practices",
                "pandas query optimization recipes"
            ]
        }
        
        # Get queries for the specified pattern type or use general if not found
        queries = search_queries.get(pattern_type, search_queries["general"])
        
        # Execute searches and collect sources
        all_added_sources = []
        for query in queries:
            # Limit to 2-3 results per query to avoid overwhelming the system
            sources = self.search_and_add_sources(query, num_results=2, source_type="web")
            all_added_sources.extend(sources)
            
            # Pause between queries
            time.sleep(random.uniform(2.0, 4.0))
        
        return all_added_sources