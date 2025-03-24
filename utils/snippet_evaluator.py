import os
import pandas as pd
import logging
from .code_execution import execute_code

class SnippetEvaluator:
    """Evaluates and maintains the quality of code snippets in the retrieval store"""
    
    def __init__(self, config, retriever):
        self.config = config
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)
        self.snippets_dir = os.path.join(config.RETRIEVAL_STORAGE_PATH, "generated_snippets")
        self.snippets_metadata = os.path.join(config.RETRIEVAL_STORAGE_PATH, "snippets_metadata.csv")
        self.metadata_df = self._load_metadata()
        
    def _load_metadata(self):
        """Load or create the snippets metadata file"""
        if os.path.exists(self.snippets_metadata):
            try:
                return pd.read_csv(self.snippets_metadata)
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
        
        # Create new metadata dataframe
        return pd.DataFrame(columns=[
            'snippet_id', 'filepath', 'description', 'tags', 
            'execution_time', 'memory_usage', 'usage_count', 
            'success_rate', 'last_used', 'quality_score'
        ])
    
    def _save_metadata(self):
        """Save the metadata dataframe"""
        try:
            self.metadata_df.to_csv(self.snippets_metadata, index=False)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def add_snippet_usage(self, snippet_id, success, execution_time=None, memory_usage=None):
        """Record usage of a snippet and update its statistics"""
        if snippet_id not in self.metadata_df['snippet_id'].values:
            self.logger.warning(f"Snippet {snippet_id} not found in metadata")
            return
        
        # Update usage statistics
        idx = self.metadata_df.index[self.metadata_df['snippet_id'] == snippet_id].tolist()[0]
        self.metadata_df.at[idx, 'usage_count'] = self.metadata_df.at[idx, 'usage_count'] + 1
        self.metadata_df.at[idx, 'last_used'] = pd.Timestamp.now()
        
        # Update success rate
        current_success = self.metadata_df.at[idx, 'success_rate'] * (self.metadata_df.at[idx, 'usage_count'] - 1)
        new_success_rate = (current_success + (1 if success else 0)) / self.metadata_df.at[idx, 'usage_count']
        self.metadata_df.at[idx, 'success_rate'] = new_success_rate
        
        # Update performance metrics if provided
        if execution_time is not None:
            # Weighted average to gradually shift toward newer measurements
            current_time = self.metadata_df.at[idx, 'execution_time']
            if pd.notna(current_time):
                self.metadata_df.at[idx, 'execution_time'] = 0.7 * current_time + 0.3 * execution_time
            else:
                self.metadata_df.at[idx, 'execution_time'] = execution_time
                
        if memory_usage is not None:
            current_memory = self.metadata_df.at[idx, 'memory_usage']
            if pd.notna(current_memory):
                self.metadata_df.at[idx, 'memory_usage'] = 0.7 * current_memory + 0.3 * memory_usage
            else:
                self.metadata_df.at[idx, 'memory_usage'] = memory_usage
        
        # Update quality score
        self._update_quality_score(snippet_id)
        self._save_metadata()
    
    def _update_quality_score(self, snippet_id):
        """Calculate and update quality score for a snippet"""
        if snippet_id not in self.metadata_df['snippet_id'].values:
            return
            
        idx = self.metadata_df.index[self.metadata_df['snippet_id'] == snippet_id].tolist()[0]
        
        # Calculate quality score (0-100):
        # 40% based on success rate
        # 30% based on execution time (lower is better)
        # 20% based on memory usage (lower is better)
        # 10% based on recency (newer is better)
        
        success_component = self.metadata_df.at[idx, 'success_rate'] * 40
        
        # Time component (normalized against all snippets)
        time_values = self.metadata_df['execution_time'].dropna()
        if not time_values.empty and not pd.isna(self.metadata_df.at[idx, 'execution_time']):
            max_time = time_values.max()
            min_time = time_values.min()
            if max_time > min_time:
                norm_time = 1 - ((self.metadata_df.at[idx, 'execution_time'] - min_time) / (max_time - min_time))
                time_component = norm_time * 30
            else:
                time_component = 15  # Neutral if all times are equal
        else:
            time_component = 15  # Neutral if no data
            
        # Memory component (normalized)
        memory_values = self.metadata_df['memory_usage'].dropna()
        if not memory_values.empty and not pd.isna(self.metadata_df.at[idx, 'memory_usage']):
            max_memory = memory_values.max()
            min_memory = memory_values.min()
            if max_memory > min_memory:
                norm_memory = 1 - ((self.metadata_df.at[idx, 'memory_usage'] - min_memory) / (max_memory - min_memory))
                memory_component = norm_memory * 20
            else:
                memory_component = 10  # Neutral
        else:
            memory_component = 10  # Neutral
            
        # Recency component
        last_used = pd.to_datetime(self.metadata_df.at[idx, 'last_used'])
        now = pd.Timestamp.now()
        days_since_use = (now - last_used).days
        recency_component = max(0, 10 - min(10, days_since_use / 30 * 10))  # 0-10 score, decays over 30 days
        
        # Calculate total quality score
        quality_score = success_component + time_component + memory_component + recency_component
        self.metadata_df.at[idx, 'quality_score'] = quality_score
        
    def evaluate_snippets(self):
        """Evaluate all snippets in the store and update their metadata"""
        if not os.path.exists(self.snippets_dir):
            self.logger.warning(f"Snippets directory {self.snippets_dir} does not exist")
            return
            
        # Get all Python files in the snippets directory
        snippet_files = [f for f in os.listdir(self.snippets_dir) if f.endswith('.py')]
        
        for snippet_file in snippet_files:
            filepath = os.path.join(self.snippets_dir, snippet_file)
            snippet_id = os.path.splitext(snippet_file)[0]
            
            # Check if snippet is already in metadata
            if snippet_id not in self.metadata_df['snippet_id'].values:
                # Add new snippet to metadata
                description, tags = self._extract_metadata_from_file(filepath)
                
                new_row = {
                    'snippet_id': snippet_id,
                    'filepath': filepath,
                    'description': description,
                    'tags': tags,
                    'execution_time': None,
                    'memory_usage': None,
                    'usage_count': 0,
                    'success_rate': 0.0,
                    'last_used': pd.Timestamp.now(),
                    'quality_score': 50.0  # Default neutral score
                }
                
                self.metadata_df = pd.concat([self.metadata_df, pd.DataFrame([new_row])], ignore_index=True)
                
        # Save updated metadata
        self._save_metadata()
        
    def _extract_metadata_from_file(self, filepath):
        """Extract description and tags from file comments"""
        description = "Generated code snippet"
        tags = ""
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            for line in lines[:5]:  # Check first 5 lines for comments
                if line.startswith('# '):
                    if 'Tags:' in line:
                        tags = line.replace('# Tags:', '').strip()
                    elif 'Metrics:' not in line:  # Skip metrics line
                        description = line.replace('#', '').strip()
        except Exception as e:
            self.logger.error(f"Error reading file {filepath}: {e}")
            
        return description, tags
        
    def get_top_snippets(self, tag=None, limit=10):
        """Get the top snippets by quality score, optionally filtered by tag"""
        if tag:
            filtered_df = self.metadata_df[self.metadata_df['tags'].str.contains(tag, na=False)]
        else:
            filtered_df = self.metadata_df
            
        return filtered_df.sort_values('quality_score', ascending=False).head(limit)