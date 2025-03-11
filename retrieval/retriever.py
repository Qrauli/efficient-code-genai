#from langchain.vectorstores import Chroma
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class Retriever:
    def __init__(self, config):
        self.config = config
        """self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=config.LLM_API_KEY
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name="code_snippets",
            embedding_function=self.embedding_model,
            persist_directory=f"{config.RESULTS_PATH}/vector_db"
        )"""
        
        # Initialize sparse retriever (BM25)
        self.sparse_retriever = self._init_sparse_retriever()
        
        """# Initialize dense retriever
        self.dense_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RETRIEVAL_TOP_K}
        )"""
        
        # Initialize hybrid retriever
        """self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.sparse_retriever, self.dense_retriever],
            weights=[1-config.HYBRID_ALPHA, config.HYBRID_ALPHA]
        )"""
    
    def _init_sparse_retriever(self):
        """Initialize sparse retriever with BM25"""
        # In a real implementation, this would load documents from the corpus
        # For demonstration purposes, using empty list
        documents = ["test document 1"]
        return BM25Retriever.from_texts(documents)
        # return BM25Retriever.from_documents(documents)
        # return []
    def retrieve(self, query, retrieval_type="hybrid"):
        """Retrieve relevant documents based on the query"""
        #if retrieval_type == "sparse":
        results = self.sparse_retriever.invoke(query)
        #elif retrieval_type == "dense":
        #    results = self.dense_retriever.get_relevant_documents(query)
        #else:  # hybrid
        #    results = self.ensemble_retriever.get_relevant_documents(query)
        
        # Format results for use in prompts
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return formatted_results