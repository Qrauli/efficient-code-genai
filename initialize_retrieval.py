import os
import logging
import argparse
from config import Config
from utils.context_retrieval import ContextRetriever, RetrievalSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Initialize the retrieval system")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the vector database")
    parser.add_argument("--no-web", action="store_true", help="Skip web search")
    args = parser.parse_args()
    
    config = Config()
    retriever = ContextRetriever(config)
    
    try:
        # Force rebuild if requested
        if args.rebuild:
            logger.info("Forcing rebuild of retrieval system...")
            include_web_search = not args.no_web and config.WEB_SEARCH_ENABLED
            retriever.initialize_comprehensive_retrieval(include_web_search=include_web_search)
        else:
            # Check if we already have a populated vectorstore
            vectorstore_path = os.path.join(config.RETRIEVAL_STORAGE_PATH, "vectorstore")
            
            if os.path.exists(vectorstore_path) and os.listdir(vectorstore_path):
                logger.info("Loading existing retrieval system...")
                retriever.initialize(force_reload=False)
                logger.info(f"Retrieval system loaded with {retriever.vectorstore._collection.count()} documents")
            else:
                logger.info("Building retrieval system from scratch...")
                include_web_search = not args.no_web and config.WEB_SEARCH_ENABLED
                retriever.initialize_comprehensive_retrieval(include_web_search=include_web_search)
        
        logger.info("Retrieval system initialization complete!")
        
    except Exception as e:
        logger.error(f"Error initializing retrieval system: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())