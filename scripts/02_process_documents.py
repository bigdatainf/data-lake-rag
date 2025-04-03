"""
Script to check document processing in the RAG system.
With the Elasticsearch-only implementation, embeddings are created during document processing.
"""
import requests
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG service URL
RAG_SERVICE_URL = "http://rag-service:8000"

def list_indexes():
    """List available document indexes"""
    try:
        response = requests.get(f"{RAG_SERVICE_URL}/indexes/list")
        if response.status_code == 200:
            return response.json().get("indexes", [])
        else:
            logger.error(f"Failed to get indexes: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        return []

def list_documents(index_name=None):
    """List documents in an index"""
    try:
        url = f"{RAG_SERVICE_URL}/documents/list"
        if index_name:
            url += f"?index_name={index_name}"

        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("documents", [])
        else:
            logger.error(f"Failed to list documents: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return []

def check_all_indexes():
    """Check all available indexes and their documents"""
    # Get all indexes
    indexes = list_indexes()
    logger.info(f"Found {len(indexes)} indexes to check")

    if not indexes:
        logger.warning("No indexes found. Make sure documents have been ingested first.")
        return

    # Check documents in each index
    for index_name in indexes:
        logger.info(f"Checking index: {index_name}")

        # List documents in index
        documents = list_documents(index_name)
        logger.info(f"Index {index_name} contains {len(documents)} documents")

        if documents:
            logger.info(f"Document examples:")
            for i, doc in enumerate(documents[:3], 1):  # Show first 3 documents
                logger.info(f"  {i}. {doc['filename']} (from {doc['source']}) - {doc['chunk_count']} chunks")

    logger.info("Document check complete")

if __name__ == "__main__":
    logger.info("Starting document processing check")
    check_all_indexes()