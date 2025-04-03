"""
Script to check document processing in the RAG system and process any unindexed documents.
Uses the RAG service API to process documents and create indexes if needed.
"""
import requests
import logging
import time
import os
from minio import Minio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG service URL
RAG_SERVICE_URL = "http://rag-service:8000"

# Configure MinIO client
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

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

def list_unindexed_documents():
    """List documents in MinIO that are not yet indexed"""
    try:
        # First, get all indexed documents to compare
        indexed_docs = []
        for index in list_indexes():
            docs = list_documents(index)
            indexed_docs.extend([doc['filename'] for doc in docs])

        indexed_filenames = set(indexed_docs)
        logger.info(f"Found {len(indexed_filenames)} already indexed filenames")

        # Now check MinIO for documents
        unindexed = []

        if minio_client.bucket_exists("raw-ingestion-zone"):
            objects = minio_client.list_objects("raw-ingestion-zone", prefix="documents/", recursive=True)

            for obj in objects:
                filename = os.path.basename(obj.object_name)
                if filename not in indexed_filenames:
                    unindexed.append({
                        "bucket": "raw-ingestion-zone",
                        "object_path": obj.object_name,
                        "filename": filename,
                        "size": obj.size,
                        "last_modified": obj.last_modified
                    })

        return unindexed
    except Exception as e:
        logger.error(f"Error listing unindexed documents: {e}")
        return []

def process_documents(documents):
    """Process unindexed documents from MinIO using the RAG service API"""
    logger.info(f"Processing {len(documents)} unindexed documents")

    for doc in documents:
        try:
            logger.info(f"Processing {doc['object_path']} from bucket {doc['bucket']}")

            # Use the RAG service API to fetch and process the document
            # CORRECTED: Use params instead of json
            response = requests.post(
                f"{RAG_SERVICE_URL}/documents/fetch-from-minio",
                params={
                    "bucket": doc['bucket'],
                    "object_path": doc['object_path']
                }
            )

            if response.status_code == 200:
                logger.info(f"Successfully queued {doc['filename']} for processing")
                # Give the service some time to process the document
                time.sleep(2)
            else:
                logger.error(f"Failed to process {doc['filename']}: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error processing {doc['object_path']}: {e}")

def check_all_indexes():
    """Check all available indexes and their documents, process any unindexed documents"""
    # First, look for unindexed documents and process them
    unindexed = list_unindexed_documents()

    if unindexed:
        logger.info(f"Found {len(unindexed)} unindexed documents in MinIO")
        for i, doc in enumerate(unindexed[:5], 1):  # Show first 5
            logger.info(f"  {i}. {doc['filename']} ({doc['size']} bytes)")

        if len(unindexed) > 5:
            logger.info(f"  ... and {len(unindexed) - 5} more")

        # Auto process documents
        process_documents(unindexed)

        # Give the service some time to complete processing
        logger.info("Waiting for processing to complete...")
        time.sleep(5)
    else:
        logger.info("No unindexed documents found in MinIO")

    # Now get all indexes (including any newly created ones)
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

def trigger_minio_scan():
    """Alternative method: Trigger a full MinIO scan through the RAG service API"""
    try:
        logger.info("Triggering full MinIO scan...")
        response = requests.post(
            f"{RAG_SERVICE_URL}/minio/scan",
            params={
                "bucket": "raw-ingestion-zone",
                "prefix": "documents/"
            }
        )

        if response.status_code == 200:
            logger.info("Successfully triggered MinIO scan. Waiting for processing...")
            # Wait for processing to complete
            time.sleep(10)
            return True
        else:
            logger.error(f"Failed to trigger MinIO scan: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error triggering MinIO scan: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting document processing check")

    # Option 1: Process documents individually
    check_all_indexes()

    # Option 2: Or use the built-in scan functionality
    # if trigger_minio_scan():
    #    check_all_indexes()