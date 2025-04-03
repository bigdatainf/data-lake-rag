"""
Script to ingest unstructured data into the data lake.
This script adds unstructured documents to the raw-ingestion-zone.
"""
import os
import io
from minio import Minio
import time
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MinIO client
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

def create_sample_documents():
    """Create sample documents for testing"""
    sample_docs = [
        {
            "filename": "customer_feedback.txt",
            "content": """
            Customer Feedback for Product X:
            
            I love the product, but I think it could be improved in terms of durability.
            The delivery was quick but the packaging was damaged. Customer service was
            excellent when I reported the issue and they sent a replacement immediately.
            
            I would recommend this product but with a warning about the packaging.
            """
        },
        {
            "filename": "product_manual.txt",
            "content": """
            PRODUCT MANUAL - MODEL X200
            
            USAGE INSTRUCTIONS:
            1. Connect the device to a power source.
            2. Press the power button and wait for the indicator light to turn green.
            3. Configure the options according to your preferences.
            4. To turn off, press and hold the power button for 3 seconds.
            
            TROUBLESHOOTING:
            - If the device doesn't turn on, check the electrical connection.
            - If the light blinks red, restart the device.
            - For technical assistance, contact our customer service.
            """
        },
        {
            "filename": "market_analysis.txt",
            "content": """
            MARKET TREND ANALYSIS 2023
            
            Summary: This document analyzes the main market trends observed
            during the first quarter of 2023. Emerging patterns in consumer
            behavior are identified and strategic recommendations are offered.
            
            Key findings:
            - 23% increase in online purchases compared to the same period last year
            - Greater preference for sustainable and eco-friendly products
            - Increase in demand for subscription services
            
            Recommendations:
            1. Strengthen digital presence and user experience on online platforms
            2. Highlight sustainable attributes in marketing communications
            3. Evaluate opportunities for subscription-based business models
            """
        }
    ]

    # Create data directory
    os.makedirs("/data/temp", exist_ok=True)

    # Create sample files
    file_paths = []
    for doc in sample_docs:
        file_path = f"/data/temp/{doc['filename']}"
        with open(file_path, "w") as f:
            f.write(doc["content"])
        file_paths.append(file_path)
        logger.info(f"Created sample file: {file_path}")

    return file_paths

def ingest_to_raw_zone():
    """Ingest documents to raw-ingestion-zone"""
    # Create bucket if it doesn't exist
    bucket_name = "raw-ingestion-zone"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        logger.info(f"Created bucket: {bucket_name}")

    # Create sample documents
    file_paths = create_sample_documents()

    # Upload to MinIO
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            file_data = f.read()
            filename = os.path.basename(file_path)
            object_name = f"documents/{filename}"

            minio_client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=io.BytesIO(file_data),
                length=len(file_data),
                content_type="text/plain"
            )

            logger.info(f"Uploaded to MinIO: {bucket_name}/{object_name}")

    # Process documents with RAG service
    rag_service_url = "http://rag-service:8000"
    response = requests.post(
        f"{rag_service_url}/minio/scan?bucket={bucket_name}&prefix=documents/",
    )

    if response.status_code == 200:
        logger.info("Successfully initiated document scanning in RAG service")
    else:
        logger.error(f"Failed to scan documents: {response.status_code} - {response.text}")

    # Return paths for further processing
    return file_paths

if __name__ == "__main__":
    logger.info("Starting unstructured data ingestion")
    ingest_to_raw_zone()
    logger.info("Unstructured data ingestion complete")