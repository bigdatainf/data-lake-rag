from elasticsearch import Elasticsearch
from minio import Minio
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clients
es_client = Elasticsearch("http://elasticsearch:9200")
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

def ensure_index_exists(index_name):
    """Ensure an index exists in Elasticsearch"""
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name)
        logger.info(f"Created index: {index_name}")
    return True

def list_elasticsearch_indexes(pattern="*"):
    """List Elasticsearch indexes matching a pattern"""
    try:
        response = es_client.indices.get(index=pattern)
        return list(response.keys())
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        return []

def get_file_extension(filename):
    """Get file extension"""
    return os.path.splitext(filename)[1].lower()

def ensure_minio_bucket_exists(bucket_name):
    """Ensure a bucket exists in MinIO"""
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        logger.info(f"Created bucket: {bucket_name}")
    return True

def get_minio_object(bucket_name, object_name):
    """Get an object from MinIO"""
    try:
        return minio_client.get_object(bucket_name, object_name)
    except Exception as e:
        logger.error(f"Error getting object from MinIO: {e}")
        raise