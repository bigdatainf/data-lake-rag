"""
Script to update data governance information for unstructured data.
This aligns with the Govern Zone in the data lake architecture.
"""
import json
import logging
import os
import io
from minio import Minio
import yaml
import requests
from datetime import datetime

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

def ensure_bucket_exists(bucket_name):
    """Ensure a bucket exists in MinIO"""
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        logger.info(f"Created bucket: {bucket_name}")
    return True

def update_metadata_catalog():
    """Update metadata catalog for unstructured data"""
    try:
        # Ensure govern-zone-metadata bucket exists
        ensure_bucket_exists("govern-zone-metadata")

        # Get information about available indexes and documents
        response = requests.get(f"{RAG_SERVICE_URL}/indexes/list")
        if response.status_code != 200:
            logger.error(f"Failed to get indexes: {response.status_code} - {response.text}")
            return

        indexes = response.json().get("indexes", [])

        for index_name in indexes:
            # Get documents in this index
            doc_response = requests.get(f"{RAG_SERVICE_URL}/documents/list?index_name={index_name}")
            if doc_response.status_code != 200:
                logger.error(f"Failed to get documents: {doc_response.status_code} - {doc_response.text}")
                continue

            documents = doc_response.json().get("documents", [])

            # Create metadata entries
            metadata = {
                "index_name": index_name,
                "document_count": len(documents),
                "created_at": datetime.now().isoformat(),
                "description": f"Unstructured data index for {index_name.replace('documents_', '')}",
                "documents": documents,
                "data_type": "unstructured",
                "access_patterns": [
                    {
                        "name": "semantic_search",
                        "description": "Semantic search using vector embeddings",
                        "endpoint": f"{RAG_SERVICE_URL}/retrieval/query"
                    },
                    {
                        "name": "access_zone_views",
                        "description": "Pre-generated views in access-zone bucket",
                        "location": "access-zone/unstructured/*"
                    }
                ]
            }

            # Upload metadata to MinIO
            metadata_json = json.dumps(metadata, indent=2).encode('utf-8')

            minio_client.put_object(
                bucket_name="govern-zone-metadata",
                object_name=f"metadata/unstructured/{index_name}.json",
                data=io.BytesIO(metadata_json),
                length=len(metadata_json),
                content_type="application/json"
            )

            logger.info(f"Updated metadata for {index_name}")

    except Exception as e:
        logger.error(f"Error updating metadata catalog: {e}")

def update_data_lineage():
    """Update data lineage information"""
    try:
        # Ensure govern-zone-metadata bucket exists
        ensure_bucket_exists("govern-zone-metadata")

        # Get information about available indexes
        response = requests.get(f"{RAG_SERVICE_URL}/indexes/list")
        if response.status_code != 200:
            logger.error(f"Failed to get indexes: {response.status_code} - {response.text}")
            return

        indexes = response.json().get("indexes", [])

        for index_name in indexes:
            # Create lineage information
            lineage = {
                "source": {
                    "zone": "raw-ingestion-zone",
                    "path": "documents/",
                    "type": "unstructured_documents"
                },
                "transformations": [
                    {
                        "name": "document_chunking",
                        "description": "Split documents into chunks for semantic search",
                        "service": "rag-service",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "name": "embedding_generation",
                        "description": "Create vector embeddings for semantic search",
                        "service": "rag-service",
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "targets": [
                    {
                        "zone": "elasticsearch",
                        "path": index_name,
                        "type": "indexed_documents"
                    },
                    {
                        "zone": "vector-db",
                        "path": index_name,
                        "type": "vector_embeddings"
                    },
                    {
                        "zone": "access-zone",
                        "path": "unstructured/",
                        "type": "analytics_views"
                    }
                ]
            }

            # Upload lineage information to MinIO
            lineage_json = json.dumps(lineage, indent=2).encode('utf-8')

            minio_client.put_object(
                bucket_name="govern-zone-metadata",
                object_name=f"lineage/unstructured/{index_name}.json",
                data=io.BytesIO(lineage_json),
                length=len(lineage_json),
                content_type="application/json"
            )

            logger.info(f"Updated lineage for {index_name}")

    except Exception as e:
        logger.error(f"Error updating data lineage: {e}")

def update_security_policies():
    """Update security policies for unstructured data"""
    try:
        # Ensure govern-zone-security bucket exists
        ensure_bucket_exists("govern-zone-security")

        # Create security policy
        policy = {
            "version": "1.0",
            "policies": [
                {
                    "name": "unstructured_data_access",
                    "description": "Security policy for unstructured data in the data lake",
                    "resources": [
                        "raw-ingestion-zone/documents/*",
                        "access-zone/unstructured/*",
                        "elasticsearch/documents_*",
                        "vector-db/*"
                    ],
                    "roles": {
                        "data_scientist": {
                            "permissions": ["read"],
                            "description": "Data scientists can only read unstructured data"
                        },
                        "data_engineer": {
                            "permissions": ["read", "write", "update"],
                            "description": "Data engineers can manage unstructured data"
                        },
                        "admin": {
                            "permissions": ["read", "write", "update", "delete"],
                            "description": "Administrators have full access"
                        }
                    },
                    "retention": {
                        "raw-ingestion-zone": "90 days",
                        "elasticsearch": "365 days",
                        "vector-db": "365 days",
                        "access-zone": "30 days"
                    },
                    "encryption": {
                        "at_rest": True,
                        "in_transit": True
                    }
                }
            ]
        }

        # Upload policy to MinIO
        policy_yaml = yaml.dump(policy, default_flow_style=False).encode('utf-8')

        minio_client.put_object(
            bucket_name="govern-zone-security",
            object_name="policies/unstructured_data_security.yaml",
            data=io.BytesIO(policy_yaml),
            length=len(policy_yaml),
            content_type="application/yaml"
        )

        logger.info("Updated security policies for unstructured data")

    except Exception as e:
        logger.error(f"Error updating security policies: {e}")

def update_governance():
    """Update all governance information"""
    logger.info("Starting governance update for unstructured data")

    # Update metadata catalog
    update_metadata_catalog()

    # Update data lineage
    update_data_lineage()

    # Update security policies
    update_security_policies()

    logger.info("Governance update complete")

if __name__ == "__main__":
    update_governance()