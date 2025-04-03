"""
Script to create access views for unstructured data.
This makes the RAG system's data available in the Access Zone of the data lake.
"""
import pandas as pd
import requests
import logging
import io
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

def get_document_queries():
    """Define queries for creating access views"""
    return [
        {"name": "customer_feedback", "query": "customer feedback complaints", "description": "Customer feedback and complaints"},
        {"name": "product_information", "query": "product specifications instructions manual", "description": "Product manuals and information"},
        {"name": "market_analysis", "query": "market trends analysis research", "description": "Market analysis and trends"}
    ]

def create_views_for_index(index_name, queries):
    """Create access views for an index based on predefined queries"""

    # Ensure access-zone bucket exists
    if not minio_client.bucket_exists("access-zone"):
        minio_client.make_bucket("access-zone")
        logger.info("Created access-zone bucket")

    for query_info in queries:
        logger.info(f"Creating view for: {query_info['name']}")

        # Query the RAG service
        try:
            response = requests.post(
                f"{RAG_SERVICE_URL}/retrieval/query",
                json={
                    "query": query_info["query"],
                    "index_name": index_name,
                    "top_k": 20
                }
            )

            if response.status_code != 200:
                logger.error(f"Failed to query documents: {response.status_code} - {response.text}")
                continue

            query_results = response.json()
            results = query_results.get("results", [])

            if not results:
                logger.warning(f"No results found for query: {query_info['name']}")
                continue

            # Create DataFrame from results
            df_data = []
            for result in results:
                df_data.append({
                    "content": result["content"],
                    "source": result["metadata"].get("source", "unknown"),
                    "filename": result["metadata"].get("filename", "unknown"),
                    "score": result.get("normalized_score", result.get("score", 0)),
                    "search_type": result.get("search_type", "unknown")
                })

            df = pd.DataFrame(df_data)

            # Create CSV and Parquet files in access-zone
            for file_format in ["csv", "parquet"]:
                # Define object path
                object_path = f"unstructured/{query_info['name']}.{file_format}"

                # Convert to proper format
                if file_format == "csv":
                    buffer = io.StringIO()
                    df.to_csv(buffer, index=False)
                    content = buffer.getvalue().encode('utf-8')
                    content_type = "text/csv"
                else:  # parquet
                    buffer = io.BytesIO()
                    df.to_parquet(buffer)
                    buffer.seek(0)
                    content = buffer.getvalue()
                    content_type = "application/octet-stream"

                # Upload to MinIO
                minio_client.put_object(
                    bucket_name="access-zone",
                    object_name=object_path,
                    data=io.BytesIO(content),
                    length=len(content),
                    content_type=content_type
                )

                logger.info(f"Created access view: access-zone/{object_path}")

        except Exception as e:
            logger.error(f"Error creating view for {query_info['name']}: {e}")

def create_access_views():
    """Create all access views for unstructured data"""
    # Get all indexes
    try:
        response = requests.get(f"{RAG_SERVICE_URL}/indexes/list")
        if response.status_code != 200:
            logger.error(f"Failed to get indexes: {response.status_code} - {response.text}")
            return

        indexes = response.json().get("indexes", [])
        if not indexes:
            logger.warning("No indexes found. Make sure documents have been ingested first.")
            return

        logger.info(f"Found {len(indexes)} indexes")

        # Get predefined queries
        queries = get_document_queries()

        # Create views for each index
        for index_name in indexes:
            logger.info(f"Creating access views for index: {index_name}")
            create_views_for_index(index_name, queries)

        logger.info("Access views creation complete")

    except Exception as e:
        logger.error(f"Error creating access views: {e}")

if __name__ == "__main__":
    logger.info("Starting access views creation")
    create_access_views()