"""
Script to demonstrate querying unstructured data using RAG.
Shows how to access unstructured data in the data lake.
"""
import requests
import logging
import pandas as pd
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

def query_rag_service():
    """Demonstrate querying the RAG service directly"""
    logger.info("\n=== Direct RAG Service Query ===")

    # Get available indexes
    try:
        response = requests.get(f"{RAG_SERVICE_URL}/indexes/list")
        if response.status_code != 200:
            logger.error(f"Failed to get indexes: {response.status_code} - {response.text}")
            return

        indexes = response.json().get("indexes", [])
        if not indexes:
            logger.warning("No indexes found. Make sure documents have been ingested first.")
            return

        # Select the first index for demonstration
        index_name = indexes[0]
        logger.info(f"Using index: {index_name}")

        # Example queries
        example_queries = [
            "How to troubleshoot product issues",
            "What are the key market trends",
            "Customer complaints about packaging",
            "How to make QA over an existing KG"
        ]

        # Execute each query
        for query in example_queries:
            logger.info(f"\nQuery: {query}")

            try:
                response = requests.post(
                    f"{RAG_SERVICE_URL}/retrieval/query",
                    json={
                        "query": query,
                        "index_name": index_name,
                        "top_k": 3
                    }
                )

                if response.status_code != 200:
                    logger.error(f"Query failed: {response.status_code} - {response.text}")
                    continue

                results = response.json()

                logger.info(f"Retrieved {len(results.get('results', []))} results")

                # Display results
                for i, result in enumerate(results.get("results", []), 1):
                    logger.info(f"Result {i}:")
                    logger.info(f"  Source: {result['metadata'].get('filename', 'Unknown')}")
                    logger.info(f"  Score: {result.get('normalized_score', result.get('score', 0)):.4f}")
                    logger.info(f"  Search Type: {result.get('search_type', 'Unknown')}")

                    # Truncate content for display
                    content = result["content"]
                    if len(content) > 200:
                        content = content[:200] + "..."
                    logger.info(f"  Content: {content}")

            except Exception as e:
                logger.error(f"Error executing query: {e}")

    except Exception as e:
        logger.error(f"Error in RAG service demo: {e}")

def load_access_zone_views():
    """Demonstrate accessing pre-generated views in access-zone"""
    logger.info("\n=== Access Zone Views ===")

    try:
        # List objects in unstructured folder
        objects = minio_client.list_objects("access-zone", prefix="unstructured/", recursive=True)

        object_paths = []
        for obj in objects:
            if obj.object_name.endswith(".parquet"):
                object_paths.append(obj.object_name)

        if not object_paths:
            logger.warning("No access views found in access-zone/unstructured/")
            return

        logger.info(f"Found {len(object_paths)} access views")

        # Load each view with pandas
        for object_path in object_paths:
            logger.info(f"\nLoading view: {object_path}")

            try:
                # Get object from MinIO
                response = minio_client.get_object("access-zone", object_path)

                # Read into pandas
                df = pd.read_parquet(io.BytesIO(response.read()))

                # Display information
                logger.info(f"View contains {len(df)} records")
                logger.info(f"Columns: {', '.join(df.columns)}")

                # Show top result
                if not df.empty:
                    top_result = df.iloc[0]
                    logger.info("\nTop result:")
                    for col in df.columns:
                        if col == "content":
                            # Truncate content for display
                            content = top_result[col]
                            if len(content) > 200:
                                content = content[:200] + "..."
                            logger.info(f"  {col}: {content}")
                        else:
                            logger.info(f"  {col}: {top_result[col]}")

            except Exception as e:
                logger.error(f"Error loading view {object_path}: {e}")

    except Exception as e:
        logger.error(f"Error accessing views: {e}")

def run_demo():
    """Run the complete demo"""
    logger.info("Starting RAG query demonstration")

    # Demo direct querying of RAG service
    query_rag_service()

    # Demo access to pre-generated views
    load_access_zone_views()

    logger.info("\nRAG query demonstration complete")

if __name__ == "__main__":
    run_demo()