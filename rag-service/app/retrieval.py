from utils import es_client
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load embeddings model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def retrieve_documents(query, index_name, top_k=5):
    """Retrieve documents using hybrid search (semantic vector + keyword)"""
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.embed_query(query)

        # 1. Semantic search with vector similarity
        vector_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "size": top_k
        }

        try:
            vector_response = es_client.search(
                index=index_name,
                body=vector_query
            )

            vector_docs = [
                {
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"]["metadata"],
                    "score": hit["_score"],
                    "search_type": "semantic"
                } for hit in vector_response["hits"]["hits"]
            ]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            vector_docs = []

        # 2. Keyword search with text matching
        keyword_query = {
            "query": {
                "match": {
                    "content": query
                }
            },
            "size": top_k
        }

        try:
            keyword_response = es_client.search(
                index=index_name,
                body=keyword_query
            )

            keyword_docs = [
                {
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"]["metadata"],
                    "score": hit["_score"],
                    "search_type": "keyword"
                } for hit in keyword_response["hits"]["hits"]
            ]
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            keyword_docs = []

        # 3. Combine and rank results
        all_results = vector_docs + keyword_docs

        # Remove duplicates by content
        unique_results = []
        seen_content = set()

        for doc in all_results:
            content_hash = hash(doc["content"])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)

        # Sort by score (higher is better)
        sorted_results = sorted(
            unique_results,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        # Limit to requested number
        final_results = sorted_results[:top_k]

        return {
            "query": query,
            "index": index_name,
            "result_count": len(final_results),
            "results": final_results
        }

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise