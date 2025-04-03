from utils import es_client
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load embeddings model
# BAAI/bge-large-en-v1.5
# sentence-transformers/all-MiniLM-L6-v2
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def retrieve_documents(query, index_name, top_k=5):
    """Retrieve documents using hybrid search (semantic vector + keyword) with normalized scores and reranking"""
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
            "size": top_k * 2  # Get more candidates for reranking
        }

        try:
            vector_response = es_client.search(
                index=index_name,
                body=vector_query
            )

            # Get max score for normalization
            vector_max_score = 2.0  # Cosine similarity + 1.0 has a max of 2.0

            vector_docs = [
                {
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"]["metadata"],
                    "raw_score": hit["_score"],
                    "score": hit["_score"] / vector_max_score * 10,  # Normalize to 0-10 scale
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
            "size": top_k * 2  # Get more candidates for reranking
        }

        try:
            keyword_response = es_client.search(
                index=index_name,
                body=keyword_query
            )

            # Get max score for normalization
            keyword_max_score = max([hit["_score"] for hit in keyword_response["hits"]["hits"]]) if keyword_response["hits"]["hits"] else 1.0

            keyword_docs = [
                {
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"]["metadata"],
                    "raw_score": hit["_score"],
                    "score": hit["_score"] / keyword_max_score * 10,  # Normalize to 0-10 scale
                    "search_type": "keyword"
                } for hit in keyword_response["hits"]["hits"]
            ]
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            keyword_docs = []

        # 3. Combine results
        all_results = vector_docs + keyword_docs

        # Remove duplicates by content
        unique_results = []
        seen_content = set()

        for doc in all_results:
            content_hash = hash(doc["content"])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)

        # 4. Simple reranking: combine semantic and keyword relevance
        for result in unique_results:
            # Apply a simple reranking formula
            # This prioritizes documents that match both semantically and lexically
            content = result["content"]

            # Calculate lexical similarity (exact word matches)
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            word_overlap = len(query_words.intersection(content_words))
            query_coverage = word_overlap / len(query_words) if query_words else 0

            # Calculate content length factor (prefer more complete chunks)
            length_factor = min(1.0, len(content) / 1000)  # Normalize up to 1000 chars

            # Weight factors
            semantic_weight = 0.6  # Emphasis on semantic understanding
            lexical_weight = 0.3   # Some weight on direct word matches
            length_weight = 0.1    # Small weight for longer, more complete content

            # Compute reranked score - bias towards semantic results
            if result["search_type"] == "semantic":
                reranked_score = (
                        semantic_weight * result["score"] +
                        lexical_weight * query_coverage * 10 +
                        length_weight * length_factor * 10
                )
            else:  # keyword results
                reranked_score = (
                        (semantic_weight * 0.7) * result["score"] +  # Slightly reduce impact of keyword scores
                        lexical_weight * query_coverage * 10 +
                        length_weight * length_factor * 10
                )

            result["rerank_factors"] = {
                "original_score": result["score"],
                "query_coverage": query_coverage,
                "length_factor": length_factor
            }
            result["score"] = reranked_score

        # Sort by reranked score
        sorted_results = sorted(
            unique_results,
            key=lambda x: x["score"],
            reverse=True
        )

        # Limit to requested number
        final_results = sorted_results[:top_k]

        # Clean up fields we don't want to expose in the API
        for result in final_results:
            if "raw_score" in result:
                del result["raw_score"]
            if "rerank_factors" in result:
                del result["rerank_factors"]

        return {
            "query": query,
            "index": index_name,
            "result_count": len(final_results),
            "results": final_results
        }

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise