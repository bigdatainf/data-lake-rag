from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
from utils import es_client, minio_client, get_file_extension, ensure_index_exists
import logging
import io
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Load embeddings model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def _get_loader_for_file(file_path):
    """Select the appropriate loader based on file type"""
    ext = get_file_extension(file_path)

    if ext in ['.txt', '.md', '.html']:
        return TextLoader(file_path)
    elif ext == '.csv':
        return CSVLoader(file_path)
    elif ext == '.pdf':
        return PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def _create_index_with_mapping(index_name):
    """Create an Elasticsearch index with vector mapping"""
    if not es_client.indices.exists(index=index_name):
        # Define mapping with vector field
        mapping = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text"
                    },
                    "metadata": {
                        "type": "object"
                    },
                    "vector": {
                        "type": "dense_vector",
                        "dims": 384,  # Dimension of the all-MiniLM-L6-v2 model
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }

        # Create index with mapping
        es_client.indices.create(index=index_name, body=mapping)
        logger.info(f"Created index with vector mapping: {index_name}")
    return True

def scan_minio_bucket(bucket, prefix="documents/"):
    """Scan a MinIO bucket for documents to process"""
    try:
        # Ensure bucket exists
        if not minio_client.bucket_exists(bucket):
            logger.warning(f"Bucket {bucket} does not exist")
            return []

        # List objects in the bucket with the given prefix
        objects = minio_client.list_objects(bucket, prefix=prefix, recursive=True)

        processed_docs = []
        for obj in objects:
            try:
                # Process each document
                logger.info(f"Processing {obj.object_name} from bucket {bucket}")
                result = process_minio_document(bucket, obj.object_name)
                processed_docs.append({
                    "object_name": obj.object_name,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error processing {obj.object_name}: {e}")
                processed_docs.append({
                    "object_name": obj.object_name,
                    "error": str(e)
                })

        logger.info(f"Processed {len(processed_docs)} documents from {bucket}/{prefix}")
        return processed_docs

    except Exception as e:
        logger.error(f"Error scanning MinIO bucket: {e}")
        raise

def process_minio_document(bucket, object_path):
    """Process a document from MinIO"""
    try:
        # Get object from MinIO
        obj = minio_client.get_object(bucket, object_path)
        content = obj.read()

        # Get original filename from object path
        original_filename = os.path.basename(object_path)

        # Create a temporary file for processing
        ext = get_file_extension(object_path)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(content)

        # Override source and filename for better traceability
        source = f"minio/{bucket}"
        description = f"Document from MinIO: {bucket}/{object_path}"

        # Process the file with the original filename in metadata
        result = process_document(
            file_path=temp_file_path,
            source=source,
            description=description,
            original_filename=original_filename  # Pass the original filename
        )

        return result

    except Exception as e:
        logger.error(f"Error processing MinIO document: {e}")
        raise

def list_documents(index_name=None):
    """List indexed documents"""
    try:
        if index_name:
            indexes = [index_name]
        else:
            # Get all document indices
            indices_response = es_client.indices.get(index="documents_*")
            indexes = list(indices_response.keys())

        all_docs = []

        for idx in indexes:
            try:
                # Query for all documents in the index
                res = es_client.search(
                    index=idx,
                    body={
                        "query": {"match_all": {}},
                        "size": 1000,
                        "_source": ["metadata"]
                    }
                )

                # Group by source file to avoid duplicates
                unique_files = {}
                for hit in res["hits"]["hits"]:
                    metadata = hit["_source"].get("metadata", {})
                    file_id = f"{metadata.get('source', 'unknown')}/{metadata.get('filename', 'unknown')}"

                    if file_id not in unique_files:
                        unique_files[file_id] = {
                            "id": hit["_id"].split("_")[0] if "_" in hit["_id"] else hit["_id"],
                            "filename": metadata.get("filename", "Unknown"),
                            "source": metadata.get("source", "Unknown"),
                            "description": metadata.get("description", ""),
                            "index": idx,
                            "chunk_count": 1
                        }
                    else:
                        unique_files[file_id]["chunk_count"] += 1

                all_docs.extend(list(unique_files.values()))

            except Exception as e:
                logger.error(f"Error listing documents from index {idx}: {e}")

        return all_docs

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise

def process_document(file_path, source="upload", description=None, original_filename=None):
    """Process a document and index it in Elasticsearch with vector embeddings"""
    try:
        # Load document based on type
        loader = _get_loader_for_file(file_path)
        documents = loader.load()

        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)

        # Use original filename if provided, otherwise use the basename of file_path
        filename = original_filename or os.path.basename(file_path)

        # Prepare metadata
        metadata = {
            "source": source,
            "filename": filename,
            "description": description or ""
        }

        # Add metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update(metadata)

        # Determine index name based on source
        source_slug = source.replace("/", "_").replace(" ", "_").lower()
        index_name = f"documents_{source_slug}"

        # Create index with vector mapping
        _create_index_with_mapping(index_name)

        # Process chunks in batches
        batch_size = 50
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(total_chunks-1)//batch_size + 1}")

            # Create embeddings for all documents in batch
            texts = [doc.page_content for doc in batch]
            embeddings = embedding_model.embed_documents(texts)

            # Index documents with their embeddings
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                es_client.index(
                    index=index_name,
                    id=f"{filename.replace('.', '_')}_{i+j}",
                    document={
                        "content": chunk.page_content,
                        "metadata": chunk.metadata,
                        "vector": embedding
                    }
                )

        # Refresh index
        es_client.indices.refresh(index=index_name)

        # If the source is not already from MinIO, store the document in MinIO
        if not source.startswith("minio/"):
            # Ensure document zone buckets exist
            raw_zone = "raw-ingestion-zone"
            if not minio_client.bucket_exists(raw_zone):
                minio_client.make_bucket(raw_zone)

            # Store in raw-ingestion-zone only if not already there
            with open(file_path, "rb") as f:
                file_data = f.read()
                object_name = f"documents/{filename}"

                minio_client.put_object(
                    bucket_name=raw_zone,
                    object_name=object_name,
                    data=io.BytesIO(file_data),
                    length=len(file_data),
                    content_type="application/octet-stream"
                )

            logger.info(f"Successfully indexed {len(chunks)} chunks with vectors to {index_name} and stored in MinIO")
        else:
            logger.info(f"Successfully indexed {len(chunks)} chunks with vectors to {index_name} (document already in MinIO)")

        # Delete temporary file if needed
        if os.path.dirname(file_path) == os.path.join("/data", "temp"):
            os.remove(file_path)

        return {
            "status": "success",
            "indexed_chunks": len(chunks),
            "index_name": index_name,
            "filename": filename
        }

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        # Clean up temporary files in case of error
        if os.path.exists(file_path) and os.path.dirname(file_path) == os.path.join("/data", "temp"):
            os.remove(file_path)
        raise