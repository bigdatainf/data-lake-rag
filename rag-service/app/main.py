from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import document_manager
import retrieval
import utils

app = FastAPI(title="Data Lake RAG Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    index_name: str
    top_k: Optional[int] = 5

@app.get("/")
async def root():
    return {"message": "Data Lake RAG Service"}

@app.post("/embeddings/create")
async def create_embeddings(
        background_tasks: BackgroundTasks,
        index_name: str
):
    """Create vector embeddings for an index (no-op as embeddings are created during document processing)"""
    try:
        return {
            "status": "success",
            "message": f"Embeddings are already created during document processing for {index_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")

@app.post("/documents/upload")
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        source: str = Form("upload"),
        description: Optional[str] = Form(None)
):
    """Upload a document to be processed and indexed"""
    try:
        # Save file temporarily
        temp_file_path = os.path.join("/data", "temp", file.filename)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        # Process in background
        background_tasks.add_task(
            document_manager.process_document,
            temp_file_path,
            source,
            description
        )

        return {
            "status": "success",
            "message": f"File {file.filename} uploaded and queued for processing"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/documents/fetch-from-minio")
async def fetch_from_minio(
        background_tasks: BackgroundTasks,
        bucket: str,
        object_path: str
):
    """Fetch and process a document from MinIO"""
    try:
        # Process MinIO document in background
        background_tasks.add_task(
            document_manager.process_minio_document,
            bucket,
            object_path
        )

        return {
            "status": "success",
            "message": f"Document {bucket}/{object_path} queued for processing"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch document: {str(e)}")

@app.get("/documents/list")
async def list_documents(index_name: Optional[str] = None):
    """List indexed documents"""
    try:
        documents = document_manager.list_documents(index_name)
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.post("/retrieval/query")
async def query_documents(request: QueryRequest):
    """Query documents using RAG"""
    try:
        results = retrieval.retrieve_documents(
            query=request.query,
            index_name=request.index_name,
            top_k=request.top_k
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/indexes/list")
async def list_indexes():
    """List available indexes"""
    try:
        indexes = utils.list_elasticsearch_indexes("documents_*")
        return {"indexes": indexes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list indexes: {str(e)}")

@app.post("/minio/scan")
async def scan_minio_bucket(
        background_tasks: BackgroundTasks,
        bucket: str,
        prefix: Optional[str] = "documents/"
):
    """Scan a MinIO bucket for documents and process them"""
    try:
        background_tasks.add_task(
            document_manager.scan_minio_bucket,
            bucket,
            prefix
        )

        return {
            "status": "success",
            "message": f"Started scanning {bucket}/{prefix} in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan bucket: {str(e)}")