from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document  
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np
import os
import tempfile
import logging
from docx import Document  # Import the Document class from python-docx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the FAISS vectorstore
vectorstore = None

class SearchQuery(BaseModel):
    query: str

# Custom embeddings class that uses sentence-transformers directly
class CustomEmbeddings:
    def __init__(self, model_name="distilbert-base-nli-stsb-mean-tokens"):  # Changed model for better performance
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        """Embed a list of documents"""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [embedding.tolist() for embedding in embeddings]
    
    def embed_query(self, text):
        """Embed a query"""
        if not isinstance(text, str):
            raise ValueError("Query must be a string")
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def embed_search(self, text):
        """Compatibility method for similarity search"""
        return self.embed_query(text)
    
    def __call__(self, text):
        """Make the class callable - required for FAISS"""
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)

# Initialize embeddings
try:
    embeddings = CustomEmbeddings()
    logger.info("Successfully initialized embeddings model")
except Exception as e:
    logger.error(f"Error initializing embeddings: {e}")
    raise

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    try:
        # Create a temporary file to store the uploaded document
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf' if file.filename.endswith('.pdf') else '.docx') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Load and process the document
        documents = []
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            logger.info(f"Loaded PDF with {len(documents)} pages")
        elif file.filename.endswith('.docx'):
    # Load DOCX content
            content = load_docx(temp_path)
            # Create Document and set metadata separately
            doc = Document()  # Initialize without parameters
            doc.page_content = content  # Set page_content attribute directly
            doc.metadata = {"filename": file.filename}
            documents = [doc]
            logger.info(f"Loaded DOCX with {len(content.splitlines())} paragraphs")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        # Create embeddings and store in FAISS
        global vectorstore
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("Successfully created FAISS index")

        # Clean up temporary file
        os.unlink(temp_path)

        return {
            "message": "File uploaded and processed successfully",
            "document_count": len(documents),
            "chunk_count": len(chunks)
        }

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(search_query: SearchQuery):
    logger.info(f"Received query: {search_query.query}")
    if not vectorstore:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Please upload a PDF first."
        )
    
    try:
        # Use similarity_search_with_score to get relevance scores
        results = vectorstore.similarity_search_with_score(search_query.query, k=5)
        logger.info(f"Found {len(results)} results")
        
        # Normalize scores
        max_score = max(float(score) for _, score in results)  # Ensure scores are converted to float
        formatted_results = [{
            "content": doc.page_content,  # Accessing properties directly
            "metadata": doc.metadata,
            "score": float(score) / max_score if max_score > 0 else 0.0,  # Normalize score
            "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        } for doc, score in results]
        
        return {
            "query": search_query.query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "embeddings_loaded": embeddings is not None}

def load_docx(file_path):
    """Load content from a DOCX file."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)