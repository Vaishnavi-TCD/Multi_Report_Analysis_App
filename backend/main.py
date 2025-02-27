from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services import store_document_in_faiss, retrieve_relevant_text, compare_reports
import shutil
import os
import fitz 


app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Uploads and stores document in FAISS"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text based on file type
    if file.filename.endswith(".pdf"):
        content = extract_text_from_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

    store_document_in_faiss(file.filename, content)
    return {"message": "File uploaded successfully", "filename": file.filename}

from pydantic import BaseModel

class QueryRequest(BaseModel):
    report_name: str
    query: str

@app.post("/query/")
async def query_report(request: QueryRequest):
    """Retrieve insights from a single report"""
    relevant_text = retrieve_relevant_text(request.report_name, request.query)
    return {"report": request.report_name, "query": request.query, "response": relevant_text}

# Define request model for compare endpoint
class CompareRequest(BaseModel):
    report1: str
    report2: str
    query: str

@app.post("/compare/")
async def compare_reports_api(request: CompareRequest):
    """Compare insights from two reports"""
    comparison_result = compare_reports(request.report1, request.report2, request.query)
    return {"report1": request.report1, "report2": request.report2, "query": request.query, "response": comparison_result}
    
    