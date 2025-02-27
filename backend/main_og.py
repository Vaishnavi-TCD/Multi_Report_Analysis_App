from fastapi import FastAPI, UploadFile, File, HTTPException
from services import get_insights
from models import QueryRequest
from typing import List
from services import get_insights, store_document, document_embeddings, retrieve_relevant_text
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# ✅ CORS Middleware to Fix Frontend Connection Issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change to specific origin in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Temporary storage for reports
reports = {}

#@app.post("/upload/")
#async def upload_report(file: UploadFile = File(...)):
#    """Endpoint to upload a market research report."""
#    content = await file.read()
#    reports[file.filename] = content.decode("utf-8")
#    return {"message": "File uploaded successfully", "filename": file.filename}
 #22/02/25   
#@app.post("/upload/")
#async def upload_report(file: UploadFile = File(...)):
#    """Endpoint to upload a market research report."""
#    if not file.filename:
#        raise HTTPException(status_code=400, detail="No filename provided")

#    content = await file.read()
#    reports[file.filename] = content.decode("utf-8")

#    return {"message": "File uploaded successfully", "filename": file.filename}
    
    
from services import store_document

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    report_text = content.decode("utf-8")

    # Ensure the report is stored in FAISS
    store_document(file.filename, report_text)

    # Also store in reports dictionary for reference
    reports[file.filename] = report_text  

    return {"message": "File uploaded successfully", "filename": file.filename}




@app.get("/reports/")
def list_reports():
    """Endpoint to list all uploaded reports."""
    return {"available_reports": list(reports.keys())}

#@app.post("/query/")
#def query_report(report_name: str, query: str):
#    """Endpoint to process user queries on a given report."""
#    if report_name not in reports:
#        raise HTTPException(status_code=404, detail="Report not found")
    
#    response = get_insights(reports[report_name], query)
#    return {"report": report_name, "query": query, "response": response}
    
from pydantic import BaseModel

class QueryRequest(BaseModel):
    report_name: str
    query: str

#@app.post("/query/")
#def query_report(request: QueryRequest):
#    """Process user queries on a given report."""
#    if request.report_name not in reports:
#        raise HTTPException(status_code=404, detail="Report not found")

#    response = get_insights(reports[request.report_name], request.query)
#    return {"report": request.report_name, "query": request.query, "response": response}
 
# search document encodings instead of report 
#@app.post("/query/")
#def query_report(request: QueryRequest):
#    """Process user queries using FAISS instead of raw reports{}."""
#    if request.report_name not in document_embeddings:
#        raise HTTPException(status_code=404, detail="Report not found in FAISS")

#    response = get_insights(request.report_name, request.query)
#    return {"report": request.report_name, "query": request.query, "response": response}
    
    
@app.post("/query/")
def query_report(request: QueryRequest):
    """Process user queries using FAISS instead of reports{}."""
    global document_embeddings  # Ensure we reference FAISS stored embeddings

    # Ensure FAISS is loaded
    if not document_embeddings:
        from services import load_faiss  # Import FAISS loading function
        load_faiss()  # Load stored FAISS index

    if request.report_name not in document_embeddings:
        raise HTTPException(status_code=404, detail="Report not found in FAISS")

    #response = get_insights(request.report_name, request.query)
    #return {"report": request.report_name, "query": request.query, "response": response}
    relevant_text = retrieve_relevant_text(request.report_name, request.query)
    return {
        "report": request.report_name,
        "query": request.query,
        "response": relevant_text
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

