from pydantic import BaseModel

class QueryRequest(BaseModel):
    report_name: str
    query: str

