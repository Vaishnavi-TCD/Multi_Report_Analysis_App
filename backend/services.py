import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# Load Sentence Transformer Model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS index setup
dimension = 384  # Model output dimension
faiss_indices = {}
document_embeddings = {}


def store_document_in_faiss(report_name, document_text):
    """Encodes and stores document in FAISS index."""
    global document_embeddings, faiss_indices

    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", document_text)
    sentence_vectors = sentence_model.encode(sentences)

    if report_name in document_embeddings:
        print(f"⚠️ Overwriting existing FAISS index for {report_name}")

    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(sentence_vectors, dtype=np.float32))

    document_embeddings[report_name] = {"sentences": sentences, "vectors": sentence_vectors}
    faiss_indices[report_name] = faiss_index  # Ensure index is stored per report

    print(f"✅ FAISS Index Updated with {len(sentences)} sentences from {report_name}")

def retrieve_relevant_text(report_name, query):
    """Retrieves relevant insights from FAISS"""
    if report_name not in faiss_indices:
        return f"⚠️ No FAISS index found for {report_name}. Please upload the report first."

    faiss_index = faiss_indices[report_name]
    
    sentences = document_embeddings[report_name]["sentences"]

    query_vector = sentence_model.encode([query])
    _, indices = faiss_index.search(np.array(query_vector, dtype=np.float32), k=3)

     # Display only the top 1 result
#    retrieved_text = sentences[indices[0][0]] if indices[0][0] < len(sentences) else "No relevant results found."
#    return retrieved_text
    
    retrieved_text = ". ".join([sentences[i] for i in indices[0] if i < len(sentences)])
    return retrieved_text if retrieved_text else "No relevant results found."

  

    return retrieved_sentences


def compare_reports(report1, report2, query):
    """Retrieves insights from two reports for comparison"""
    text1 = retrieve_relevant_text(report1, query)
    text2 = retrieve_relevant_text(report2, query)

    comparison_result = f"🔹 {report1}: {text1}\n\n🔹 {report2}: {text2}"
    return comparison_result
    
