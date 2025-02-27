#import openai
#import os
#from dotenv import load_dotenv

# Load API keys from .env file
#load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")

#def get_insights(report_text, query):
#    """Uses GPT to generate insights based on the report and user query."""
#    if not openai.api_key:
#        raise ValueError("Missing OpenAI API Key. Make sure OPENAI_API_KEY is set in .env")

#    prompt = f"Given the following report:\n\n{report_text}\n\nAnswer the query: {query}"
    
#    response = openai.ChatCompletion.create(
        
        #model="gpt-4",
#        model="gpt-3.5-turbo",
#        messages=[{"role": "system", "content": "You are an AI assistant."},
#                  {"role": "user", "content": prompt}],
#        temperature=0.7
#    )

#    return response.choices[0].message["content"]


#from transformers import AutoModelForCausalLM, AutoTokenizer
#import torch

# Load LLaMA 2 model from Hugging Face
#model_name = "meta-llama/Llama-2-7b-chat-hf"  # You can use a smaller model like "meta-llama/Llama-2-7b"
#model_name = "google/flan-t5-small"  # Or "mistralai/Mistral-7B-Instruct"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

#def get_insights(report_text, query):
#    """Generates insights using LLaMA 2 instead of OpenAI's API."""
    
#    prompt = f"Given the following report:\n\n{report_text}\n\nAnswer the query: {query}"
    
    # Tokenize input
#    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate response
#    outputs = model.generate(**inputs, max_length=500)
#    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#    return response
    
    
#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#import torch

#model_name = "google/flan-t5-small"  
#model_name = "mistralai/Mistral-7B-Instruct"

#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

#def get_insights(report_text, query):
#    """Generates insights using Hugging Face models."""
    
#    prompt = f"Given the following report:\n\n{report_text}\n\nAnswer the query: {query}"
    
#    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
#    outputs = model.generate(**inputs, max_length=500)
#    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#    return response


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re

# Load model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load language model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# FAISS Index
#index = faiss.IndexFlatL2(384)  # 384 is embedding size for MiniLM
#document_embeddings = {}

#22/02/25
#def store_document(report_name, report_text):
#    """Convert report into embeddings and store in FAISS index."""
#    global index
#    sentences = report_text.split(". ")  # Splitting report into sentences
#    embeddings = embedding_model.encode(sentences)

    # Print debug info
#    print(f"🔹 Storing {len(sentences)} sentences in FAISS for {report_name}")
#    for i, sentence in enumerate(sentences):
#        print(f"📝 Sentence {i}: {sentence}")

#    document_embeddings[report_name] = (sentences, embeddings)

    # Reset FAISS index and add new embeddings
#    index = faiss.IndexFlatL2(embeddings.shape[1])
#    index.add(np.array(embeddings, dtype=np.float32))

#    print("✅ FAISS Indexing Complete")




#def retrieve_relevant_text(report_name, query):
#    """Retrieve the most relevant sentences from a report based on query."""
#    if report_name not in document_embeddings:
#        return "Report not found."

#    sentences, embeddings = document_embeddings[report_name]
#    query_embedding = embedding_model.encode([query])
    
    # Find the closest match
#    _, indices = index.search(np.array(query_embedding, dtype=np.float32), 3)
    
#    retrieved_text = ". ".join([sentences[i] for i in indices[0]])
#    return retrieved_text
    
#def retrieve_relevant_text(report_name, query):
#    """Find the most relevant sentences using FAISS."""
#    if report_name not in document_embeddings:
#        return "Report not found."

#    sentences, embeddings = document_embeddings[report_name]
#    query_embedding = embedding_model.encode([query])

    # Find closest match
#    _, indices = index.search(np.array(query_embedding, dtype=np.float32), 3)
#    
#    retrieved_text = ". ".join([sentences[i] for i in indices[0]])
    
    # Debugging: Print retrieved text
#    print("FAISS Retrieved Text:", retrieved_text)

#    return retrieved_text
#def store_document(report_name, report_text):
#    """Convert report into embeddings and store in FAISS index."""
#    sentences = report_text.split("\n")  # Ensure each line is a separate embedding
#    embeddings = embedding_model.encode(sentences)
    
#    document_embeddings[report_name] = (sentences, embeddings)
#    index.add(np.array(embeddings, dtype=np.float32))

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import os
import faiss
import numpy as np
import pickle

FAISS_INDEX_FILE = "faiss_index.bin"
EMBEDDINGS_FILE = "document_embeddings.pkl"

# Load FAISS index if it exists
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(EMBEDDINGS_FILE, "rb") as f:
        document_embeddings = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)  # 384 is the embedding size for MiniLM
    document_embeddings = {}

def save_faiss():
    """Persist FAISS index and embeddings to disk."""
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(document_embeddings, f)

def load_faiss():
    """Load FAISS index and embeddings from disk."""
    global index, document_embeddings

    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            document_embeddings = pickle.load(f)

    logger.debug("✅ FAISS index and embeddings loaded successfully.")


def store_document(report_name, report_text):
    """Store embeddings of the report in FAISS and persist it."""
    global index, document_embeddings

    #sentences = report_text.split(". ")  # Ensure sentences are separated
    sentences = re.split(r'(?<!\d)\.\s+|\n+', report_text.strip())
    embeddings = embedding_model.encode(sentences)
    
        # Print debug info
    print(f"🔹 Storing {len(sentences)} sentences in FAISS for {report_name}")
    for i, sentence in enumerate(sentences):
        print(f"📝 Sentence {i}: {sentence}")

    if len(sentences) == 0 or len(embeddings) == 0:
        logger.warning(f"🚨 No valid sentences found in {report_name}. Skipping FAISS storage.")
        return

    # Store in memory
    document_embeddings[report_name] = (sentences, embeddings)

    # Add to FAISS index
    #index.reset()
    index.add(np.array(embeddings, dtype=np.float32))

    # Save FAISS index and embeddings
    save_faiss()
    logger.debug(f"✅ FAISS Index Updated with {len(sentences)} sentences from {report_name}")

def debug_faiss_retrieval(report_name, query_embedding):
    """Debugging function to print FAISS index details."""
    print(f"🔎 Debugging FAISS retrieval for: {report_name}")
    print(f"Query Vector Shape: {query_embedding.shape}")

    if report_name not in document_embeddings:
        print(f"🚨 Report {report_name} not found in FAISS.")
        return

    _, indices = index.search(np.array(query_embedding, dtype=np.float32), 5)
    print(f"📊 Retrieved FAISS Indices: {indices[0]}")




def retrieve_relevant_text(report_name, query):
    """Retrieve relevant text from FAISS index safely."""
    global index, document_embeddings

    # Ensure FAISS is loaded
    if not document_embeddings:
        if os.path.exists(FAISS_INDEX_FILE):
            index = faiss.read_index(FAISS_INDEX_FILE)
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                document_embeddings = pickle.load(f)

    if report_name not in document_embeddings:
        logger.warning("🚨 Report not found in FAISS.")
        return "Report not found."

    sentences, embeddings = document_embeddings[report_name]
    query_embedding = embedding_model.encode([query])
    debug_faiss_retrieval(report_name, query_embedding) 
    
    
    # Find the closest match
    #_, indices = index.search(np.array(query_embedding, dtype=np.float32), 3)
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), 3)  # Retrieve top 5 instead of 3


    # **Fix: Check if indices are valid**
    if len(indices[0]) == 0 or indices[0][0] < 0:
        logger.warning("🚨 FAISS retrieval returned no results.")
        return "No relevant results found."

    #retrieved_text = ". ".join([sentences[i] for i in indices[0] if i < len(sentences)])
    #retrieved_text = ". ".join([sentences[i].strip() for i in indices[0] if i < len(sentences)])
    #retrieved_text = ". ".join(
    #    [sentences[i].strip().replace("\n", " ").replace("\r", "").split(maxsplit=1)[-1]
    #     for i in indices[0] if i < len(sentences)]
    #)
    
    retrieved_text = ". ".join(
        [re.sub(r"^\d+\.\s*", "", sentences[i].strip().replace("\n", " ").replace("\r", ""))
         for i in indices[0] if i < len(sentences)]
    )

    logger.debug(f"🔍 Query: {query}")
    logger.debug(f"📊 Retrieved indices: {indices[0]}")
    for idx in indices[0]:
        if idx < len(sentences):
            logger.debug(f"✅ Retrieved sentence: {sentences[idx]}")

    return retrieved_text if retrieved_text else "No relevant results found."




def get_insights(report_name, query):
    """Retrieve relevant text & generate response using AI."""
    relevant_section = retrieve_relevant_text(report_name, query)
    
    if not relevant_section.strip():
        return {"error": "No relevant information found in the report."}

    prompt = f"Based on the following report section:\n\n{relevant_section}\n\nAnswer the query: {query}"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_length=500)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response, "source": relevant_section}
