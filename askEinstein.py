import os
import glob
import json
import pdfplumber
import pandas as pd
import faiss
import pickle
import streamlit as st
from tqdm import tqdm
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import tiktoken
import math

# === Configuration ===
# Embed your API key directly here (or replace with environment variable retrieval)
openai.api_key = st.secrets["OPENAI_API_KEY"] 
enc = tiktoken.get_encoding("cl100k_base")

# Constants
MAX_EMBED_TOKENS = 8000
CHUNK_SIZE = 3000        # for PDF/text splitting
CHUNK_OVERLAP = 50
BATCH_SIZE = 10          # reduced to avoid token-limit errors
TOP_K = 5

CHUNKS_FILE = "chunks.pkl"
INDEX_FILE = "index.faiss"

@st.cache_resource(show_spinner="Loading FAISS index …", persist="disk")
def load_assets():
    """
    Load chunks.pkl and index.faiss if they exist.  
    Memory-map the FAISS file so only the pages you query are pulled into RAM.
    If the files are missing, build them from scratch.
    """
    if os.path.exists(CHUNKS_FILE) and os.path.exists(INDEX_FILE):
        chunks_local = load_pickle(CHUNKS_FILE)
        index_local  = faiss.read_index(INDEX_FILE, faiss.IO_FLAG_MMAP)
    else:
        chunks_local, index_local = load_files()         # your existing builder
    return chunks_local, index_local

SYSTEM_PROMPT = (
    "You are a senior property performance consultant. "
    "You help analyze Accounts, Properties, Surveys, Reviews and identify problem points. "
    "For any flagged property, do the following:\n"
    "- Clearly list the JTA ID, Property name, and relevant failing metrics with scores.\n"
    "- The metrics should be from the various files you have trained on - Prospect, Resident(TALi), ORA, Survey Response Rate, Review Respond Time, Work Order score, and so on. They should not be clubbed together as one score.\n"
    "- For each property, explain the key pain points concisely.\n"
    "- For each, create a detailed, practical action plan covering Reputation, Maintenance, Leasing, Resident Engagement, or other relevant areas.\n"
    "- Structure the answer using Markdown or JSON tables if possible.\n"
    "- Use clear headings and bullet points.\n"
    "- End with a short 'Next Steps' section: How to operationalize this plan regionally.\n"
    "Always think step by step and produce well-structured output."
)

# Logging utility
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Extract text from PDFs (with OCR fallback)
def extract_pdf_text(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        pass
    if text.strip():
        return text
    try:
        doc = fitz.open(file)
        for page in doc:
            text += page.get_text()
    except Exception:
        pass
    try:
        doc = fitz.open(file)
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img)
    except Exception:
        pass
    return text

# Robust embedding that splits on errors
try:
    from openai.error import InvalidRequestError
except ImportError:
    class InvalidRequestError(Exception):
        """Fallback placeholder for specific OpenAI exception"""
        pass

def safe_embed(batch):
    if not batch:
        return []
    try:
        log(f"Embedding batch of {len(batch)} chunks...")
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=batch
        )
        log("Batch embedded successfully.")
        return [r.embedding for r in response.data]
    except InvalidRequestError:
        if len(batch) == 1:
            raise
        mid = len(batch) // 2
        log("Batch too large, splitting further...")
        return safe_embed(batch[:mid]) + safe_embed(batch[mid:])

# Load and preprocess all docs, chunk and embed
def load_files():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    files = glob.glob("docs/*")
    all_chunks = []
    index = faiss.IndexFlatL2(3072)

    for file in tqdm(files, desc="Processing files"):
        log(f"Starting file: {file}")
        ext = os.path.splitext(file)[1].lower()
        file_chunks = []
        file_embeddings = []

        if ext in [".csv", ".xlsx", ".xlsm"]:
            try:
                if ext == ".csv":
                    try:
                        df = pd.read_csv(file, encoding="utf-8")
                    except UnicodeDecodeError:
                        log(f"⚠️ UTF-8 failed, trying ISO-8859-1 for {file}")
                        df = pd.read_csv(file, encoding="ISO-8859-1", on_bad_lines='skip')
                else:
                    df = pd.read_excel(file)
            except Exception as e:
                log(f"❌ Failed to load {file}: {e}")
                continue

            rows = []
            for i in range(0, len(df), 100):
                chunk = df.iloc[i:i+100].to_string(index=False)
                if len(enc.encode(chunk)) > MAX_EMBED_TOKENS:
                    parts = splitter.split_text(chunk)
                    rows.extend(parts)
                else:
                    rows.append(chunk)
            log(f"Total chunks for {file}: {len(rows)}")

            num_batches = math.ceil(len(rows) / BATCH_SIZE)
            for idx in tqdm(range(num_batches), desc=f"Embedding batches for {os.path.basename(file)}"):
                batch = rows[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                log(f"Embedding batch {idx+1}/{num_batches} for {file}")
                embeddings = safe_embed(batch)
                file_chunks.extend(batch)
                file_embeddings.extend(embeddings)

        elif ext == ".pdf":
            text = extract_pdf_text(file)
            parts = splitter.split_text(text)
            log(f"Total text chunks for {file}: {len(parts)}")

            num_batches = math.ceil(len(parts) / BATCH_SIZE)
            for idx in tqdm(range(num_batches), desc=f"Embedding batches for {os.path.basename(file)}"):
                batch = parts[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                log(f"Embedding batch {idx+1}/{num_batches} for {file}")
                embeddings = safe_embed(batch)
                file_chunks.extend(batch)
                file_embeddings.extend(embeddings)

        all_chunks.extend(file_chunks)
        if file_embeddings:
            index.add(pd.DataFrame(file_embeddings).values.astype('float32'))
            faiss.write_index(index, INDEX_FILE)
            save_pickle(all_chunks, CHUNKS_FILE)
            log(f"💾 Progress saved after {file}")

    return all_chunks, index

# Utility to save/load pickle
def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Streamlit UI
st.title("📊 Property Consultant Chatbot")

# LOCAL
# if os.path.exists(CHUNKS_FILE) and os.path.exists(INDEX_FILE):
#    chunks = load_pickle(CHUNKS_FILE)
#    index = faiss.read_index(INDEX_FILE)
# else:
#    chunks, index = load_files()

# STREAMLIT
chunks, index = load_assets()

st.session_state.chunks = chunks
st.session_state.index = index

query = st.chat_input("Ask your question:")

if query:
    vec = safe_embed([query])[0]
    D, I = index.search(pd.DataFrame([vec]).values.astype('float32'), TOP_K)
    context = "\n".join([chunks[i] for i in I[0]])
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    st.markdown(response.choices[0].message.content)
