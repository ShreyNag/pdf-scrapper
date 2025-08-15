# main.py

# 1. Import necessary tools
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import fitz
from typing import Optional

# --- NEW IMPORT for CORS ---
from fastapi.middleware.cors import CORSMiddleware
# --- END NEW IMPORT ---

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import google.generativeai as genai

# --- CONFIGURE GEMINI API KEY ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set!")
genai.configure(api_key=GOOGLE_API_KEY)

# 2. Create the FastAPI app instance
app = FastAPI()

# --- NEW CORS MIDDLEWARE CONFIGURATION ---
# This allows our frontend (running on a different "origin")
# to communicate with our backend.
origins = ["*"] # Allow all origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)
# --- END NEW CONFIGURATION ---


# This dictionary will act as our simple in-memory database.
document_store = {}

class ChatRequest(BaseModel):
    filename: str
    question: str

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    pdf_document = fitz.open(stream=contents, filetype="pdf")
    full_text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        full_text += page.get_text()
    pdf_document.close()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(full_text)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    document_store[file.filename] = vector_store

    return {
        "filename": file.filename,
        "status": "Successfully processed and indexed.",
        "total_chunks": len(text_chunks)
    }

@app.post("/chat/")
async def chat_with_doc(request: ChatRequest):
    vector_store = document_store.get(request.filename)
    if not vector_store:
        raise HTTPException(status_code=404, detail="Document not found. Please upload the PDF first.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(request.question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    You are a helpful assistant. Answer the following question based ONLY on the context provided below.
    If the answer is not found in the context, say "I cannot answer this question based on the provided document."

    Context:
    {context}

    Question:
    {request.question}
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from Gemini: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Docu-Chat API is running!"}
