# main.py

# 1. Import necessary tools
import os
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import fitz
from typing import Optional

# Import for CORS middleware
from fastapi.middleware.cors import CORSMiddleware

# Imports for LangChain and Gemini
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import google.generativeai as genai

# --- CONFIGURE GEMINI API KEY ---
# Load environment variables from .env, then read the API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set!")
genai.configure(api_key=GOOGLE_API_KEY)

# Singleton: loading the ~90MB sentence-transformer model from disk is expensive,
# and the model itself is stateless, so we load it once at startup and reuse it
# for every upload instead of reloading it per-request.
EMBEDDINGS = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Create the FastAPI app instance
app = FastAPI()

# --- CORS MIDDLEWARE CONFIGURATION ---
# This allows our frontend to communicate with our backend.
origins = ["*"] # Allow all origins for development

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# This is our simple in-memory database: it lives only in this process's
# memory, so it does NOT survive a restart, and it does NOT work across
# multiple gunicorn workers, since an upload handled by one worker is
# invisible to the others. The real fix for either problem is persisting
# the FAISS index (FAISS.save_local/load_local) or using a hosted vector
# store — out of scope for this project.
#
# It's capped at MAX_DOCUMENTS via an OrderedDict acting as an LRU cache:
# document_store.move_to_end() on access keeps active documents alive,
# and the oldest (least-recently-used) entry is evicted once the cap is
# exceeded, so memory doesn't grow unbounded.
MAX_DOCUMENTS = 20
document_store: "OrderedDict[str, dict]" = OrderedDict()

# Pydantic model to define the structure of chat requests
class ChatRequest(BaseModel):
    doc_id: str
    question: str

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB

# L2 distance cutoff below which a chunk counts as "relevant" — chosen by
# inspecting scores for on-topic vs. off-topic questions against this
# embedding model, not derived from any formula. Re-check by hand if you
# swap embedding models, since distance scales differ between models.
RELEVANCE_THRESHOLD = 1.0

@app.post("/upload-pdf/")
def upload_pdf(file: UploadFile = File(...)):
    """
    Processes a PDF, extracts its text, creates a vector store,
    and stores it in memory.
    """
    # def, not async def: PDF parsing, embedding, and FAISS indexing below
    # are all synchronous, blocking, CPU-bound work. FastAPI runs a sync
    # endpoint in a threadpool automatically, which is exactly what that
    # needs; leaving it async def would block the whole event loop (and
    # every other in-flight request) for the duration of one upload.
    if file.content_type != "application/pdf" or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = file.file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.")

    try:
        pdf_document = fitz.open(stream=contents, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open file as a PDF: {e}")

    try:
        if pdf_document.is_encrypted:
            raise HTTPException(status_code=400, detail="This PDF is encrypted/password-protected and cannot be processed.")

        # Extract page-by-page (rather than into one big string) so page
        # numbers survive as metadata and citations are possible later.
        pages = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pages.append(Document(page_content=page.get_text(), metadata={"page": page_num + 1}))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {e}")
    finally:
        pdf_document.close()

    full_text = "".join(p.page_content for p in pages)
    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this PDF. It is probably a scanned document that needs OCR.",
        )

    # Chunk the pages; split_documents (rather than split_text) propagates
    # each page's metadata onto the chunks it produces.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    text_chunks = text_splitter.split_documents(pages)

    # Create the vector store using the shared, module-level embeddings model
    vector_store = FAISS.from_documents(documents=text_chunks, embedding=EMBEDDINGS)

    # Key by a server-generated id rather than the filename: two users
    # uploading "resume.pdf" must not overwrite each other's document.
    doc_id = uuid.uuid4().hex
    document_store[doc_id] = {
        "vector_store": vector_store,
        "filename": file.filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    if len(document_store) > MAX_DOCUMENTS:
        document_store.popitem(last=False)  # evict the oldest (least-recently-used) entry

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "Successfully processed and indexed.",
        "total_chunks": len(text_chunks)
    }

@app.post("/chat/")
def chat_with_doc(request: ChatRequest):
    """
    Answers a question based on the content of a previously uploaded PDF.
    """
    # def, not async def: FAISS search and model.generate_content() are both
    # blocking calls. FastAPI offloads sync endpoints to a threadpool, so
    # this keeps one slow Gemini call from stalling every other request.
    entry = document_store.get(request.doc_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"No document found for id '{request.doc_id}'. Please upload the PDF first.")
    document_store.move_to_end(request.doc_id)  # mark as recently used so it survives eviction
    vector_store = entry["vector_store"]

    # Retrieve relevant context, along with each chunk's similarity score
    # so we can cite pages and report source excerpts back to the caller.
    #
    # NOTE: this FAISS index (default distance strategy, i.e. raw L2) returns
    # a DISTANCE, not a similarity — LOWER scores mean MORE similar. Do not
    # flip this comparison without re-checking, or the threshold silently
    # inverts and every query either always or never passes.
    results = vector_store.similarity_search_with_score(request.question, k=6)
    relevant_results = [(doc, score) for doc, score in results if score <= RELEVANCE_THRESHOLD]

    if not relevant_results:
        return {
            "answer": "I couldn't find any content in this document relevant to your question.",
            "sources": [],
            "context_found": False,
        }

    context = "\n\n".join(f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}" for doc, _ in relevant_results)
    sources = [
        {"page": doc.metadata.get("page"), "score": float(score), "excerpt": doc.page_content[:200]}
        for doc, score in relevant_results
    ]

    # Augment the prompt
    prompt = f"""
    You are a helpful assistant. Use the context below (taken from the user's uploaded document)
    to answer the question, which may ask for facts, a summary, or advice/help based on the document
    (e.g. interview prep, feedback, or suggestions).

    Ground your answer in the context — don't invent facts about the document that aren't there —
    but you may reason about and build on the context to be genuinely helpful.

    If the context has nothing relevant to the question at all, say
    "I cannot answer this question based on the provided document."

    Context:
    {context}

    Question:
    {request.question}
    """

    # Generate the answer
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt)
        return {"answer": response.text, "sources": sources, "context_found": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from Gemini: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Docu-Chat API is running!"}
