# main.py

# 1. Import necessary tools
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF
from typing import Optional, List
from PIL import Image
import io

from fastapi.middleware.cors import CORSMiddleware

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

# --- CORS MIDDLEWARE CONFIGURATION ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for vector stores
document_store = {}

class ChatRequest(BaseModel):
    filename: str
    question: str

# --- NEW HELPER FUNCTION: Get Image Description ---
def get_image_description(image_bytes: bytes) -> str:
    """
    Uses Gemini 1.5 Flash to generate a description for an image.
    """
    try:
        # Prepare the image parts for the Gemini API
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
        ]
        prompt_parts = [
            "Describe this image in detail. If it is a graph or chart, explain what the data shows. If it is a diagram, explain what it represents.",
            *image_parts
        ]
        
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        print(f"Could not get description for an image: {e}")
        return "Could not analyze this image."
# --- END NEW HELPER FUNCTION ---


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Processes a PDF, extracts text, tables, and images, creates a multimodal
    vector store, and stores it in memory.
    """
    contents = await file.read()
    pdf_document = fitz.open(stream=contents, filetype="pdf")
    
    # --- NEW MULTIMODAL EXTRACTION LOGIC ---
    documents = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        # 1. Extract plain text
        documents.append(page.get_text())
        
        # 2. Extract and format tables
        tables = page.find_tables()
        for table in tables:
            table_data = table.extract()
            # Convert table data to Markdown format for better context
            markdown_table = "| " + " | ".join(map(str, table_data[0])) + " |\n"
            markdown_table += "| " + " | ".join(["---"] * len(table_data[0])) + " |\n"
            for row in table_data[1:]:
                markdown_table += "| " + " | ".join(map(str, row)) + " |\n"
            documents.append(f"The following is a table:\n{markdown_table}")

        # 3. Extract images, get descriptions, and add them
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get an AI-generated description of the image
            image_description = get_image_description(image_bytes)
            documents.append(f"The following is a description of an image on this page: {image_description}")
            
    pdf_document.close()
    # --- END NEW EXTRACTION LOGIC ---

    # Join all extracted content into a single text block
    full_content = "\n\n".join(documents)

    # Chunk the combined content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(full_content)

    # Create embeddings and the vector store
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    document_store[file.filename] = vector_store

    return {
        "filename": file.filename,
        "status": "Successfully processed and indexed.",
        "total_chunks": len(text_chunks)
    }

# The /chat/ endpoint and root endpoint remain exactly the same
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
