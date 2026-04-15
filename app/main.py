import sys
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_google_genai import ChatGoogleGenerativeAI

from app.chunker import process_document
from app.embedder import encode_texts, encode_text
from app.retriever import VectorStore
from app.cache import query_cache

load_dotenv()

app = FastAPI(title="ContextRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore()
chunk_store = []

try:
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=gemini_key)
except Exception as e:
    print(f"Warning: Failed to initialize LLM: {e}")
    llm = None

@app.on_event("startup")
def startup_event():
    store.load()
    global chunk_store
    chunk_store = store.documents.copy()

class QueryRequest(BaseModel):
    question: str
    chat_history: list[dict] = []
    mode: str = "hybrid"

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    content = await file.read()
    try:
        chunks = process_document(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
        
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in document.")
        
    embeddings = encode_texts(chunks)
    store.add_vectors(embeddings, chunks)
    store.save()
    
    global chunk_store
    chunk_store.extend(chunks)
    query_cache.clear()
    
    return {
        "message": f"Successfully processed '{file.filename}'",
        "chunks_indexed": len(chunks)
    }

@app.post("/query")
async def query_index(request: QueryRequest):
    query_text = request.question
    if not query_text.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    # Phase 3: Conversational Memory and Hybrid Search integration
    history = request.chat_history
    mode = request.mode
    
    # Check cache (now hashed by question + history bounds)
    cached_response = query_cache.get(query_text, history)
    if cached_response:
        return cached_response
        
    # Hybrid Retrieval (semantic|keyword|hybrid)
    results = store.retrieve(query_text, mode=mode, top_k=5)
    
    # Format chunks logic
    source_texts = []
    for res in results:
        chunk_text = res.get('chunk', '')
        # Incorporate breakdown into source for frontend parsing if needed
        # We will return the dict payload to preserve scores
        source_texts.append(res)
        
    joined_chunks = "\n\n".join([r['chunk'] for r in source_texts])
    
    # Inject conversational history into Prompt
    history_block = ""
    if history:
        history_block = "--- Conversation So Far ---\n"
        # History is expected to be [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
        # Take the last 4 turns (8 messages)
        recent_history = history[-8:]
        for msg in recent_history:
            role_label = "User" if msg.get("role") == "user" else "Assistant"
            history_block += f"{role_label}: {msg.get('content')}\n"
        history_block += "\n"
    
    prompt = f"""You are a strict Retrieval-Augmented Generation (RAG) assistant designed to produce fully grounded answers.

Your task is to answer the user’s question using ONLY the provided context.

STRICT RULES:

1. CONTEXT-ONLY ANSWERING

* Use ONLY the given context.
* Do NOT use prior knowledge or external information.
* Do NOT guess, infer, or assume anything not explicitly stated.

2. NEGATIVE / BOUNDARY AWARENESS

* Pay special attention to negative statements in the context (e.g., “does NOT use labeled data”).
* Do NOT confuse similar concepts.
* Ensure distinctions (e.g., supervised vs unsupervised vs reinforcement learning) are preserved exactly as stated.

3. UNANSWERABLE QUESTIONS

* If the answer is NOT explicitly present in the context, respond EXACTLY with:
  "Not found in context."
* Do NOT attempt partial answers.
* Do NOT use outside knowledge.

4. WEAK OR IRRELEVANT CONTEXT

* If the provided context is insufficient, vague, or unrelated to the question, respond EXACTLY with:
  "Insufficient context to answer."

5. NO HALLUCINATION POLICY

* Every part of your answer must be directly supported by the context.
* If you cannot find clear supporting information → do NOT answer.

6. ANSWER FORMAT
   Answer: <concise, context-based answer>
   Source: <exact supporting sentence or chunk from context>

INPUT:

{history_block}Context:
{joined_chunks}

Question:
{query_text}"""

    if llm is None:
        raise HTTPException(status_code=500, detail="LLM not initialized. Check GOOGLE_API_KEY.")
        
    try:
        response = llm.invoke(prompt)
        answer_text = response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer from Gemini: {str(e)}")
    
    final_response = {
        "answer": answer_text,
        "sources": source_texts
    }
    
    query_cache.set(query_text, history, final_response)
    
    return final_response

