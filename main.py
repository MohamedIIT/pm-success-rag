from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Any
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import time
from contextlib import asynccontextmanager

# Global resources
_resources = {"index": None, "metadata": None, "model": None}

def load_resources():
    if _resources["model"] is None:
        start_time = time.time()
        print(">>> START: Loading RAG resources...")
        try:
            # 1. Load FAISS (Fast)
            _resources["index"] = faiss.read_index("faiss.index")
            print(f">>> Index loaded in {time.time() - start_time:.2f}s")
            
            # 2. Load Metadata (Fast)
            with open("metadata.json", "r", encoding="utf-8") as f:
                _resources["metadata"] = json.load(f)
            print(f">>> Metadata loaded in {time.time() - start_time:.2f}s")
            
            # 3. Load Model (Slow - ~80MB)
            print(">>> Loading SentenceTransformer (all-MiniLM-L6-v2)...")
            _resources["model"] = SentenceTransformer("all-MiniLM-L6-v2")
            print(f">>> Resources READY in {time.time() - start_time:.2f}s total.")
        except Exception as e:
            print(f">>> CRITICAL ERROR during resource load: {e}")
            raise e
    return _resources["index"], _resources["metadata"], _resources["model"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm on startup
    try:
        load_resources()
    except:
        pass
    yield

app = FastAPI(lifespan=lifespan)

@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "message": "Alemni RAG Server is running", "ready": _resources["model"] is not None}

@app.post("/ask")
async def ask(request: Request):
    req_start = time.time()
    try:
        req_data = await request.json()
    except:
        return {"answer": "Erreur : JSON invalide.", "citations": []}
        
    question = req_data.get("question", "")
    user_id = req_data.get("user_id", 0)
    allowed_ids = req_data.get("allowed_course_ids", [])
    
    print(f"\n--- New Request from User {user_id} ---")
    print(f"Question: {question}")
    
    try:
        index, metadata, model = load_resources()
    except Exception as e:
        return {"answer": f"Erreur serveur (chargement) : {str(e)}", "citations": []}

    # 1. Vector Search
    embed_start = time.time()
    q_emb = model.encode([str(question)])
    q_emb = np.array(q_emb).astype("float32")
    print(f"Embedding completed in {time.time() - embed_start:.2f}s")

    # 2. FAISS Search
    search_start = time.time()
    distances, ids = index.search(q_emb, 30)
    print(f"FAISS search completed in {time.time() - search_start:.2f}s")

    context_chunks = []
    citations = []
    seen_urls = set()

    # Ensure allowed_ids is a list
    if not isinstance(allowed_ids, list):
        allowed_ids = []

    # 3. Filtering
    filter_start = time.time()
    for rank, idx in enumerate(ids[0]):
        if idx == -1 or idx >= len(metadata):
            continue
            
        item = metadata[idx]
        # Check both doc_id and course_id for robustness
        cid = item.get("doc_id") or item.get("course_id")

        if allowed_ids and cid not in allowed_ids:
            continue

        if len(context_chunks) < 5:
            context_chunks.append(item.get("text", ""))
            
            url = item.get("url")
            if url and url not in seen_urls:
                citations.append({
                    "type": "lesson",
                    "title": item.get("title", "Resource"),
                    "url": url,
                    "locator": item.get("chunk_id")
                })
                seen_urls.add(url)
    print(f"Filtering completed in {time.time() - filter_start:.2f}s")

    if not context_chunks:
        print("Result: No matching context found.")
        return {
            "answer": "Je n'ai pas trouvé d'information spécifique dans vos cours actuels.",
            "citations": []
        }

    # Answer Construction
    answer = context_chunks[0]
    if len(answer) > 600:
        answer = answer[:600] + "..."
    
    print(f"Total processing time: {time.time() - req_start:.2f}s")
    return {
        "answer": answer,
        "citations": citations
    }
