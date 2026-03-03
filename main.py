from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Any
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()

# Global resources loaded lazily
_resources = {"index": None, "metadata": None, "model": None}

def load_resources():
    if _resources["index"] is None:
        print("Loading RAG resources...")
        try:
            _resources["index"] = faiss.read_index("faiss.index")
            with open("metadata.json", "r", encoding="utf-8") as f:
                _resources["metadata"] = json.load(f)
            # This is the heavy part
            _resources["model"] = SentenceTransformer("all-MiniLM-L6-v2")
            print("Resources loaded successfully.")
        except Exception as e:
            print(f"Error loading resources: {e}")
            raise e
    return _resources["index"], _resources["metadata"], _resources["model"]

@app.api_route("/", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "message": "Alemni RAG Server is running"}

@app.post("/ask")
async def ask(request: Request):
    # Use raw request to avoid Pydantic 422 errors
    try:
        req_data = await request.json()
    except:
        return {"answer": "Erreur : JSON invalide.", "citations": []}
        
    question = req_data.get("question", "")
    user_id = req_data.get("user_id", 0)
    allowed_ids = req_data.get("allowed_course_ids", [])
    
    print(f"Received request: {question} for user {user_id}")
    
    try:
        index, metadata, model = load_resources()
    except Exception as e:
        print(f"Failed to load resources: {e}")
        return {"answer": "Erreur serveur : ressources non disponibles.", "citations": []}

    # 1. Vector Search
    q_emb = model.encode([str(question)])
    q_emb = np.array(q_emb).astype("float32")

    # Retrieve candidates
    distances, ids = index.search(q_emb, 30)

    context_chunks = []
    citations = []
    seen_urls = set()

    for rank, idx in enumerate(ids[0]):
        if idx == -1 or idx >= len(metadata):
            continue
            
        item = metadata[idx]
        cid = item.get("doc_id") or item.get("course_id")

        # Filtering
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

    if not context_chunks:
        return {
            "answer": "Je n'ai pas trouvé d'information spécifique dans vos cours actuels.",
            "citations": []
        }

    # Better summary answer
    answer = context_chunks[0]
    if len(answer) > 500:
        answer = answer[:500] + "..."
    
    return {
        "answer": answer,
        "citations": citations
    }
