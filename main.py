from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()

# Configuration
INDEX_FILE = "faiss.index"
META_FILE = "metadata.json"

# Global resources
_resources = {"index": None, "metadata": None, "model": None}

def load_resources():
    if _resources["index"] is None:
        print("Loading RAG resources...")
        _resources["index"] = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            _resources["metadata"] = json.load(f)
        _resources["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _resources["index"], _resources["metadata"], _resources["model"]

class AskRequest(BaseModel):
    question: str
    user_id: int
    allowed_course_ids: List[int]
    context: Optional[dict] = None

@app.post("/ask")
async def ask(req: AskRequest):
    index, metadata, model = load_resources()

    # 1. Vector Search
    q_emb = model.encode([req.question])
    q_emb = np.array(q_emb).astype("float32")

    # Retrieve 50 candidates to allow for filtering
    distances, ids = index.search(q_emb, 50)

    # 2. Filtering & Citation Extraction
    context_chunks = []
    citations = []
    seen_urls = set()

    for rank, idx in enumerate(ids[0]):
        if idx == -1 or idx >= len(metadata):
            continue
            
        item = metadata[idx]
        doc_id = item.get("doc_id")

        # SECURITY: Only include if doc_id matches an allowed course
        if doc_id not in req.allowed_course_ids:
            continue

        if len(context_chunks) < 5:
            context_chunks.append(item.get("text", ""))
            
            # Citation logic
            url = item.get("url")
            if url not in seen_urls:
                citations.append({
                    "type": "lesson", # Default; refine if PDF/Video metadata exists
                    "title": item.get("title"),
                    "url": url,
                    "locator": item.get("chunk_id")
                })
                seen_urls.add(url)

    if not context_chunks:
        return {
            "answer": "Désolé, je n'ai pas trouvé d'information pertinente dans les cours auxquels vous êtes inscrit.",
            "citations": []
        }

    # 3. LLM Generation (Placeholder)
    # TODO: Integrate OpenAI or local LLM here
    # prompt = f"Context: {' '.join(context_chunks)}\nQuestion: {req.question}"
    answer = f"Ceci est une réponse simulée basée sur le contenu de vos cours. (Question : {req.question})"

    return {
        "answer": answer,
        "citations": citations
    }

if __name__ == "__main__":
    load_resources() # Pre-load
    uvicorn.run(app, host="0.0.0.0", port=5000)
