from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List
import requests
import os
from sentence_transformers import SentenceTransformer
from chromadb import HttpClient

app = FastAPI()

# === Config ===
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "chat-mistral"
CHROMA_HOST = "http://chromadb:8000"

# === Init embedding model and Chroma ===
embedder = SentenceTransformer(EMBEDDING_MODEL)
client = HttpClient(host=CHROMA_HOST)
collection = client.get_or_create_collection(CHROMA_COLLECTION)

SYSTEM_PROMPT = "You are a helpful assistant."

# === Models list (Open WebUI expects just a list) ===
@app.get("/api/tags")
async def get_models():
    return [
        {"model": "chat-mistral", "modelfile": "", "details": {}},
        {"model": "ha-mistral", "modelfile": "", "details": {}}
    ]

# === Chat endpoint ===
class GenerateRequest(BaseModel):
    model: Optional[str] = "chat-mistral"
    prompt: str

@app.post("/api/generate")
async def generate(data: GenerateRequest):
    user_id = "webui"

    # Embed prompt
    try:
        embedding = embedder.encode(data.prompt).tolist()
    except Exception as e:
        return {"error": f"Embedding failed: {str(e)}"}

    # Query memory
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3,
            where={"user_id": user_id}
        )
        memory_snippets = results.get("documents", [[]])[0]
    except Exception:
        memory_snippets = []

    memory_block = "\n".join(f"- {m}" for m in memory_snippets)

    full_prompt = f"""
{SYSTEM_PROMPT}

Relevant past entries:
{memory_block}

Current message:
{data.prompt}
"""

    # Call Ollama (or mimic)
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": data.model, "prompt": full_prompt, "stream": False}
        )
        response.raise_for_status()
        response_text = response.json().get("response", "No response returned.")
    except Exception as e:
        return {"error": f"Ollama call failed: {str(e)}"}

    # Store interaction
    try:
        collection.add(
            documents=[data.prompt],
            embeddings=[embedding],
            metadatas=[{"user_id": user_id}],
            ids=[f"{user_id}_{hash(data.prompt)}"]
        )
    except Exception:
        pass  # Do not fail on storage errors

    return {
        "response": response_text,
        "done": True
    }
