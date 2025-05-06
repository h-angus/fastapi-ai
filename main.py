from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import chromadb
import requests
from sentence_transformers import SentenceTransformer

app = FastAPI()

# === Config ===
import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "chat-mistral"

# === Init embedding model ===
embedder = SentenceTransformer(EMBEDDING_MODEL)

# === Init Chroma ===
import chromadb.config

chroma_client_settings = chromadb.config.Settings(
    chroma_api_impl="rest",
    chroma_server_host="chromadb.chromadb.svc.cluster.local",
    chroma_server_http_port=8000,
)

client = chromadb.Client(chroma_client_settings)

collection = client.get_or_create_collection(CHROMA_COLLECTION)

# === System Prompt ===
SYSTEM_PROMPT = "You are a helpful assistant."

# === API: Return available models (for Open WebUI) ===
@app.get("/api/tags")
async def get_models():
    return {"models": ["chat-mistral", "ha-mistral"]}

# === API: Main Chat Endpoint (Open WebUI-compatible) ===
@app.post("/api/generate")
async def generate(request: Request):
    data = await request.json()
    model = data.get("model", "chat-mistral")
    prompt = data.get("prompt", "")
    user_id = "webui"  # You can expand this later to support multiple users

    # Embed the current message
    embedding = embedder.encode(prompt).tolist()

    # Retrieve memory snippets from Chroma
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3,
        where={"user_id": user_id}
    )
    memory_snippets = results.get("documents", [[]])[0]
    memory_block = "\n".join(f"- {m}" for m in memory_snippets)

    # Build final prompt
    full_prompt = f"""
{SYSTEM_PROMPT}

Relevant past entries:
{memory_block}

Current message:
{prompt}
"""

    # Call Ollama to get a response
    ollama_response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": model, "prompt": full_prompt, "stream": False}
    )
    response_text = ollama_response.json().get("response")

    # Save message to Chroma
    collection.add(
        documents=[prompt],
        embeddings=[embedding],
        metadatas=[{"user_id": user_id}],
        ids=[f"{user_id}_{hash(prompt)}"]
    )

    return {"response": response_text}
