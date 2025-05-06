from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List
import requests
import os
from chromadb import HttpClient

app = FastAPI()

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],  # Or specify your frontend domains
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)


# === Config ===
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_HOST = os.environ.get("EMBEDDING_HOST", "http://embedder-service:8001")
CHROMA_COLLECTION = "chat-mistral"
CHROMA_HOST = "http://chromadb.chromadb.svc.cluster.local:8000"

# === Init Chroma ===
client = HttpClient(host=CHROMA_HOST)
try:
   collection = client.get_or_create_collection(CHROMA_COLLECTION)
except Exception as e:
   raise RuntimeError(f"Failed to connect to ChromaDB: {str(e)}")

SYSTEM_PROMPT = "You are a helpful assistant."

# === Health check ===
@app.get("/")
async def root():
   return {"status": "ok"}

# === Required Ollama endpoints ===
@app.get("/api/tags")
async def get_models():
   return {
      "models": [
         {
            "name": "chat-mistral",
            "model": "chat-mistral",
            "version": "0.1.31",  # 👈 REQUIRED to fix KeyError
            "modelfile": "Modelfile",
            "details": {
               "format": "gguf",
               "family": "mistral",
               "parameter_size": "7B",
               "quantization_level": "Q4_0"
            }
         },
         {
            "name": "ha-mistral",
            "model": "ha-mistral",
            "version": "0.1.31",  # 👈 Add here too
            "modelfile": "Modelfile",
            "details": {
               "format": "gguf",
               "family": "mistral",
               "parameter_size": "7B",
               "quantization_level": "Q4_0"
            }
         }
      ]
   }



@app.get("/api/models")
async def get_all_models():
   return {
      "models": [
         {
            "name": "chat-mistral",
            "model": "chat-mistral",
            "version": "0.1.31",  # ✅ Required
            "modified_at": "2024-01-01T00:00:00.000Z",
            "parameters": {},
            "template": ""
         },
         {
            "name": "ha-mistral",
            "model": "ha-mistral",
            "version": "0.1.31",  # ✅ Required
            "modified_at": "2024-01-01T00:00:00.000Z",
            "parameters": {},
            "template": ""
         }
      ]
   }


@app.get("/api/show")
async def show_model():
   return {}

@app.post("/api/pull")
async def pull_model():
   return {"status": "success"}

@app.post("/api/delete")
async def delete_model():
   return {"status": "success"}

@app.get("/ollama/api/version")
async def get_ollama_version():
   # Option 1 — Mock
   return {"version": "0.1.31"}

   # Option 2 — Proxy to real Ollama (if you prefer)
   # try:
   #    r = requests.get(f"{OLLAMA_HOST}/api/version", timeout=5)
   #    r.raise_for_status()
   #    return r.json()
   # except Exception as e:
   #    return {"error": f"Version check failed: {str(e)}"}


# === Chat endpoint ===
class GenerateRequest(BaseModel):
   model: Optional[str] = "chat-mistral"
   prompt: str

def get_embeddings(texts: List[str]) -> List[List[float]]:
   try:
      response = requests.post(f"{EMBEDDING_HOST}/embed", json={"texts": texts}, timeout=10)
      response.raise_for_status()
      return response.json().get("embeddings", [])
   except Exception as e:
      print(f"Embedding service error: {e}")
      return []

@app.post("/api/generate")
async def generate(data: GenerateRequest):
   user_id = "webui"

   if not data.prompt or not data.prompt.strip():
      return {"error": "Prompt cannot be empty."}

   # Embed prompt
   try:
      embedding_list = get_embeddings([data.prompt])
      if not embedding_list:
         raise ValueError("Empty embedding returned.")
      embedding = embedding_list[0]
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
      pass

   return {
      "response": response_text,
      "done": True
   }
