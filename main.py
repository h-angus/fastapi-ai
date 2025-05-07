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
            "model": "chat-mistral:latest",
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
            "model": "ha-mistral:latest",
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
            "model": "chat-mistral:latest",
            "version": "0.1.31",
            "modified_at": "2024-01-01T00:00:00.000Z",
            "parameters": {},
            "template": ""
         },
         {
            "name": "ha-mistral",
            "model": "ha-mistral:latest",
            "version": "0.1.31",
            "modified_at": "2024-01-01T00:00:00.000Z",
            "parameters": {},
            "template": ""
         }
      ]
   }

@app.get("/api/version")
async def real_ollama_version():
   print("🔥 /api/version was hit")
   return {"version": "0.1.31"}


@app.get("/api/show")
async def show_model():
   return {}

@app.post("/api/pull")
async def pull_model():
   return {"status": "success"}

@app.post("/api/delete")
async def delete_model():
   return {"status": "success"}

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

@app.post("/api/chat")
async def chat_with_memory(request: Request):
   user_id = "webui"
   body = await request.json()
   model = body.get("model", "chat-mistral:latest")
   messages = body.get("messages", [])

   if not messages or not isinstance(messages, list):
      return {"error": "Missing or invalid 'messages' field."}

   # Get latest user message (to embed)
   latest_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)
   if not latest_user_msg:
      return {"error": "No user message found."}

   try:
      embedding_list = get_embeddings([latest_user_msg["content"]])
      if not embedding_list:
         raise ValueError("Empty embedding returned.")
      embedding = embedding_list[0]
   except Exception as e:
      return {"error": f"Embedding failed: {str(e)}"}

   # Query ChromaDB
   try:
      results = collection.query(
         query_embeddings=[embedding],
         n_results=3,
         where={"user_id": user_id}
      )
      memory_snippets = results.get("documents", [[]])[0]
   except Exception:
      memory_snippets = []

   # Format memory as a system message
   if memory_snippets:
      memory_block = "\n".join(f"- {m}" for m in memory_snippets)
      memory_msg = {
         "role": "system",
         "content": f"Relevant past entries:\n{memory_block}"
      }

      # Inject memory after system prompt (if exists), or at the top
      sys_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), -1)
      insert_idx = sys_idx + 1 if sys_idx != -1 else 0
      messages.insert(insert_idx, memory_msg)

   # Proxy to Ollama’s /api/chat
   try:
      response = requests.post(
         f"{OLLAMA_HOST}/api/chat",
         json={"model": model, "messages": messages, "stream": False}
      )
      response.raise_for_status()
      ollama_reply = response.json()
   except Exception as e:
      return {"error": f"Ollama call failed: {str(e)}"}

   # Store long-term memory entry
   try:
      collection.add(
         documents=[latest_user_msg["content"]],
         embeddings=[embedding],
         metadatas=[{"user_id": user_id}],
         ids=[f"{user_id}_{hash(latest_user_msg['content'])}"]
      )
   except Exception:
      pass

   return ollama_reply

