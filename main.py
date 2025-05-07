from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel
from typing import Optional, List
import requests
import os
from hashlib import sha256
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

@app.get("/inspect/chroma")
async def inspect_chroma():
   try:
      collections = client.list_collections()
      return {"collections": [c.name for c in collections]}
   except Exception as e:
      return {"error": str(e)}

@app.get("/inspect/chroma/docs")
async def inspect_chroma_docs():
   try:
      docs = collection.get()
      return {
         "ids": docs.get("ids", []),
         "documents": docs.get("documents", []),
         "metadatas": docs.get("metadatas", [])
      }
   except Exception as e:
      return {"error": str(e)}

@app.get("/api/clear_memory")
async def clear_user_memory_get(user_id: str = Query("webui")):
   try:
      deleted_ids = collection.delete(where={"user_id": user_id})
      print(f"🗑️ Cleared memory for user_id '{user_id}': {deleted_ids}")
      return {"status": "success", "deleted_ids": deleted_ids}
   except Exception as e:
      print(f"❌ Failed to clear memory: {e}")
      return {"status": "error", "detail": str(e)}


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
   body = await request.json()
   user_id = body.get("user_id", "webui")  # ← now dynamic
   model = body.get("model", "chat-mistral:latest")
   messages = body.get("messages", [])

   print(f"💬 Model: {model}")
   print(f"📨 Incoming messages: {[m['content'] for m in messages if m['role'] == 'user']}")

   if not messages or not isinstance(messages, list):
      return {"error": "Missing or invalid 'messages' field."}

   # Get latest user message
   latest_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)
   if not latest_user_msg:
      return {"error": "No user message found."}

   # Check if it's a metadata prompt (Open WebUI special prompt)
   is_metadata_prompt = (
      latest_user_msg["content"].startswith("### Task:") or
      "<chat_history>" in latest_user_msg["content"]
   )

   # === Memory embedding & retrieval only for real chats ===
   if not is_metadata_prompt:
      try:
         embedding_list = get_embeddings([latest_user_msg["content"]])
         print(f"🧠 Embedding returned: {embedding_list}")
         if not embedding_list:
            raise ValueError("Empty embedding returned.")
         embedding = embedding_list[0]
      except Exception as e:
         return {"error": f"Embedding failed: {str(e)}"}

      # Query ChromaDB with similarity + distance filtering
      try:
         results = collection.query(
            query_embeddings=[embedding],
            n_results=10,  # you can raise this to get more to filter from
            where={"user_id": user_id},
            include=["documents", "distances"]
         )
         print(f"🧾 Raw Chroma results: {results}")
         memory_snippets = []
      
         for doc, dist in zip(results.get("documents", [[]])[0], results.get("distances", [[]])[0]):
            print(f"📏 Distance: {dist:.4f} — Doc: {doc[:60]}...")
            if dist < 0.5:  # Only include relevant results
               memory_snippets.append(doc)
      
      except Exception as e:
         print(f"❌ Chroma query failed: {e}")
         memory_snippets = []
      
      # Inject memory
      if memory_snippets:
         memory_block = "\n".join(f"- {m}" for m in memory_snippets)
         memory_msg = {
            "role": "system",
            "content": f"Relevant past entries:\n{memory_block}"
         }
         sys_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), -1)
         insert_idx = sys_idx + 1 if sys_idx != -1 else 0
         messages.insert(insert_idx, memory_msg)

   
   # === Store only if not metadata ===
   if not is_metadata_prompt:
      try:
         msg_hash = sha256(latest_user_msg["content"].encode()).hexdigest()
         doc_id = f"{user_id}_{msg_hash}"
         print(f"🔁 Adding to Chroma: {doc_id}")
         print(f"🧠 Content: {latest_user_msg['content']}")
         print(f"📏 Embedding length: {len(embedding)}")

         collection.upsert(
            documents=[latest_user_msg["content"]],
            embeddings=[embedding],
            metadatas=[{"user_id": user_id}],
            ids=[doc_id]
         )
      except Exception as e:
         print(f"❌ Failed to store memory: {e}")
   else:
      print("⚠️ Skipping Chroma (metadata prompt)")

   # === Send to Ollama ===
   try:
      response = requests.post(
         f"{OLLAMA_HOST}/api/chat",
         json={"model": model, "messages": messages, "stream": False}
      )
      response.raise_for_status()
      ollama_reply = response.json()
   except Exception as e:
      return {"error": f"Ollama call failed: {str(e)}"}

   return ollama_reply

class HARequest(BaseModel):
   prompt: str
   user_id: Optional[str] = "home-assistant"



# HA CODE:

@app.post("/api/generate")
async def ha_generate(req: HARequest):
   prompt = req.prompt
   user_id = req.user_id

   # === Embed prompt
   embedding_list = get_embeddings([prompt])
   if not embedding_list:
      return {"response": "Embedding failed."}
   embedding = embedding_list[0]

   # === Retrieve memory
   try:
      results = collection.query(
         query_embeddings=[embedding],
         n_results=5,
         where={"user_id": user_id},
         include=["documents", "distances"]
      )
      memory_snippets = [
         doc for doc, dist in zip(results.get("documents", [[]])[0], results.get("distances", [[]])[0])
         if dist < 0.5
      ]
   except Exception as e:
      memory_snippets = []
      print(f"Chroma query failed: {e}")

   memory_block = "\n".join(memory_snippets)
   full_prompt = f"{SYSTEM_PROMPT}\n\n"
   if memory_block:
      full_prompt += f"Relevant memory:\n{memory_block}\n\n"
   full_prompt += f"User: {prompt}"

   # === Generate reply
   try:
      response = requests.post(f"{OLLAMA_HOST}/api/generate", json={"prompt": full_prompt})
      response.raise_for_status()
      ollama_reply = response.json().get("response", "No response.")
   except Exception as e:
      return {"response": f"Ollama error: {str(e)}"}

   # === Store memory
   try:
      msg_hash = sha256(prompt.encode()).hexdigest()
      doc_id = f"{user_id}_{msg_hash}"
      collection.upsert(
         documents=[prompt],
         embeddings=[embedding],
         metadatas=[{"user_id": user_id}],
         ids=[doc_id]
      )
   except Exception as e:
      print(f"Failed to store memory: {e}")

   return {"response": ollama_reply}

