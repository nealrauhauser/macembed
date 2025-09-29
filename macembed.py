#!/usr/bin/env python
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from contextlib import asynccontextmanager
import os
import torch
import numpy as np
import logging
import sys
import traceback
from sentence_transformers import SentenceTransformer, CrossEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('macembed.log')
    ]
)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("EMBED_SERVER_API_KEY", "changeme")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Initialize device with logging
try:
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")
except Exception as e:
    logger.error(f"Error detecting device: {e}")
    DEVICE = "cpu"
    logger.info("Falling back to CPU")

# Global variables for models
model = None
rerank_model = None

def load_models():
    """Load models with comprehensive error handling"""
    global model, rerank_model

    try:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

    try:
        logger.info(f"Loading rerank model: {RERANK_MODEL_NAME}")
        rerank_model = CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)
        logger.info("Rerank model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load rerank model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting macembed server...")
    load_models()
    logger.info("FastAPI server started successfully")
    logger.info(f"Models loaded: embedding={model is not None}, rerank={rerank_model is not None}")
    logger.info(f"Using device: {DEVICE}")
    yield
    # Shutdown
    logger.info("FastAPI server shutting down")

app = FastAPI(lifespan=lifespan)

# Pydantic models
class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    encoding_format: Optional[str] = None

class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str
    documents: List[str]
    top_k: Optional[int] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False

@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_model": MODEL_NAME,
        "rerank_model": RERANK_MODEL_NAME,
        "device": DEVICE
    }

@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest, authorization: Optional[str] = Header(None)):
    try:
        logger.info(f"Embeddings request received for {len(req.input) if isinstance(req.input, list) else 1} text(s)")

        if API_KEY and (not authorization or not authorization.startswith("Bearer ") or authorization.split(" ",1)[1] != API_KEY):
            raise HTTPException(401, "Unauthorized")

        if model is None:
            logger.error("Embedding model is not loaded")
            raise HTTPException(500, "Embedding model not available")

        texts = [req.input] if isinstance(req.input, str) else list(req.input)

        with torch.no_grad():
            if DEVICE == "mps":
                with torch.autocast("mps", dtype=torch.float16):
                    vectors = model.encode(texts, convert_to_numpy=False, normalize_embeddings=False)
            else:
                vectors = model.encode(texts, convert_to_numpy=False, normalize_embeddings=False)

        # Force cosine normalization (unit-length vectors)
        if isinstance(vectors, torch.Tensor):
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1).cpu().numpy()
        else:
            vectors = np.array(vectors)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-9, None)

        data = []
        for i, v in enumerate(vectors):
            data.append({"object": "embedding", "index": i, "embedding": v.tolist()})

        logger.info(f"Successfully generated {len(data)} embeddings")
        return {
            "object": "list",
            "data": data,
            "model": req.model or MODEL_NAME,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in embeddings endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/v1/rerank")
def rerank(req: RerankRequest, authorization: Optional[str] = Header(None)):
    try:
        logger.info(f"Rerank request received for query with {len(req.documents)} documents")

        if API_KEY and (not authorization or not authorization.startswith("Bearer ") or authorization.split(" ",1)[1] != API_KEY):
            raise HTTPException(401, "Unauthorized")

        if rerank_model is None:
            logger.error("Rerank model is not loaded")
            raise HTTPException(500, "Rerank model not available")

        if not req.documents:
            return {"results": []}

        # Create query-document pairs
        pairs = [[req.query, doc] for doc in req.documents]

        with torch.no_grad():
            if DEVICE == "mps":
                with torch.autocast("mps", dtype=torch.float16):
                    scores = rerank_model.predict(pairs)
            else:
                scores = rerank_model.predict(pairs)

        # Convert to numpy if tensor
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        # Create results with original indices
        results = []
        for i, (doc, score) in enumerate(zip(req.documents, scores)):
            results.append({
                "index": i,
                "document": doc,
                "relevance_score": float(score)
            })

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply top_k if specified
        if req.top_k is not None:
            results = results[:req.top_k]

        logger.info(f"Successfully reranked {len(req.documents)} documents, returning top {len(results)}")
        return {
            "results": results,
            "model": req.model or RERANK_MODEL_NAME
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in rerank endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, authorization: Optional[str] = Header(None)):
    """Minimal chat completions endpoint that returns a simple response"""
    try:
        logger.info(f"Chat completions request received with {len(req.messages)} messages")

        if API_KEY and (not authorization or not authorization.startswith("Bearer ") or authorization.split(" ",1)[1] != API_KEY):
            raise HTTPException(401, "Unauthorized")

        # Extract the last user message or use a default
        user_message = "Hello"
        for msg in reversed(req.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        # Return a minimal OpenAI-compatible response
        return {
            "id": "chatcmpl-placeholder",
            "object": "chat.completion",
            "created": 1677652288,
            "model": req.model or "macembed-chat",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"I'm a minimal chat endpoint. You said: {user_message[:100]}..."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions endpoint: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

# Handle OPTIONS requests for CORS
@app.options("/v1/chat/completions")
def chat_completions_options():
    return {"message": "OK"}

if __name__ == "__main__":
    try:
        import uvicorn
        logger.info("Starting server with uvicorn on 0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        logger.warning("uvicorn not available, run with: uvicorn macembed:app --host 0.0.0.0 --port 8000")
        uvicorn = None
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
