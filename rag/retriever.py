import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "rag" / "kb_index.faiss"
CHUNKS_PATH = BASE_DIR / "rag" / "kb_chunks.json"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

_model = None
_index = None
_chunks = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_index():
    global _index
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
    return _index


def get_chunks():
    global _chunks
    if _chunks is None:
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return _chunks


def retrieve(query: str, top_k: int = 3):
    model = get_model()
    index = get_index()
    chunks = get_chunks()

    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(q_emb, dtype=np.float32), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        item = chunks[idx]
        results.append({
            "doc_name": item["doc_name"],
            "chunk_id": item["chunk_id"],
            "score": float(score),
            "text": item["text"]
        })
    return results