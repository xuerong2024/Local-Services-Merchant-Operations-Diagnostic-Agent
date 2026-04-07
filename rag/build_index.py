import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge_base"
INDEX_PATH = BASE_DIR / "rag" / "kb_index.faiss"
CHUNKS_PATH = BASE_DIR / "rag" / "kb_chunks.json"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def split_text(text: str, chunk_size: int = 200, overlap: int = 50):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def load_documents():
    docs = []
    for path in KB_DIR.glob("*"):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append({
                "doc_name": path.name,
                "chunk_id": i,
                "text": chunk
            })
    return docs


def main():
    docs = load_documents()
    if not docs:
        raise ValueError("knowledge_base 目录下没有可用文档")

    texts = [d["text"] for d in docs]
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"索引已生成: {INDEX_PATH}")
    print(f"chunk元数据已生成: {CHUNKS_PATH}")
    print(f"共写入 {len(docs)} 个 chunks")


if __name__ == "__main__":
    main()