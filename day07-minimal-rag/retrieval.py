import pickle

import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
index = faiss.read_index("knowledge.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = query_vec.astype("float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({"content": chunks[idx], "score": float(score)})
    return results


if __name__ == "__main__":
    query = "我年資三年可以請幾天特休?"
    results = retrieve(query)
    for r in results:
        print(f"[score={r['score']:.3f}] {r['content']}")
