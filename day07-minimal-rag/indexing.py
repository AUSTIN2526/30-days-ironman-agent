import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from corpus import documents

# 選用支援中文的 Embedding 模型
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")


def naive_chunk(text: str, max_len: int = 100) -> list[str]:
    """
    最簡單的固定長度切割,先求能動。
    真正的 Chunking 策略(語意切割、Overlap 設計)Day 8 會重寫這個函式。
    """
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]


def main():
    # 1. 把所有文件切成 chunk
    chunks = []
    for doc in documents:
        chunks.extend(naive_chunk(doc))

    # 2. 把每個 chunk 轉成向量
    embeddings = model.encode(chunks, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # 3. 建立 FAISS 索引(用內積模擬 cosine similarity,因為向量已經正規化)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 存起來給檢索階段用
    faiss.write_index(index, "knowledge.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"索引完成,共 {len(chunks)} 個 chunk")


if __name__ == "__main__":
    main()
