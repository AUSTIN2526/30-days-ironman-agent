# evaluate.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chunkers import semantic_chunk
from compare_chunks import handbook, strategies

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 語意切割需要 Embedding 模型，所以另外加進來
strategies["semantic"] = lambda t: semantic_chunk(t, model, threshold=0.6)

# 測試題：問題 + 答案裡「一定要出現」的關鍵片段
test_cases = [
    ("我年資三年可以請幾天特休？", "滿三年可請 14 天"),
    ("特休沒休完會怎樣？", "遞延到隔年"),
    ("請病假超過幾天要附證明？", "超過 3 天，需要附上醫院開立的診斷證明"),
    ("報帳金額超過多少要給經理簽？", "超過 5,000 元則需要再經過部門經理簽核"),
    ("報帳之後多久會撥款？", "5 個工作天內撥款"),
    ("遠端工作的時候幾點要能被聯繫到？", "上午 10 點到下午 4 點"),
    ("離職最後一天要還什麼東西？", "歸還筆電與門禁卡"),
]


def build_index(chunks: list[str]) -> faiss.Index:
    vecs = model.encode(chunks, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def evaluate(chunks: list[str], top_k: int = 3) -> tuple[float, float]:
    index = build_index(chunks)
    queries = [q for q, _ in test_cases]
    q_vecs = model.encode(queries, normalize_embeddings=True).astype("float32")
    _, indices = index.search(q_vecs, top_k)   # 一次查全部問題，形狀是 (題數, top_k)

    hit1 = hitk = 0
    for (_, answer), idxs in zip(test_cases, indices):
        retrieved = [chunks[i] for i in idxs if i != -1]
        hit1 += answer in retrieved[0]
        hitk += any(answer in c for c in retrieved)
    n = len(test_cases)
    return hit1 / n, hitk / n


if __name__ == "__main__":
    print(f"{'策略':<10}{'塊數':>6}{'平均長度':>8}{'Hit@1':>8}{'Hit@3':>8}")
    for name, chunker in strategies.items():
        chunks = chunker(handbook)
        avg_len = np.mean([len(c) for c in chunks])
        hit1, hit3 = evaluate(chunks)
        print(f"{name:<10}{len(chunks):>6}{avg_len:>8.0f}{hit1:>8.0%}{hit3:>8.0%}")
