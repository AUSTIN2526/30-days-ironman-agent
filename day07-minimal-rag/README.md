# Day 7:最小可運行的本地 RAG Pipeline

完全跑在本地,不需要任何 API Key。

## 安裝

```bash
pip install -r requirements.txt
```

## 執行順序

```bash
# 1. 建立向量索引(離線階段,只需跑一次;知識庫更新時重跑)
python indexing.py

# 2. 測試檢索(可選,確認檢索結果對不對)
python retrieval.py

# 3. 端到端問答(檢索 + 本地模型生成)
python generation.py
```

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `corpus.py` | 測試用的知識庫文件 |
| `indexing.py` | 切割文件、轉向量、建立 FAISS 索引 |
| `retrieval.py` | 輸入查詢,回傳最相關的 Chunk |
| `generation.py` | 組合 Prompt,呼叫本地 LLM 生成回答 |

## 備註

- 第一次執行 `generation.py` 會自動從 Hugging Face 下載 `Qwen/Qwen2.5-1.5B-Instruct` 模型權重,之後會使用本地快取
- 沒有 GPU 也能跑,會自動退回 CPU 推論;如果速度太慢,可以把 `generation.py` 裡的 `MODEL_NAME` 換成更小的 `Qwen/Qwen2.5-0.5B-Instruct`
- `indexing.py` 裡的 `naive_chunk` 是刻意先求能動的簡化版本,Day 8 會換成更完整的 Chunking 策略
