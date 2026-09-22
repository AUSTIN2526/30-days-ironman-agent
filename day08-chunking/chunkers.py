# chunkers.py
import re

import numpy as np


# ---------- 策略 1：固定長度切割（Day 7 的版本） ----------
def fixed_chunk(text: str, chunk_size: int = 100) -> list[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# ---------- 策略 2：固定長度 + Overlap ----------
def overlap_chunk(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    assert 0 <= overlap < chunk_size, "overlap 必須小於 chunk_size"
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= len(text):  # 已經切到結尾就停，避免最後多一塊純重疊的碎片
            break
    return chunks


# ---------- 策略 3：遞迴切割（依照標點與換行） ----------
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，"]


def _split_keep_sep(text: str, sep: str) -> list[str]:
    """用 sep 切開，但把 sep 留在前一段的尾巴，句號才不會不見。"""
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    return [p for p in pieces if p.strip()]


def recursive_chunk(text: str, chunk_size: int = 100, separators: list[str] = SEPARATORS) -> list[str]:
    # 1. 本身就夠小，直接回傳
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # 2. 找出第一個出現在文字裡的分隔符號；都沒有就只能硬切
    sep = next((s for s in separators if s in text), None)
    if sep is None:
        return fixed_chunk(text, chunk_size)
    next_seps = separators[separators.index(sep) + 1:]

    # 3. 切成小片段後，由左到右「貪婪合併」，塞得下就繼續塞
    chunks, buffer = [], ""
    for piece in _split_keep_sep(text, sep):
        if len(piece) > chunk_size:
            # 這一片自己就太大，先把 buffer 收掉，再用更細的分隔符號往下切
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = ""
            chunks.extend(recursive_chunk(piece, chunk_size, next_seps))
        elif len(buffer) + len(piece) <= chunk_size:
            buffer += piece
        else:
            chunks.append(buffer.strip())
            buffer = piece
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


# ---------- 策略 4：依文件結構切割（Markdown 標題） ----------
def markdown_chunk(markdown: str, chunk_size: int = 100) -> list[str]:
    chunks = []
    headers = {}          # 例如 {1: "員工手冊", 2: "請假制度", 3: "特休"}
    body_lines = []

    def flush():
        body = "\n".join(body_lines).strip()
        if not body:
            return
        # 標題路徑只留第 2 層以下，第 1 層是整份文件的名稱，每塊都一樣就沒有鑑別度
        path = " > ".join(headers[k] for k in sorted(headers) if k >= 2)
        prefix = f"【{path}】" if path else ""
        for piece in recursive_chunk(body, chunk_size - len(prefix)):
            chunks.append(prefix + piece)

    for line in markdown.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            flush()
            body_lines = []
            level = len(match.group(1))
            headers = {k: v for k, v in headers.items() if k < level}  # 換章節時，把更深層的標題清掉
            headers[level] = match.group(2).strip()
        else:
            body_lines.append(line)
    flush()
    return chunks


# ---------- 策略 5：語意切割 ----------
def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[。！？\n])", text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(text: str, model, threshold: float = 0.6, chunk_size: int = 200) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    vecs = model.encode(sentences, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype="float32")

    chunks, current = [], sentences[0]
    for i in range(1, len(sentences)):
        similarity = float(vecs[i - 1] @ vecs[i])   # 相鄰兩句的 cosine similarity
        too_long = len(current) + len(sentences[i]) > chunk_size
        if similarity < threshold or too_long:     # 話題轉了、或塞不下了，就切一刀
            chunks.append(current)
            current = sentences[i]
        else:
            current += sentences[i]
    chunks.append(current)
    return chunks
