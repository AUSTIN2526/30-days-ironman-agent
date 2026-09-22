# compare_chunks.py
from chunkers import fixed_chunk, markdown_chunk, overlap_chunk, recursive_chunk

with open("handbook.md", encoding="utf-8") as f:
    handbook = f.read()

strategies = {
    "fixed": lambda t: fixed_chunk(t, 100),
    "overlap": lambda t: overlap_chunk(t, 100, 20),
    "recursive": lambda t: recursive_chunk(t, 100),
    "markdown": lambda t: markdown_chunk(t, 100),
}

if __name__ == "__main__":
    for name, chunker in strategies.items():
        chunks = chunker(handbook)
        avg_len = sum(len(c) for c in chunks) / len(chunks)
        print(f"===== {name}：共 {len(chunks)} 塊，平均 {avg_len:.0f} 字 =====")
        for i, c in enumerate(chunks):
            print(f"[{i}] {c!r}")
        print()
