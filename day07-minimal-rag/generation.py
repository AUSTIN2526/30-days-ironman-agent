import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieval import retrieve

# 選用一個中文能力不錯、體積夠小、本機也能跑的指令模型
# 沒有 GPU 的話可以換成 Qwen/Qwen2.5-0.5B-Instruct,推論會更輕量
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("正在載入本地模型,第一次執行會下載權重,請稍候...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    model = model.to(device)


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n".join(f"- {c['content']}" for c in retrieved_chunks)
    user_prompt = f"""請根據以下參考資料回答問題,如果參考資料中沒有相關資訊,請明確說明查不到,不要自行猜測。

參考資料:
{context}

問題:{query}"""
    return user_prompt


def call_local_llm(user_prompt: str, max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": "你是一個嚴謹的助理，只根據提供的參考資料回答問題，不會自行編造內容。"},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,          # 明確要求回傳 dict，新舊版行為一致
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,              # 同時傳入 input_ids 和 attention_mask
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()


def answer(query: str) -> str:
    retrieved = retrieve(query, top_k=3)
    prompt = build_prompt(query, retrieved)
    response = call_local_llm(prompt)
    return response


if __name__ == "__main__":
    query = "我年資三年可以請幾天特休?"
    print(f"問題:{query}\n")
    print(f"回答:{answer(query)}")
