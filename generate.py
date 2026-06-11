import os
from query import embed, retrieveTopK
from models import ChunkResult
from openai import OpenAI
from dotenv import load_dotenv
from typing import Generator


load_dotenv()
base_url = os.getenv("LLM_BASE_URL")
model = os.getenv("LLM_MODEL")
api_key = os.getenv("LLM_API_KEY")

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

def generate(question: str, chunks: list[ChunkResult]) -> str:
    # 1. 萃取並串接所有 chunk 的文字，組成參考文本
    #    這裡我們用換行符號 \n\n 把每個片段隔開，並加上編號讓 LLM 好閱讀
    context_list = []
    for i, chunk in enumerate(chunks, 1):
        # 批判性思考：加入來源檔案名稱（filename），可以幫助 LLM 回答時更精確，或在需要時標註出處
        context_list.append(f"[文件片段 {i} (來源: {chunk.filename})]:\n{chunk.text}")
    
    context = "\n\n".join(context_list)
    
    # 2. 組出問 LLM 的 Prompt 模板
    #    使用 f-string 把 context 和 question 塞進去
    prompt = f"""你是一個專業的 AI 助理。請根據下方提供的「參考文字」，嚴謹且精確地回答使用者的「問題」。
如果參考文字中的資訊不足以回答問題，請直接說「根據目前資料無法回答」，切勿胡言亂語或憑空捏造。

# 參考文字：
{context}

# 問題：
{question}

# 回答："""

    #3. 呼叫LLM API 取得回應
    try:
        response = client.chat.completions.create(
            model=model,  # 修正：直接使用上面讀到的 model 變數，而不是寫死 "gpt-5.5"
            messages=[
                {"role": "system", "content": "你是一個只根據提供資料回答的誠實助理。"},
                {"role": "user", "content": prompt}  # 修正：把 prompt 變數帶進來
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"呼叫 LLM 時發生錯誤: {e}"

def generate_stream(question: str, chunks: list[ChunkResult]) -> Generator[str, None, None]:
    # 1. 萃取並串接所有 chunk 的文字
    context_list = []
    for i, chunk in enumerate(chunks, 1):
        context_list.append(f"[文件片段 {i} (來源: {chunk.filename})]:\n{chunk.text}")
    
    context = "\n\n".join(context_list)
    
    # 2. 組出 Prompt
    prompt = f"""你是一個專業的 AI 助理。請根據下方提供的「參考文字」，嚴謹且精確地回答使用者的「問題」。
如果參考文字中的資訊不足以回答問題，請直接說「根據目前資料無法回答」，切勿胡言亂語或憑空捏造。

# 參考文字：
{context}

# 問題：
{question}

# 回答："""

    # 3. 呼叫 LLM API，開啟 stream=True
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一個只根據提供資料回答的誠實助理。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            stream=True  # 👈 關鍵設定
        )

        # 透過 yield 一個字一個字吐出來
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"呼叫 LLM 時發生錯誤: {e}"

if __name__ == "__main__":
    question = "TPIX是什麼"
    vector = embed(question)
    results = retrieveTopK(vector, k=3)
    llm_answer = generate(question, results)
    print(llm_answer)
