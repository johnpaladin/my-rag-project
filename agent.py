from generate import generate
import os, json, datetime
from query import embed, retrieveTopK, hybrid_search
from models import ChunkResult
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
base_url = os.getenv("LLM_BASE_URL")
model = os.getenv("LLM_MODEL")
api_key = os.getenv("LLM_API_KEY")

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


def decompose_query(question: str) -> list[str]:
    """
    使用 LLM 將複雜問題拆解成多個簡單的子問題。
    """
    prompt = f"""你是一個專業的問題拆解助理。請將使用者的「原始問題」拆解成 2-3 個簡單、單一的問題。
目標是讓這些子問題更容易從企業內部文件中找到答案。

請嚴格遵循以下 JSON 格式回傳，不要包含任何開頭或結尾的敘述：
["子問題1", "子問題2", "子問題3"]

# 原始問題：
{question}

# 拆解後的問題："""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你只會輸出 JSON 格式的列表。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # 拆解問題不需要創造力，越穩定越好
        )
        
        # 解析回傳的 JSON 字串
        content = response.choices[0].message.content.strip()
        # 處理可能的 markdown code block 格式
        content = content.replace("```json", "").replace("```", "")
        return json.loads(content)

    except Exception as e:
        print(f"問題拆解失敗，直接回傳原始問題: {e}")
        return [question]


def agentic_rag(question: str) -> str:
    print(f"--- 開始處理複雜問題: {question} ---")
    
    # 1. 拆解問題
    sub_queries = decompose_query(question)
    print(f"拆解出子問題: {sub_queries}")
    
    # 2. 針對每個子問題進行混合檢索
    all_chunks = []
    seen_texts = set() # 用於去重
    
    for sub_q in sub_queries:
        print(f"正在檢索: {sub_q}")
        chunks = hybrid_search(sub_q, k=5)
        
        for chunk in chunks:
            if chunk.text not in seen_texts:
                all_chunks.append(chunk)
                seen_texts.add(chunk.text)
    
    print(f"共收集到 {len(all_chunks)} 個不重複的參考片段。")
    for i, c in enumerate(all_chunks, 1):
        print(f"  片段 {i}: {c.text[:50]}...")
    
    # 3. 生成最終答案
    final_answer = generate(question, all_chunks)
    
    return final_answer

if __name__ == "__main__":
    # 1. 定義一個複雜問題來測試 Agent 的能力
    question = "在雙軌制計畫中，RAG 試點的 MVP 預期開發週期是多久？而在第幾個月的里程碑需要決定是否擴展或加入 Agent？"
    
    # 2. 直接執行 Agentic RAG 流程
    # 這個函數內部已經包含了 decompose_query、hybrid_search 與 generate
    answer = agentic_rag(question)
    
    # 3. 輸出最終結果
    print("\n" + "="*30)
    print("【最終生成答案】")
    print(answer)
    print("="*30)
