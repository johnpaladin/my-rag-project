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

def run_agent(question: str) -> str:
    # 1. 定義工具清單
    tools = [
        {
            "type": "function",
            "function": {
                "name": "hybrid_search",
                "description": "當需要從企業內部文件中查詢特定知識或資訊時呼叫此函數",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "使用者的查詢語句"},
                        "k": {"type": "integer", "description": "檢索的片段數量"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # 2. 第一次呼叫 LLM，詢問是否需要使用工具
    messages = [{"role": "user", "content": question}]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto" # 讓 LLM 自己決定
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 3. 判斷是否需要呼叫工具
    if tool_calls:
        print(f"Agent 決定呼叫工具: {tool_calls[0].function.name}")
        
        # 這裡執行工具 (這裡直接呼叫 query.py 的函數)
        function_args = json.loads(tool_calls[0].function.arguments)
        search_results = hybrid_search(query=function_args.get("query"), k=5)
        
        # 將工具執行結果餵回給 LLM
        context = "\n".join([chunk.text for chunk in search_results])
        messages.append(response_message)
        messages.append({
            "tool_call_id": tool_calls[0].id,
            "role": "tool",
            "name": "hybrid_search",
            "content": context
        })
        
        # 4. 最後一次呼叫 LLM 產生最終答案
        final_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return final_response.choices[0].message.content
    
    else:
        # 不需要搜尋，直接回答 (例如問「你好嗎？」)
        return response_message.content


if __name__ == "__main__":
    answer = run_agent("在雙軌制計畫中，RAG 試點的 MVP 預期開發週期是多久？")
    print(answer)
