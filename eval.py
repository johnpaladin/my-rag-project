from generate import generate
import os
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


# 定義 RAG 測試資料集
eval_dataset = [
    {
        "question": "在雙軌制計畫中，RAG 試點的 MVP 預期開發週期是多久？而在第幾個月的里程碑需要決定是否擴展或加入 Agent？",
        "ground_truth": "RAG MVP 的 POC 預期週期為 3 個月。在 M6（第六個月）的決策點，需要評估雙軌成效，並決定是否擴展或加入 Agent。"
    },
    {
        "question": "計畫書中提到管理外部 RAG 廠商時，有哪三條不可逾越的紅線條款？",
        "ground_truth": "三條紅線條款分別為：1. 資料可攜權 (Data Portability)：向量資料、metadata、文件切片必須可匯出，合約終止 30 天內完整移轉，且不得收贖回費。2. API 開放規格 (Open API Spec)：RAG 核心檢索功能須以標準 REST/OpenAPI 提供，公司可自行呼叫此 API。3. 模型解耦 (Model Decoupling)：雲端 LLM 由公司自行決定，廠商不得內建「只能用某家 LLM」的硬編碼邏輯。"
    },
    {
        "question": "雙軌制方案與原包山包海的採購案相比，總投入預算預估是多少？且雙軌制的驗收標準是什麼？",
        "ground_truth": "雙軌制方案的總投入預估是原計畫的 10-20%，其驗收標準為「行為指標 + 使用數據」（而非僅僅是功能上線）。"
    }
]

def score_with_llm(question, ground_truth, generated):
    prompt = f"""你是一個裁判會根據客戶提問的question去比對groud_truth
跟llm generated的答案相似度 有沒有回答到  只回傳一個0-1的分數，不要說明文字。

#問題:
{question}

#ground truth:
{ground_truth}

#llm generated answer:
{generated}


"""

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



if __name__ == "__main__":
    print(f"成功載入測試資料集，共包含 {len(eval_dataset)} 個測試題目。\n")
    for i, data in enumerate(eval_dataset, 1):
        print(f"【測試題目 {i}】")
        print(f"問題: {data['question']}")
        print(f"標準答案: {data['ground_truth']}\n")
        question = data['question']
        vector = embed(question)
        results = hybrid_search(question,10)
        llm_answer = generate(question, results)
        print(f"LLM回答: {llm_answer}")
        print(score_with_llm(question,data['ground_truth'],llm_answer))
