from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import agentic_rag  # 直接使用你已經寫好的 agentic_rag 邏輯
from fastapi.responses import StreamingResponse # 確保導入這個
from query import hybrid_search
from generate import generate_stream


# 1. 定義 FastAPI 應用
app = FastAPI(title="RAG Agentic API")

# 2. 定義 Request 與 Response 的結構 (這讓你的 API 自動生成文件)
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

# 3. 定義 API 路徑
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        # 呼叫你已經寫好的核心邏輯
        answer = agentic_rag(request.question)
        
        return {
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. 這是新的串流 API
@app.post("/ask/stream")
async def ask_stream(request: QueryRequest) -> StreamingResponse:
    try:
        # 第一步：先進行檢索 (這必須同步完成，否則沒參考資料)
        results = hybrid_search(request.question, k=5)
        
        # 第二步：將 generate_stream 生成器交給 StreamingResponse
        # media_type 設定為 text/event-stream 或 text/plain 均可
        return StreamingResponse(
            generate_stream(request.question, results), 
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# 4. 簡單的根目錄測試
@app.get("/")
def read_root():
    return {"message": "RAG API is running. Go to /docs for API documentation."}
