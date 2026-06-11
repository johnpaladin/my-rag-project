from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import agentic_rag  # 直接使用你已經寫好的 agentic_rag 邏輯

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

# 4. 簡單的根目錄測試
@app.get("/")
def read_root():
    return {"message": "RAG API is running. Go to /docs for API documentation."}
