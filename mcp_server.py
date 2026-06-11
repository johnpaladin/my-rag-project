from mcp.server.fastmcp import FastMCP
from query import hybrid_search  # 引用你原本寫好的檢索核心
from models import ChunkResult

# 1. 初始化 MCP Server
mcp = FastMCP("My-RAG-Server")

# 2. 定義 MCP Tool
@mcp.tool()
def search_internal_knowledge(query: str, k: int = 3) -> str:
    """
    查詢企業內部知識庫。
    使用混合檢索（向量+關鍵字）來獲取最相關的文檔片段。
    """
    results = hybrid_search(query=query, k=k)
    
    # 將搜尋結果轉換成字串，以便 AI 閱讀
    formatted_results = []
    for i, res in enumerate(results, 1):
        formatted_results.append(f"片段 {i} (來源: {res.filename}):\n{res.text}")
    
    return "\n\n".join(formatted_results)

if __name__ == "__main__":
    # 執行 Server
    mcp.run()
