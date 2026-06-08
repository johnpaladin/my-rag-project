import os, chromadb, jieba
from query import embed, retrieveTopK
from models import ChunkResult
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))

chroma = chromadb.PersistentClient(path="./chroma_db")
# ?? 提示：為了避免重複執行時報錯，改用 get_or_create_collection
collection = chroma.get_collection(name="rag_docs")
all_data = collection.get()

token_lists = [jieba.lcut(doc) for doc in all_data["documents"] if doc]

bm25 = BM25Okapi(token_lists)

def bm25_search(query: str, k: int = 5):
    """使用 BM25 演算法搜尋最相關的文本

    Args:
        query (str): 搜尋關鍵字或句子
        k (int): 回傳的前 k 筆最高分結果

    Returns:
        list: 包含 (得分, 原始文件) 的元組列表
    """
    # 1. 將輸入的 query 同樣使用 jieba 切詞
    query_tokens = jieba.lcut(query)

    # 2. 計算 query 與所有文件的 BM25 分數
    doc_scores = bm25.get_scores(query_tokens)

    # 3. 結合分數與原始文件，並依分數由高到低排序
    results = list(zip(doc_scores, all_data["documents"]))
    results.sort(key=lambda x: x[0], reverse=True)

    # 4. 回傳前 k 個結果
    return results[:k]

def reciprocal_rank_fusion(bm25_results: list, vector_results: list, k_constant: int = 60) -> list:
    """
    RRF 演算法：將兩個不同來源的排序結果合併。
    bm25_results: [(score, text), ...]
    vector_results: [ChunkResult, ...]
    """
    fused_scores = {}

    # 1. 處理 BM25 結果 (排名索引 i 從 0 開始)
    for rank, (score, text) in enumerate(bm25_results):
        # 這裡我們用 text 作為 key，如果文本重複出現，分數會累加
        fused_scores[text] = fused_scores.get(text, 0) + 1 / (k_constant + rank)

    # 2. 處理 Vector 結果 (排名索引 i 從 0 開始)
    for rank, chunk in enumerate(vector_results):
        # 使用 chunk.text 作為 key
        fused_scores[chunk.text] = fused_scores.get(chunk.text, 0) + 1 / (k_constant + rank)

    # 3. 將合併後的結果重新排序
    # fused_scores.items() -> [(text, score), ...]
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    return reranked_results


# --- 測試搜尋 ---
question = "在雙軌制計畫中，RAG 試點的 MVP 預期開發週期是多久？"
vector = embed(question)
v_results = retrieveTopK(vector, k=5)
b_results = bm25_search(question, k=5)
fused = reciprocal_rank_fusion(b_results, v_results)

for rank, (text, score) in enumerate(fused[:3], 1):
    print(f"排名 {rank} | RRF分數: {score:.4f}")
    print(f"內容: {text[:100]}...")
    print("-"*30)
