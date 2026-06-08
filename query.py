import os, chromadb, jieba
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from models import ChunkResult

load_dotenv()
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)


chroma=chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("rag_docs")
all_data = collection.get()

# 把所有的 text 和 metadata 對應起來
text_to_meta = {doc: meta for doc, meta in zip(all_data["documents"], all_data["metadatas"])}

# 初始化 BM25
token_lists = [jieba.lcut(doc) for doc in all_data["documents"] if doc]
bm25 = BM25Okapi(token_lists)

def embed(text: str) -> list[float]:
    #用embedding model把問題轉向量
    response = client.embeddings.create(
        model=os.getenv("LLM_EMBED_MODEL"),
        input=[text]
    )
    return response.data[0].embedding

def retrieveTopK(query_vector: list[float], k: int) -> list[ChunkResult]:
    #用chorma去抓出最相關的chunk top k
    results = collection.query(
        query_embeddings=[query_vector],
        n_results = k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]      # 注意：原始欄位通常是 "metadatas"，請依你實際欄位名為主
    dists = results["distances"][0]

    chunk_results = []

    # 2. 用 zip 同步迭代，並依序填入 ChunkResult
    for text, metadata, score in zip(docs, metas, dists):
        # 這裡的 filename 假設存在 metadata 的 'filename' 欄位中，請根據你實際的 key 調整
        filename = metadata.get("filename", "unknown") 
    
        # 建立物件並加入 list
        chunk = ChunkResult(text=text, filename=filename, score=score)
        chunk_results.append(chunk)

    return chunk_results


def bm25_search(query: str, k: int = 5):
    query_tokens = jieba.lcut(query)
    doc_scores = bm25.get_scores(query_tokens)
    
    # 結合分數與原始文件
    results = list(zip(doc_scores, all_data["documents"]))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]


def reciprocal_rank_fusion(bm25_results: list, vector_results: list[ChunkResult], k_constant: int = 60) -> list:
    fused_scores = {}

    # 處理 BM25 (排名 i)
    for rank, (score, text) in enumerate(bm25_results):
        fused_scores[text] = fused_scores.get(text, 0) + 1 / (k_constant + rank)

    # 處理 Vector (排名 i)
    for rank, chunk in enumerate(vector_results):
        fused_scores[chunk.text] = fused_scores.get(chunk.text, 0) + 1 / (k_constant + rank)

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)



def hybrid_search(query: str, k: int = 3) -> list[ChunkResult]:
    # 1. 向量搜尋
    vector = embed(query)
    v_res = retrieveTopK(vector, k=k)
    
    # 2. BM25 搜尋
    b_res = bm25_search(query, k=k)
    
    # 3. RRF 合併
    fused = reciprocal_rank_fusion(b_res, v_res, k_constant=60)
    
    # 4. 轉回 ChunkResult (保留最高的分數作為 Hybrid Score)
    final_chunks = []
    for text, score in fused[:k]:
        # 從對照表取出 metadata，再從 metadata 取出 filename
        meta = text_to_meta.get(text, {})
        filename = (meta or {}).get("filename", "unknown")
        
        final_chunks.append(ChunkResult(text=text, filename=filename, score=score))
        
    return final_chunks


if __name__ == "__main__":
    test_q = "雙軌制預算"
    results = hybrid_search(test_q, k=3)
    for r in results:
        print(f"分數: {r.score:.4f} | 內容: {r.text[:30]}...")
