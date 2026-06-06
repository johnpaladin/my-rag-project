import os, chromadb
from dotenv import load_dotenv
from openai import OpenAI
from models import ChunkResult

load_dotenv()
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

chroma=chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("rag_docs")

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

if __name__ == "__main__":
    question = "TPIX是什麼"
    vector = embed(question)
    results = retrieveTopK(vector, k=3)
    for r in results:
        print(r)
