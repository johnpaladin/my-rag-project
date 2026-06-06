import os, chromadb
from dotenv import load_dotenv
from openai import OpenAI
from docx import Document                        # 讀 docx
from pdfminer.high_level import extract_text     # 讀 pdf

load_dotenv()
client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))

chroma = chromadb.PersistentClient(path="./chroma_db")
# ?? 提示：為了避免重複執行時報錯，改用 get_or_create_collection
collection = chroma.get_or_create_collection("rag_docs")

def read_file(path: str) -> str:
    # 修正：判斷副檔名(.docx / .pdf),用對的方式回傳純文字字串
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".docx":
        doc = Document(path)
        # 把每個段落的文字用換行符號串接起來
        return "\n".join([p.text for p in doc.paragraphs])
        
    elif ext == ".pdf":
        # pdfminer 直接傳入路徑就能抓出文字
        return extract_text(path)
        
    else:
        print(f"?? 不支援的檔案格式: {ext}")
        return ""

def chunk_text(text: str, size: int = 300) -> list[str]:
    # 修正：把長文字切成片段,每段 <= size 個字
    # 先按換行切成小段落
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        # 如果當前快取加上新段落超過 size，就把舊的存起來，開新的 chunk
        if len(current_chunk) + len(p) > size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = p
        else:
            # 如果不是第一個，就加個空格或換行串接
            current_chunk = f"{current_chunk}\n{p}" if current_chunk else p
            
    # 最後剩下沒滿的碎屑也要丟進去
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def embed(text: str) -> list[float]:
    # 修正：呼叫 client.embeddings.create(), 回傳 vector
    # 注意：這裡的 model 要換成你外部設定支援的 Embedding 模型名稱（例如 text-embedding-3-small 或 text-embedding-ada-002）
    response = client.embeddings.create(
        model=os.getenv("LLM_EMBED_MODEL", "text-embedding-3-small"), 
        input=[text]
    )
    return response.data[0].embedding

# --- 主流程 (這段不用改) ---
# 確保 docs 資料夾存在，才不會報錯
if not os.path.exists("docs"):
    os.makedirs("docs")

for fname in os.listdir("docs"):
    path = os.path.join("docs", fname)
    if os.path.isdir(path): # 略過資料夾
        continue
    text = read_file(path)
    if not text:
        continue
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        vec = embed(chunk)
        collection.add(documents=[chunk], embeddings=[vec], ids=[f"{fname}_{i}"],metadatas=[{"filename": fname}])

print(f"? 存了 {collection.count()} 個 chunk")
