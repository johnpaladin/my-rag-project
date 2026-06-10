## 專案描述
旨在透過嚴謹的「內部知識 RAG」架構，將專屬知識文件進行高效切片與向量化，
並結合 LLM 實現精準、誠實且拒絕幻覺的企業級問答系統，同時建立自動化的測試評估流程。



## 專案架構與檔案說明

本專案將 RAG 流程完全解耦，由以下五隻核心 Python 檔案協同運作：

1. **`models.py`**：定義資料結構。包含核心的 `ChunkResult` 資料類別（Data Class），用來規範從向量資料庫檢索出來的文字片段、來源檔名（`filename`）與相似度分數（`score`），確保後續模組的資料型態一致性。
2. **`ingest.py`**：資料清理與向量化。負責讀取 `docs/` 資料夾中的企業知識文件（如 PDF 或 Word），進行文本切片（Chunking），並將生成的 Embedding 向量與 metadata 儲存至 Chroma DB 向量資料庫。
3. **`query.py`**：向量檢索模組。負責處理使用者的原生問題，計算其向量後至 Chroma DB 中進行相似度檢索（Similarity Search），利用 `zip()` 函數將 Column-major 的資料結構對齊，並回傳 `list[ChunkResult]`。
4. **`generate.py`**：Prompt 工程與 LLM 生成。負責將檢索出的 `ChunkResult` 陣列格式化串接為結構化的多行參考文本，利用 Python 的 `f"""` 多行格式化字串嵌入 Prompt 模板，最後呼叫 LLM API（降低 Temperature）產出精確且合規的解答。
5. **`eval.py`**：自動化評估測試集。將精心萃取的專案關鍵測試題目與標準答案（Ground Truth）封裝為 `eval_dataset` 清單，作為系統上線前的回歸測試與準確度評估基準。

## 如何運行專案

請確保您的開發環境已安裝相關依賴（如 `openai`, `chromadb` 等），並在根目錄建立 `.env` 檔案設定您的 API Key：

```bash
# 1. 配置環境變數
.env

LLM_BASE_URL=http://xxx.xxx.xxx.xxx:4000/v1
LLM_MODEL=Gemma4-26B-A4B-IT
LLM_API_KEY=sk-xxxxxxxxxx
LLM_EMBED_MODEL="xxxx"

# 2. 將文件放入 docs/ 資料夾，並執行資料寫入（向量化）
python ingest.py

# 3. 執行自動化評估測試，驗證 RAG 系統回答品質
python eval.py


## Eval 結果（基準分數）

| 題目 | 問題摘要 | LLM-as-judge 分數 |
|------|----------|-------------------|
| 1 | RAG MVP 開發週期與決策時程 | 0.0 |
| 2 | 廠商管理三條紅線 | 0.3 |
| 3 | 預算結構與驗收標準 | 1.0 |
| **平均** | | **0.43** |

> 題目 1 失敗原因：時程資訊分散在不同 chunk，k=3 未能撈到相關片段（retrieval failure，非 LLM failure）。


## Eval 結果對比

| 方法 | k值 | 平均分 |
|------|-----|--------|
| 純向量（baseline） | 3 | 0.43 |
| Hybrid Search (BM25 + RRF) | 10 | 1.0 |
| Agentic RAG (Query Decomposition) | 5×2 | 1.0 |

> Hybrid 靠拉大 k 覆蓋更多 chunks；Agentic 靠拆問題讓每個子查詢更精準。
