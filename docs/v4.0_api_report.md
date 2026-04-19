# SAILER 專案進度報告：情緒分類器 REST API 建置

**報告對象：** 系統整合團隊 / LangGraph 面試模擬器開發人員
**負責人：** [你的名字/團隊]

---

## 📅 進度摘要 (Executive Summary)

因應 LangGraph 面試模擬器之整合需求，本週我們成功地將 SAILER 情緒識別模型（純語音 Baseline 版本）封裝為獨立的 REST API 微服務。目前整體 API 服務已開發完成、通過所有邊界測試（Edge-case tests），並已完整發布至 GitHub (`feature/official-whisper-fusion` 分支)。下游的 LangGraph 面試官系統現在已經可以透過標準的 HTTP POST 請求，無縫且即時地獲取應徵者的情緒狀態，為面試模擬系統補齊了最核心的「情緒感知」能力。

---

## 🛠️ 開發項目與實作細節

### 1. 核心模型封裝 (Model Wrapping)
為了追求系統的穩定度與推理效率，本次 API 我們選用**效能最佳純語音基線模型 (Speech-Only Baseline) **，採用官方預先訓練的 `WhisperWrapper` 架構：
*   **模型源：** 使用 HuggingFace `tiantiaf/whisper-large-v3-msp-podcast-emotion`，包含 1.54 億參數的巨大聽覺模型。
*   **推理速度：** 針對一段 3 秒的音檔，經測試模型推論耗時僅約 **0.068 秒**（測試於 RTX 3090），具備極佳的 Real-time 應用潛力，不會造成面試對話的延遲。
*   **記憶體管理：** 將龐大的模型權重配置於 FastAPI 的 `Lifespan` 事件中一次性全預載至 GPU，確保後續 API 請求能直接調用，避免重複冷啟動。

### 2. `/classify-emotion` 核心端點實作
完全根據 `api-spec.md` Section 3a 之規範實作了情緒分類的核心 API：
*   **穩健的資料預處理 (Robust Preprocessing)：**
    *   負責接收 LangGraph (或前端 STT 模組) 傳來的 `multipart/form-data` 音檔。
    *   **格式容錯：** 透過 SoundFile 引擎，支援市面上主流的 WAV、MP3、FLAC、OGG 等多種音檔格式解碼。
    *   **規格對齊：** API 收到音檔後會進行三大自動修正：**強制轉單聲道 (mono)**、**重新採樣 (Resample) 至 16kHz**、**長度截斷至 15 秒**。確保所有送進 Inference Engine 的資料皆符合模型訓練時的最佳分佈。
*   **標籤映射系統 (Label Re-mapping)：**
    *   官方模型輸出為 9 個維度的概率 (包含一個實用度不高的 `Other` 類別)。
    *   API 層面實作了軟性分配邏輯 (Soft-redistribution)，將 `Other` 類別的預測概率依比例打散，強制映射回面試系統所要求的 **8 大核心情緒標籤**（Neutral, Angry, Sad, Happy, Fear, Disgust, Surprise, Contempt）。

### 3. API 輸出結構標準化 (JSON Payload)
API 的 JSON 輸出精準符合串接規格，單次請求即可回傳多維度的情感數據：
*   **`primary_label` / `primary_probabilities` (主情緒)：** 8 大分類的強預測與概率分佈，用以決定面試官的主要應對語氣。
*   **`secondary_probabilities` (次情緒)：** 額外提供多達 17 種細緻情感（例如：Other-Concerned, Other-Annoyed）的概率觀察值。
*   **`arousal`, `valence`, `dominance` (AVD 回歸模型)：** 提供情感強烈度、正負向、控制感的三連數值 (值域 0~1)，協助 LLM 調校更細微的對話提示詞 (Prompt tuning)。
*   **`embedding` (聲紋特徵)：** 返回高階的 256 維聲學特徵向量 (Acoustic embeddings)，保留未來跨服務特徵比對的擴展性。

### 4. 工程與安全防護機制 (Engineering & Edge Cases)
*   **Exception Handling (例外處理)：**
    *   嚴格阻擋過短的無效音訊（< 3 秒報錯 `400 Bad Request`）。
    *   對於惡意的假檔案或純文字檔報錯 `422 Unprocessable Entity`。
*   **Health Check：** 加入 `/health` 基礎端點，讓 K8s 或外部 Orchestrator 能隨時存取伺服器的活性與顯卡掛載狀態。
*   **自動化測試庫：** 撰寫了完整的測試腳本 `test_api.py`，支援合成靜音測試、真實音檔測試與錯誤案例驗證。

---

## 🚀 未來展望與 Action Items

1. **部署準備：** 本體程式碼與說明文件（見 `api/README.md`）已就緒，可隨時著手包裝 Docker Image 部署上線。
2. **多模態擴展 (Phase 2 預留)：** 為了簡化第一次串接的複雜度，本次釋出版本為「純語音」，暫不需要輸入文字。然而在程式架構上，未來可隨時升級為原本正在迭代的 **「語音 + 文本 (RoBERTa)」雙模態聯合辨識系統**（方案 B），進一步提升辨識準確率 (Macro-F1)。

*(報告完畢)*
