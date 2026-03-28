# SAILER 模型實作專業程式碼審查與重構計畫 (Code Review & Implementation Plan)

分析了您的 `train.py`、`src/sailer_model.py`、`src/msp_dataset.py` 與 `src/experiment_tracker.py` 後，以下是以「最高規格與專業」進行的程式碼審查 (Code Review) 以及與論文一致性 (Paper Alignment) 的對照。

## 📍 總評
您的實作有極大的潛力，特別是在這幾個地方寫得很棒：
1. **目標函數 (KL Divergence)** 使用得非常準確。
2. **Audio Mixing (資料增強)** 的實作邏輯相當符合論文中「逆類別頻率抽樣」和「靜音/重疊」的描述。
3. **Mixed Precision (混合精度訓練)** 及 **torch.compile** 的使用展現了良好的工程優化思維。

然而，您的實作在**「基礎模型選擇」**、**「權重計算邏輯 (Re-weighting)」**及**「工程整潔度 (Clean Code)」**上，與論文的描述及業界實務有些許落差。以下是詳細的問題點及建議。

---

## 🛑 論文規格對齊問題 (Methodology Discrepancies)

> [!WARNING]
> 以下是與論文方法論有嚴重出入的地方，會直接影響模型的表現與公平性。

### 1. 語音基礎模組 (Speech Foundation Model) 用錯了
- **論文**：論文的核心架構 (Figure 1 與內文) 明確指出，語音編碼器使用的是 **WavLM Large**，且其參數是**會被微調 (Fine-tuned)** 的（並且取所有隱藏層的加權平均）。Whisper Large-V3 是負責將語音轉為文字 (ASR)，再由凍結的 RoBERTa 處理。
- **您的程式碼**：`train.py` 中直接使用了 `WhisperModel.from_pretrained("openai/whisper-large-v3").encoder` 當用語音特徵提取（而且還被 **凍結 `requires_grad = False`** ）。且資料集裡讀取的是 `Whisper_Features_15s`。
- **影響**：這實作比較偏向論文中的 System 2 變體，而不是核心的 SAILER (System 1 + 語言擴展)。若要拿最高分或是實質的 "SAILER" 架構，必須將其改為讀取原始音訊 waveform，並傳入 WavLM。

### 2. 標籤重加權 (Distribution Re-weighting) 時機錯誤
- **論文**：論文明確表示重加權策略 **「只在訓練階段 (Training) 使用，在驗證 (Validation) 或測試 (Testing) 階段不需要也不應該使用」**，以確保評估的客觀性。
- **您的程式碼**：在 `msp_dataset.py` 的 `_get_target_distribution` 中，`d_prime = d * self.w_norm` 是無條件執行的。這代表您的 Validation Set 也被套用了不平衡權重，這會導致 `val_loss` 失真。
- **解法**：應該將 Re-weighting 加入判斷 `if is_training:` 中。

### 3. Annotation Dropout 邏輯略有偏差
- **論文**：隨機隱藏部分標註者的意見（隨機 Drop votes）。
- **您的程式碼**：`v[drop_idx] -= 1.0` 只有針對 `self.majority_classes` 做 Dropout。論文並沒有說只丟棄 majority class 的標註，而是完全隨機丟棄任何標註者的獨立投票來增加魯棒性。

### 4. 欠缺多任務學習 (Multi-task Learning) (選擇性)
- **論文**：論文在 Section 4.3 證明了最佳單一系統 (Best Single System, 0.411) 是建立在學習次要情緒 (Secondary emotions) 與屬性 (Arousal/Valence/Dominance) 上。
- **您的程式碼**：`SAILER_Model` 的 Classifier 只有針對 8 分類 (`num_classes=8`) 輸出。若有餘裕，加入輔助的 Regression head 會大幅提升準星。

---

## 🧹 Clean Code 與工程實踐建議 (Clean Code Review)

> [!TIP]
> 您的程式碼具有基本結構，但可以透過以下調整達到更模組化、易讀的業界水準。

1. **`wandb.init` 雙重呼叫衝突**
   - `train.py` 呼叫了 `wandb.init()`，但 `ExperimentTracker.__init__` 裡面又呼叫了一次 `wandb.init()`。這將產生兩個獨立或衝突的 tracking sessions。
   - **解法**：統一由 `tracker` 物件來初始化 `wandb`。
2. **Hardcoded Paths (路徑硬編碼)**
   - `data_dir = "/home/brant/Project/SAILER_test/MSP_Podcast_Data"` 被寫死在 `train.py`。
   - **解法**：應該善用 `argparse` 建構 CLI 工具，透過 `python train.py --data_dir ...` 處理環境變數。
3. **變數命名需更具表達性**
   - `w_feat`, `t_ids`, `w_seq` 雖然您看得懂，但在 Clean Code 原則中會被判定為「過度簡稱 (Cryptic naming)」。
   - **解法**：改為 `speech_features`, `text_input_ids`, `speech_embeddings`，可讀性更佳。
4. **模組解耦 (Decoupling)**
   - 在 `train.py` 的訓練迴圈裡塞有資料擴增/轉錄的邏輯判斷是可以接受的，但在 `ExperimentTracker` 內部匯入了特定 ML 套件，可以嘗試把 Tracker 做成更純粹的 Logger。

---

## 🛠️ Proposed Changes (預計修改方案)

如果您同意，我將透過工具修改以下檔案以達標最高規格：

### 1. 修改 `train.py`
- 修復 `wandb` 雙重啟動的問題。
- 改用 `argparse` 將設定檔化。
- 變更變數命名，使 code self-explanatory。

### 2. 修改 `src/msp_dataset.py`
- 修復 **Distribution Re-weighting 污染驗證集**的問題 (`val_loss` 計算才會正確)。
- 修正 Annotation Dropout 的邏輯為隨機所有類別的 Dropout。

### 3. [Open Question] 關於語音編碼器 (WavLM vs Whisper Features)
您的特徵目錄叫做 `Whisper_Features_15s`，代表您**已經把語音預先抽成了 Whisper 的 Log-Mel Spectrogram**。
- **選項 A (維持現狀但優化架構)**：承認這是一個 Whisper-based System 2，我會幫您把程式碼弄得很乾淨，並補齊訓練機制。
- **選項 B (完全看齊論文 SAILER)**：您需要重啟特徵萃取流程，直接餵入 `.wav` 給 `WavLMModel` 在訓練迴圈中微調。

請告訴我您希望採用 **選項 A** 還是 **選項 B**？
收到您的確認與回饋後，我會立刻為您重構相對應的程式碼。
