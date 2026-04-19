# SAILER 專案實作紀錄與程式碼說明 (Implementation Summary)

這份文件為您整理了截至目前為止，我們針對 SAILER 論文在您的專案中所實作的核心邏輯。我們會依照論文的方法論結構，將對應的程式碼檔案進行整理說明。

---

## 1. 語音基礎模型與特徵提取 (Speech Foundation Model)
**論文對照**：對應 Section 2.1 & 4.4 的 **Best Single System (System 3)**
**相關檔案**：`scripts/extract_features.py`

* **負責功能**：將原始音檔轉換成模型能讀取的特徵，我們採用了 openai 的 `whisper-large-v3` 模型前處理器。
* **重點實作**：
  * **重採樣 (Resampling)**：為了防範 MSP-Podcast 中的檔案擁有不同的取樣率 (Sampling Rate)，我們透過 `librosa.load(path, sr=16000)` 強制把所有聲音壓成 16kHz。
  * **特徵生成**：將這些 16kHz 的語音透過 Whisper Processor 提取，轉存成 `.pt` 特徵矩陣 (`Whisper_Features_30s`)。這能省去我們在每次訓練當下由基礎網路運算的時間，這就是您目前這個架構跑起來很省資源的原因。

## 2. 軟標籤學習目標 (Learning Objective - KL Divergence)
**論文對照**：對應 Section 2.2 的 **Soft Labeling**
**相關檔案**：`train.py` (Line 57)

* **負責功能**：計算模型預測與真實標籤之間的誤差 (Loss)。
* **重點實作**：
  * 我們直接採用了 `nn.KLDivLoss(reduction='batchmean')`。
  * 不同於傳統只會有一個絕對標準答案 (One-Hot Encoder) 的 Cross Entropy，KL Divergence 允許我們訓練「機率分佈」。
  * 舉例來說，我們不會強迫模型認定某句話為「100% 生氣」，而是告訴模型它的 Ground Truth 是「60% 生氣、20% 期待、20% 難過」。

## 3. 資料增強：音訊混合 (Data Augmentation - Audio Mixing)
**論文對照**：對應 Section 2.3 的 **Audio Mixing**
**相關檔案**：`src/msp_dataset.py` (內部的方法 `__getitem__`)

* **負責功能**：增加稀有情緒（如 害怕、反感）的曝光率，解決「多數情緒與少數情緒的極度不平衡」。
* **重點實作**：
  * **條件觸發**：當抽中「多數類別（Neutral, Happy, Angry 等）」的資料時，有 `50%` 機率會對它進行混合。
  * **混合對象**：會從「少數類別」中，依據機率反比（數量越少越容易被抽中）抽一條出來混！
  * **連接方式**：模擬現實生活中的說話場景：
    1. **Silence (靜音插入)**：中間隨機插入 50~150 幀的空白，再接著說下一句話。
    2. **Overlap (重疊說話)**：兩段聲音的尾端跟頭端隨機重疊 (50~150 幀)，並且算特徵的平均。
  * **標籤重整**：新產生的音檔因為混了兩個不同情緒，所以它的 Soft Label 就會是 `(d1 + d2) / 2`。 

## 4. 解決資料不平衡：加權分配與驗證機制 (Distribution Re-weighting)
**論文對照**：對應 Section 2.4 的 **Distribution Re-weighting & Validation Metrics**
**相關檔案**：`src/msp_dataset.py` & `train.py`

* **負責功能**：讓模型在訓練時，猜錯「樣本越少的類別」扣分越重。
* **重點實作**：
  * **Train 專用權重 (在 dataset 中實作)**：算出了整體訓練資料的情緒分佈（經驗分佈 $q$），然後以 $w = 1/q$ 將標籤放大（$d\_prime\_normalized$）。**我們已經修復了它，確保它絕對不會在 Validation 階段使用，以保持公正評估！**
  * **少數類別平均精度 (Min. mAP)** (在 `train.py` 實作)：除了看大局的 Macro-F1 以外，我們特別設立了 `minority_classes` 的監控。一旦模型對於這幾個少數類別（如 Fear, Disgust）的預測更準確時，就會存下這個權重 `best_model_min_map.pth`。

---

整體來說，除了還沒有額外預測**次要情緒** (Multi-task learning) 以外，您現在的架構已經**完美移植了 SAILER 系統的最強單體配置**！
