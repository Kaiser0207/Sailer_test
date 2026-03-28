# SAILER 專案實作與程式碼詳細解說 (Implementation Summary)

這份文件為您整理了截至目前為止，本專案實作 SAILER 論文核心邏輯的詳細過程，並逐段對應程式碼為您解說。本文件會直接存放在您的 `SAILER_test` 資料夾中，方便您隨時查閱與做研究紀錄。

---

## 1. 語音基礎模型與特徵擷取 (Speech Foundation Model)
**📂 對應檔案**：`scripts/extract_features.py`
**📖 論文章節**：Section 2.1 & Section 4.4 (System 3 變體)
**🎯 目的**：使用 OpenAI 的 Whisper 模型將龐大、大小不一的原始音波轉換成機器看得懂的頻譜圖特徵 (Log-Mel Features)。這能讓後續訓練時免去重複提卡特徵的時間。

**詳細程式碼解說**：
```python
import librosa
from transformers import WhisperProcessor

# ====== (1) 安全重採樣 ====== 
# 讀取音訊檔。為了避免讀到 44.1kHz 或 48kHz 的音訊導致訓練報廢，
# 我們使用 librosa 強制將所有的音檔洗成 16,000Hz 的一致格式。
speech, sr = librosa.load(path, sr=16000)

# ====== (2) Whisper 특徵提取 ======
# 將 16kHz 的音訊丟給 Whisper 處理器。
# 它會截取音訊最前面 30 秒內的資料，並轉換做成 80-dim 的 Mel 頻譜圖特徵。
# squeeze(0) 是用來去掉多餘的維度，讓 Batch 訓練時不會形狀錯誤。
inputs = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)

# ====== (3) 儲存特徵以備訓練 ======
# half() 會將特徵轉為 float16 (半精度浮點數)，
# 在不損失太多資訊的狀況下，大大減少硬碟佔用量 (從幾百 GB 變成 35 GB)。
torch.save(inputs.half(), save_path)
```

---

## 2. 學習目標 (Learning Objective - 軟標籤學習)
**📂 對應檔案**：`train.py`
**📖 論文章節**：Section 2.2 KL-Divergence
**🎯 目的**：情緒是主觀且複雜的。我們不逼迫模型在一道題中只選一個「標準答案 (Hard Label)」，而是教它學會這句話中的「情緒機率佔比 (Soft Label)」。

**詳細程式碼解說**：
```python
import torch.nn as nn
import torch.nn.functional as F

# ====== (1) 宣告損失函數 ======
# KL Divergence 專門用來計算「兩個機率分佈的差異」。
# reduction='batchmean' 表示會將這一個 Batch 內所有樣本的誤差取平均，這是 PyTorch 規定的正確寫法。
criterion = nn.KLDivLoss(reduction='batchmean')

# ... 在 Training 迴圈中 ...

# ====== (2) 透過模型得到預測值 ======
# 這裡模型出來的 logits (未正規化分數)，形狀是 [Batch, 8種情緒]
logits = model(w_seq, t_seq, t_mask) 

# ====== (3) 計算誤差 ======
# - F.log_softmax(logits): 會先把模型的猜測值轉成「對數機率 (Log Probability)」。 
# - label_dists: 真實答案的機率分佈 (如: 0.8 生氣, 0.2 難過)，來自於資料集裡的標籤重組。
loss = criterion(F.log_softmax(logits, dim=-1), label_dists)
```

---

## 3. 資料增強：音訊混合 (Data Augmentation - Audio Mixing)
**📂 對應檔案**：`src/msp_dataset.py`
**📖 論文章節**：Section 2.3 Audio Mixing
**🎯 目的**：極端情緒 (害怕、厭惡) 的資料太少。為了讓模型「多聽一點」少數類別的聲音，我們透過機率將兩段不同情緒的人聲進行混合。

**詳細程式碼解說** (位在 `__getitem__` 中)：
```python
# ====== (1) 觸發條件 (50% 機率且當前音檔為多數類別) ======
if self.apply_aug and record['consensus_label'] in self.majority_classes and random.random() < 0.5:
    
    # ====== (2) 逆向抽取少數類別 ======
    # random.choices 會根據先前算好的自我權重 (self.record_weights)
    # 進行不公平抽籤。也就是說，資料越稀有的情緒 (例如: 恐懼) 被抽中作為干擾項的機率越高！
    min_record = random.choices(self.minority_records, weights=self.record_weights, k=1)[0]
    min_feat = torch.load(min_record["feat_path"]).float()

    # ====== (3) 決定兩段人聲如何結合 ======
    mix_type = random.choice(["silence", "overlap"])
    
    # [A] 靜音結合法 (Silence)：模擬一個人講完，另一個人停頓後接著講
    if mix_type == "silence":
        silence_len = random.randint(50, 150) # 隨機停頓長度
        silence = torch.zeros((mel_bins, silence_len))
        # 像火車車廂一樣把三段 (前音訊 + 空白 + 後音訊) 接在一起
        mixed_feat = torch.cat([feat_first, silence, feat_second], dim=-1)
        
    # [B] 重疊結合法 (Overlap)：模擬兩個人搶話 (聲音重疊)
    else:
        overlap_len = random.randint(50, 150)
        # 把前音檔尾巴跟後音檔頭部算平均，做漸層融合
        overlap_zone = (feat_first[:, -overlap_len:] + feat_second[:, :overlap_len]) / 2.0
        mixed_feat = torch.cat([front, overlap_zone, back], dim=-1)

    # ====== (4) 重新融合情緒標籤 ======
    # 既然聲音混在一起，答案也要平分！如果原音檔是 100% 生氣，混了 100% 中性
    # 那這條新產生的資料，Ground Truth 就會變成 50% 生氣 + 50% 中性。
    label_dist = (label_dist + min_dist) / 2.0
```

---

## 4. 解決資料不平衡：加權與審查機制 (Class Handling)
**📂 對應檔案**：`src/msp_dataset.py` & `train.py`
**📖 論文章節**：Section 2.4 Engineering Design Choices
**🎯 目的**：即便有資料增強，不平衡一樣存在。我們在資料集中放大冷門情緒的學習價值，並在 `train.py` 設下驗證機制的防線。

**詳細程式碼解說**：
```python
# --- [1] msp_dataset.py 的目標重新分配 ---
def _get_target_distribution(self, votes, is_training):
    d = v / (v.sum() + 1e-8)
    
    # ====== (1) 嚴格把關：只有 Training 才能被干涉 ======
    if self.split == "Train":
        # self.w_norm 是從訓練集跑過一遍統計出來的「頻率反比權重」。
        # 把少數情緒的分數加倍放大，讓模型知道「猜錯冷門題會扣很多分」。
        d_prime = d * self.w_norm
        d_prime_normalized = d_prime / (d_prime.sum() + 1e-8)
    else:
        # Validation 用原汁原味的真實分佈考核！
        d_prime_normalized = d
```

```python
# --- [2] train.py 的防破功監控 ---
# ====== (2) 專為少數類別設計的分數指標 ======
# 每跑完一個 epoch，除了看整體的平均外，要把那些「超冷門情緒 (如恐懼、厭惡)」
# 單獨拉出來審查 (Minority Classes mAP)。
minority_classes = [4, 5, 6, 7]
min_aps = []

for c in minority_classes:
    # 只把答案是這個分類的部分抓出來對答案
    y_true_c = (np.array(all_labels) == c).astype(int)
    y_score_c = np.array(all_preds_probs)[:, c]
    if np.sum(y_true_c) > 0:
        min_aps.append(average_precision_score(y_true_c, y_score_c))

min_map = np.mean(min_aps) if min_aps else 0.0

# 模型只有在這個少數派也考得高的時候，才值得我們把權重存下來！
# 這確保模型不是個只會狂猜中性跟生氣而拿高分的騙子。
if min_map > best_min_map:
    torch.save(model.state_dict(), "best_model_min_map.pth")
```
