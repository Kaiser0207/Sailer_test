# 🔬 SAILER 論文合規性全面審計報告 (Full Compliance Audit)

本報告以最高學術規格，逐項比對您目前的實作與論文原文 + 官方開源碼 (`vox-profile-release`) 之間的差距。

---

## ✅ 已正確實作的部分 (Fully Compliant)

| # | 論文章節 | 項目 | 您的實作位置 | 狀態 |
|---|---------|------|-------------|------|
| 1 | §2.1 | Whisper-Large-V3 作為語音基礎模型 | `train.py:34` | ✅ |
| 2 | §2.1 | 3-layer Pointwise Conv (filter=256) | `sailer_model.py:15-24` | ✅ |
| 3 | §2.1 | 2-layer MLP Classifier | `sailer_model.py:27-31` | ✅ |
| 4 | §2.1 | RoBERTa-Large 做文字模型 | `train.py:32,35` | ✅ |
| 5 | §2.1 | Whisper Encoder 凍結、只訓練下游 | `train.py:37-40` | ✅ |
| 6 | §2.2 | KL-Divergence Loss (Soft Labeling) | `train.py:57` | ✅ |
| 7 | §2.2 | 從 `labels_detailed.csv` 聚合投票做 Soft Label | `msp_dataset.py:59-73` | ✅ |
| 8 | §2.3 | Annotation Dropout (20%, majority only) | `msp_dataset.py:120-132` | ✅ |
| 9 | §2.3 | Audio Mixing (silence / overlap) | `msp_dataset.py:154-184` | ✅ |
| 10 | §2.4 | Distribution Re-weighting (Train only) | `msp_dataset.py:34-40, 137-141` | ✅ |
| 11 | §2.4 | Min. mAP 少數類別驗證指標 | `train.py:131-140` | ✅ |
| 12 | §3.2 | 語音最大長度限 15 秒 | `extract_features.py` (已修正) | ✅ |
| 13 | §3.2 | 訓練 15 Epochs | `train.py:19` | ✅ |
| 14 | §2.2 | 訓練使用「No Agreement」樣本 | `msp_dataset.py:79-83` | ✅ |

---

## 🔴 Critical 等級差距 (嚴重影響最終分數)

### GAP-1：缺少 Multi-task Learning (次要情緒 + 情感屬性預測)
**論文依據**：Section 4.3, Table 4

> *"Incorporating additional prediction targets can improve the overall macro-F1 score."*

論文中的 **Best Single System (Table 5, Macro-F1 = 0.411)** 同時預測了：
1. **Primary Emotion** (8 類) — 您目前唯一有做的
2. **Secondary Emotion** (17 類) — ❌ 缺少
3. **Arousal / Valence / Dominance** (AVD，3 個回歸值) — ❌ 缺少

> [!CAUTION]
> 從官方程式碼 `whisper_emotion.py:198-229` 可以看到，作者有 5 個輸出頭：`emotion_layer`、`detailed_out_layer`、`arousal_layer`、`valence_layer`、`dominance_layer`。
> **您的模型只有 1 個輸出頭**，這代表您目前的系統對應到的是 Table 4 的 **「Not Included」那一行 (Macro-F1 = 0.406)**，而非 Best Single System 的 0.411。

**影響**：少了 Multi-task，您的 Macro-F1 上限將被壓在約 0.406 左右，無法觸及論文的最佳 0.411。

---

### GAP-2：文字模型的 Temporal Averaging 缺少 Weighted Average
**論文依據**：Section 2.1, Figure 1

> *"Like speech modeling, we apply a weighted average to all encoder outputs, then the temporal averaging."*

您目前 `train.py:77` 的做法：
```python
t_seq = roberta_model(input_ids=t_ids, attention_mask=t_mask).last_hidden_state
```
這只取了 RoBERTa 的**最後一層**輸出。但論文說的是對**所有層的輸出做 Weighted Average**（學習每一層的權重）。

> [!IMPORTANT]
> 對於 Whisper，論文 Section 3.2 明確說 *"we use only the representations from the last layer"*，所以取 last layer 是對的。
> 但對於**文字模型 (RoBERTa)**，論文 Figure 1 和 Section 2.1 明確畫出了 "Weighted Average Pooling" 在文字特徵端也存在。

**影響**：中高。RoBERTa 的不同層包含不同語義粒度的資訊（低層是語法、高層是語意），學習一組權重能讓模型自動選擇最有利的混合比例。

---

### GAP-3：Learning Rate 不一致
**論文依據**：Section 3.2

> *"All of our systems are trained with a learning rate of 0.0005 for 15 epochs."*

您目前 `train.py:21`：
```python
"learning_rate": 0.0004  # ← 論文要求 0.0005
```

**影響**：LR 偏低，模型收斂速度較慢，在只有 15 個 Epoch 的情況下，可能導致模型還沒有訓練到最佳狀態就停止了。

---

## 🟡 Important 等級差距 (會影響模型品質)

### GAP-4：Whisper Encoder 的 `max_source_positions` 應該限制為 750
**論文依據**：官方程式碼 `whisper_emotion.py:142-149`

```python
# 官方寫法
self.backbone_model = WhisperModel.from_pretrained(
    "openai/whisper-large-v3",
    output_hidden_states=True,
    ignore_mismatched_sizes=True,
    max_source_positions=750,  # ← 限制最大位置編碼為 750 (對應 15 秒)
)
```

您目前的寫法沒有加上 `max_source_positions=750`。

**影響**：位置編碼 (Positional Embedding) 如果保持預設的 1500（即 30 秒長度），但您輸入只有 15 秒的特徵，可能會造成位置編碼表的前半截與後半截被不均勻地使用，影響模型的注意力機制效果。

---

### GAP-5：語音特徵應該使用 Masked Temporal Average（依有效長度），而非直接 `mean`
**論文依據**：官方程式碼 `whisper_emotion.py:288-296`

```python
# 官方寫法：只對有效長度做 mean，忽略 padding 的部分
for snt_id in range(features.shape[0]):
    actual_size = length[snt_id]
    mean.append(torch.mean(features[snt_id, 0:actual_size, ...], dim=0))
```

您目前的做法 (`sailer_model.py:38`)：
```python
s_emb = s_out.mean(dim=-1)  # ← 對整個序列做平均，包含 padding 的 0
```

**影響**：MSP-Podcast 中大約 99% 的音檔只有 2-10 秒。如果一段 5 秒的語音被填充到 15 秒，您的 `mean` 會把 10 秒的空白 padding 也算進去，稀釋了真正有意義的語音資訊。論文的做法是計算每條音訊的真實有效長度，**只對有效區域做平均**。

---

### GAP-6：Audio Mixing 中的 `t ∈ [0, 2]` 秒應該是以時間為單位
**論文依據**：Section 2.3

> *"We sample a time value t ∈ [0, 2] to determine the duration of the silence or overlap."*

您目前的寫法 (`msp_dataset.py:169,174`)：
```python
silence_len = random.randint(50, 150)  # ← 以 frame 為單位
overlap_len = random.randint(50, 150)
```

論文的意思是取一個 0 到 2 秒之間的隨機時間。在 Whisper 的特徵空間中，1 秒 ≈ 50 frame，所以 `[0, 2]` 秒應該對應 `[0, 100]` frame。您目前的範圍 `[50, 150]` frame = `[1, 3]` 秒，所以靜音和重疊的長度**偏長了**。

**影響**：低中。可能導致混合後的音訊特徵中，靜音或重疊的比例過大，稍微偏離論文的實驗設定。

---

## 🟢 Nice-to-Have 等級 (錦上添花)

### GAP-7：`pa` (Audio Mixing 的觸發機率) 未明確對照論文
**論文依據**：Section 2.3

> *"For each majority-class speech sample x_maj, we apply this augmentation with a probability of p_a."*

論文沒有給出 `pa` 的具體數值。您目前設定為 `0.5`（50%），這是一個合理的預設值。但如果您想完美重現，可以去查看官方程式碼的訓練腳本中是否有明確定義。

---

## 📊 影響力排名總結

| 優先序 | GAP | 描述 | 預估 F1 影響 | 修改難度 |
|--------|-----|------|-------------|---------|
| 🥇 | GAP-1 | Multi-task (2nd Emo + AVD) | +0.005 ~ +0.01 | ⬛⬛⬛ 高 |
| 🥈 | GAP-3 | Learning Rate 0.0005 | +0.002 ~ +0.005 | ⬜ 極低 |
| 🥉 | GAP-5 | Masked Temporal Average | +0.001 ~ +0.005 | ⬛⬛ 中 |
| 4 | GAP-2 | RoBERTa Weighted Average | +0.001 ~ +0.003 | ⬛⬛ 中 |
| 5 | GAP-4 | max_source_positions=750 | +0.001 | ⬜ 低 |
| 6 | GAP-6 | Audio Mixing 時間範圍 [0,100] | < +0.001 | ⬜ 極低 |

> [!TIP]
> **建議策略**：先修 GAP-3 (1 秒改一個數字)，然後修 GAP-5 (語音 Masked Average)，跑一輪看分數，再考慮是否做 GAP-1 (Multi-task Learning)，因為 Multi-task 改動量大且需要修 model + dataset + train 三個檔案。
