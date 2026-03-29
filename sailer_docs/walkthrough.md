# SAILER Best Single System 完整升級報告

本次修改將您的系統從 Table 4「Not Included」升級為論文的 **Best Single System (Table 5)**。

---

## 修改總覽

| 檔案 | 改動量 | 涵蓋 GAP |
|------|--------|---------|
| `src/sailer_model.py` | 完整重寫 | GAP-5, GAP-2, GAP-1 |
| `src/msp_dataset.py` | 完整重寫 | GAP-5, GAP-1 |
| `train.py` | 完整重寫 | GAP-2, GAP-1 |

---

## GAP-5: Masked Temporal Average ✅

**問題**：原本語音特徵的 `mean()` 把 padding 的 0 也算進去了，稀釋了有意義的語音資訊。

**修復**：
- `msp_dataset.py`：計算並回傳 `effective_length`（padding 前的真實幀數）
- `sailer_model.py`：`forward()` 接收 `lengths` 參數，語音特徵只對有效幀做 mean

```python
# 新寫法：只計算有效區域
for i in range(s_out.shape[0]):
    actual_len = min(lengths[i].item(), s_out.shape[2])
    mean_list.append(s_out[i, :, :actual_len].mean(dim=-1))
```

---

## GAP-2: RoBERTa Learnable Weighted Average ✅

**問題**：原本只取 RoBERTa 最後一層，丟失了低層的語法資訊。

**修復**：
- `train.py`：呼叫 RoBERTa 時傳 `output_hidden_states=True`，取得所有 25 層
- `sailer_model.py`：新增 `self.text_layer_weights = nn.Parameter(torch.ones(25)/25)`，用 softmax 做 learnable weighted average

```python
stacked = torch.stack(t_hidden_states, dim=0)         # [25, B, T, 1024]
norm_weights = torch.softmax(self.text_layer_weights, dim=0)
t_seq = (norm_weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)  # [B, T, 1024]
```

---

## GAP-1: Multi-task Learning ✅

**問題**：原本只有 1 個輸出頭 (Primary Emotion 8 類)，對應 Table 4「Not Included」。

**修復**：新增 4 個輸出頭，對應官方碼 `whisper_emotion.py:198-229`：

| 輸出頭 | 類型 | 維度 | Loss |
|--------|------|------|------|
| Primary Emotion | 分類 | 8 | KLDivLoss |
| Secondary Emotion | 分類 | 17 | KLDivLoss |
| Arousal | 回歸 | 1 (Sigmoid) | MSELoss |
| Valence | 回歸 | 1 (Sigmoid) | MSELoss |
| Dominance | 回歸 | 1 (Sigmoid) | MSELoss |

**Total Loss** = `loss_primary + loss_secondary + loss_avd`

### Dataset 變更：
- 新增 17 類 secondary emotion 分佈（從 `labels_detailed.csv` 的 `EmoClass_Major` 統計）
- 新增 AVD 標籤（從 `labels_consensus.csv` 的 `EmoAct/EmoVal/EmoDom`，正規化到 [0,1]）
- 回傳格式：`(w_feat, t_ids, t_mask, label_dist, secondary_dist, avd_target, effective_length)`

---

## Dry-run 驗證結果 ✅

```
Model created successfully!
Primary logits:   torch.Size([4, 8])
Secondary logits: torch.Size([4, 17])
Arousal:          torch.Size([4, 1])
Valence:          torch.Size([4, 1])
Dominance:        torch.Size([4, 1])
Primary Loss:   0.3263
Secondary Loss: 0.4135
AVD Loss:       0.1185
Total Loss:     0.8582
All checks passed! ✅
```

---

## GAP 完整狀態

| GAP | 狀態 |
|-----|------|
| GAP-1 Multi-task Learning | ✅ Done |
| GAP-2 RoBERTa Weighted Average | ✅ Done |
| GAP-3 Learning Rate 0.0005 | ✅ Done (user) |
| GAP-4 max_source_positions=750 | ✅ Done |
| GAP-5 Masked Temporal Average | ✅ Done |
| GAP-6 Audio Mixing [0,100] | ✅ Done |
| GAP-7 pa probability | ⬜ Skip (合理預設) |
