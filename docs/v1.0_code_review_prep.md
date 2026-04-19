# 🔍 Code Review 預審報告 — 您朋友一定會挑的問題

## 總體評價
您的程式碼已經**遠超一般初學者水準**。模組化切分 (`sailer_model.py` / `msp_dataset.py` / `train.py`) 是非常乾淨的設計，變數命名也很清楚。以下是站在「嚴格 Reviewer」的角度，列出他會提出的問題。

---

## 🔴 嚴重問題 (他一定會說「這個必須改」)

### 問題 1：Hardcoded 絕對路徑
**檔案**：[train.py:31](file:///home/brant/Project/SAILER_test/train.py#L31)
```python
data_dir = "/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data"
```
> [!CAUTION]
> 把您個人電腦的絕對路徑直接寫死在程式碼裡。如果另一個人 clone 這份 repo，他必須手動改這一行，否則直接報錯。這是 Clean Code 的第一大忌。

**建議改法**：使用 `argparse` 或相對路徑
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./datasets/MSP_Podcast_Data")
args = parser.parse_args()
data_dir = args.data_dir
```

---

### 問題 2：GPU 上的 Python for 迴圈 (效能殺手)
**檔案**：[sailer_model.py:91-99](file:///home/brant/Project/SAILER_test/src/sailer_model.py#L91-L99)
```python
for i in range(s_out.shape[0]):
    actual_len = min(lengths[i].item(), s_out.shape[2])
```
> [!WARNING]
> `.item()` 會強制 CPU-GPU 同步，在 Batch 迴圈中呼叫 64 次會造成嚴重的效能降低。PyTorch 中應盡量避免在 forward 裡用 Python for 迴圈。此外，這段程式碼也導致了 `torch.compile` 產生 graph break（訓練日誌裡有對應的 warning）。

**建議改法**：向量化 (Vectorized Masking)
```python
if lengths is not None:
    B, D, T = s_out.shape
    mask = torch.arange(T, device=s_out.device).unsqueeze(0) < lengths.unsqueeze(1)
    s_out_masked = s_out * mask.unsqueeze(1)
    s_emb = s_out_masked.sum(dim=-1) / lengths.clamp(min=1).unsqueeze(1).float()
else:
    s_emb = s_out.mean(dim=-1)
```

---

### 問題 3：WandB 初始化了兩次
**檔案**：[train.py:26](file:///home/brant/Project/SAILER_test/train.py#L26) 和 [experiment_tracker.py:28](file:///home/brant/Project/SAILER_test/src/experiment_tracker.py#L28)

`train.py` 裡面呼叫了 `wandb.init()`，但 `ExperimentTracker.__init__()` 裡面**也呼叫了一次 `wandb.init()`**。這代表每次訓練啟動時，WandB 會被初始化兩次，可能會產生衝突或覆蓋設定。

**建議**：擇一保留。建議把 `experiment_tracker.py` 中的 `wandb.init()` 移除，統一由 `train.py` 管理。

---

## 🟡 中等問題 (他可能會「建議」你改)

### 問題 4：Magic Numbers（魔法數字）
**多處出現**：
- `1500`（目標幀數）散佈在 [msp_dataset.py:253](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L253)
- `128`（文字最大長度）在 [msp_dataset.py:265](file:///home/brant/Project/SAILER_test/src/msp_dataset.py#L265)
- `0.2`（Dropout）在 [train.py:57](file:///home/brant/Project/SAILER_test/train.py#L57) 卻沒放進 config dict

這些數字不集中管理的話，改一處忘改另一處就會出 bug。

**建議**：全部收進 `config` 字典，或在 dataset `__init__` 加上 `target_frames` 參數。

---

### 問題 5：半殘的英文註解
**檔案**：[sailer_model.py](file:///home/brant/Project/SAILER_test/src/sailer_model.py)
```python
# Arousal Regression: 
# Valence Regression: 
# Dominance Regression:
```
冒號後面空空的，看起來像是有人刪掉了一半的說明。Reviewer 會覺得這是不完整的 (Incomplete Documentation)。

**建議**：要嘛補完，要嘛乾脆刪掉多餘的冒號和空格讓它看起來更乾淨。

---

### 問題 6：`__pycache__` 殘留
**位置**：根目錄和 `src/` 底下都有 `__pycache__/` 資料夾。
雖然 `.gitignore` 有排除它，但如果之前不小心 commit 過，它就會留在 git 歷史裡。

**建議**：
```bash
git rm -r --cached __pycache__ src/__pycache__
```

---

## 🟢 額外加分項（您已經做得很好的地方）

| 項目 | 評價 |
|------|------|
| 模組化 (Separation of Concerns) | ✅ 非常乾淨，三個主要檔案各有職責 |
| 變數命名 | ✅ `primary_logits`, `fused_emb`, `effective_length` 等命名都非常直觀 |
| `.gitignore` | ✅ 設定精確，大檔案都有排除 |
| Docstring | ✅ `sailer_model.py` 的 `forward()` 有完整的 Args/Returns 文件 |
| 實驗追蹤 | ✅ 同時有 TensorBoard + WandB + 本地存檔，三重保險 |
| 雙重權重儲存 | ✅ 同時追蹤 Best F1 和 Best Min.mAP，非常專業 |

---

## 修改優先級

| 優先 | 問題 | 難度 | 影響 |
|------|------|------|------|
| 🔴 1 | Hardcoded 路徑 | 🟢 簡單 | 可攜性 |
| 🔴 2 | for 迴圈向量化 | 🟡 中等 | 效能 +50% |
| 🔴 3 | WandB 雙重初始化 | 🟢 簡單 | 避免衝突 |
| 🟡 4 | Magic Numbers | 🟢 簡單 | 可維護性 |
| 🟡 5 | 半殘註解 | 🟢 簡單 | 整潔度 |
