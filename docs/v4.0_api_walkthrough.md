# SAILER Emotion Classifier REST API — Walkthrough

## 概要

建立了一個 FastAPI REST API 服務，將 SAILER 官方 WhisperWrapper (speech-only) 模型包裝成 HTTP 端點，供你朋友的 LangGraph 面試模擬器串接。

所有檔案都收在 `api/` 資料夾內。

## 建立的檔案

| 檔案 | 用途 |
|---|---|
| [api/app.py](file:///home/brant/Project/SAILER_test/api/app.py) | FastAPI 主程式（模型載入 + 端點定義） |
| [api/config.py](file:///home/brant/Project/SAILER_test/api/config.py) | 常數定義（標籤映射、路徑、音檔限制） |
| [api/test_api.py](file:///home/brant/Project/SAILER_test/api/test_api.py) | 測試腳本（含合成音檔、錯誤處理測試） |
| [api/requirements.txt](file:///home/brant/Project/SAILER_test/api/requirements.txt) | API 專用 Python 依賴 |
| [api/README.md](file:///home/brant/Project/SAILER_test/api/README.md) | 使用說明（可直接給你朋友看） |
| [api/__init__.py](file:///home/brant/Project/SAILER_test/api/__init__.py) | Python package init |

## 架構

```
用戶端 (LangGraph)                           SAILER API (port 8001)
┌──────────────────────┐                    ┌─────────────────────────────────────┐
│ POST /classify-emotion│ ─── 音檔 WAV ──▶ │ soundfile 解碼 → 16kHz mono         │
│                      │                    │ → WhisperWrapper 推理                │
│                      │ ◀── JSON ──────── │ → 9類 softmax → 映射到8類            │
│ 收到情緒+AVD結果     │                    │ → 回傳 JSON (labels, AVD, embedding) │
└──────────────────────┘                    └─────────────────────────────────────┘
```

## 測試結果

```
🧪 SAILER Emotion Classifier API 測試

✅ Health check 通過!
✅ POST /classify-emotion — 合成音檔推理成功 (Neutral, 0.3809)
✅ 回應格式驗證通過 (8類primary + 17類secondary + AVD + 256d embedding)
✅ 正確拒絕太短音檔 (400)
✅ 正確拒絕無效格式 (422)

🎉 所有測試完成!
```

## 使用方式

```bash
# 啟動 server
cd /home/brant/Project/SAILER_test
source .venv/bin/activate
python -m uvicorn api.app:app --host 0.0.0.0 --port 8001

# 測試
python api/test_api.py
```

## 關鍵設計決策

1. **用 `soundfile` 而非 `torchaudio`**：torchaudio 2.11 移除了 soundfile backend，需要額外裝 torchcodec。改用 soundfile 直接解碼，與 `evaluate_official_baseline.py` 一致。

2. **9 類 → 8 類映射**：官方模型輸出 9 類（含 Other），API 規格要求 8 類。Other 類的機率被重新分配給其他 8 類，然後正規化。

3. **完全遵守 api-spec.md**：回應格式包含 `primary_label`, `primary_index`, `primary_confidence`, `primary_probabilities` (8類), `secondary_probabilities` (17類), `arousal`, `valence`, `dominance`, `embedding` (256維)。
