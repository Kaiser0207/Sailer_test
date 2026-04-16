# SAILER Emotion Classifier — REST API

基於 SAILER 官方 WhisperWrapper (speech-only) 的音頻情緒分類 REST API 服務。

遵守 `paper/api-spec.md` Section 3a 的規格，供 LangGraph 面試模擬器串接。

## 快速啟動

### 1. 安裝依賴

```bash
cd /home/brant/Project/SAILER_test
pip install -r api/requirements.txt
```

### 2. 啟動 API Server

```bash
cd /home/brant/Project/SAILER_test
python -m uvicorn api.app:app --host 0.0.0.0 --port 8001
```

Server 啟動後會自動載入模型（首次啟動需要幾秒鐘），載入完成後會顯示：

```
SAILER Emotion Classifier API 已啟動 (port 8001)
```

### 3. 測試

```bash
# 基本測試 (合成音檔)
python api/test_api.py

# 用真實音檔測試
python api/test_api.py --audio /path/to/your/audio.wav
```

## API 端點

### `GET /health`

健康檢查。

```bash
curl http://localhost:8001/health
```

```json
{"status": "ok", "model": "tiantiaf/whisper-large-v3-msp-podcast-emotion", "device": "cuda"}
```

### `POST /classify-emotion`

情緒分類。接受 `multipart/form-data` 音檔。

**要求：**
- 格式：WAV, MP3, FLAC, OGG
- 取樣率：自動 resample 到 16kHz
- 通道：自動轉 mono
- 長度：3~15 秒

**範例：**

```bash
curl -X POST http://localhost:8001/classify-emotion \
  -F "audio=@candidate_answer.wav"
```

```python
import requests

with open("candidate_answer.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8001/classify-emotion",
        files={"audio": ("answer.wav", f, "audio/wav")}
    )

result = response.json()
print(result["primary_label"], result["primary_confidence"])
# "Neutral" 0.82
```

**回應格式：**

```json
{
  "primary_label": "Neutral",
  "primary_index": 0,
  "primary_confidence": 0.82,
  "primary_probabilities": {
    "Neutral": 0.82, "Angry": 0.05, "Sad": 0.04, "Happy": 0.03,
    "Fear": 0.02, "Disgust": 0.01, "Surprise": 0.02, "Contempt": 0.01
  },
  "secondary_probabilities": {
    "Neutral": 0.79, "Angry": 0.04, "Sad": 0.04, "Happy": 0.03,
    "Fear": 0.02, "Disgust": 0.01, "Surprise": 0.02, "Contempt": 0.01,
    "Other-Concerned": 0.01, "Other-Annoyed": 0.01, "Other-Frustrated": 0.01,
    "Other-Confused": 0.00, "Other-Amused": 0.00, "Other-Disappointed": 0.00,
    "Other-Excited": 0.00, "Other-Bored": 0.00, "Other": 0.01
  },
  "arousal": 0.35,
  "valence": 0.61,
  "dominance": 0.52,
  "embedding": [0.12, -0.04, ...]
}
```

## 錯誤碼

| Status | 說明 |
|---|---|
| `400` | 音檔太短 (< 3 秒) |
| `422` | 無法解碼的檔案格式 |
| `500` | 模型推理失敗 |
| `503` | 模型尚未載入完成 |

## 8 類情緒標籤

| Index | Label | 說明 |
|---|---|---|
| 0 | Neutral | 中性 |
| 1 | Angry | 憤怒 |
| 2 | Sad | 悲傷 |
| 3 | Happy | 開心 |
| 4 | Fear | 恐懼 |
| 5 | Disgust | 厭惡 |
| 6 | Surprise | 驚訝 |
| 7 | Contempt | 蔑視 |

## 模型資訊

- **架構**: Whisper-Large-V3 encoder + LoRA fine-tuning + 1D Conv temporal pooling
- **HuggingFace**: `tiantiaf/whisper-large-v3-msp-podcast-emotion`
- **訓練資料**: MSP-Podcast corpus
- **模式**: Speech-only (純語音)
