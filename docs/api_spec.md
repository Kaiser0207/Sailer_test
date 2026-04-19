# LangGraph Interview Simulator — API Specification

**Version:** 1.0  
**Date:** 2026-04-13

---

## Overview

The LangGraph interview simulator orchestrates 4 external services:

| # | Service | Protocol | Base URL |
|---|---|---|---|
| 1 | LLM (via LangChain) | REST | Provider API (OpenAI / Anthropic) |
| 2 | Fish Audio S2 TTS | REST chunked streaming | `http://localhost:8080` |
| 2b | Fish Audio S2 TTS (real-time) | WebSocket | `ws://localhost:8080` or cloud |
| 3 | Audio Emotion Classifier (SAILER) | REST | `http://localhost:8001` (to be built) |
| 4 | O*NET Job Skills | REST | `https://services.onetcenter.org/ws/` |

> **Local Fish Audio WebSocket note:** The `/v1/tts/live` WebSocket endpoint is part of the Fish Audio **cloud** API. The local `tools/api_server.py` only exposes REST `POST /v1/tts`. Either extend the local server or use the REST streaming endpoint for local deployments.

---

## 1. LLM — LangChain Chat Model Interface

LangGraph nodes call the LLM through LangChain's `ChatModel` abstraction (no custom endpoint). This section documents the node I/O contract and prompt structure.

### Provider Configuration

| Field | Value |
|---|---|
| OpenAI | `ChatOpenAI(model="gpt-4o")`, env: `OPENAI_API_KEY` |
| Anthropic | `ChatAnthropic(model="claude-opus-4-6")`, env: `ANTHROPIC_API_KEY` |
| Streaming | `streaming=True` for real-time token output |

### LangGraph Node I/O (Shared State Dict)

**Interviewer Node** — generates the next interview question

```python
# Reads from state
{
    "job_title": str,                       # e.g. "Software Engineer"
    "onet_skills": list[str],               # pulled from O*NET
    "conversation_history": list[Message],  # [{"role": "...", "content": "..."}]
    "candidate_answer": str | None          # None on first turn
}

# Writes to state
{
    "interviewer_question": str,
    "tts_input_text": str,                  # same as interviewer_question (with emotion tags)
    "conversation_history": list[Message]   # appended
}
```

**Feedback Node** — evaluates the candidate's response

```python
# Reads from state
{
    "candidate_answer": str,
    "detected_emotion": str,        # primary label from emotion classifier
    "emotion_confidence": float,    # 0.0–1.0
    "emotion_arousal": float,       # 0.0–1.0
    "emotion_valence": float,       # 0.0–1.0
    "onet_skills": list[str],
    "conversation_history": list[Message]
}

# Writes to state
{
    "feedback_text": str,
    "skill_gap_notes": str,
    "conversation_history": list[Message]   # appended
}
```

### System Prompt Template

```
System: You are an expert interviewer for the role of {job_title}.
        Required skills per O*NET: {onet_skills}.
        Current candidate emotional state: {detected_emotion}
        (confidence: {emotion_confidence:.0%}, arousal: {emotion_arousal:.2f},
         valence: {emotion_valence:.2f}).
        Tailor follow-up questions and feedback accordingly.
```

---

## 2. Fish Audio S2 TTS

### Deployment

```bash
# Local server startup
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --decoder-config-name modded_dac_vq \
  --device cuda \
  --listen 0.0.0.0:8080

# Docker (GPU compile mode)
COMPILE=1 docker compose --profile server up
```

Default listen address: `0.0.0.0:8080`

### 2a. REST Endpoint — `POST /v1/tts`

Generates speech from text. Supports chunked streaming (WAV only) and single-shot responses.

**Auth:** Optional Bearer token  
`Authorization: Bearer <api_key>` (only required if server started with `--api-key`)

**Content-Type:** `application/msgpack` (preferred) or `application/json`

**Request Body** (`ServeTTSRequest`)

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `text` | `string` | required | max configurable | Text to synthesize. Supports inline emotion tags (see below). |
| `format` | `"wav"\|"pcm"\|"mp3"\|"opus"` | `"wav"` | — | Output audio format. Streaming requires `"wav"`. |
| `chunk_length` | `integer` | `200` | 100–1000 | Token chunk size for generation. |
| `latency` | `"normal"\|"balanced"` | `"normal"` | — | Latency mode (relevant for cloud API). |
| `references` | `ServeReferenceAudio[]` | `[]` | — | Voice cloning reference audio + transcript pairs. |
| `reference_id` | `string\|null` | `null` | — | ID of a pre-loaded server-side reference voice. |
| `seed` | `integer\|null` | `null` | — | Fixed seed for deterministic output. |
| `use_memory_cache` | `"on"\|"off"` | `"off"` | — | Cache encoded reference codes in memory. |
| `normalize` | `boolean` | `true` | — | Normalize text (numbers, EN/ZH). |
| `streaming` | `boolean` | `false` | — | Chunked streaming response (WAV only). |
| `max_new_tokens` | `integer` | `1024` | — | Generation limit. |
| `top_p` | `float` | `0.8` | 0.1–1.0 | Nucleus sampling. |
| `repetition_penalty` | `float` | `1.1` | 0.9–2.0 | Repetition penalty. |
| `temperature` | `float` | `0.8` | 0.1–1.0 | Sampling temperature. |

**`ServeReferenceAudio` object**

| Field | Type | Description |
|---|---|---|
| `audio` | `bytes` (or base64 string) | Reference audio bytes (WAV/MP3/etc.) |
| `text` | `string` | Transcript of the reference audio |

**Inline Emotion Tags**

Fish Audio supports emotion control via inline text tags. Examples:

```
[whisper] This is said in a whisper.
[excited] Great news everyone!
[warm] Thank you so much for coming today.
[sad] I'm really sorry to hear that.
```

Over 15,000 tags are supported including tonal and emotional variants.

**Response**

- **Non-streaming:** Full audio bytes returned in one response body.
- **Streaming (`streaming: true`):** Chunked transfer encoding; WAV format only.

| Scenario | Content-Type | Body |
|---|---|---|
| `format=wav` | `audio/wav` | WAV audio bytes |
| `format=mp3` | `audio/mpeg` | MP3 audio bytes |
| `format=opus` | `audio/ogg` | Opus audio bytes |
| `format=pcm` | `application/octet-stream` | Raw PCM float16 bytes |

**Performance:** RTF ~0.4 on RTX 5090 (generates faster than real-time).

**Example Request (msgpack, Python)**

```python
import ormsgpack
import requests
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

req = ServeTTSRequest(
    text="[warm] Welcome to your interview today. Tell me about yourself.",
    format="wav",
    streaming=True,
    reference_id="interviewer-voice"   # pre-loaded server-side voice
)

response = requests.post(
    "http://localhost:8080/v1/tts",
    data=ormsgpack.packb(req, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
    headers={
        "Content-Type": "application/msgpack",
        "Authorization": "Bearer <api_key>"
    },
    stream=True
)

for chunk in response.iter_content(chunk_size=4096):
    audio_player.write(chunk)
```

**Example Request (JSON)**

```json
POST /v1/tts
Content-Type: application/json

{
  "text": "[warm] Welcome to your interview today. Tell me about yourself.",
  "format": "wav",
  "streaming": false,
  "reference_id": "interviewer-voice",
  "temperature": 0.8,
  "top_p": 0.8,
  "repetition_penalty": 1.1
}
```

**Error Responses**

| Status | Condition |
|---|---|
| `400 Bad Request` | Text too long, or `streaming=true` with non-WAV format |
| `401 Unauthorized` | Invalid or missing Bearer token |
| `500 Internal Server Error` | Generation failure |

---

### 2b. WebSocket Endpoint — `WS /v1/tts/live`

Bidirectional real-time TTS stream. Preferred for lowest latency. Uses **MessagePack** serialization for all frames.

> This endpoint is defined in the Fish Audio cloud API. For local deployments, use the REST endpoint above unless the local server is extended.

**Connection:** `ws://<host>:8080/v1/tts/live`

**Auth:** Pass API key as query param or header on upgrade.

#### Client → Server Message Types

**`start`** — Initialize a synthesis session

```json
{
  "event": "start",
  "request": {
    "text": "",
    "format": "wav",
    "reference_id": "interviewer-voice",
    "latency": "normal",
    "streaming": true,
    "temperature": 0.8,
    "top_p": 0.8,
    "repetition_penalty": 1.1,
    "chunk_length": 200
  }
}
```

**`text`** — Stream text chunks to synthesize

```json
{
  "event": "text",
  "text": "[warm] Tell me about your experience with distributed systems."
}
```

**`flush`** — Force synthesis of any buffered text

```json
{
  "event": "flush"
}
```

**`stop`** — End the session

```json
{
  "event": "stop"
}
```

#### Server → Client Message Types

**`audio`** — Binary audio chunk

```json
{
  "event": "audio",
  "audio": "<binary bytes>"
}
```

**`finish`** — Synthesis complete for flushed segment

```json
{
  "event": "finish"
}
```

All messages are MessagePack-encoded. The `audio` payload is raw audio bytes in the requested format.

---

### 2c. Reference Voice Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/health` | Server health check |
| `POST` | `/v1/references/add` | Upload a new voice reference |
| `GET` | `/v1/references/list` | List all reference voice IDs |
| `DELETE` | `/v1/references/delete` | Delete a reference voice |
| `POST` | `/v1/references/update` | Rename a reference voice |

**Add Reference** (`POST /v1/references/add`, `multipart/form-data`)

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique ID, pattern `[a-zA-Z0-9\-_ ]+`, max 255 chars |
| `audio` | `file` | WAV audio file |
| `text` | `string` | Transcript of the reference audio |

---

## 3. Audio Emotion Classifier (SAILER / vox-profile)

> **Status: Not yet implemented as a service.** The team must build a REST API wrapper around the existing model code. This section specifies the API contract to build.

### Model Details

| Attribute | Value |
|---|---|
| Architecture | Whisper-Large-V3 encoder (decoder discarded) + LoRA fine-tuning + 1D Conv temporal pooling |
| HuggingFace model | `tiantiaf/whisper-large-v3-msp-podcast-emotion` |
| Input | Mono audio, 16 kHz, 3–15 seconds |
| Primary output | 8-class emotion probabilities + argmax label |
| Secondary output | 17-class fine-grained emotion probabilities |
| AVD output | Arousal, Valence, Dominance (each 0–1, Sigmoid) |
| Embedding | 256-dim acoustic feature vector |
| Training data | MSP-Podcast corpus |

### Emotion Labels

**Primary Classes (8)** — `num_classes: 8`

| Index | Label | Short Code |
|---|---|---|
| 0 | Neutral | N |
| 1 | Angry | A |
| 2 | Sad | S |
| 3 | Happy | H |
| 4 | Fear | F |
| 5 | Disgust | D |
| 6 | Surprise | U |
| 7 | Contempt | C |

**Secondary / Fine-grained Classes (17)** — `secondary_class_num: 17`

| Index | Label |
|---|---|
| 0 | Neutral |
| 1 | Angry |
| 2 | Sad |
| 3 | Happy |
| 4 | Fear |
| 5 | Disgust |
| 6 | Surprise |
| 7 | Contempt |
| 8 | Other-Concerned |
| 9 | Other-Annoyed |
| 10 | Other-Frustrated |
| 11 | Other-Confused |
| 12 | Other-Amused |
| 13 | Other-Disappointed |
| 14 | Other-Excited |
| 15 | Other-Bored |
| 16 | Other |

### 3a. Non-Real-Time REST Endpoint — `POST /classify-emotion`

**Base URL:** `http://localhost:8001`

**Request**

```
POST /classify-emotion
Content-Type: multipart/form-data
```

| Field | Type | Constraints | Description |
|---|---|---|---|
| `audio` | `file` | WAV/MP3, 16kHz mono, 3–15 sec | Audio to classify |

> Alternatively, accept `application/octet-stream` with raw PCM float32 at 16kHz mono, and a query param `?sample_rate=16000`.

**Response** `200 OK`, `Content-Type: application/json`

```json
{
  "primary_label": "Neutral",
  "primary_index": 0,
  "primary_confidence": 0.82,
  "primary_probabilities": {
    "Neutral": 0.82,
    "Angry": 0.05,
    "Sad": 0.04,
    "Happy": 0.03,
    "Fear": 0.02,
    "Disgust": 0.01,
    "Surprise": 0.02,
    "Contempt": 0.01
  },
  "secondary_probabilities": {
    "Neutral": 0.79,
    "Angry": 0.04,
    "Sad": 0.04,
    "Happy": 0.03,
    "Fear": 0.02,
    "Disgust": 0.01,
    "Surprise": 0.02,
    "Contempt": 0.01,
    "Other-Concerned": 0.01,
    "Other-Annoyed": 0.01,
    "Other-Frustrated": 0.01,
    "Other-Confused": 0.00,
    "Other-Amused": 0.00,
    "Other-Disappointed": 0.00,
    "Other-Excited": 0.00,
    "Other-Bored": 0.00,
    "Other": 0.01
  },
  "arousal": 0.35,
  "valence": 0.61,
  "dominance": 0.52,
  "embedding": [0.12, -0.04, ...]
}
```

| Field | Type | Description |
|---|---|---|
| `primary_label` | `string` | Argmax of primary softmax (8-class) |
| `primary_index` | `integer` | 0–7 |
| `primary_confidence` | `float` | Softmax probability of predicted class |
| `primary_probabilities` | `object` | Full 8-class softmax distribution |
| `secondary_probabilities` | `object` | Full 17-class softmax distribution |
| `arousal` | `float` | Activation level, 0–1 |
| `valence` | `float` | Positive/negative affect, 0–1 |
| `dominance` | `float` | Sense of control, 0–1 |
| `embedding` | `float[]` | 256-dim acoustic embedding (optional; omit if not needed) |

**Error Responses**

| Status | Condition |
|---|---|
| `400 Bad Request` | Audio shorter than 3s, longer than 15s, or wrong sample rate |
| `422 Unprocessable Entity` | Invalid file format |
| `500 Internal Server Error` | Inference failure |

**Example (Python)**

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

---

### 3b. Real-Time Chunked REST (Polling Mode)

For real-time use during a live session, send sliding 2-second windows as repeated POSTs.

```
POST /classify-emotion
Content-Type: multipart/form-data

audio: <2-second PCM window at 16kHz>
```

**Recommended cadence:** Every 1–2 seconds (send 2s window, 1s stride).  
**Minimum window:** 3 seconds (pad with silence if needed to meet minimum).  
**Response:** Same schema as above.

The LangGraph node should maintain a rolling buffer, emit when the buffer reaches 3+ seconds, and use the latest result to update `detected_emotion` in the shared state.

---

### 3c. Suggested Server Implementation Skeleton

```python
# app.py — FastAPI wrapper for the SAILER model
import torch
import torchaudio
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, HTTPException
from src.model.emotion.whisper_emotion import WhisperWrapper

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRIMARY_LABELS = ["Neutral","Angry","Sad","Happy","Fear","Disgust","Surprise","Contempt"]
SECONDARY_LABELS = [
    "Neutral","Angry","Sad","Happy","Fear","Disgust","Surprise","Contempt",
    "Other-Concerned","Other-Annoyed","Other-Frustrated","Other-Confused",
    "Other-Amused","Other-Disappointed","Other-Excited","Other-Bored","Other"
]

model = WhisperWrapper.from_pretrained(
    "tiantiaf/whisper-large-v3-msp-podcast-emotion"
).to(device)
model.eval()

@app.post("/classify-emotion")
async def classify_emotion(audio: UploadFile):
    data, sr = torchaudio.load(audio.file)
    if sr != 16000:
        data = torchaudio.functional.resample(data, sr, 16000)
    data = data[:1, :15 * 16000].float().to(device)  # mono, max 15s

    duration = data.shape[1] / 16000
    if duration < 3.0:
        raise HTTPException(400, f"Audio too short: {duration:.1f}s (min 3s)")

    with torch.no_grad():
        logits, embedding, sec_logits, arousal, valence, dominance = model(
            data, return_feature=True
        )

    primary_probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    secondary_probs = F.softmax(sec_logits, dim=1)[0].cpu().tolist()
    pred_idx = int(torch.argmax(logits).item())

    return {
        "primary_label": PRIMARY_LABELS[pred_idx],
        "primary_index": pred_idx,
        "primary_confidence": primary_probs[pred_idx],
        "primary_probabilities": dict(zip(PRIMARY_LABELS, primary_probs)),
        "secondary_probabilities": dict(zip(SECONDARY_LABELS, secondary_probs)),
        "arousal": float(arousal[0].item()),
        "valence": float(valence[0].item()),
        "dominance": float(dominance[0].item()),
    }
```

Run with: `uvicorn app:app --host 0.0.0.0 --port 8001`

---

## 4. O*NET Web Services

**Base URL:** `https://services.onetcenter.org/ws/`  
**Auth:** HTTP Basic — `Authorization: Basic base64(username:password)`  
Register at: https://services.onetcenter.org/developer/

### 4a. Occupation Keyword Search

Find an O*NET occupation code from a job title.

```
GET /mnm/search?keyword={job_title}&client=<username>
```

**Query Parameters**

| Param | Type | Description |
|---|---|---|
| `keyword` | `string` | Job title to search (e.g. "Software Engineer") |
| `client` | `string` | Your O*NET username |
| `start` | `integer` | Pagination start (default 1) |
| `end` | `integer` | Pagination end (default 20) |

**Response** `200 OK`, `Content-Type: application/json`

```json
{
  "keyword": "software engineer",
  "start": 1,
  "end": 5,
  "total": 12,
  "occupation": [
    {
      "code": "15-1252.00",
      "title": "Software Developers",
      "tags": {"bright_outlook": true, "green": false}
    }
  ]
}
```

---

### 4b. Occupation Skills Lookup

Retrieve skills for a given O*NET occupation code.

```
GET /occupations/{onet_code}/summary/skills?client=<username>
```

| Param | Type | Description |
|---|---|---|
| `onet_code` | `string` (path) | O*NET-SOC code, e.g. `15-1252.00` |

**Response**

```json
{
  "occupation": {"code": "15-1252.00", "title": "Software Developers"},
  "element": [
    {
      "id": "2.A.1.b",
      "name": "Active Learning",
      "score": {"value": 3.62, "important": true},
      "category": 2
    },
    {
      "id": "2.B.3.g",
      "name": "Programming",
      "score": {"value": 4.25, "important": true},
      "category": 1
    }
  ]
}
```

**Key fields**

| Field | Description |
|---|---|
| `element[].name` | Skill name — pass these as `onet_skills` to the LLM prompt |
| `element[].score.value` | Importance score 1–5 |
| `element[].score.important` | `true` if importance >= 3.0 |

---

### 4c. Knowledge Lookup

```
GET /occupations/{onet_code}/summary/knowledge?client=<username>
```

Same response shape as skills. Use to augment LLM prompt with domain knowledge requirements.

---

### 4d. CareerOneStop Skill Gap (Optional)

Richer endpoint with wage + education data plus skill-gap comparison.

**Base URL:** `https://api.careeronestop.org/v1/`  
**Auth:** `Authorization: Bearer <token>` (separate registration at careeronestop.org)

```
GET /skillsmatcher/{userid}?sourceOccupation={code1}&targetOccupation={code2}
```

Returns skills present in target but missing or lower-scored in source — useful for generating targeted feedback.

---

## 5. LangGraph Node Wiring Summary

```
[User speaks]
    │
    ▼
[STT / transcript node]
    │ candidate_answer
    ▼
[Emotion Classifier node] ──── POST /classify-emotion ────► SAILER service
    │ detected_emotion, arousal, valence, dominance
    ▼
[Feedback / Interviewer LLM node] ── ChatModel ──────────► OpenAI / Anthropic
    │ interviewer_question + tts_input_text
    ▼
[TTS node] ────────────────────── POST /v1/tts ──────────► Fish Audio S2
    │ audio bytes (streamed)
    ▼
[Audio playback]
    │
    ▼
[O*NET node] (run once at session start, not per turn)
    └─── GET /mnm/search → GET /occupations/{code}/summary/skills
         stores result in session state as onet_skills
```

---

## 6. Authentication Summary

| Service | Method | Where |
|---|---|---|
| LLM (OpenAI) | API key | `OPENAI_API_KEY` env var |
| LLM (Anthropic) | API key | `ANTHROPIC_API_KEY` env var |
| Fish Audio (local) | Optional Bearer | `Authorization: Bearer <token>` header |
| Fish Audio (cloud) | Bearer | `Authorization: Bearer <token>` header |
| SAILER service | None (internal) | — |
| O*NET Web Services | HTTP Basic | `Authorization: Basic base64(user:pass)` |
| CareerOneStop | Bearer | `Authorization: Bearer <token>` |

---

## 7. Environment Variables Reference

```bash
# LLM
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Fish Audio (local)
FISH_AUDIO_URL=http://localhost:8080
FISH_AUDIO_API_KEY=          # leave blank if no --api-key set

# Fish Audio (cloud, if used)
FISH_AUDIO_CLOUD_URL=https://api.fish.audio
FISH_AUDIO_CLOUD_KEY=...

# SAILER
EMOTION_CLASSIFIER_URL=http://localhost:8001

# O*NET
ONET_USERNAME=your_username
ONET_PASSWORD=your_password

# CareerOneStop (optional)
CAREERONESTOP_USERID=your_userid
CAREERONESTOP_TOKEN=...
```
