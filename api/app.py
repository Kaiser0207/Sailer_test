"""
SAILER Emotion Classifier API — FastAPI Server
===============================================
基於 SAILER 官方 WhisperWrapper (speech-only) 的情緒分類 REST API。
遵守 api-spec.md Section 3a 規格。

啟動方式:
    cd /home/brant/Project/SAILER_test
    python -m uvicorn api.app:app --host 0.0.0.0 --port 8001

或直接:
    python api/app.py
"""

import sys
import os
import io
import time
import logging

import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# 將專案根目錄與 vox-profile-release 加入 Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "vox-profile-release"))

from api.config import (
    WHISPER_MODEL_ID,
    SAMPLE_RATE,
    MIN_AUDIO_SEC, MAX_AUDIO_SEC,
    MIN_AUDIO_SAMPLES, MAX_AUDIO_SAMPLES,
    OFFICIAL_9_LABELS, PRIMARY_LABELS, SECONDARY_LABELS,
    OFFICIAL_TO_API_MAP,
    DEFAULT_HOST, DEFAULT_PORT,
)

# ==========================================
# 日誌設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sailer_api")

# ==========================================
# 全域模型容器
# ==========================================
model_container = {
    "model": None,
    "device": None,
}


def load_model():
    """載入官方 WhisperWrapper 模型到 GPU"""
    from src.model.emotion.whisper_emotion import WhisperWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"裝置: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info(f"正在載入 SAILER 官方模型: {WHISPER_MODEL_ID} ...")
    model = WhisperWrapper.from_pretrained(WHISPER_MODEL_ID).float().to(device)
    model.eval()

    # 統計參數量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型載入完成！總參數: {total_params:,}")

    # Sanity Check
    logger.info("啟動 Sanity Check ...")
    test_audio = torch.randn(1, 48000).float().to(device)  # 3 秒假音
    with torch.no_grad():
        logits, _, _, _, _, _ = model(test_audio, return_feature=True)
    assert logits.shape == (1, 9), f"模型輸出 shape 異常: {logits.shape}"
    logger.info("Sanity Check 通過！模型就緒。")

    model_container["model"] = model
    model_container["device"] = device


# ==========================================
# FastAPI Lifespan (啟動時載入模型)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info(f"🚀 SAILER Emotion Classifier API 已啟動 (port {DEFAULT_PORT})")
    yield
    logger.info("API 伺服器關閉")


# ==========================================
# FastAPI App
# ==========================================
app = FastAPI(
    title="SAILER Emotion Classifier API",
    description=(
        "基於 SAILER 官方 WhisperWrapper (speech-only) 的音頻情緒分類服務。\n"
        "輸入一段 3~15 秒的音檔，回傳 8 類情緒機率分布、AVD 維度值、以及 256 維聲學嵌入向量。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (讓前端或其他服務可以跨域呼叫)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 工具函數
# ==========================================
def remap_9class_to_8class(probs_9: np.ndarray) -> np.ndarray:
    """
    將官方 WhisperWrapper 的 9 類機率重新映射為 API 規格的 8 類。
    Other 類別的機率按比例重新分配給其他 8 類。
    """
    reordered = np.zeros(8)
    for official_idx, api_idx in OFFICIAL_TO_API_MAP.items():
        if api_idx != -1:
            reordered[api_idx] = probs_9[official_idx]

    # 正規化，使 8 類機率總和為 1
    total = reordered.sum()
    if total > 1e-8:
        reordered = reordered / total

    return reordered


# ==========================================
# Endpoints
# ==========================================
@app.get("/health")
async def health_check():
    """健康檢查端點"""
    model_loaded = model_container["model"] is not None
    return {
        "status": "ok" if model_loaded else "model_not_loaded",
        "model": WHISPER_MODEL_ID,
        "device": str(model_container.get("device", "unknown")),
    }


@app.post("/classify-emotion")
async def classify_emotion(audio: UploadFile = File(...)):
    """
    音頻情緒分類端點 (遵守 api-spec.md Section 3a)

    接受: multipart/form-data，欄位名稱 "audio"
    支援格式: WAV, MP3, FLAC, OGG 等 torchaudio 可解碼的格式
    要求: 16kHz mono, 3~15 秒

    回傳: JSON，包含 primary/secondary emotion probabilities、AVD 值、embedding
    """
    model = model_container["model"]
    device = model_container["device"]

    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未載入完成")

    # ------------------------------------------
    # 1. 讀取上傳的音檔
    # ------------------------------------------
    try:
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="上傳的音檔為空")

        # 用 soundfile 解碼 (支援 WAV, FLAC, OGG 等)
        audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"無法解碼音檔格式: {str(e)}。支援 WAV, FLAC, OGG。"
        )

    # ------------------------------------------
    # 2. 預處理
    # ------------------------------------------
    # 轉 mono (如果是多聲道)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample 到 16kHz
    if sr != SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)

    # 截斷到最大 15 秒
    audio_data = audio_data[:MAX_AUDIO_SAMPLES]

    # 檢查最短長度
    duration = len(audio_data) / SAMPLE_RATE
    if duration < MIN_AUDIO_SEC:
        raise HTTPException(
            status_code=400,
            detail=f"音檔太短: {duration:.1f}s (最低要求 {MIN_AUDIO_SEC}s)"
        )

    # 轉為 torch tensor [1, num_samples]
    waveform = torch.tensor(audio_data).unsqueeze(0)

    # ------------------------------------------
    # 3. 模型推理
    # ------------------------------------------
    try:
        data = waveform.float().to(device)
        start_time = time.time()

        with torch.no_grad():
            logits, embedding, detailed_logits, arousal, valence, dominance = model(
                data, return_feature=True
            )

        inference_time = time.time() - start_time
        logger.info(f"推理完成 ({duration:.1f}s 音檔, 耗時 {inference_time:.3f}s)")

    except Exception as e:
        logger.error(f"推理失敗: {e}")
        raise HTTPException(status_code=500, detail=f"模型推理失敗: {str(e)}")

    # ------------------------------------------
    # 4. 後處理：機率計算 + 標籤映射
    # ------------------------------------------
    # 9 類機率
    probs_9 = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 映射到 API 規格的 8 類
    primary_probs = remap_9class_to_8class(probs_9)
    pred_idx = int(np.argmax(primary_probs))

    # 17 類次要情緒機率
    # 官方模型的 detailed_logits 通常也是 17 類，直接 softmax
    secondary_probs_raw = F.softmax(detailed_logits, dim=1).squeeze(0).cpu().numpy()

    # 建構 secondary probabilities dict
    # 如果模型輸出的類別數與 SECONDARY_LABELS 數量不同，做安全處理
    secondary_probs_dict = {}
    for i, label in enumerate(SECONDARY_LABELS):
        if i < len(secondary_probs_raw):
            secondary_probs_dict[label] = round(float(secondary_probs_raw[i]), 4)
        else:
            secondary_probs_dict[label] = 0.0

    # Embedding (256 維)
    embedding_list = embedding.squeeze(0).cpu().numpy().tolist()

    # AVD 值
    arousal_val = float(arousal.squeeze().cpu().item())
    valence_val = float(valence.squeeze().cpu().item())
    dominance_val = float(dominance.squeeze().cpu().item())

    # ------------------------------------------
    # 5. 組裝回應 (完全遵守 api-spec.md Section 3a)
    # ------------------------------------------
    response = {
        "primary_label": PRIMARY_LABELS[pred_idx],
        "primary_index": pred_idx,
        "primary_confidence": round(float(primary_probs[pred_idx]), 4),
        "primary_probabilities": {
            label: round(float(prob), 4)
            for label, prob in zip(PRIMARY_LABELS, primary_probs)
        },
        "secondary_probabilities": secondary_probs_dict,
        "arousal": round(arousal_val, 4),
        "valence": round(valence_val, 4),
        "dominance": round(dominance_val, 4),
        "embedding": embedding_list,
    }

    return response


# ==========================================
# 直接執行入口
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=False,
        log_level="info",
    )
