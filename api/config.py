"""
SAILER Emotion Classifier API — Configuration
=============================================
集中管理 API 伺服器的所有常數、標籤映射、路徑設定。
"""

import os

# ==========================================
# 路徑設定
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOX_PROFILE_PATH = os.path.join(PROJECT_ROOT, "vox-profile-release")

# 官方 HuggingFace 模型 ID (Speech-Only Baseline)
WHISPER_MODEL_ID = "tiantiaf/whisper-large-v3-msp-podcast-emotion"

# ==========================================
# 音訊限制
# ==========================================
SAMPLE_RATE = 16000
MIN_AUDIO_SEC = 3.0
MAX_AUDIO_SEC = 15.0
MIN_AUDIO_SAMPLES = int(MIN_AUDIO_SEC * SAMPLE_RATE)   # 48000
MAX_AUDIO_SAMPLES = int(MAX_AUDIO_SEC * SAMPLE_RATE)    # 240000

# ==========================================
# 標籤定義
# ==========================================

# 官方 WhisperWrapper 的 9 類標籤順序 (index 0~8)
OFFICIAL_9_LABELS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise", "Other"
]

# API 回傳的 8 類標籤順序 (遵守 api-spec.md Section 3)
PRIMARY_LABELS = [
    "Neutral", "Angry", "Sad", "Happy",
    "Fear", "Disgust", "Surprise", "Contempt"
]

# 17 類次要標籤 (遵守 api-spec.md Section 3)
SECONDARY_LABELS = [
    "Neutral", "Angry", "Sad", "Happy",
    "Fear", "Disgust", "Surprise", "Contempt",
    "Other-Concerned", "Other-Annoyed", "Other-Frustrated",
    "Other-Confused", "Other-Amused", "Other-Disappointed",
    "Other-Excited", "Other-Bored", "Other"
]

# 官方 9 類 index → API 8 類 index 映射
# (排除 Other=8，改為在 8 類中選最高機率)
OFFICIAL_TO_API_MAP = {
    0: 1,   # Anger     → Angry   (index 1)
    1: 7,   # Contempt  → Contempt(index 7)
    2: 5,   # Disgust   → Disgust (index 5)
    3: 4,   # Fear      → Fear    (index 4)
    4: 3,   # Happiness → Happy   (index 3)
    5: 0,   # Neutral   → Neutral (index 0)
    6: 2,   # Sadness   → Sad     (index 2)
    7: 6,   # Surprise  → Surprise(index 6)
    8: -1,  # Other     → 排除，機率重新分配給 8 類
}

# ==========================================
# 伺服器設定
# ==========================================
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
