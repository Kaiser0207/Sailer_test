"""
SAILER Emotion Classifier API — 測試腳本
========================================
用來驗證 API 是否正常運作。

使用方式:
    # 先啟動 server (另一個終端):
    cd /home/brant/Project/SAILER_test
    python -m uvicorn api.app:app --host 0.0.0.0 --port 8001

    # 然後執行測試:
    python api/test_api.py
    python api/test_api.py --audio /path/to/your/audio.wav
"""

import argparse
import json
import sys
import os
import requests
import numpy as np

# 預設 URL，可透過 --url 參數覆蓋
_config = {"base_url": "http://localhost:8001"}

def get_url():
    return _config["base_url"]


def test_health():
    """測試健康檢查端點"""
    print("=" * 50)
    print("測試 GET /health")
    print("=" * 50)

    try:
        resp = requests.get(f"{get_url()}/health", timeout=10)
        print(f"   Status: {resp.status_code}")
        print(f"   Response: {json.dumps(resp.json(), indent=2)}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        print("   Health check 通過!")
        return True
    except requests.exceptions.ConnectionError:
        print("   無法連接到 API server。請確認 server 已啟動。")
        return False
    except Exception as e:
        print(f"   失敗: {e}")
        return False


def test_classify_with_synthetic_audio():
    """用合成的測試音檔 (3秒靜音) 測試分類端點"""
    print()
    print("=" * 50)
    print("測試 POST /classify-emotion (合成 3 秒測試音)")
    print("=" * 50)

    import io
    import wave
    import struct

    # 產生 3 秒 16kHz mono 靜音 WAV
    sample_rate = 16000
    duration = 3
    num_samples = sample_rate * duration

    # 加一些白噪音讓它不是完全靜音
    np.random.seed(42)
    samples = (np.random.randn(num_samples) * 0.01).astype(np.float32)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # 轉成 int16
        int_samples = (samples * 32767).astype(np.int16)
        wf.writeframes(int_samples.tobytes())

    buf.seek(0)

    try:
        resp = requests.post(
            f"{get_url()}/classify-emotion",
            files={"audio": ("test_synthetic.wav", buf, "audio/wav")},
            timeout=60,
        )
        print(f"   Status: {resp.status_code}")

        if resp.status_code == 200:
            result = resp.json()
            print(f"   Primary Label:      {result['primary_label']}")
            print(f"   Primary Confidence: {result['primary_confidence']:.4f}")
            print(f"   Primary Index:      {result['primary_index']}")
            print(f"   Arousal:            {result['arousal']:.4f}")
            print(f"   Valence:            {result['valence']:.4f}")
            print(f"   Dominance:          {result['dominance']:.4f}")
            print(f"   Embedding dim:      {len(result['embedding'])}")
            print()
            print("   Primary Probabilities:")
            for label, prob in result["primary_probabilities"].items():
                bar = "█" * int(prob * 40)
                print(f"     {label:>10}: {prob:.4f} {bar}")
            print()

            # 驗證回應格式
            assert "primary_label" in result
            assert "primary_index" in result
            assert "primary_confidence" in result
            assert "primary_probabilities" in result
            assert "secondary_probabilities" in result
            assert "arousal" in result
            assert "valence" in result
            assert "dominance" in result
            assert "embedding" in result
            assert len(result["primary_probabilities"]) == 8
            assert len(result["secondary_probabilities"]) == 17
            assert 0 <= result["arousal"] <= 1
            assert 0 <= result["valence"] <= 1
            assert 0 <= result["dominance"] <= 1
            print("   ✅ 回應格式驗證通過!")
            return True
        else:
            print(f"   失敗: {resp.text}")
            return False

    except Exception as e:
        print(f"   失敗: {e}")
        return False


def test_classify_with_real_audio(audio_path: str):
    """用真實音檔測試分類端點"""
    print()
    print("=" * 50)
    print(f"測試 POST /classify-emotion (真實音檔)")
    print(f"   音檔: {audio_path}")
    print("=" * 50)

    if not os.path.exists(audio_path):
        print(f"   音檔不存在: {audio_path}")
        return False

    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{get_url()}/classify-emotion",
                files={"audio": (os.path.basename(audio_path), f, "audio/wav")},
                timeout=60,
            )

        print(f"   Status: {resp.status_code}")

        if resp.status_code == 200:
            result = resp.json()
            print(f"   Primary Label:      {result['primary_label']}")
            print(f"   Primary Confidence: {result['primary_confidence']:.4f}")
            print(f"   Arousal:            {result['arousal']:.4f}")
            print(f"   Valence:            {result['valence']:.4f}")
            print(f"   Dominance:          {result['dominance']:.4f}")
            print(f"   Embedding dim:      {len(result['embedding'])}")
            print()
            print("   Primary Probabilities:")
            for label, prob in result["primary_probabilities"].items():
                bar = "█" * int(prob * 40)
                print(f"     {label:>10}: {prob:.4f} {bar}")
            print()
            print("   真實音檔測試通過!")
            return True
        else:
            print(f"   失敗: {resp.text}")
            return False

    except Exception as e:
        print(f"   失敗: {e}")
        return False


def test_error_cases():
    """測試錯誤處理"""
    print()
    print("=" * 50)
    print("測試錯誤處理")
    print("=" * 50)

    import io
    import wave

    # 測試 1: 太短的音檔 (1 秒)
    print("   [1] 音檔太短 (1 秒) ...")
    sample_rate = 16000
    num_samples = sample_rate * 1  # 1 秒
    np.random.seed(42)
    samples = (np.random.randn(num_samples) * 0.01 * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    buf.seek(0)

    resp = requests.post(
        f"{get_url()}/classify-emotion",
        files={"audio": ("short.wav", buf, "audio/wav")},
        timeout=30,
    )
    assert resp.status_code == 400, f"期望 400，實際 {resp.status_code}"
    print(f"       Status: {resp.status_code} — {resp.json()['detail']}")
    print("       正確拒絕太短音檔!")

    # 測試 2: 無效格式
    print("   [2] 無效檔案格式 (文字檔) ...")
    fake_file = io.BytesIO(b"this is not audio")
    resp = requests.post(
        f"{get_url()}/classify-emotion",
        files={"audio": ("fake.txt", fake_file, "text/plain")},
        timeout=30,
    )
    assert resp.status_code == 422, f"期望 422，實際 {resp.status_code}"
    print(f"       Status: {resp.status_code} — {resp.json()['detail']}")
    print("       正確拒絕無效格式!")

    print()
    print("   所有錯誤處理測試通過!")
    return True


def main():
    parser = argparse.ArgumentParser(description="SAILER API 測試腳本")
    parser.add_argument(
        "--audio", type=str, default=None,
        help="真實音檔路徑 (WAV, 16kHz mono, 3~15s)"
    )
    parser.add_argument(
        "--url", type=str, default=get_url(),
        help=f"API base URL (預設: {get_url()})"
    )
    args = parser.parse_args()

    _config["base_url"] = args.url

    print()
    print("SAILER Emotion Classifier API 測試")
    print(f"Target: {get_url()}")
    print()

    # 1. Health check
    if not test_health():
        print("\nServer 未就緒，中止測試。")
        sys.exit(1)

    # 2. 合成音檔測試
    test_classify_with_synthetic_audio()

    # 3. 真實音檔測試 (如果有提供)
    if args.audio:
        test_classify_with_real_audio(args.audio)

    # 4. 錯誤處理測試
    test_error_cases()

    print()
    print("=" * 50)
    print("所有測試完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
