"""
Whisper Encoder 離線特徵提取腳本 (Offline Feature Extraction)

目的：將所有預先提取的 Mel 頻譜圖 (.pt) 一次性通過凍結的 Whisper Encoder，
      將輸出的高維度特徵向量儲存為 .pt 檔案，供後續訓練直接載入。
      
效果：訓練時完全移除 Whisper Encoder，每 Epoch 從 ~2 小時降至 ~2-5 分鐘。

用法：
    python scripts/extract_whisper_features.py --config configs/default_config.json
    python scripts/extract_whisper_features.py --config configs/default_config.json --delete_mel
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import WhisperModel
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract Whisper Encoder features offline")
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for extraction")
    parser.add_argument("--delete_mel", action="store_true",
                        help="Delete original Mel .pt files after extraction to free disk space")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    data_dir = config["data_dir"]
    mel_dir = os.path.join(data_dir, "Whisper_Features_15s")
    out_dir = os.path.join(data_dir, "Whisper_Encoder_Features")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入 Whisper Encoder (凍結狀態)
    print("Loading Whisper-Large-V3 Encoder...")
    whisper_enc = WhisperModel.from_pretrained("openai/whisper-large-v3").encoder.to(device)
    whisper_enc.eval()
    for p in whisper_enc.parameters():
        p.requires_grad = False

    # 掃描所有 Mel 檔案
    mel_files = sorted([f for f in os.listdir(mel_dir) if f.endswith(".pt")])
    print(f"Total Mel files to process: {len(mel_files)}")

    # 過濾已經提取過的檔案 (支援斷點續接)
    remaining = [f for f in mel_files if not os.path.exists(os.path.join(out_dir, f))]
    print(f"Already extracted: {len(mel_files) - len(remaining)}, Remaining: {len(remaining)}")

    if args.delete_mel:
        # 先刪除已經成功提取過的 Mel 檔案 (斷點續接時清理)
        already_done = [f for f in mel_files if os.path.exists(os.path.join(out_dir, f))]
        deleted_count = 0
        for f in already_done:
            mel_path = os.path.join(mel_dir, f)
            if os.path.exists(mel_path):
                os.remove(mel_path)
                deleted_count += 1
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} already-extracted Mel files.")

    if len(remaining) == 0:
        print("All features already extracted. Nothing to do.")
        return

    WHISPER_INPUT_FRAMES = 3000  # Whisper 固定要求的 Mel 輸入長度
    MAX_AUDIO_FRAMES = 1500     # 論文規格：最大有效音訊 = 15 秒

    # 逐 Batch 提取特徵
    for i in tqdm(range(0, len(remaining), args.batch_size), desc="Extracting"):
        batch_files = remaining[i : i + args.batch_size]
        batch_mels = []
        batch_lengths = []

        for fname in batch_files:
            mel = torch.load(os.path.join(mel_dir, fname), weights_only=True).float()

            # 記錄有效幀數 (截斷至 15 秒上限)
            effective_frames = min(mel.shape[-1], MAX_AUDIO_FRAMES)

            # 截斷至 15 秒
            if mel.shape[-1] > MAX_AUDIO_FRAMES:
                mel = mel[:, :MAX_AUDIO_FRAMES]

            # 補零至 3000 幀 (Whisper 的固定要求)
            if mel.shape[-1] < WHISPER_INPUT_FRAMES:
                pad_len = WHISPER_INPUT_FRAMES - mel.shape[-1]
                mel = F.pad(mel, (0, pad_len))

            batch_mels.append(mel)
            batch_lengths.append(effective_frames // 2)

        # Stack batch: [B, 128, 3000]
        batch_tensor = torch.stack(batch_mels).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            encoder_output = whisper_enc(batch_tensor).last_hidden_state

        # 逐樣本儲存 (只保存有效幀段，以 float16 壓縮空間)
        for j, fname in enumerate(batch_files):
            valid_len = batch_lengths[j]
            feat = encoder_output[j, :valid_len, :].cpu().half()
            torch.save(feat, os.path.join(out_dir, fname))

            # 提取完成後立即刪除對應的源 Mel 檔案，邊跑邊釋放空間
            if args.delete_mel:
                mel_path = os.path.join(mel_dir, fname)
                if os.path.exists(mel_path):
                    os.remove(mel_path)

    print(f"\nExtraction complete! Features saved to: {out_dir}")
    print(f"Total files extracted: {len(remaining)}")

    sample = torch.load(os.path.join(out_dir, remaining[0]), weights_only=True)
    print(f"Sample feature shape: {sample.shape}, dtype: {sample.dtype}")


if __name__ == "__main__":
    main()
