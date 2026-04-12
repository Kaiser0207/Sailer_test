"""
SAILER 官方 Baseline 評估腳本 (Production Grade)
====================================================
使用 SAILER 官方釋出的 Whisper-Large-V3 預訓練模型 (tiantiaf/whisper-large-v3-msp-podcast-emotion)
在 MSP-Podcast v2.0 的 Test1 / Test2 上進行 8 類情緒分類評估。

【功能特色】
- W&B 完整整合（Config、指標、圖表、Artifacts 全面同步）
- ExperimentTracker 相容（Local 日誌 + TensorBoard + W&B 三路同步）
- Confusion Matrix（Raw / Normalized）自動產生
- Per-Class F1/Precision/Recall 柱狀圖
- 少數類別 Average Precision 與 Min. mAP 追蹤
- 完整的例外處理與清理機制

Usage:
    python evaluate_official_baseline.py --split Test1
    python evaluate_official_baseline.py --split Test2
    python evaluate_official_baseline.py --split Test1 --max_samples 100  # 快速測試
"""

import os
import sys
import json
import argparse
import logging
import datetime
import time

import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    classification_report, f1_score, average_precision_score,
    confusion_matrix, precision_recall_fscore_support
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import wandb

# 加入 vox-profile-release 的路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vox-profile-release'))

# ==========================================
# 常數定義
# ==========================================
DATA_DIR = "/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data"
AUDIO_DIR = os.path.join(DATA_DIR, "Audios")
CONSENSUS_PATH = os.path.join(DATA_DIR, "Labels", "labels_consensus.csv")

# 官方模型的 9 類標籤 (含 Other)
OFFICIAL_9_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise', 'Other']

# 我們使用的 8 類標籤 (不含 Other)
EMOTION_8_MAP = {
    'N': 'Neutral', 'A': 'Angry', 'S': 'Sad', 'H': 'Happy',
    'F': 'Fear', 'D': 'Disgust', 'U': 'Surprise', 'C': 'Contempt'
}

# 官方 9 類 index -> 我們的 8 類 index 映射
OFFICIAL_TO_OUR_MAP = {
    0: 1,   # Anger -> Angry
    1: 7,   # Contempt -> Contempt
    2: 5,   # Disgust -> Disgust
    3: 4,   # Fear -> Fear
    4: 3,   # Happiness -> Happy
    5: 0,   # Neutral -> Neutral
    6: 2,   # Sadness -> Sad
    7: 6,   # Surprise -> Surprise
    8: -1,  # Other -> 從 8 類中選機率最高的
}

OUR_8_LABELS = ['Neutral', 'Angry', 'Sad', 'Happy', 'Fear', 'Disgust', 'Surprise', 'Contempt']
MINORITY_CLASSES = {4: 'Fear', 5: 'Disgust', 6: 'Surprise', 7: 'Contempt'}


# ==========================================
# 日誌系統設定
# ==========================================
def setup_logger(log_dir, name="eval_baseline"):
    """建立專業級雙路日誌系統（螢幕 + 檔案）"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_file = os.path.join(log_dir, "eval.log")

        fh = logging.FileHandler(log_file, encoding='utf-8')
        ch = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.propagate = False

    return logger


# ==========================================
# W&B 初始化
# ==========================================
def init_wandb(exp_name, eval_config, log_dir):
    """初始化 W&B Run，支援斷點續傳"""
    wandb_id_file = os.path.join(log_dir, "wandb_id.txt")
    run_id = None

    if os.path.exists(wandb_id_file):
        with open(wandb_id_file, "r") as f:
            run_id = f.read().strip()

    wandb.init(
        project="SAILER_Emotion_Recognition",
        name=exp_name,
        id=run_id,
        resume="allow" if run_id else None,
        config=eval_config,
        tags=["evaluation", "official_baseline", "speech_only"],
        notes="使用 SAILER 官方 Whisper-Large-V3 預訓練模型進行 8 類情緒分類測試",
    )

    if not run_id:
        with open(wandb_id_file, "w") as f:
            f.write(wandb.run.id)


# ==========================================
# 模型載入
# ==========================================
def load_official_whisper_model(device, logger):
    """載入 SAILER 官方 Whisper 模型（含相容性修復）"""
    from src.model.emotion.whisper_emotion import WhisperWrapper

    logger.info("正在從 HuggingFace 載入官方 SAILER Whisper 模型 (tiantiaf/whisper-large-v3-msp-podcast-emotion)...")
    model = WhisperWrapper.from_pretrained(
        "tiantiaf/whisper-large-v3-msp-podcast-emotion"
    ).float().to(device)
    model.eval()

    # 統計模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"官方模型載入成功！總參數量: {total_params:,} | 可訓練: {trainable_params:,}")

    return model


# ==========================================
# 資料載入
# ==========================================
def load_test_data(split, logger):
    """載入 Test1 或 Test2 的資料索引 (只保留 8 類有共識的樣本)"""
    df = pd.read_csv(CONSENSUS_PATH)
    df_split = df[df['Split_Set'] == split].copy()

    total_before_filter = len(df_split)

    # 只保留 8 種主情緒
    valid_emo = list(EMOTION_8_MAP.keys())
    df_split = df_split[df_split['EmoClass'].isin(valid_emo)]

    records = []
    missing_files = 0
    for _, row in df_split.iterrows():
        filename = row['FileName']
        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            missing_files += 1
            continue

        emo_label = EMOTION_8_MAP[row['EmoClass']]
        emo_idx = OUR_8_LABELS.index(emo_label)

        records.append({
            'filename': filename,
            'audio_path': audio_path,
            'emo_label': emo_label,
            'emo_idx': emo_idx,
        })

    logger.info(f"[{split}] 總樣本: {total_before_filter} → 篩選 8 類: {len(df_split)} → 有效音檔: {len(records)} (缺失: {missing_files})")

    # 統計各類別分布
    class_dist = {}
    for r in records:
        class_dist[r['emo_label']] = class_dist.get(r['emo_label'], 0) + 1
    logger.info(f"[{split}] 類別分布: {json.dumps(class_dist, indent=2)}")

    return records


# ==========================================
# 推理核心
# ==========================================
def evaluate_speech_only(model, records, device, logger, max_audio_sec=15):
    """
    使用官方 WhisperWrapper 進行純語音推理。
    紀錄每一筆的推理結果，支援錯誤恢復。
    """
    all_preds = []
    all_labels = []
    all_probs = []
    skipped_short = 0
    skipped_error = 0

    max_audio_length = max_audio_sec * 16000

    logger.info(f"🎙️ 開始 Speech-Only 推理 ({len(records)} 筆)...")
    start_time = time.time()

    for idx, record in enumerate(tqdm(records, desc="Evaluating")):
        try:
            # 載入音檔
            audio, sr = sf.read(record['audio_path'])

            # 確保 mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # 確保 16kHz
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # 截斷至 15 秒
            audio = audio[:max_audio_length]

            # 跳過太短的音檔 (< 1 秒 → 不可靠)
            if len(audio) < 16000:
                skipped_short += 1
                continue

            data = torch.tensor(audio).unsqueeze(0).float().to(device)

            with torch.no_grad():
                logits, embedding, _, _, _, _ = model(data, return_feature=True)

            # 取得 9 類機率 → 映射到 8 類
            probs_9 = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_9 = int(torch.argmax(logits, dim=1).cpu().item())

            # 重新排列到我們的 8 類順序
            reordered = np.zeros(8)
            for official_idx, our_idx in OFFICIAL_TO_OUR_MAP.items():
                if our_idx != -1:
                    reordered[our_idx] = probs_9[official_idx]
            reordered = reordered / (reordered.sum() + 1e-8)

            # 最終預測 = 8 類中機率最高的
            pred_8 = int(np.argmax(reordered))

            all_preds.append(pred_8)
            all_labels.append(record['emo_idx'])
            all_probs.append(reordered)

            # 每 5000 筆印一次中間統計
            if (idx + 1) % 5000 == 0:
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed
                interim_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                logger.info(
                    f"  進度: {idx+1}/{len(records)} | "
                    f"速度: {speed:.1f} it/s | "
                    f"暫時 Macro F1: {interim_f1:.4f} | "
                    f"跳過(短): {skipped_short} | 錯誤: {skipped_error}"
                )

        except Exception as e:
            skipped_error += 1
            if skipped_error <= 5:  # 只印前 5 個錯誤
                logger.warning(f"推理失敗 [{record['filename']}]: {e}")
            continue

    elapsed = time.time() - start_time
    logger.info(
        f"推理完成！耗時: {elapsed:.1f}s | "
        f"成功: {len(all_preds)} | 跳過(短): {skipped_short} | 錯誤: {skipped_error}"
    )

    return all_preds, all_labels, all_probs


# ==========================================
# 指標計算
# ==========================================
def compute_metrics(all_preds, all_labels, all_probs, split_name, logger):
    """計算全套分類指標並記錄到日誌"""
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 1. Classification Report (文字版)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(8)),
        target_names=OUR_8_LABELS,
        digits=4,
        zero_division=0
    )

    # 2. Macro / Weighted / Micro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # 3. Accuracy
    accuracy = (all_preds == all_labels).mean()

    # 4. Per-class F1, Precision, Recall
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(8)), zero_division=0
    )

    # 5. Min mAP (少數類別的平均精確度)
    all_probs_arr = np.array(all_probs) if len(all_probs) > 0 else np.zeros((0, 8))
    ap_per_class = {}
    for c in range(8):
        y_true_c = (all_labels == c).astype(int)
        if y_true_c.sum() > 0 and len(all_probs_arr) > 0:
            ap = average_precision_score(y_true_c, all_probs_arr[:, c])
            ap_per_class[OUR_8_LABELS[c]] = ap

    min_map_classes = [ap_per_class.get(MINORITY_CLASSES[c], 0.0) for c in MINORITY_CLASSES]
    min_map = np.mean(min_map_classes) if min_map_classes else 0.0

    # 印出完整報告
    header = f"\n{'='*60}"
    header += f"\n {split_name} 評估結果 (官方 SAILER Whisper - Speech Only)"
    header += f"\n Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f} | Accuracy: {accuracy:.4f}"
    header += f"\n Min. mAP (少數類別): {min_map:.4f}"
    header += f"\n 總評估樣本數: {len(all_preds)}"
    header += f"\n{'='*60}"
    logger.info(header)
    logger.info(f"\n{report}")
    logger.info(f"{'='*60}")

    # 各類別 AP
    if ap_per_class:
        ap_str = " | ".join([f"{k}: {v:.4f}" for k, v in ap_per_class.items()])
        logger.info(f"各類別 Average Precision: {ap_str}")

    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': accuracy,
        'min_map': min_map,
        'report': report,
        'precisions': precisions,
        'recalls': recalls,
        'f1s': f1s,
        'supports': supports,
        'ap_per_class': ap_per_class,
    }


# ==========================================
# 可視化圖表產生
# ==========================================
def plot_confusion_matrix(all_preds, all_labels, split_name, output_dir, logger):
    """產生 Confusion Matrix (Raw + Normalized) 並上傳到 W&B"""
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(8)))

    # ====== 1. Raw Counts ======
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=OUR_8_LABELS, yticklabels=OUR_8_LABELS,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'Confusion Matrix — {split_name}\n(Raw Counts)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f'{split_name}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    wandb.log({f"{split_name}/confusion_matrix_raw": wandb.Image(fig)})
    plt.close()
    logger.info(f"Confusion Matrix (Raw) 已保存至: {cm_path}")

    # ====== 2. Normalized (Row = Recall per class) ======
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd',
                xticklabels=OUR_8_LABELS, yticklabels=OUR_8_LABELS,
                linewidths=0.5, vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'Normalized Confusion Matrix — {split_name}\n(Row-Normalized: Recall per Class)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_norm_path = os.path.join(output_dir, f'{split_name}_confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=150)
    wandb.log({f"{split_name}/confusion_matrix_normalized": wandb.Image(fig)})
    plt.close()
    logger.info(f"Normalized Confusion Matrix 已保存至: {cm_norm_path}")

    return cm


def plot_per_class_metrics(metrics, split_name, output_dir, logger):
    """產生 Per-Class F1/Precision/Recall 柱狀圖並上傳到 W&B"""
    precisions = metrics['precisions']
    recalls = metrics['recalls']
    f1s = metrics['f1s']

    x = np.arange(len(OUR_8_LABELS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#4C72B0', alpha=0.85)
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#DD8452', alpha=0.85)
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#55A868', alpha=0.85)

    ax.set_xlabel('Emotion Category', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Per-Class Performance — {split_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(OUR_8_LABELS, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 在每個柱子上方標記數值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    bar_path = os.path.join(output_dir, f'{split_name}_per_class_metrics.png')
    plt.savefig(bar_path, dpi=150)
    wandb.log({f"{split_name}/per_class_metrics": wandb.Image(fig)})
    plt.close()
    logger.info(f"Per-Class Metrics 柱狀圖已保存至: {bar_path}")


def plot_class_distribution(all_labels, split_name, output_dir, logger):
    """產生真實標籤分布圖"""
    unique, counts = np.unique(all_labels, return_counts=True)
    labels = [OUR_8_LABELS[i] for i in unique]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("husl", len(labels))
    bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Emotion Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Class Distribution — {split_name}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.annotate(f'{count:,}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    dist_path = os.path.join(output_dir, f'{split_name}_class_distribution.png')
    plt.savefig(dist_path, dpi=150)
    wandb.log({f"{split_name}/class_distribution": wandb.Image(fig)})
    plt.close()
    logger.info(f"Class Distribution 已保存至: {dist_path}")


# ==========================================
# W&B 指標上傳
# ==========================================
def log_to_wandb(metrics, split_name, all_preds, all_labels):
    """將所有指標一次性上傳到 W&B"""
    # 主要指標
    log_dict = {
        f"{split_name}/macro_f1": metrics['macro_f1'],
        f"{split_name}/weighted_f1": metrics['weighted_f1'],
        f"{split_name}/accuracy": metrics['accuracy'],
        f"{split_name}/min_mAP": metrics['min_map'],
    }

    # 各類別 F1
    for i, label in enumerate(OUR_8_LABELS):
        log_dict[f"{split_name}/f1_{label}"] = metrics['f1s'][i]
        log_dict[f"{split_name}/precision_{label}"] = metrics['precisions'][i]
        log_dict[f"{split_name}/recall_{label}"] = metrics['recalls'][i]

    # 各類別 AP
    for label, ap in metrics['ap_per_class'].items():
        log_dict[f"{split_name}/AP_{label}"] = ap

    # 少數類別 AP
    for c_idx, c_name in MINORITY_CLASSES.items():
        if c_name in metrics['ap_per_class']:
            log_dict[f"{split_name}/minority_AP_{c_name}"] = metrics['ap_per_class'][c_name]

    wandb.log(log_dict)

    # W&B 原生 Confusion Matrix (互動式)
    wandb.log({
        f"{split_name}/wandb_confusion_matrix": wandb.plot.confusion_matrix(
            y_true=list(all_labels),
            preds=list(all_preds),
            class_names=OUR_8_LABELS,
            title=f"Confusion Matrix — {split_name}"
        )
    })

    # 摘要指標 (顯示在 W&B Run 總覽)
    wandb.run.summary[f"{split_name}_macro_f1"] = metrics['macro_f1']
    wandb.run.summary[f"{split_name}_weighted_f1"] = metrics['weighted_f1']
    wandb.run.summary[f"{split_name}_accuracy"] = metrics['accuracy']
    wandb.run.summary[f"{split_name}_min_mAP"] = metrics['min_map']


# ==========================================
# 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="SAILER 官方 Baseline 評估 (Production Grade)")
    parser.add_argument('--split', type=str, default='Test1', choices=['Test1', 'Test2'],
                        help='測試集劃分')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='限制最大評估樣本數 (用於快速測試)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='停用 W&B (離線模式)')
    args = parser.parse_args()

    # ==========================================
    # 1. 初始化實驗環境
    # ==========================================
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"Official_Baseline_{args.split}"
    exp_dir = os.path.join("experiments", "official_baseline")
    plots_dir = os.path.join(exp_dir, "plots")
    logs_dir = os.path.join(exp_dir, "logs")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    logger = setup_logger(logs_dir, name=f"eval_{args.split}")
    logger.info(f"{'='*60}")
    logger.info(f"  SAILER 官方 Baseline 評估啟動")
    logger.info(f"  實驗名稱: {exp_name}")
    logger.info(f"  測試分割: {args.split}")
    logger.info(f"  時間戳: {now}")
    logger.info(f"{'='*60}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"使用裝置: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU 型號: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ==========================================
    # 2. 評估配置 (Config)
    # ==========================================
    eval_config = {
        "model_source": "tiantiaf/whisper-large-v3-msp-podcast-emotion",
        "model_type": "SAILER_Official_Whisper_Large_V3",
        "modality": "speech_only",
        "split": args.split,
        "num_classes": 8,
        "max_audio_sec": 15,
        "min_audio_sec": 1,
        "sampling_rate": 16000,
        "dataset": "MSP-Podcast_v2.0",
        "class_labels": OUR_8_LABELS,
        "max_samples": args.max_samples,
    }
    logger.info(f"評估配置: {json.dumps(eval_config, indent=2, ensure_ascii=False)}")

    # 保存 Config 到本地
    config_path = os.path.join(exp_dir, f"eval_config_{args.split}.json")
    with open(config_path, 'w') as f:
        json.dump(eval_config, f, indent=4, ensure_ascii=False)

    # ==========================================
    # 3. 初始化 W&B
    # ==========================================
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("W&B 已停用 (離線模式)")

    init_wandb(exp_name, eval_config, logs_dir)
    logger.info(f"W&B Run ID: {wandb.run.id}")
    logger.info(f"W&B Run URL: {wandb.run.get_url()}")

    # ==========================================
    # 4. 載入資料
    # ==========================================
    records = load_test_data(args.split, logger)

    if args.max_samples:
        records = records[:args.max_samples]
        logger.info(f"快速測試模式：只評估前 {args.max_samples} 筆")

    # ==========================================
    # 5. 載入模型
    # ==========================================
    model = load_official_whisper_model(device, logger)

    # ==========================================
    # 6. Sanity Check (防呆乾跑)
    # ==========================================
    logger.info("🔍 啟動 Sanity Check...")
    try:
        test_audio = torch.randn(1, 32000).float().to(device)  # 2 秒假音檔
        with torch.no_grad():
            logits, _, _, _, _, _ = model(test_audio, return_feature=True)
        assert logits.shape == (1, 9), f"輸出 shape 不正確: {logits.shape}"
        logger.info(f"Sanity Check 通過！模型輸出 shape: {logits.shape}")
    except Exception as e:
        logger.error(f"Sanity Check 失敗: {e}")
        wandb.finish(exit_code=1)
        return

    # ==========================================
    # 7. 執行推理
    # ==========================================
    preds, labels, probs = evaluate_speech_only(model, records, device, logger)

    if len(preds) == 0:
        logger.error("沒有任何成功的推理結果！請檢查音檔路徑和模型。")
        wandb.finish(exit_code=1)
        return

    # ==========================================
    # 8. 計算指標
    # ==========================================
    metrics = compute_metrics(preds, labels, probs, f"{args.split} (speech_only)", logger)

    # ==========================================
    # 9. 產生可視化圖表
    # ==========================================
    logger.info("正在產生可視化圖表...")
    split_key = f"{args.split}_speech_only"

    plot_class_distribution(np.array(labels), split_key, plots_dir, logger)
    plot_confusion_matrix(np.array(preds), np.array(labels), split_key, plots_dir, logger)
    plot_per_class_metrics(metrics, split_key, plots_dir, logger)

    # ==========================================
    # 10. 上傳到 W&B
    # ==========================================
    logger.info("正在同步指標到 W&B...")
    log_to_wandb(metrics, split_key, np.array(preds), np.array(labels))

    # ==========================================
    # 11. 保存完整結果報告 (本地)
    # ==========================================
    result_path = os.path.join(exp_dir, f"{args.split}_speech_only_results.log")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"SAILER Official Baseline Evaluation Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: {eval_config['model_source']}\n")
        f.write(f"Modality: {eval_config['modality']}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Timestamp: {now}\n")
        f.write(f"Device: {device}\n")
        f.write(f"W&B Run ID: {wandb.run.id}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Macro F1:    {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n")
        f.write(f"Accuracy:    {metrics['accuracy']:.4f}\n")
        f.write(f"Min. mAP:    {metrics['min_map']:.4f}\n\n")
        f.write(f"Classification Report:\n")
        f.write(f"{metrics['report']}\n\n")
        f.write(f"Per-Class Average Precision:\n")
        for label, ap in metrics['ap_per_class'].items():
            f.write(f"  {label}: {ap:.4f}\n")

    logger.info(f"結果報告已保存至: {result_path}")

    # ==========================================
    # 12. 上傳 Artifact (完整結果打包)
    # ==========================================
    artifact = wandb.Artifact(
        name=f"eval_{args.split}_speech_only",
        type="evaluation",
        description=f"官方 SAILER Whisper 模型在 MSP-Podcast {args.split} 上的完整評估結果"
    )
    artifact.add_dir(exp_dir)
    wandb.log_artifact(artifact)
    logger.info(f"W&B Artifact 已上傳: eval_{args.split}_speech_only")

    # ==========================================
    # 13. 清理收尾
    # ==========================================
    wandb.finish()
    logger.info(f"\n所有評估完成！請查看:")
    logger.info(f"   本地結果: {exp_dir}/")
    logger.info(f"   W&B: {wandb.run.get_url() if wandb.run else 'N/A'}")


if __name__ == "__main__":
    main()
