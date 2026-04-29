"""
SAILER V4.0 — 4-Class 面試情緒偵測訓練腳本 (純語音)
策略：載入官方 SAILER Whisper → 凍結一切 → 砍掉 9 類頭 → 接上 4 類新頭 → 只訓練新頭

Usage:
    python train_4class.py --config configs/interview_4class_config.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import json
import os
import sys
import numpy as np
import wandb
from tqdm import tqdm
from transformers import set_seed, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

# 追加 vox-profile-release 路徑以載入官方 WhisperWrapper
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vox-profile-release'))

from src.experiment_tracker import ExperimentTracker
from src.interview_dataset import InterviewEmotionDataset


def collate_fn(batch):
    """
    自定義 collate：將不等長的 raw audio 打包成 list of tensors，
    讓官方 WhisperWrapper 的 forward 自行處理 feature extraction。
    """
    waveforms, labels, lengths = zip(*batch)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    # waveforms 保持為 list of 1D tensors（官方模型 forward 會自行 .detach().cpu().numpy()）
    return list(waveforms), labels, lengths


def main():
    # ==========================================
    # 1. 配置與環境
    # ==========================================
    parser = argparse.ArgumentParser(description="SAILER V4.0 — 4-Class Interview Emotion Training")
    parser.add_argument("--config", type=str, default="configs/interview_4class_config.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tracker = ExperimentTracker(experiment_name=config["model_name"], config=config)
    tracker.save_config(config)

    # ==========================================
    # 2. 載入官方 SAILER Whisper 模型
    # ==========================================
    tracker.logger.info("正在載入官方 SAILER Whisper 模型 (tiantiaf/whisper-large-v3-msp-podcast-emotion)...")
    from src.model.emotion.whisper_emotion import WhisperWrapper
    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion")
    model = model.float()  # 確保 dtype 一致

    # ==========================================
    # 3. 砍頭重接：替換 9 類分類頭為 4 類
    # ==========================================
    num_classes = config["num_classes"]  # 4
    hidden_dim = 256  # 官方模型的 model_seq 輸出維度

    tracker.logger.info(f"替換 emotion_layer: 9 類 → {num_classes} 類 (面試情緒)")

    # 建立新的 4 類分類頭
    model.emotion_layer = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=config.get("dropout_rate", 0.2)),
        nn.Linear(hidden_dim, num_classes),
    )

    # ==========================================
    # 4. 凍結策略：只訓練新的 emotion_layer
    # ==========================================
    # 先凍結一切
    for param in model.parameters():
        param.requires_grad = False

    # 解凍新的 emotion_layer
    for param in model.emotion_layer.parameters():
        param.requires_grad = True

    # 統計可訓練參數量
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    tracker.logger.info(f"可訓練參數: {trainable:,} / 總參數: {total:,} ({trainable/total*100:.2f}%)")

    model = model.to(device)

    # ==========================================
    # 5. 資料集與 DataLoader
    # ==========================================
    data_dir = config["data_dir"]

    train_dataset = InterviewEmotionDataset(data_dir, split="Train")
    val_dataset = InterviewEmotionDataset(data_dir, split="Development")

    # Minority Oversampling: WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # ==========================================
    # 6. Loss / Optimizer / Scheduler
    # ==========================================
    # Class-Weighted CrossEntropyLoss
    class_weights = train_dataset.class_weights.to(device)
    tracker.logger.info(f"Class Weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 只優化可訓練的參數 (emotion_layer)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01)
    )

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0.0
    no_improve_count = 0
    emotion_names = InterviewEmotionDataset.EMOTION_NAMES

    # ==========================================
    # 7. Sanity Check
    # ==========================================
    tracker.logger.info("啟動 Sanity Check...")
    model.eval()
    with torch.no_grad():
        for waveforms, labels, lengths in val_loader:
            # 官方模型的 forward 接受 raw waveform list + lengths
            lengths = lengths.to(device)
            labels = labels.to(device)
            # 官方模型需要 list of numpy arrays
            x_list = waveforms  # 官方模型內部會自行 .detach().cpu().numpy()
            with torch.amp.autocast('cuda'):
                # 只取 emotion predictions (第一個返回值)
                outputs = model(x_list, length=lengths)
                logits = outputs[0]  # [B, 4]
            tracker.logger.info(f"✅ Sanity Check 通過！Output shape: {logits.shape}")
            break

    # ==========================================
    # 8. 主訓練迴圈
    # ==========================================
    for epoch in range(config["epochs"]):
        tracker.logger.info(f"====== [{epoch+1}/{config['epochs']}] 啟動 Epoch 訓練 ======")
        model.train()

        # 凍結層保持 eval 模式（BatchNorm / Dropout 行為正確）
        model.backbone_model.eval()
        model.model_seq.eval()

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (waveforms, labels, lengths) in enumerate(tqdm(train_loader, desc="Training")):
            lengths = lengths.to(device)
            labels = labels.to(device)
            x_list = waveforms  # 官方模型內部會自行 .detach().cpu().numpy()

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    # 前向傳播：官方模型內部會自動做 feature extraction + encoder + model_seq + pooling
                    # 我們需要拿到 pooling 後的特徵 (return_feature=True)
                    outputs = model(x_list, length=lengths, return_feature=True)
                    # outputs: (predicted, features, detailed_predicted, arousal, valence, dominance)
                    features = outputs[1]  # [B, 256] — pooling 後的特徵

                # 只有 emotion_layer 有梯度
                logits = model.emotion_layer(features)  # [B, 4]
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.emotion_layer.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            # Log batch metrics
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }, step=global_step)

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # ==========================================
        # 9. 驗證循環
        # ==========================================
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for waveforms, labels, lengths in tqdm(val_loader, desc="Validation"):
                lengths = lengths.to(device)
                labels = labels.to(device)
                x_list = waveforms  # 官方模型內部會自行 .detach().cpu().numpy()

                with torch.amp.autocast('cuda'):
                    outputs = model(x_list, length=lengths, return_feature=True)
                    features = outputs[1]
                    logits = model.emotion_layer(features)
                    loss = criterion(logits, labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()

        # 驗證報告
        report = f"\n{'='*50}\n"
        report += f" Epoch {epoch+1} 驗證報告 (4-Class Interview Emotion)\n"
        report += f" Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
        report += f" Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Macro F1: {val_macro_f1:.4f}\n"
        report += f"{'-'*50}\n"
        report += classification_report(
            all_labels, all_preds,
            labels=list(range(num_classes)),
            target_names=emotion_names,
            zero_division=0
        )
        report += f"\n{'='*50}"

        tracker.logger.info(report)
        tracker.log_metrics(epoch, avg_train_loss, avg_val_loss, val_acc)

        epoch_end_step = (epoch + 1) * len(train_loader) - 1
        wandb.log({
            "Loss_Train/Loss": avg_train_loss,
            "Loss_Val/Loss": avg_val_loss,
            "Accuracy/Train": train_acc,
            "Accuracy/Validation": val_acc,
            "Macro_F1/Validation": val_macro_f1,
        }, step=epoch_end_step)

        # ==========================================
        # 10. Early Stopping + 模型儲存
        # ==========================================
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            no_improve_count = 0
            # 只存 emotion_layer 的權重（輕量檔案）
            torch.save(
                model.emotion_layer.state_dict(),
                os.path.join(tracker.weights_dir, "best_emotion_head.pth")
            )
            tracker.logger.info(f"🎉 發現更佳模型 (Macro F1: {val_macro_f1:.4f}), 已儲存 emotion_layer。")
        else:
            no_improve_count += 1
            tracker.logger.info(f"模型已連續 {no_improve_count} 次無改善 (上限: {config.get('early_stop_patience', 8)})")

        # 儲存最新 checkpoint
        checkpoint_path = os.path.join(tracker.weights_dir, "checkpoint_latest.pth")
        checkpoint_data = {
            'epoch': epoch,
            'emotion_layer_state_dict': model.emotion_layer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_f1': best_f1,
            'no_improve_count': no_improve_count,
        }
        tmp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint_data, tmp_path)
        os.replace(tmp_path, checkpoint_path)

        # Early Stop 判定
        if no_improve_count >= config.get("early_stop_patience", 8):
            tracker.logger.info(f"已達到早停門檻 ({config.get('early_stop_patience', 8)} 次沒改善)，正式中止訓練。")
            break

    tracker.logger.info(f"\n🏁 訓練完成！最佳 Macro F1: {best_f1:.4f}")
    tracker.close()


if __name__ == "__main__":
    main()
