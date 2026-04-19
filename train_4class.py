"""
SAILER V4.0 — 4-Class 面試情緒偵測訓練腳本 (純語音)
策略：Phase 3 — Focal Loss + 解凍 Whisper 最後 1 層 + model_seq + emotion_layer

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


class FocalLoss(nn.Module):
    """Focal Loss: 抑制簡單樣本的梯度，聚焦困難樣本 (如 Fear)。
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # 模型對正確類別的信心度
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


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
    # 4. 凍結策略：Phase 3 — 訓練 Whisper最後1層 + model_seq + emotion_layer
    # ==========================================
    # 先凍結一切
    for param in model.parameters():
        param.requires_grad = False

    # 解凍 emotion_layer (新的 4 類頭)
    for param in model.emotion_layer.parameters():
        param.requires_grad = True

    # 解凍 model_seq (3 層 Conv1d，讓 256 維特徵空間重新優化)
    for param in model.model_seq.parameters():
        param.requires_grad = True

    # 🔥 Phase 3: 解凍 Whisper Encoder 最後 1 層
    whisper_last_layer = model.backbone_model.encoder.layers[-1]
    for param in whisper_last_layer.parameters():
        param.requires_grad = True
    num_encoder_layers = len(model.backbone_model.encoder.layers)
    tracker.logger.info(f"🔥 Phase 3: 解凍 Whisper Encoder 最後 1 層 (Layer {num_encoder_layers})")

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

    # 不使用 WeightedRandomSampler（配合 Class Weights 已足夠，雙重策略會導致少數類過度膨脹）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
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
    # 🔥 Phase 3: Focal Loss (取代 CrossEntropyLoss)
    class_weights = train_dataset.class_weights.to(device)
    focal_gamma = config.get("focal_gamma", 2.0)
    tracker.logger.info(f"Class Weights: {class_weights.tolist()}")
    tracker.logger.info(f"使用 Focal Loss (gamma={focal_gamma})")
    criterion = FocalLoss(weight=class_weights, gamma=focal_gamma)

    # 🔥 Phase 3: 三層 Discriminative LR
    whisper_lr = config.get("whisper_lr", 1e-5)
    optimizer = torch.optim.AdamW([
        {'params': whisper_last_layer.parameters(), 'lr': whisper_lr},                  # Whisper最後層: 1e-5
        {'params': model.model_seq.parameters(), 'lr': config["learning_rate"] * 0.1},  # Conv1d: 1e-4
        {'params': model.emotion_layer.parameters(), 'lr': config["learning_rate"]},    # Head: 1e-3
    ], weight_decay=config.get("weight_decay", 0.01))
    tracker.logger.info(f"Discriminative LR: whisper_last={whisper_lr}, model_seq={config['learning_rate']*0.1}, emotion_layer={config['learning_rate']}")

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

        # Whisper 保持 eval 模式（BatchNorm / Dropout 行為正確）
        # 但最後一層的 LayerNorm 需要 train 模式以更新 running stats
        model.backbone_model.eval()
        whisper_last_layer.train()

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (waveforms, labels, lengths) in enumerate(tqdm(train_loader, desc="Training")):
            lengths = lengths.to(device)
            labels = labels.to(device)
            x_list = waveforms  # 官方模型內部會自行 .detach().cpu().numpy()

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # Phase 3: 梯度流過 Whisper最後層 → model_seq → emotion_layer
                outputs = model(x_list, length=lengths, return_feature=True)
                # outputs: (predicted, features, detailed_predicted, arousal, valence, dominance)
                features = outputs[1]  # [B, 256] — model_seq 處理後的特徵

                # 三層都有梯度：whisper_last_layer + model_seq + emotion_layer
                logits = model.emotion_layer(features)  # [B, 4]
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            trainable_params = (list(whisper_last_layer.parameters()) +
                               list(model.model_seq.parameters()) +
                               list(model.emotion_layer.parameters()))
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
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
            # 儲存所有可訓練模組的權重
            torch.save(
                model.emotion_layer.state_dict(),
                os.path.join(tracker.weights_dir, "best_emotion_head.pth")
            )
            torch.save(
                model.model_seq.state_dict(),
                os.path.join(tracker.weights_dir, "best_model_seq.pth")
            )
            torch.save(
                whisper_last_layer.state_dict(),
                os.path.join(tracker.weights_dir, "best_whisper_last_layer.pth")
            )
            tracker.logger.info(f"🎉 發現更佳模型 (Macro F1: {val_macro_f1:.4f}), 已儲存 emotion_layer + model_seq + whisper_last_layer。")
        else:
            no_improve_count += 1
            tracker.logger.info(f"模型已連續 {no_improve_count} 次無改善 (上限: {config.get('early_stop_patience', 8)})")

        # 儲存最新 checkpoint
        checkpoint_path = os.path.join(tracker.weights_dir, "checkpoint_latest.pth")
        checkpoint_data = {
            'epoch': epoch,
            'emotion_layer_state_dict': model.emotion_layer.state_dict(),
            'model_seq_state_dict': model.model_seq.state_dict(),
            'whisper_last_layer_state_dict': whisper_last_layer.state_dict(),
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
