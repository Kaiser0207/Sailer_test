import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import json
from transformers import RobertaTokenizer, RobertaModel, WhisperModel, set_seed, get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, classification_report
import numpy as np
import os
import wandb
import random

from src.experiment_tracker import ExperimentTracker

from src.msp_dataset import MSP_Podcast_Dataset
from src.sailer_model import SAILER_Model

def get_clean_state_dict(model):
    state_dict = model.state_dict()
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

def main():
    """
    SAILER 系統核心訓練迴圈 
    負責組裝巨型預訓練模型 (Whisper, RoBERTa)、載入特徵與標籤，執行多任務聯合訓練，並追蹤指標
    """
    # ==========================================
    # 1. 訓練參數與基礎環境配置 (Configuration & Setup)
    # ==========================================
    parser = argparse.ArgumentParser(description="SAILER Training Script")
    parser.add_argument("--config", type=str, default="configs/default_config.json", help="Path to configuration JSON file")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    args = parser.parse_args()

    # 讀取外部 JSON Config 檔案
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 採用正規套件鎖定一切亂數，確保實驗結果絕對可重現 (Reproducibility)
    set_seed(config.get("seed", 42))

    # 2. 實驗追蹤器
    resume_dir = None
    if args.resume:
        resume_dir = ExperimentTracker.find_latest_experiment(config["model_name"])
    
    tracker = ExperimentTracker(experiment_name=config["model_name"], resume_dir=resume_dir, config=config)
    tracker.save_config(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = config["data_dir"]
    use_cached = config.get("use_cached_features", True)

    # ==========================================
    # 2. 載入預訓練模型架構 (Load Pre-trained Encoders)
    # ==========================================
    r_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    # 載入 Whisper 編碼器 (僅在 V2 模式時需要，V3 已預提取特徵則跳過)
    if use_cached:
        whisper_enc = None
        print("[V3 Mode] Whisper Encoder 已跳過載入，使用預提取特徵。")
    else:
        whisper_enc = WhisperModel.from_pretrained(
            "openai/whisper-large-v3"
        ).encoder.to(device)
        whisper_enc.eval()
        for p in whisper_enc.parameters():
            p.requires_grad = False
    
    roberta_model = RobertaModel.from_pretrained("roberta-large").to(device)

    # RoBERTa 設為 eval 並凍結權重
    roberta_model.eval()
    for p in roberta_model.parameters():
        p.requires_grad = False

    # ==========================================
    # 3. 準備資料集加載器 (DataLoader Pipeline)
    # Train: 套用資料增強 (Audio Mixing) 與動態過採樣，增加資料多樣性保護少數群體
    # Validation: 純淨資料供給，嚴格依賴自然資料分佈作為評估基準
    # ==========================================
    train_dataset = MSP_Podcast_Dataset(data_dir, split="Train", roberta_tokenizer=r_tokenizer, apply_aug=True, use_cached_features=use_cached)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    val_dataset = MSP_Podcast_Dataset(data_dir, split="Development", roberta_tokenizer=r_tokenizer, apply_aug=False, use_cached_features=use_cached)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    model = SAILER_Model(
        num_classes=config["num_classes"],
        secondary_class_num=config["secondary_class_num"],
        dropout_rate=config.get("dropout_rate", 0.2)
    ).to(device)

    # ==========================================
    # 4.1 定義訓練組件 (Criterion, Optimizer, Scheduler)
    # ==========================================
    criterion_primary = nn.KLDivLoss(reduction='batchmean')
    criterion_secondary = nn.KLDivLoss(reduction='batchmean')
    criterion_avd = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config.get("weight_decay", 1e-4)
    )
    
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    scaler = torch.amp.GradScaler('cuda')
    start_epoch = 0
    best_f1 = 0.0
    best_min_map = 0.0
    no_improve_count = 0 # ✨ 早停計數器

    # ==========================================
    # 4.2 執行斷點續傳載入 (Resume Checkpoint)
    # ==========================================
    checkpoint_path = os.path.join(tracker.weights_dir, "checkpoint_latest.pth")
    if args.resume and os.path.exists(checkpoint_path):
        tracker.logger.info(f"正在從斷點載入狀態: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 載入模型權重
        model.load_state_dict(checkpoint['model_state_dict']) # 這裡存的時候已經清理過了，直接載入即可
        
        # 重點：使用記錄的 total_steps 重建 scheduler
        saved_total_steps = checkpoint.get('total_steps', total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(saved_total_steps * config.get("warmup_ratio", 0.1)),
            num_training_steps=saved_total_steps
        )

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        best_min_map = checkpoint.get('best_min_map', 0.0)
        no_improve_count = checkpoint.get('no_improve_count', 0) # 恢復計數器
        tracker.logger.info(f"成功恢復進度！將從 Epoch {start_epoch + 1} 開始訓練 (當前最佳 F1: {best_f1:.4f}, 忍耐次數: {no_improve_count})")

    # ==========================================
    # 4.3 啟動編譯優化 (torch.compile)
    # ==========================================
    try:
        print("正在啟動編譯優化 (torch.compile)...")
        model = torch.compile(model)
        print("編譯成功！")
    except Exception as e:
        print(f"編譯失敗，將使用一般動態圖模式訓練。錯誤: {e}")

    # ==========================================
    # 4.5 Sanity Check (防呆乾跑驗證機制)
    # ==========================================
    tracker.logger.info("啟動 Sanity Check (抽取一筆驗證資料預跑)...")
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            w_feat, t_ids, t_mask, label_dists, sec_dists, avd_targets, lengths = [b.to(device) for b in batch]
            with torch.amp.autocast('cuda'):
                if use_cached:
                    # V3: w_feat 已是 Whisper Encoder 輸出 [B, 1280, 750]，直接遞給 speech_conv
                    w_seq = w_feat.transpose(1, 2)  # [B, 750, 1280]
                else:
                    # V2: w_feat 是 Mel [B, 128, 3000]，需過 Whisper Encoder
                    w_seq = whisper_enc(w_feat).last_hidden_state
                roberta_out = roberta_model(input_ids=t_ids, attention_mask=t_mask, output_hidden_states=True)
                # V2 模式下 lengths 是 Mel 幀數，需除以 2 對齊 Whisper Encoder stride
                enc_lengths = lengths if use_cached else lengths // 2
                model(w_seq, roberta_out.hidden_states, t_mask, enc_lengths)
            break
    tracker.logger.info("Sanity Check 通過！網路無 Shape Mismatch，正式進入訓練迴圈！")

    # ==========================================
    # 5. 主訓練迴圈 (Main Epoch Execution)
    # ==========================================
    for epoch in range(start_epoch, config["epochs"]):
        tracker.logger.info(f"====== [{epoch+1}/{config['epochs']}] 啟動 Epoch 訓練 ======")
        model.train()
        total_train_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # 解離並裝載這包 Batch 產生的 7 個輸入張量至 GPU
            w_feat, t_ids, t_mask, label_dists, sec_dists, avd_targets, lengths = [b.to(device) for b in batch]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # ==== 階段一：預訓練前端提取 (No Grad Context) ====
                with torch.no_grad():
                    if use_cached:
                        # V3: w_feat 已是 Encoder 輸出 [B, 1280, 750]
                        w_seq = w_feat.transpose(1, 2)  # [B, 750, 1280]
                    else:
                        # V2: 即時過 Whisper Encoder
                        w_seq = whisper_enc(w_feat).last_hidden_state
                    roberta_out = roberta_model(input_ids=t_ids, attention_mask=t_mask, output_hidden_states=True)
                    t_hidden_states = roberta_out.hidden_states  
                
                # ==== 階段二：SAILER 特徵融合網絡 (Train Context) ====
                enc_lengths = lengths if use_cached else lengths // 2
                primary_logits, secondary_logits, arousal, valence, dominance = model(
                    w_seq, t_hidden_states, t_mask, enc_lengths
                )

                # 計算 5 大任務的 Loss 差異
                loss_primary = criterion_primary(F.log_softmax(primary_logits, dim=-1), label_dists)
                loss_secondary = criterion_secondary(F.log_softmax(secondary_logits, dim=-1), sec_dists)
                avd_pred = torch.cat([arousal, valence, dominance], dim=-1) 
                loss_avd = criterion_avd(avd_pred, avd_targets)
                
                # Unweighted Sum 無權重總和
                loss = loss_primary + loss_secondary + loss_avd

            # ==== 階段三：傳播梯度更新 ====
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 更新排程器 (Scheduler Step)
            scheduler.step()

            # 使用 global_step 統一 W&B 步數，避免圖表錯位
            global_step = epoch * len(train_loader) + batch_idx
            wandb.log({
                "batch_loss": loss.item(),
                "batch_primary_loss": loss_primary.item(),
                "batch_secondary_loss": loss_secondary.item(),
                "batch_avd_loss": loss_avd.item(),
            }, step=global_step)

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ==========================================
        # 6. 階段性驗證循環 (Validation Inference & Metrics)
        # 用於衡量模型在未見過資料 (Dev Set) 中的表現，嚴防Overfitting
        # ==========================================
        model.eval()
        total_val_loss = 0
        all_preds, all_labels, all_preds_probs = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                w_feat, t_ids, t_mask, label_dists, sec_dists, avd_targets, lengths = [b.to(device) for b in batch]

                with torch.amp.autocast('cuda'):
                    if use_cached:
                        w_seq = w_feat.transpose(1, 2)
                    else:
                        w_seq = whisper_enc(w_feat).last_hidden_state
                    roberta_out = roberta_model(input_ids=t_ids, attention_mask=t_mask, output_hidden_states=True)
                    t_hidden_states = roberta_out.hidden_states

                    primary_logits, secondary_logits, arousal, valence, dominance = model(
                        w_seq, t_hidden_states, t_mask, lengths if use_cached else lengths // 2
                    )
                    
                    val_loss_primary = criterion_primary(F.log_softmax(primary_logits, dim=-1), label_dists)
                    val_loss_secondary = criterion_secondary(F.log_softmax(secondary_logits, dim=-1), sec_dists)
                    avd_pred = torch.cat([arousal, valence, dominance], dim=-1)
                    val_loss_avd = criterion_avd(avd_pred, avd_targets)
                    val_loss = val_loss_primary + val_loss_secondary + val_loss_avd
                    total_val_loss += val_loss.item()

                probs = F.softmax(primary_logits, dim=-1)
                _, predicted = torch.max(primary_logits, 1)
                _, true_labels = torch.max(label_dists, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(true_labels.cpu().numpy())
                all_preds_probs.extend(probs.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        
        # 驗證整體精準指標
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()

        emotion_names = ['Neutral', 'Angry', 'Sad', 'Happy', 'Fear', 'Disgust', 'Surprise', 'Contempt']
        minority_classes = [4, 5, 6, 7]
        minority_labels = {4: "Fear", 5: "Disgust", 6: "Surprise", 7: "Contempt"}
        min_aps = []
        
        log_dict = {
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss
        }

        # 針對極端少數類別單獨計算平均精度 (Average Precision, AP)。
        # 在高度失衡的資料集中，比起單看 F1 這個指標能在微觀保護力上給予更清晰的反饋。
        for c in minority_classes:
            y_true_c = (np.array(all_labels) == c).astype(int)
            y_score_c = np.array(all_preds_probs)[:, c]
            if np.sum(y_true_c) > 0:
                ap = average_precision_score(y_true_c, y_score_c)
                min_aps.append(ap)
                log_dict[f"val_AP_{minority_labels[c]}"] = ap
                
        min_map = np.mean(min_aps) if min_aps else 0.0
        log_dict["val_min_mAP"] = min_map

        # 統一交給 Logging 紀錄
        report = f"\n{'='*45}\n"
        report += f" Epoch {epoch+1} 驗證報告 (Validation Report)\n"
        report += f" Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Min. mAP: {min_map:.4f}\n"
        report += f"{'-'*45}\n"
        report += classification_report(all_labels, all_preds, labels=list(range(8)), target_names=emotion_names, zero_division=0)
        report += f"\n{'='*45}"
        
        tracker.logger.info(report)

        tracker.log_metrics(epoch, avg_train_loss, avg_val_loss, val_acc)
        epoch_end_step = (epoch + 1) * len(train_loader) - 1
        wandb.log(log_dict, step=epoch_end_step)

        # 7. 早停判定與模型備份 (雙指標監控)
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            no_improve_count = 0
            torch.save(get_clean_state_dict(model), os.path.join(tracker.weights_dir, "best_model_f1.pth"))
            tracker.logger.info(f"發現更佳模型 (F1: {val_macro_f1:.4f}), 已儲存。")
        elif min_map > best_min_map:
            no_improve_count = 0 
            tracker.logger.info(f"F1 未進進步，但 Min mAP 有所提升，重置計數器。")
        else:
            no_improve_count += 1
            tracker.logger.info(f"模型已連續 {no_improve_count} 次無改善 (上限: {config.get('early_stop_patience', 4)})")

        if min_map > best_min_map:
            best_min_map = min_map
            torch.save(get_clean_state_dict(model), os.path.join(tracker.weights_dir, "best_model_min_map.pth"))
            tracker.logger.info(f"發現更佳模型 (Min mAP: {min_map:.4f}), 已儲存。")

        # ==========================================
        # 8. 保存最新斷點 (Latest Checkpoint for Resume)
        # ==========================================
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': get_clean_state_dict(model),  
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_f1': best_f1,
            'best_min_map': best_min_map,
            'no_improve_count': no_improve_count,
            'total_steps': total_steps   
        }
        
        # 【修改這裡】加入原子存檔機制防損毀
        tmp_checkpoint_path = checkpoint_path + ".tmp"
        torch.save(checkpoint_data, tmp_checkpoint_path) # 先存入暫存檔
        os.replace(tmp_checkpoint_path, checkpoint_path) # 瞬間替換 (作業系統層級的安全操作)
        
        tracker.logger.info(f"Epoch {epoch+1} 狀態 (含 Best 紀錄) 已備份至: {checkpoint_path}")
        
        # Early Stopping 判定
        if no_improve_count >= config.get("early_stop_patience", 4):
            tracker.logger.info(f"已達到早停門檻 ({config.get('early_stop_patience', 4)} 次沒改善)，正式中止訓練。")
            break
        

    tracker.close()

if __name__ == "__main__":
    main()
