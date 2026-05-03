## THIS ONE IS AN ABLATION USING MODEL WITH NO CROSS MODAL ATTENTION
# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
import glob
import librosa
import copy
import logging
from datetime import datetime
import wandb
from sklearn.metrics import f1_score, classification_report, recall_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer
import importlib
# Self-Written Modules

import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Get the project root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.models as net
from src.utils.losses import MultiPosConLoss
from src.data.podcast import load_cat_emo_label
from src.data.dataset.dataset import WavSet, CAT_EmoSet, CombinedSet, load_norm_stat, TxtSet, LazyWavSet
from src.utils.etc import set_deterministic
from src.data.dataset.collate_fn import collate_fn_wav_lab_mask

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--text_model_path", type=str, default="~/github/MM-ser/bin/models/roberta-large")
parser.add_argument("--pre_trained_path", type=str, default="./experiments/")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--model_path", type=str, default="./temp")
parser.add_argument("--head_dim", type=int, default=1024)
parser.add_argument("--use_tp", action="store_true", default=False)
parser.add_argument("--tp_prob", type=float, default=0.8)
parser.add_argument("--df_path", type=str, default="dataframes")
parser.add_argument("--wav_base_dir", type=str, default="")
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--constrastive_loss", action="store_true", default=False)
parser.add_argument("--balanced_sampling", action="store_true", default=False)
parser.add_argument("--pooling_type", type=str, default="AttentionPoolingBatched")
parser.add_argument("--text_max_len", type=int, default=128)
parser.add_argument("--fusion_hidden_dim", type=int, default=512)
parser.add_argument("--classes_list", nargs='+', type=str, 
                    default=['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral'],
                    help="List of emotion classes")
parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in model_path")

args = parser.parse_args()

# Setup logging
MODEL_PATH = args.model_path
os.makedirs(MODEL_PATH, exist_ok=True)

# Create logging filename with current datetime
log_filename = f"logging_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_filepath = os.path.join(MODEL_PATH, log_filename)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Initialize W&B
wandb.init(
    project="SAILER_Emotion_Recognition",
    name=f"Crab_MSP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config=vars(args),
)

# Log initial configuration
logger.info("="*80)
logger.info("Training Started - Bimodal Model with Contrastive Loss")
logger.info("="*80)
logger.info(f"Log file: {log_filepath}")
logger.info(f"W&B Run: {wandb.run.get_url()}")
logger.info("\nAll Arguments:")
for arg, value in vars(args).items():
    logger.info(f"  {arg}: {value}")

set_deterministic(args.seed)
SSL_TYPE = args.ssl_type
TEXT_MODEL_PATH = os.path.expanduser(args.text_model_path)
assert SSL_TYPE != None, "Invalid SSL type!"
BATCH_SIZE = args.batch_size
ACCUMULATION_STEP = args.accumulation_steps
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
EPOCHS=args.epochs
LR=args.lr
USE_TP = args.use_tp
TP_PROB = args.tp_prob
CONTRASTIVE_LOSS = args.constrastive_loss
BALANCED_SAMPLING = args.balanced_sampling
TEXT_MAX_LEN = args.text_max_len
FUSION_HIDDEN_DIM = args.fusion_hidden_dim
PRE_TRAINED_PATH = args.pre_trained_path

# Use classes from args
classes = args.classes_list
logger.info(f"Classes: {classes}")

emo_list = copy.deepcopy(classes)

#run in debug mode
debug = args.debug
if(debug):
    logger.info("Running in debug mode!")
    BATCH_SIZE = 2
    EPOCHS = 2

import json
from collections import defaultdict
audio_path = args.wav_base_dir
label_path = args.df_path

# Load the CSV file
df = pd.read_csv(label_path)

# Check if Text column exists
if 'Text' not in df.columns:
    raise ValueError("The dataframe must contain a 'Text' column with transcriptions!")

# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']

# Calculate class frequencies for loss weighting
class_frequencies = train_df[classes].sum().to_dict()
total_samples = len(train_df)
class_weights = {cls: total_samples / (len(classes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
logger.info("Class weights for loss function:")
logger.info(class_weights)

# Convert to list in the order of classes
weights_list = [class_weights[cls] for cls in classes]
# Convert to PyTorch tensor for loss function
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_weights_tensor = torch.tensor(weights_list, device=device, dtype=torch.float)
logger.info(f"Loss weights tensor: {class_weights_tensor}")
logger.info(f"Device: {device}")

# Function to calculate sample weights for balanced sampling
def calculate_sample_weights(labels_data, classes):
    """
    Calculate weight for each sample based on its emotion labels.
    Multi-label samples get averaged weights.
    """
    # Get class frequencies
    class_counts = {cls: labels_data[cls].sum() for cls in classes}
    total_samples = len(labels_data)
    
    # Calculate weight for each class (inverse frequency)
    class_sample_weights = {}
    for cls, count in class_counts.items():
        if count > 0:
            # Weight inversely proportional to class frequency
            class_sample_weights[cls] = total_samples / count
        else:
            class_sample_weights[cls] = 0.0
    
    # Normalize weights
    max_weight = max(class_sample_weights.values())
    if max_weight > 0:
        class_sample_weights = {k: v/max_weight for k, v in class_sample_weights.items()}
    
    logger.info("\nClass sample weights for balanced sampling:")
    for cls, weight in class_sample_weights.items():
        logger.info(f"  {cls}: {weight:.4f} (count: {class_counts[cls]})")
    
    # Calculate weight for each sample
    sample_weights = []
    for idx, row in labels_data.iterrows():
        # Get all emotions for this sample
        sample_emotions = [cls for cls in classes if row[cls] == 1]
        
        if sample_emotions:
            # Average weight across all emotions in this sample
            weight = np.mean([class_sample_weights[cls] for cls in sample_emotions])
        else:
            # If no emotion labeled, give minimal weight
            weight = 0.1
        
        sample_weights.append(weight)
    
    return sample_weights

# Load text data function
def load_text_data(df_path, dtype, debug=False):
    """Load text data for given data type"""
    df = pd.read_csv(df_path)
    
    if dtype == "train":
        df_filtered = df[df['Split_Set'] == 'Train']
    elif dtype == "dev":
        df_filtered = df[df['Split_Set'] == 'Development']
    elif dtype == "test":
        df_filtered = df[df['Split_Set'] == 'Test']
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
    
    if debug:
        df_filtered = df_filtered.sample(n=min(100, len(df_filtered)), random_state=42).reset_index(drop=True)
    
    # Extract texts
    texts = df_filtered['Text'].fillna("").to_numpy()
    
    return texts

# Initialize text tokenizer
logger.info(f"Loading text tokenizer from: {TEXT_MODEL_PATH}")
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)

# Modified collate function for bimodal data
def collate_fn_bimodal(data):
    """
    Collate function for bimodal data (audio + text)
    data: list of [(wav, dur), (input_ids, attention_mask), lab, utt]
    """
    n_batch = len(data)
    
    # Process audio data
    wav_list = [x[0][0] for x in data]
    dur_list = [x[0][1] for x in data]
    max_len = max(dur_list)
    
    wav_arr = torch.zeros((n_batch, max_len))
    mask_arr = torch.zeros((n_batch, max_len))
    
    for i, (wav, dur) in enumerate(zip(wav_list, dur_list)):
        wav_arr[i, :dur] = torch.tensor(wav[:dur])
        mask_arr[i, :dur] = 1
    
    # Process text data
    input_ids_list = [x[1][0] for x in data]
    attention_mask_list = [x[1][1] for x in data]
    
    input_ids_arr = torch.stack(input_ids_list)
    text_attention_mask_arr = torch.stack(attention_mask_list)
    
    # Process labels
    lab_list = [x[2] for x in data]
    lab_arr = torch.tensor(np.array(lab_list))
    
    # Process utterance names
    utt_list = [x[3] for x in data]
    
    return (wav_arr, mask_arr), (input_ids_arr, text_attention_mask_arr), lab_arr, utt_list

# Create datasets and dataloaders
total_dataset = dict()
total_dataloader = dict()

for dtype in ["train", "dev"]:
    cur_utts, cur_labs = load_cat_emo_label(label_path, dtype, debug=debug, emolist=emo_list)
    # Build file paths instead of loading all audio into RAM
    cur_wav_paths = [os.path.join(audio_path, utt) for utt in cur_utts]
    cur_texts = load_text_data(label_path, dtype, debug=debug)
    
    # Create audio dataset using lazy loading (on-demand from disk)
    if dtype == "train":
        cur_wav_set = LazyWavSet(
            cur_wav_paths, use_tp=USE_TP, tp_prob=TP_PROB
        )
        # Compute normalization stats from a sample (not entire dataset)
        logger.info("Computing normalization stats from 5000 random samples...")
        cur_wav_set.compute_norm_stats(sample_size=5000)
        cur_wav_set.save_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
    else:
        if dtype == "dev":
            wav_mean = total_dataset["train"].datasets[0].wav_mean
            wav_std = total_dataset["train"].datasets[0].wav_std
        elif dtype == "test":
            wav_mean, wav_std = load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_wav_set = LazyWavSet(
            cur_wav_paths, wav_mean=wav_mean, wav_std=wav_std
        )
    
    # Create text dataset
    cur_txt_set = TxtSet(cur_texts, text_tokenizer, max_len=TEXT_MAX_LEN)
    
    # Create emotion dataset
    cur_emo_set = CAT_EmoSet(cur_labs)
    
    # Combine all datasets
    total_dataset[dtype] = CombinedSet([cur_wav_set, cur_txt_set, cur_emo_set, cur_utts])
    
    # Create dataloader with balanced sampling for training
    if dtype == "train":
        # Get the emotion labels DataFrame for current split
        cur_df = df[df['Split_Set'] == 'Train']

        if(debug):
            cur_df = cur_df.sample(n=100, random_state=42).reset_index(drop=True)
        
        if(BALANCED_SAMPLING):
            # Calculate sample weights
            sample_weights = calculate_sample_weights(cur_df, classes)
            
            # Create WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            # Create DataLoader with sampler
            cur_bs = BATCH_SIZE // ACCUMULATION_STEP
            total_dataloader[dtype] = DataLoader(
                total_dataset[dtype], 
                batch_size=cur_bs,
                sampler=sampler,
                pin_memory=True, 
                num_workers=0,
                collate_fn=collate_fn_bimodal
            )
            
            logger.info(f"\nTraining with balanced sampler:")
            logger.info(f"  Total samples: {len(sample_weights)}")
            logger.info(f"  Batch size: {cur_bs}")
            logger.info(f"  Batches per epoch: {len(total_dataloader[dtype])}")
        else:
            # For training without balanced sampling, use regular DataLoader
            cur_bs = BATCH_SIZE // ACCUMULATION_STEP
            total_dataloader[dtype] = DataLoader(
                total_dataset[dtype], 
                batch_size=cur_bs,
                shuffle=True,
                pin_memory=True, 
                num_workers=0,
                collate_fn=collate_fn_bimodal
            )
            
            logger.info(f"\nTraining without balanced sampler:")
            logger.info(f"  Total samples: {len(cur_df)}")
            logger.info(f"  Batch size: {cur_bs}")
            logger.info(f"  Batches per epoch: {len(total_dataloader[dtype])}")
        
    else:
        # For dev/test, use regular DataLoader without sampling
        total_dataloader[dtype] = DataLoader(
            total_dataset[dtype], 
            batch_size=cur_bs*4 if dtype == "dev" else 1,
            shuffle=False,
            pin_memory=True, 
            num_workers=0,
            collate_fn=collate_fn_bimodal
        )

# Load pre-trained models
logger.info(f"Loading pre-trained {SSL_TYPE} model...")
ssl_model = AutoModel.from_pretrained(SSL_TYPE)

# Load SSL weights
ssl_weights_path = os.path.join(PRE_TRAINED_PATH, "final_ssl.pt")
if os.path.exists(ssl_weights_path):
    ssl_model.load_state_dict(torch.load(ssl_weights_path, map_location=device))
    logger.info(f"Loaded SSL weights from {ssl_weights_path}")
else:
    logger.error(f"SSL weights not found at {ssl_weights_path}")
    # sys.exit(1)

# Handle different model types for freezing
if hasattr(ssl_model, 'freeze_feature_encoder'):
    ssl_model.freeze_feature_encoder()
elif hasattr(ssl_model, 'feature_extractor'):
    # For HuBERT/Wav2Vec2
    for param in ssl_model.feature_extractor.parameters():
        param.requires_grad = False
    logger.info("Manually froze feature extractor parameters")

ssl_model.eval()
ssl_model.to(device)

logger.info(f"Loading pre-trained RoBERTa model from: {TEXT_MODEL_PATH}")
text_model = AutoModel.from_pretrained(TEXT_MODEL_PATH)
text_model.eval()
text_model.to(device)

########## Remove pooling - use sequences directly ##########
audio_feat_dim = ssl_model.config.hidden_size
text_feat_dim = text_model.config.hidden_size

logger.info(f"Audio feature dim: {audio_feat_dim}")
logger.info(f"Text feature dim: {text_feat_dim}")

# Setup loss and model
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
contrastive_criterion_audio = MultiPosConLoss()  # For audio encoder
contrastive_criterion_text = MultiPosConLoss()   # For text encoder
contrastive_criterion_fusion = MultiPosConLoss() # For fusion embeddings

# Initialize bimodal model
ser_model = net.MultiModalEmotionClassifierDeep(
    features1_dim=audio_feat_dim,
    features2_dim=text_feat_dim,
    fusion_hidden_dim=FUSION_HIDDEN_DIM,
    num_emotions=len(classes),
    dropout=0.5
)
ser_model.eval()
ser_model.to(device)
ser_opt = torch.optim.AdamW(ser_model.parameters(), LR)
ser_opt.zero_grad(set_to_none=True)

# Optimizers with 10x lower learning rate for encoders
ENCODER_LR = LR / 10
ssl_opt = torch.optim.AdamW(ssl_model.parameters(), ENCODER_LR)
text_opt = torch.optim.AdamW(text_model.parameters(), ENCODER_LR)

ssl_opt.zero_grad(set_to_none=True)
text_opt.zero_grad(set_to_none=True)

# Add cosine schedulers
ser_scheduler = CosineAnnealingLR(ser_opt, T_max=EPOCHS, eta_min=1e-6)
ssl_scheduler = CosineAnnealingLR(ssl_opt, T_max=EPOCHS, eta_min=1e-7)
text_scheduler = CosineAnnealingLR(text_opt, T_max=EPOCHS, eta_min=1e-7)

min_epoch=0
min_loss=1e10
max_f1=0.0

# Resume logic
start_epoch = 0
if args.resume:
    checkpoint_path = os.path.join(MODEL_PATH, "checkpoint_latest.pt")
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ssl_model.load_state_dict(checkpoint['ssl_model_state_dict'])
        text_model.load_state_dict(checkpoint['text_model_state_dict'])
        ser_model.load_state_dict(checkpoint['ser_model_state_dict'])
        ssl_opt.load_state_dict(checkpoint['ssl_optimizer_state_dict'])
        text_opt.load_state_dict(checkpoint['text_optimizer_state_dict'])
        ser_opt.load_state_dict(checkpoint['ser_optimizer_state_dict'])
        ssl_scheduler.load_state_dict(checkpoint['ssl_scheduler_state_dict'])
        text_scheduler.load_state_dict(checkpoint['text_scheduler_state_dict'])
        ser_scheduler.load_state_dict(checkpoint['ser_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        max_f1 = checkpoint.get('max_f1', 0.0)
        min_loss = checkpoint.get('min_loss', 1e10)
        logger.info(f"Resumed from Epoch {checkpoint['epoch']}. Starting from Epoch {start_epoch}")
    else:
        # Fallback to loading final_*.pt if checkpoint doesn't exist
        ssl_path = os.path.join(MODEL_PATH, "final_ssl.pt")
        if os.path.exists(ssl_path):
            logger.info("Checkpoint not found, but final_*.pt found. Loading weights for warm start...")
            ssl_model.load_state_dict(torch.load(ssl_path, map_location=device))
            text_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "final_text.pt"), map_location=device))
            ser_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "final_ser.pt"), map_location=device))
            logger.info("Weights loaded. Note: Optimizer/Scheduler states and Epoch counter are reset.")
        else:
            logger.warning("No weights found to resume from. Starting from scratch.")

logger.info("\n" + "="*80)
logger.info("Starting Bimodal Training with Contrastive Loss")
logger.info("="*80)

for epoch in range(start_epoch, EPOCHS):
    logger.info(f"\nEpoch: {epoch}")
    ssl_model.train()
    text_model.train()
    ser_model.train()
    batch_cnt = 0

    for batch_data in tqdm(total_dataloader["train"]):
        # Unpack bimodal data
        (x_audio, mask_audio), (x_text_ids, mask_text), y, utts = batch_data
        
        # Move to device
        x_audio = x_audio.to(device, non_blocking=True).float()
        mask_audio = mask_audio.to(device, non_blocking=True).float()
        x_text_ids = x_text_ids.to(device, non_blocking=True)
        mask_text = mask_text.to(device, non_blocking=True)
        y = y.max(dim=1)[1].to(device, non_blocking=True).long()
        
        # Process audio - use sequences directly
        ssl = ssl_model(x_audio, attention_mask=mask_audio).last_hidden_state  # (B, T, D)
        
        # Process text - outputs sequence
        text_output = text_model(input_ids=x_text_ids, attention_mask=mask_text)
        text_hidden_states = text_output.last_hidden_state  # (B, T_text, D_text)
        
        # Main classification loss with embeddings
        emo_pred, embeddings = ser_model(ssl, text_hidden_states, return_embeddings=True)
        cls_loss = criterion(emo_pred, y)

        # Calculate contrastive losses for all embedding types
        # Frame-level embeddings
        contrastive_loss_speech_frame = contrastive_criterion_audio(embeddings['speech_frame_emb'], y)
        contrastive_loss_text_frame = contrastive_criterion_text(embeddings['text_frame_emb'], y)

        # Pooled embeddings (attention-pooled)
        contrastive_loss_speech_pooled = contrastive_criterion_audio(embeddings['speech_pooled_emb'], y)
        contrastive_loss_text_pooled = contrastive_criterion_text(embeddings['text_pooled_emb'], y)

        # Fusion embedding 
        contrastive_loss_fusion = contrastive_criterion_fusion(embeddings['fusion_emb'], y)

        # Combine all contrastive losses
        # Simple average
        total_contrastive_loss = 2.0 * (
            contrastive_loss_speech_frame + 
            contrastive_loss_text_frame + 
            contrastive_loss_speech_pooled + 
            contrastive_loss_text_pooled + 
            contrastive_loss_fusion
        ) / 5

        # Total loss
        loss = cls_loss + total_contrastive_loss

        if debug:
            logger.info("#### Debug Mode ##")
            logger.info(f"x_audio shape: {x_audio.shape}")
            logger.info(f"x_text_ids shape: {x_text_ids.shape}")
            logger.info(f"ssl shape: {ssl.shape}")
            logger.info(f"text_hidden_states shape: {text_hidden_states.shape}")
            logger.info(f"mask_audio shape: {mask_audio.shape}")
            logger.info(f"mask_text shape: {mask_text.shape}")
            logger.info(f"cls_loss: {cls_loss.item():.4f}")
            logger.info(f"total_contrastive_loss: {total_contrastive_loss.item():.4f}")
            logger.info(f"total loss: {loss.item():.4f}")

        total_loss = loss / ACCUMULATION_STEP
        total_loss.backward()
        
        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(total_dataloader["train"]):
            ssl_opt.step()
            text_opt.step()
            ser_opt.step()

            ssl_opt.zero_grad(set_to_none=True)
            text_opt.zero_grad(set_to_none=True)
            ser_opt.zero_grad(set_to_none=True)

        # W&B batch logging
        global_step = epoch * len(total_dataloader["train"]) + batch_cnt
        wandb.log({
            "batch/total_loss": loss.item(),
            "batch/cls_loss": cls_loss.item(),
            "batch/contrastive_loss": total_contrastive_loss.item(),
        }, step=global_step)

        batch_cnt += 1
    
    # Step schedulers at the end of each epoch
    ser_scheduler.step()
    ssl_scheduler.step()
    text_scheduler.step()
    
    # Log current learning rates
    logger.info(f"Current LRs - SER: {ser_scheduler.get_last_lr()[0]:.6f}, "
                f"SSL: {ssl_scheduler.get_last_lr()[0]:.6f}, "
                f"Text: {text_scheduler.get_last_lr()[0]:.6f}")

    train_loss = total_loss.item()
    
    # Evaluation
    ssl_model.eval()
    text_model.eval()
    ser_model.eval()
    
    total_list = []
    all_preds_logits = []
    all_labels_tensor = []

    for batch_data in tqdm(total_dataloader["dev"]):
        # Unpack bimodal data
        (x_audio, mask_audio), (x_text_ids, mask_text), y, utts = batch_data
        
        # Move to device
        x_audio = x_audio.to(device, non_blocking=True).float()
        mask_audio = mask_audio.to(device, non_blocking=True).float()
        x_text_ids = x_text_ids.to(device, non_blocking=True)
        mask_text = mask_text.to(device, non_blocking=True)
        y = y.max(dim=1)[1].to(device, non_blocking=True).long()
        
        with torch.no_grad():
            # Process audio - sequences directly
            ssl = ssl_model(x_audio, attention_mask=mask_audio).last_hidden_state
            # Process text
            text_output = text_model(input_ids=x_text_ids, attention_mask=mask_text)
            text_hidden_states = text_output.last_hidden_state

            emo_pred = ser_model(ssl, text_hidden_states)
            all_preds_logits.append(emo_pred)
            all_labels_tensor.append(y)

    # Calculate loss over all validation data
    all_preds_logits = torch.cat(all_preds_logits, dim=0)
    all_labels_tensor = torch.cat(all_labels_tensor, dim=0)
    dev_loss = criterion(all_preds_logits, all_labels_tensor).item()

    # Compute comprehensive metrics
    y_pred = torch.argmax(all_preds_logits, dim=1).cpu().numpy()
    y_true = all_labels_tensor.cpu().numpy()

    dev_war = accuracy_score(y_true, y_pred)  # WAR = overall accuracy
    dev_uar = recall_score(y_true, y_pred, average='macro')  # UAR = macro recall
    dev_macro_f1 = f1_score(y_true, y_pred, average='macro')
    dev_weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    logger.info(f"\nTrain loss: {train_loss}")
    logger.info(f"Dev loss: {dev_loss}")
    logger.info(f"Dev WAR (Accuracy): {dev_war:.4f}")
    logger.info(f"Dev UAR (Macro Recall): {dev_uar:.4f}")
    logger.info(f"Dev Macro-F1: {dev_macro_f1:.4f}")
    logger.info(f"Dev Weighted-F1: {dev_weighted_f1:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    # W&B epoch logging
    epoch_end_step = (epoch + 1) * len(total_dataloader["train"]) - 1
    wandb_log = {
        "epoch": epoch,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "dev_WAR": dev_war,
        "dev_UAR": dev_uar,
        "dev_macro_f1": dev_macro_f1,
        "dev_weighted_f1": dev_weighted_f1,
        "lr_ser": ser_scheduler.get_last_lr()[0],
        "lr_ssl": ssl_scheduler.get_last_lr()[0],
        "lr_text": text_scheduler.get_last_lr()[0],
    }

    # Per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, cls in enumerate(classes):
        if i < len(per_class_f1):
            wandb_log[f"dev_f1/{cls}"] = per_class_f1[i]
            wandb_log[f"dev_recall/{cls}"] = per_class_recall[i]

    wandb.log(wandb_log, step=epoch_end_step)

    # Log confusion matrix to W&B every 5 epochs or last epoch
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        try:
            cm = confusion_matrix(y_true, y_pred)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Dev Confusion Matrix - Epoch {epoch+1}')
            plt.tight_layout()
            wandb.log({"dev/confusion_matrix": wandb.Image(fig)}, step=epoch_end_step)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    # Save model
    if dev_macro_f1 > max_f1:
        min_epoch = epoch
        min_loss = dev_loss
        max_f1 = dev_macro_f1

        logger.info(f"New best model! Epoch: {min_epoch}, Loss: {min_loss:.4f}, Macro-F1: {max_f1:.4f}")
        logger.info("Saving model...")
        
        # Save all models
        torch.save(ssl_model.state_dict(), 
            os.path.join(MODEL_PATH, "final_ssl.pt"))
        torch.save(text_model.state_dict(), 
            os.path.join(MODEL_PATH, "final_text.pt"))
        torch.save(ser_model.state_dict(), 
            os.path.join(MODEL_PATH, "final_ser.pt"))
        
        logger.info(f"Models saved to {MODEL_PATH}")

        # Also save best metrics to W&B summary
        wandb.run.summary["best_dev_macro_f1"] = dev_macro_f1

    # Save latest checkpoint for resume (every epoch)
    checkpoint_latest = {
        'epoch': epoch,
        'ssl_model_state_dict': ssl_model.state_dict(),
        'text_model_state_dict': text_model.state_dict(),
        'ser_model_state_dict': ser_model.state_dict(),
        'ssl_optimizer_state_dict': ssl_opt.state_dict(),
        'text_optimizer_state_dict': text_opt.state_dict(),
        'ser_optimizer_state_dict': ser_opt.state_dict(),
        'ssl_scheduler_state_dict': ssl_scheduler.state_dict(),
        'text_scheduler_state_dict': text_scheduler.state_dict(),
        'ser_scheduler_state_dict': ser_scheduler.state_dict(),
        'max_f1': max_f1,
        'min_loss': min_loss,
    }
    torch.save(checkpoint_latest, os.path.join(MODEL_PATH, "checkpoint_latest.pt"))
    logger.info(f"Epoch {epoch} checkpoint saved for future resumption.")

logger.info("\n" + "="*80)
logger.info("Bimodal Training Completed!")
logger.info(f"Best epoch: {min_epoch}, Best loss: {min_loss:.4f}, Best Macro-F1: {max_f1:.4f}")
logger.info("="*80)
wandb.finish()