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
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer
import importlib
import wandb

# Get the project root directory (MM-ser)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.models as net
from src.utils.losses import MultiPosConLoss
from src.data.podcast import load_cat_emo_label
from src.data.wav import load_audio
from src.data.dataset.dataset import WavSet, CAT_EmoSet, CombinedSet, load_norm_stat, TxtSet
from src.utils.etc import set_deterministic
from src.data.dataset.collate_fn import collate_fn_wav_lab_mask

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--text_model_path", type=str, default="~/github/MM-ser/bin/models/roberta-large")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model_path", type=str, required=True, help="Path to saved models")
parser.add_argument("--fusion_hidden_dim", type=int, default=512)
parser.add_argument("--text_max_len", type=int, default=128)
parser.add_argument("--df_path", type=str, default="dataframes")
parser.add_argument("--wav_base_dir", type=str, default="")
parser.add_argument("--classes_list", nargs='+', type=str, 
                    default=['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral'],
                    help="List of emotion classes")
parser.add_argument("--dtype", type=str, default="test", help="Dataset type: train, dev, or test")
parser.add_argument("--use_wandb", action="store_true", default=False, help="Enable W&B logging")

args = parser.parse_args()

# Setup logging
MODEL_PATH = args.model_path
os.makedirs(MODEL_PATH, exist_ok=True)

# Create logging filename with current datetime
log_filename = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_filepath = os.path.join(MODEL_PATH, log_filename)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log initial configuration
logger.info("="*80)
logger.info("Bimodal Model Inference Started")
logger.info("="*80)
logger.info(f"Log file: {log_filepath}")
logger.info("\nAll Arguments:")
for arg, value in vars(args).items():
    logger.info(f"  {arg}: {value}")

# Initialize W&B if enabled
if args.use_wandb:
    wandb.init(
        project="SAILER_Emotion_Recognition",
        name=f"Crab_MSP_test_{args.dtype}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=vars(args),
    )
    logger.info(f"W&B Run: {wandb.run.get_url()}")

set_deterministic(args.seed)
SSL_TYPE = args.ssl_type
TEXT_MODEL_PATH = os.path.expanduser(args.text_model_path)
BATCH_SIZE = args.batch_size
TEXT_MAX_LEN = args.text_max_len
FUSION_HIDDEN_DIM = args.fusion_hidden_dim

# Use classes from args
classes = args.classes_list
logger.info(f"Classes: {classes}")

# Create mapping for argmax to class abbreviation (like in original second code) ONLY FOR MSP
classes_ = ['A', 'S', 'H', 'U', 'F', 'D', 'C', 'N']
map_argmax = dict()
for i, c in enumerate(classes_):
    map_argmax[i] = c

emo_list = copy.deepcopy(classes)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Device: {device}")

# Check if ser_model exists
ser_model_path = os.path.join(MODEL_PATH, "final_ser.pt")
has_ser_model = os.path.exists(ser_model_path)
logger.info(f"SER model found: {has_ser_model}")

if not has_ser_model:
    logger.error("SER model not found! Cannot generate predictions.")
    sys.exit(1)

# Load data
audio_path = args.wav_base_dir
label_path = args.df_path

# Load the CSV file
df = pd.read_csv(label_path)

# Check if Text column exists
if 'Text' not in df.columns:
    raise ValueError("The dataframe must contain a 'Text' column with transcriptions!")

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

# Create dataset based on dtype
dtype = args.dtype
cur_utts, cur_labs = load_cat_emo_label(label_path, dtype, debug=False, emolist=emo_list)
cur_wavs = load_audio(audio_path, cur_utts)
cur_texts = load_text_data(label_path, dtype, debug=False)

# Load normalization stats
wav_mean, wav_std = load_norm_stat(MODEL_PATH + "/train_norm_stat.pkl")
cur_wav_set = WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)

# Create text dataset
cur_txt_set = TxtSet(cur_texts, text_tokenizer, max_len=TEXT_MAX_LEN)

# Create emotion dataset
cur_emo_set = CAT_EmoSet(cur_labs)

# Combine all datasets
dataset = CombinedSet([cur_wav_set, cur_txt_set, cur_emo_set, cur_utts])

# Create dataloader
dataloader = DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False,
    pin_memory=True, 
    num_workers=4,
    collate_fn=collate_fn_bimodal
)

logger.info(f"{dtype} dataset size: {len(dataset)}")
logger.info(f"Number of batches: {len(dataloader)}")

# Load SSL model
logger.info(f"Loading pre-trained {SSL_TYPE} model...")
ssl_model = AutoModel.from_pretrained(SSL_TYPE)

# Load SSL weights
ssl_weights_path = os.path.join(MODEL_PATH, "final_ssl.pt")
if os.path.exists(ssl_weights_path):
    ssl_model.load_state_dict(torch.load(ssl_weights_path, map_location=device))
    logger.info(f"Loaded SSL weights from {ssl_weights_path}")
else:
    logger.error(f"SSL weights not found at {ssl_weights_path}")
    sys.exit(1)

ssl_model.eval()
ssl_model.to(device)

# Load text model
logger.info(f"Loading pre-trained RoBERTa model from: {TEXT_MODEL_PATH}")
text_model = AutoModel.from_pretrained(TEXT_MODEL_PATH)

# Load text weights
text_weights_path = os.path.join(MODEL_PATH, "final_text.pt")
if os.path.exists(text_weights_path):
    text_model.load_state_dict(torch.load(text_weights_path, map_location=device))
    logger.info(f"Loaded text weights from {text_weights_path}")
else:
    logger.error(f"Text weights not found at {text_weights_path}")

text_model.eval()
text_model.to(device)

# Get feature dimensions
audio_feat_dim = ssl_model.config.hidden_size
text_feat_dim = text_model.config.hidden_size

logger.info(f"Audio feature dim: {audio_feat_dim}")
logger.info(f"Text feature dim: {text_feat_dim}")

# Load SER model
ser_model = net.MultiModalEmotionClassifierDeep(
    features1_dim=audio_feat_dim,
    features2_dim=text_feat_dim,
    fusion_hidden_dim=FUSION_HIDDEN_DIM,
    num_emotions=len(classes),
    dropout=0.5
)
ser_model.load_state_dict(torch.load(ser_model_path, map_location=device))
ser_model.eval()
ser_model.to(device)
logger.info(f"Loaded SER model from {ser_model_path}")

# Inference
logger.info("\nStarting inference...")

total_pred = []
total_utt = []
total_labels = []  # Store ground truth labels for metrics calculation
all_embeddings = []
all_labels = []
all_predictions = []
gender_list = []

i = 0
with torch.no_grad():
    for batch_data in tqdm(dataloader):
        # Unpack bimodal data
        (x_audio, mask_audio), (x_text_ids, mask_text), y, wav_names = batch_data
        
        # Move to device
        x_audio = x_audio.to(device, non_blocking=True).float()
        mask_audio = mask_audio.to(device, non_blocking=True).float()
        x_text_ids = x_text_ids.to(device, non_blocking=True)
        mask_text = mask_text.to(device, non_blocking=True)
        
        # Process audio
        ssl_output = ssl_model(x_audio, attention_mask=mask_audio)
        ssl = ssl_output.last_hidden_state  # (B, T, D)
        
        # Process text
        text_output = text_model(input_ids=x_text_ids, attention_mask=mask_text)
        text_hidden_states = text_output.last_hidden_state  # (B, T_text, D_text)
        
        # Get predictions
        emo_pred, embeddings_dict = ser_model(ssl, text_hidden_states, return_embeddings=True)

        embeddings = embeddings_dict['normalized']  # (B, D_fusion)

        pred_labels = torch.argmax(emo_pred, dim=1)
        all_predictions.append(pred_labels.cpu().numpy())

        all_embeddings.append(embeddings.cpu().numpy())
        total_pred.append(emo_pred)
        total_utt.append(wav_names[0])  # Since batch_size=1
        
        # Store ground truth labels (convert multi-label to single label)
        y_single = y.max(dim=1)[1]  # Get argmax for multi-label
        all_labels.append(y_single.cpu().numpy())
        total_labels.append(y_single.cpu().numpy()[0])
        
        if("Gender" in df.columns):
            gender_list.append(df["Gender"].values[i])
        else:
            gender_list.append("No_gender_info")
        i+=1

all_embeddings = np.concatenate(all_embeddings, axis = 0)
all_labels = np.concatenate(all_labels, axis=0)
all_predictions = np.concatenate(all_predictions, axis=0)

# Apply UMAP
logger.info("\nApplying UMAP...")
umap = UMAP(n_components=2, random_state=args.seed)
embeddings_2d = umap.fit_transform(all_embeddings)

# Create DataFrame
if has_ser_model:
    # With predictions
    df_results = pd.DataFrame({
        # 'wav_name': all_wav_names,
        'groundtruth_label': all_labels,
        'groundtruth_class': [classes[label] for label in all_labels],
        'predicted_label': all_predictions,
        'predicted_class': [classes[label] for label in all_predictions],
        'dim1': embeddings_2d[:, 0],
        'dim2': embeddings_2d[:, 1]
    })
    
    # Calculate accuracy
    accuracy = (df_results['groundtruth_label'] == df_results['predicted_label']).mean()
    logger.info(f"\nAccuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    logger.info("\nPer-class accuracy:")
    for i, cls in enumerate(classes):
        cls_mask = df_results['groundtruth_label'] == i
        if cls_mask.sum() > 0:
            cls_acc = (df_results.loc[cls_mask, 'groundtruth_label'] == 
                      df_results.loc[cls_mask, 'predicted_label']).mean()
            logger.info(f"  {cls}: {cls_acc:.4f} ({cls_mask.sum()} samples)")
else:
    # Without predictions
    df_results = pd.DataFrame({
        # 'wav_name': all_wav_names,
        'groundtruth_label': all_labels,
        'groundtruth_class': [classes[label] for label in all_labels],
        'dim1': embeddings_2d[:, 0],
        'dim2': embeddings_2d[:, 1]
    })

# Save DataFrame
df_path = os.path.join(MODEL_PATH, 'inference_results.csv')
df_results.to_csv(df_path, index=False)
logger.info(f"\nSaved results to {df_path}")

# Create UMAP visualization
logger.info("\nCreating UMAP visualization...")

try:
    # Calculate clustering metrics on full test set
    from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

    ch_score = calinski_harabasz_score(all_embeddings, all_labels)
    silhouette = silhouette_score(all_embeddings, all_labels)
    db_score = davies_bouldin_score(all_embeddings, all_labels)

    logger.info("\n=== UMAP Clustering Metrics (Full Test Set) ===")
    logger.info(f"Calinski-Harabasz Score: {ch_score:.2f} (higher is better)")
    logger.info(f"Silhouette Score: {silhouette:.3f} (range [-1,1], higher is better)")
    logger.info(f"Davies-Bouldin Score: {db_score:.3f} (lower is better)")
    logger.info("================================================\n")
except Exception as e:
    logger.warning(f"Could not compute clustering metrics: {e}")    

# Sample for visualization: all samples except neutral (limited to 300)
balanced_dfs = []
for i in range(len(classes)):
    class_df = df_results[df_results['groundtruth_label'] == i]
    if classes[i].lower() == 'neutral' and len(class_df) > 300:
        # Sample only 300 for neutral class
        class_df = class_df.sample(n=300, random_state=42)
    # Keep all samples for other emotions
    balanced_dfs.append(class_df)

df_balanced = pd.concat(balanced_dfs, ignore_index=True)
logger.info(f"Visualization using {len(df_balanced)} samples (neutral limited to 300)")

plt.figure(figsize=(12, 10))

# Create color palette
try:
    # For matplotlib >= 3.9
    colors = plt.colormaps['tab10'](np.linspace(0, 1, len(classes)))
except:
    # For older matplotlib versions
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(classes)))
color_dict = {i: colors[i] for i in range(len(classes))}

# Plot each class with enhanced aesthetics
for i, cls in enumerate(classes):
    mask = df_balanced['groundtruth_label'] == i
    if mask.sum() > 0:
        plt.scatter(
            df_balanced.loc[mask, 'dim1'],
            df_balanced.loc[mask, 'dim2'],
            c=[color_dict[i]],
            label=f'{cls} ({mask.sum()})',
            alpha=0.7,
            s=80,
            edgecolors='white',
            linewidth=0.5
        )

plt.xlabel('UMAP Dimension 1', fontsize=14, fontweight='bold')
plt.ylabel('UMAP Dimension 2', fontsize=14, fontweight='bold')
# plt.title('UMAP Projection of Bimodal Embeddings (Dev Set)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add subtle background
ax = plt.gca()
ax.set_facecolor('#f8f9fa')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save plot
plot_path = os.path.join(MODEL_PATH, 'umap_embeddings_by_emotion_test.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
logger.info(f"Saved UMAP plot to {plot_path}")

# Prepare data for CSV 
data = []
predictions_numeric = []

for pred, utt in zip(total_pred, total_utt):
    # Get the predicted class using argmax
    pred_numeric = np.argmax(pred.cpu().numpy().flatten())
    pred_values = map_argmax[pred_numeric]
    data.append([utt, pred_values])
    predictions_numeric.append(pred_numeric)

# Writing to CSV file (original backward compatibility)
os.makedirs(MODEL_PATH + '/results', exist_ok=True) 
csv_filename = MODEL_PATH + '/results/' + dtype + '.csv'

# Convert to pandas DataFrame and sort by filename
data = np.array(data)
df_results = pd.DataFrame({'FileName': data[:,0], 'EmoClass': data[:,1]})
df_results = df_results.sort_values(by='FileName').reset_index(drop=True)
df_results.to_csv(csv_filename, index=False)

logger.info(f"\nSaved predictions to {csv_filename}")
logger.info(f"Total predictions: {len(df_results)}")

# Display first few predictions
logger.info("\nFirst 10 predictions:")
logger.info(df_results.head(10).to_string())

# Save in submission format
submit_csv = MODEL_PATH + '/to_submit.csv'
df_results.to_csv(submit_csv, index=False)
logger.info(f"\nAlso saved submission format to {submit_csv}")

# Compute detailed metrics automatically
logger.info("\n" + "="*50)
logger.info("Computing detailed metrics...")
logger.info("="*50)

# Import metrics from sklearn
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score

# Convert lists to numpy arrays
y_true = np.array(total_labels)
y_pred = np.array(predictions_numeric)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
logger.info(f"\nAccuracy: {accuracy:.4f}")

# Per-class accuracy
logger.info("\nPer-class accuracy:")
for i, cls in enumerate(classes):
    cls_mask = y_true == i
    if cls_mask.sum() > 0:
        cls_acc = (y_true[cls_mask] == y_pred[cls_mask]).mean()
        logger.info(f"  {cls}: {cls_acc:.4f} ({cls_mask.sum()} samples)")

# Calculate F1 macro score
f1_macro = f1_score(y_true, y_pred, average='macro')
logger.info(f"\nF1 Macro Score: {f1_macro:.4f}")

# Calculate UAR (Unweighted Average Recall)
from sklearn.metrics import recall_score
uar = recall_score(y_true, y_pred, average='macro')
logger.info(f"UAR (Macro Recall): {uar:.4f}")

# Calculate per-class F1 scores
f1_per_class = f1_score(y_true, y_pred, average=None)
logger.info("\nPer-class F1 scores:")
for i, cls in enumerate(classes):
    if i < len(f1_per_class):
        logger.info(f"  {cls}: {f1_per_class[i]:.4f}")

# Generate classification report
logger.info("\nClassification Report:")
report = classification_report(
    y_true, 
    y_pred,
    target_names=classes,
    digits=4
)
logger.info(f"\n{report}")

# Save detailed results DataFrame
df_detailed = pd.DataFrame({
    'wav_name': total_utt,
    'groundtruth_label': y_true,
    'groundtruth_class': [classes[label] for label in y_true],
    'predicted_label': y_pred,
    'predicted_class': [classes[label] for label in y_pred],
    'correct': y_true == y_pred,
    'gender': gender_list
})

# Sort by filename for consistency
df_detailed = df_detailed.sort_values(by='wav_name').reset_index(drop=True)

# Save detailed results
detailed_csv = os.path.join(MODEL_PATH, 'results', f'{dtype}_detailed_results.csv')
df_detailed.to_csv(detailed_csv, index=False)
logger.info(f"\nSaved detailed results to {detailed_csv}")

# Save plots automatically
logger.info("Creating and saving plots...")

low_thr = 0.2     # values below this → red
high_thr = 0.5    # values above this → bold

plt.figure(figsize=(10, 8))

cm = confusion_matrix(df_detailed['groundtruth_class'], df_detailed['predicted_class'])
# Normalize confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


ax = sns.heatmap(
    cm_normalized,
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes,
    cbar=True
)

# Manual annotations
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        value = cm_normalized[i, j]

        color = 'black'
        weight = 'normal'

        if i == j:
            color = 'white'
            weight = 'bold'

            if value < low_thr:
                color = 'red'
                weight = 'normal'

        ax.text(
            j + 0.5,
            i + 0.5,
            f"{value:.2f}",
            ha='center',
            va='center',
            color=color,
            fontsize=20,
            fontweight=weight
        )

plt.xlabel('Predicted Label', fontsize=22)
plt.ylabel('True Label', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

cm_path = os.path.join(MODEL_PATH, 'results', f'{dtype}_confusion_matrix.png')

plt.tight_layout()
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

logger.info(f"Saved confusion matrix to {cm_path}")
plt.close()

# Create accuracy by class bar plot
plt.figure(figsize=(12, 6))
class_accuracies = []
class_names = []
class_counts = []

for i, cls in enumerate(classes):
    cls_mask = y_true == i
    if cls_mask.sum() > 0:
        cls_acc = (y_true[cls_mask] == y_pred[cls_mask]).mean()
        class_accuracies.append(cls_acc)
        class_names.append(cls)
        class_counts.append(cls_mask.sum())

bars = plt.bar(range(len(class_names)), class_accuracies, color='skyblue', alpha=0.7)
plt.xlabel('Emotion Classes', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(f'Per-Class Accuracy - {dtype.upper()} Set', fontsize=14)
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc, count in zip(bars, class_accuracies, class_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{acc:.3f}\n({count})', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
acc_path = os.path.join(MODEL_PATH, 'results', f'{dtype}_accuracy_by_class.png')
plt.savefig(acc_path, dpi=300, bbox_inches='tight')
logger.info(f"Saved accuracy plot to {acc_path}")
plt.close()

logger.info("\nMetrics computation completed!")

logger.info("\n" + "="*80)
logger.info("Bimodal Model Inference Completed!")
logger.info("="*80)

# W&B logging of final results
if args.use_wandb:
    log_dict = {
        f"{dtype}/accuracy": accuracy,
        f"{dtype}/f1_macro": f1_macro,
        f"{dtype}/UAR": uar,
    }
    # Per-class F1
    for i, cls in enumerate(classes):
        if i < len(f1_per_class):
            log_dict[f"{dtype}/f1_{cls}"] = f1_per_class[i]
    wandb.log(log_dict)
    wandb.run.summary[f"{dtype}_accuracy"] = accuracy
    wandb.run.summary[f"{dtype}_f1_macro"] = f1_macro
    wandb.run.summary[f"{dtype}_UAR"] = uar
    # Log confusion matrix image
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        cm_final = confusion_matrix(y_true, y_pred)
        cm_final_norm = cm_final.astype('float') / cm_final.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_final_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        ax_cm.set_title(f'Confusion Matrix - {dtype}')
        plt.tight_layout()
        wandb.log({f"{dtype}/confusion_matrix": wandb.Image(fig_cm)})
        plt.close(fig_cm)
    except Exception as e:
        logger.warning(f"Failed to log confusion matrix to W&B: {e}")
    wandb.finish()


# --- Additional Gender-based Metrics and Plots with Logged Matrix ---

if 'Gender' in df.columns:
    logger.info("\n" + "="*50)
    logger.info("Computing Metrics per Gender...")
    logger.info("="*50)

    df_gender_metrics = df_detailed

    genders = [g for g in df_gender_metrics['gender'].unique() if pd.notna(g)]

    for gender in genders:
        g_df = df_gender_metrics[df_gender_metrics['gender'] == gender]
        
        if len(g_df) == 0:
            continue

        g_y_true = g_df['groundtruth_label'].values
        g_y_pred = g_df['predicted_label'].values
        g_true_classes = g_df['groundtruth_class'].values
        g_pred_classes = g_df['predicted_class'].values

        # Calculate basic metrics
        g_acc = accuracy_score(g_y_true, g_y_pred)
        g_f1 = f1_score(g_y_true, g_y_pred, average='macro')

        logger.info(f"\n" + "-"*40)
        logger.info(f"GENDER REPORT: {gender} (N={len(g_df)})")
        logger.info(f"-"*40)
        logger.info(f"Overall Accuracy: {g_acc:.4f}")
        logger.info(f"F1 Macro Score:   {g_f1:.4f}")

        # Compute Confusion Matrix
        g_cm = confusion_matrix(g_true_classes, g_pred_classes, labels=classes)
        # Avoid division by zero with 1e-9
        g_cm_norm = g_cm.astype('float') / (g_cm.sum(axis=1)[:, np.newaxis] + 1e-9)

        # Log Normalized Matrix Values
        logger.info(f"\nNormalized Confusion Matrix (Recall) for {gender}:")
        header = "True \\ Pred".ljust(15) + "".join([c.center(10) for c in classes])
        logger.info(header)
        
        for i, row_label in enumerate(classes):
            row_str = f"{row_label.ljust(15)}"
            for j, val in enumerate(g_cm_norm[i]):
                row_str += f"{val:10.2f}"
            logger.info(row_str)

        # Visual Plotting
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(g_cm_norm, cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True)
        
        for i in range(g_cm_norm.shape[0]):
            for j in range(g_cm_norm.shape[1]):
                val = g_cm_norm[i, j]
                color = 'white' if i == j and val > 0.2 else 'black'
                if i == j and val < 0.2: color = 'red'
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center', 
                        color=color, fontsize=16, fontweight='bold' if i == j else 'normal')

        plt.title(f'Confusion Matrix: {gender}', fontsize=18)
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        
        g_cm_path = os.path.join(MODEL_PATH, 'results', f'{dtype}_confusion_matrix_{gender.lower()}.png')
        plt.tight_layout()
        plt.savefig(g_cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"\nSaved gender-specific confusion matrix plot to {g_cm_path}")

else:
    logger.info("\nGender column not found. Skipping gender-based metrics.")