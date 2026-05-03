#!/bin/bash
# Crab MSP-Podcast 2.0 Training Script (Local, non-SLURM)
# Reproduces the paper's training on MSP-Podcast dataset
# Adapted for RTX 3090 24GB (batch_size=8, accumulation=8, effective=64)

set -e

PROJ_DIR="/home/brant/Project/SAILER_test"
AUDIO_DIR="${PROJ_DIR}/datasets/MSP_Podcast_Data/Audios/"
CSV_PATH="${PROJ_DIR}/Crab/data/msp2_processed_labels.csv"
SAVE_DIR="${PROJ_DIR}/Crab/experiments/msp/crab"

# Create save directory
mkdir -p "$SAVE_DIR"
echo "Experiment directory: $SAVE_DIR"

# Activate uv environment
source ${PROJ_DIR}/Crab/.venv/bin/activate

# Run training
python ${PROJ_DIR}/Crab/bin/train_crab.py \
  --df_path "$CSV_PATH" \
  --wav_base_dir "$AUDIO_DIR" \
  --model_path "$SAVE_DIR" \
  --ssl_type "microsoft/wavlm-large" \
  --text_model_path "roberta-large" \
  --fusion_hidden_dim 512 \
  --text_max_len 128 \
  --head_dim 1024 \
  --batch_size 64 \
  --accumulation_steps 16 \
  --epochs 20 \
  --constrastive_loss \
  --lr 1e-5 \
  --seed 42 2>&1 | tee -a "$SAVE_DIR/console_out.log"

TRAIN_EXIT_CODE=$?
echo "Training job exited with code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Models saved at: $SAVE_DIR"
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi
