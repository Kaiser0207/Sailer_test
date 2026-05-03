#!/bin/bash
#SBATCH --job-name=train_crab
#SBATCH --output=./slurm/train_crab_%j.out
#SBATCH --error=./slurm/train_crab_%j.err
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00  
#SBATCH --mem=128G    
#SBATCH --partition=l40s  
#SBATCH --gres=gpu:1

# Load Miniconda and activate environment
source ~/miniconda3/bin/activate
conda activate crab  

# Properly expand the tilde to the home directory
HOME_DIR=$(echo ~)
AUDIO_DIR=""

# Create save directory if it doesn't exist
SAVE_DIR="${HOME_DIR}/github/Crab/experiments/meld/crab"
mkdir -p "$SAVE_DIR"
echo "Created save directory: $SAVE_DIR"

# Run the crab training script
python ~/github/Crab/bin/train_crab.py \
  --df_path "${HOME_DIR}/github/MM-ser/meld.csv" \
  --wav_base_dir "$AUDIO_DIR" \
  --model_path "$SAVE_DIR" \
  --ssl_type "${HOME_DIR}/github/MM-ser/bin/models/wavlm-large" \
  --text_model_path "${HOME_DIR}/github/MM-ser/bin/models/roberta-large" \
  --fusion_hidden_dim 512 \
  --text_max_len 128 \
  --head_dim 1024 \
  --batch_size 32 \
  --accumulation_steps 4 \
  --epochs 20 \
  --lr 1e-5 \
  --seed 42 \
  --classes_list Neutral Surprise Fear Sadness Joy Anger Disgust


TRAIN_EXIT_CODE=$?
echo "Training job exited with code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Bimodal training completed successfully!"
    echo "Models saved at: $SAVE_DIR"
else
    echo "Bimodal training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi