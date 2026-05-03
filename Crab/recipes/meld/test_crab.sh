#!/bin/bash
#SBATCH --job-name=inference_crab
#SBATCH --output=./slurm/inference_crab_%j.out
#SBATCH --error=./slurm/inference_crab_%j.err
#SBATCH --ntasks=1
#SBATCH --time=0-06:00:00 
#SBATCH --mem=64G          
#SBATCH --partition=l40s   
#SBATCH --gres=gpu:1

# Load Miniconda and activate environment
source ~/miniconda3/bin/activate
conda activate crab  

# Properly expand the tilde to the home directory
HOME_DIR=$(echo ~)
AUDIO_DIR=""

# Model directory from training
MODEL_DIR="${HOME_DIR}/github/Crab/experiments/meld/crab"

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found at $MODEL_DIR"
    echo "Please ensure the training has completed successfully"
    exit 1
fi

# Check for required model files
echo "Checking for model files..."
if [ -f "$MODEL_DIR/final_ssl.pt" ]; then
    echo "✓ Found SSL model: final_ssl.pt"
else
    echo "✗ SSL model not found!"
    exit 1
fi

if [ -f "$MODEL_DIR/final_text.pt" ]; then
    echo "✓ Found text model: final_text.pt"
else
    echo "✗ Text model not found!"
    exit 1
fi

if [ -f "$MODEL_DIR/final_ser.pt" ]; then
    echo "✓ Found SER model: final_ser.pt"
    echo "Will compute predictions and F1 scores"
else
    echo "⚠ SER model not found - will only compute embeddings"
fi

# Run the inference script
echo ""
echo "Starting bimodal inference on dev set..."
python ~/github/Crab/bin/test_crab.py \
  --df_path "${HOME_DIR}/github/MM-ser/meld.csv" \
  --wav_base_dir "$AUDIO_DIR" \
  --model_path "$MODEL_DIR" \
  --ssl_type "${HOME_DIR}/github/MM-ser/bin/models/wavlm-large" \
  --text_model_path "${HOME_DIR}/github/MM-ser/bin/models/roberta-large" \
  --fusion_hidden_dim 512 \
  --text_max_len 128 \
  --batch_size 1 \
  --seed 42 \
  --classes_list Neutral Surprise Fear Sadness Joy Anger Disgust

INFERENCE_EXIT_CODE=$?
echo "Inference job exited with code: $INFERENCE_EXIT_CODE"

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Inference completed successfully!"
    echo ""
    echo "Output files in $MODEL_DIR:"
    echo "- inference_results.csv: Results DataFrame with embeddings and predictions"
    echo "- umap_embeddings_by_emotion.png: UMAP visualization"
    if [ -f "$MODEL_DIR/final_ser.pt" ]; then
        echo "- confusion_matrix.png: Confusion matrix visualization"
    fi
    echo ""
    echo "Check the log file for detailed metrics (accuracy, F1 scores, etc.)"
    
    # Display last few lines of the results file
    echo ""
    echo "Preview of inference_results.csv:"
    head -n 5 "$MODEL_DIR/inference_results.csv"
    echo "..."
    echo "Total lines: $(wc -l < "$MODEL_DIR/inference_results.csv")"
else
    echo "Inference failed with exit code $INFERENCE_EXIT_CODE"
    exit $INFERENCE_EXIT_CODE
fi