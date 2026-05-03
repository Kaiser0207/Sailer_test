#!/bin/bash
# Crab MSP-Podcast 2.0 Test Script (Local, non-SLURM)
# Tests the trained model on Test1 and Test2 partitions

set -e

PROJ_DIR="/home/brant/Project/SAILER_test"
AUDIO_DIR="${PROJ_DIR}/datasets/MSP_Podcast_Data/Audios/"
CSV_PATH="${PROJ_DIR}/Crab/data/msp2_processed_labels.csv"
MODEL_DIR="${PROJ_DIR}/Crab/experiments/msp/crab"

# Check model directory
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found at $MODEL_DIR"
    exit 1
fi

echo "Checking model files..."
for f in final_ssl.pt final_text.pt final_ser.pt; do
    if [ -f "$MODEL_DIR/$f" ]; then
        echo "  ✓ Found: $f"
    else
        echo "  ✗ Missing: $f"
        exit 1
    fi
done

# Activate uv environment
source ${PROJ_DIR}/Crab/.venv/bin/activate

# Test on Test1 and Test2
for DTYPE in test1 test2; do
    echo ""
    echo "========================================"
    echo "Testing on ${DTYPE}..."
    echo "========================================"

    python ${PROJ_DIR}/Crab/bin/test_crab.py \
      --df_path "$CSV_PATH" \
      --wav_base_dir "$AUDIO_DIR" \
      --model_path "$MODEL_DIR" \
      --ssl_type "microsoft/wavlm-large" \
      --text_model_path "roberta-large" \
      --fusion_hidden_dim 512 \
      --text_max_len 128 \
      --batch_size 1 \
      --seed 42 \
      --dtype "$DTYPE" \
      --use_wandb

    echo "${DTYPE} inference completed!"
done

echo ""
echo "All tests completed!"
