# SAILER: Speech-Audio Integrated Learning for Emotion Recognition

This repository contains the official implementation of the SAILER framework, a top-tier system designed for the Naturalistic Conditions Challenge for Categorized Emotion Prediction (IS2025-SER).

## 🚀 Architecture Overview
SAILER is a "Best Single System" baseline that completely discards complex, slow ensembling in favor of a clean, highly optimized multi-task learning architecture:
1. **Speech Pipeline**: Uses **Whisper-Large-V3 (Encoder Only)** + Learnable Temporal Pooling to extract pristine 256D acoustic features. By discarding the Decoder, SAILER saves ~1.7GB VRAM per GPU.
2. **Text Pipeline**: Uses **RoBERTa-Large** with a 25-layer Learnable Weighted Average mechanism to compress semantic depth into a 1024D representation.
3. **Multimodal Fusion**: Concatenates L2-Normalized features to combat numerical domination, predicting Primary (8 classes), Secondary (17 classes), and AVD regressions simultaneously.

## 🛠️ Installation & Setup

Ensure you have Python 3.10+ installed.

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install requirements (Assuming PyTorch with CUDA is pre-installed)
pip install -r requirements.txt
```

## ⚙️ Configuration
All hyperparameters and paths are decoupled from the code. Before training, edit the JSON config file:
`configs/default_config.json`

```json
{
    "model_name": "SAILER_BestSingleSystem",
    "data_dir": "/path/to/MSP_Podcast_Data",
    "epochs": 15,
    "batch_size": 64,
    "learning_rate": 0.0005,
    ...
}
```

## 🧠 Training the Model

To launch the training pipeline, simply run:
```bash
python train.py --config configs/default_config.json
```

### Advanced Features Included:
- **Sanity Check**: Before burning hours of GPU time, the framework automatically performs a "dry run" forward pass on the validation set to ensure zero shape-mismatches.
- **Auto-Cleanup Exceptions**: If the training crashes or gets interrupted via `Ctrl+C` before saving any checkpoints, the `ExperimentTracker` will automatically nuke the empty garbage directory and W&B instances to keep your file system clean.
- **Cosine Warmup & Weight Decay**: Employs advanced `transformers` schedulers for maximum convergence stability.
- **Dev-Set Merging**: Extracts highly ambiguous "No Agreement" labels from the Development set and injects them into the Training pool for extreme data augmentation.

## 📊 Evaluation
Evaluation metrics (Macro-F1, Accuracy, and Minority mAP) are printed directly to the terminal, and saved alongside loss curves inside `experiments/<timestamp>_SAILER/logs/train.log`. Visual confusion matrices and Tensorboard graphs are simultaneously generated.
