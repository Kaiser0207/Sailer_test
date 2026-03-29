# SAILER Project Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [V2.0.0] - 2026-03-29
### Added
- **Config Separation System**: Implemented `argparse` and JSON configuration (`configs/default_config.json`) to control Hyperparameters externally.
- **Learning Rate Scheduler**: Added `get_cosine_schedule_with_warmup` and `weight_decay=1e-4` to combat early gradient bouncing and late-stage convergence instability.
- **Sanity Check (Dry Run)**: `train.py` now runs a single batch validation forward pass before Epoch 1 to catch `Shape Mismatch` and OOM errors instantly.
- **Vectorized Masking**: Transformed the slow `for` loop in `sailer_model.py`'s temporal mask pooling into a vectorized Broadcasting op, unlocking GPU utilization.
- **L2 Normalization Fusion**: Enforced `F.normalize(p=2)` on both text and speech embeddings before concatenation to prevent 1024D RoBERTa features from dominating 256D Whisper features.
- **Dev-Set Unagreed Data Merging**: Updated `msp_dataset.py` to fetch "No Agreement" and "Other" audio samples from the Development split and merge them into the Training set for maximal data utilization.

### Changed
- Replaced custom `seed_everything` with HuggingFace `transformers.set_seed` for comprehensive determinism across platforms.
- `train.py` correctly imports Whisper Encoder-Only architecture without `ignore_mismatched_sizes`, retaining perfect Positional Embeddings and dropping unnecessary Decoder VRAM overhead.
