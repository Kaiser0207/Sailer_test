<div align="center">

# CRAB

**Crab: Multi layer contrastive supervision to improve speech emotion recognition under both prompted and natural conditions**

[![Publication](https://img.shields.io/badge/Publication-Read%20Paper-success?style=flat-square&logo=academia)](your-link-here)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![SLURM](https://img.shields.io/badge/SLURM-HPC%20Ready-green?style=flat-square)](https://slurm.schedmd.com)
[![License](https://img.shields.io/badge/License-Research-orange?style=flat-square)]()

*Official implementation of the CRAB paper*

</div>

---

## ✨ Overview

CRAB is a Speech Emotion Recognition system based on **Contrastive Representation and Multimodal Aligned Bottleneck** — a framework that leverages contrastive learning and multimodal alignment to build robust emotional representations from speech. It is based on a Bi-modal Cross-Modal Transformer architecture on top of WavLM and RoBERTA features. It employs Multi Positive Contrastive Learning (MPCL) loss at different layers of the model to improve speech emotion recognition.

---

## 🛠️ Environment Setup

We provide a setup script that assumes a **Conda** installation. It will automatically create a new environment named `crab` and install all dependencies.

```bash
sh make_crab_env.sh
```

Alternatively, you can install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
crab/
├── bin/                        # Training and inference scripts
├── src/                        # Main source code
└── recipes/
    └── {dataset}/              # Dataset-specific recipes for training and inference
```

- **`bin/`** — Entry-point scripts for launching training and running inference.
- **`src/`** — Core model architecture, data loaders, and utilities.
- **`recipes/`** — Ready-to-use configurations for supported datasets. Use the provided examples as a starting point to adapt CRAB to your own dataset.

---

## 🚀 Training & Inference

We provide SLURM-ready scripts for HPC environments inside the `recipes/` folder.

### Using a pre-defined dataset recipe

Navigate to the corresponding recipe folder and submit the job:

```bash
cd recipes/{dataset}
sbatch train_crab.sh
sbatch test_crab.sh
```

> Each experiment will automatically create an experiment folder containing all corresponding logging files and checkpoints.

---

## 📄 Citation

> Citation coming soon — paper under review.

```bibtex
@article{ueda2026crab,
  title   = {Crab: Multi layer contrastive supervision to improve speech emotion recognition under both prompted and natural conditions},
  year    = {2026},
  author  = {Ueda, Lucas H., Lima, João G.T., Costa, Paula D.P.},
  note    = {Coming soon}
}
```
