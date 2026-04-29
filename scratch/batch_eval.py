import os
import torch
import subprocess

models = {
    "Phase 1 (Original Head)": "experiments/20260418_234018_SAILER_v4.0_Interview_4Class",
    "Phase 2 (New Peak)": "experiments/20260423_020544_SAILER_v4.0_Phase2_FinalPeak_NoSampler",
    "Phase 3 (Previous Deep)": "experiments/20260422_111122_SAILER_v4.0_Phase2_Full"
}

for name, path in models.items():
    print(f"\n>>> Evaluating {name}...")
    # Assuming evaluate_4class_models.py takes model path as input
    # Adjusting the command based on typical script usage
    subprocess.run([
        ".venv/bin/python3", "scripts/evaluate_4class_models.py", 
        "--model_path", path
    ])
