import torch
import gc
from transformers import WhisperModel

def print_vram(step_name):
    """Retrieve and print current VRAM usage in GB."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[{step_name:<30}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    return allocated

def clear_vram():
    """Force clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Cannot perform VRAM test.")
        return
        
    device = torch.device("cuda")
    clear_vram()
    
    print("======================================================")
    print("[Experiment 1] SAILER Architecture: Whisper Encoder Only")
    print("======================================================")
    print_vram("1. Initial State")
    
    encoder_only_model = WhisperModel.from_pretrained(
        "openai/whisper-large-v3"
    ).encoder.to(device)
    
    enc_vram = print_vram("2. Whisper Encoder Loaded")
    
    print("\n[System Info] Clearing memory for control experiment...\n")
    del encoder_only_model
    clear_vram()
    
    print("======================================================")
    print("[Experiment 2] Control Group: Full Whisper-Large-V3")
    print("======================================================")
    print_vram("1. Initial State")
    
    full_model = WhisperModel.from_pretrained(
        "openai/whisper-large-v3"
    ).to(device)
    
    full_vram = print_vram("2. Full Whisper Loaded")
    
    print("\n======================================================")
    print("[VRAM Consumption Comparison Report]")
    print("======================================================")
    print(f"- Full Whisper VRAM Usage    : {full_vram:.2f} GB")
    print(f"- Encoder-Only VRAM Usage    : {enc_vram:.2f} GB")
    print(f"- VRAM Saved by Encoder-Only : {full_vram - enc_vram:.2f} GB")
    print("======================================================")

if __name__ == "__main__":
    main()
