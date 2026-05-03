#!/usr/bin/env python
"""
Download speech models (WavLM, Wav2Vec2, HuBERT, Whisper) with comprehensive testing
"""

import os
import argparse
from transformers import AutoModel, AutoFeatureExtractor, AutoProcessor
from huggingface_hub import snapshot_download
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Model configurations
MODEL_CONFIGS = {
    # WavLM models
    "wavlm-base": "microsoft/wavlm-base",
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "wavlm-large": "microsoft/wavlm-large",
    
    # Wav2Vec2 models
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wav2vec2-large": "facebook/wav2vec2-large",
    "wav2vec2-large-960h": "facebook/wav2vec2-large-960h",
    "wav2vec2-xlsr-53": "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    "wav2vec2-large-lv60": "facebook/wav2vec2-large-lv60",
    "w2v-bert-2.0": "facebook/w2v-bert-2.0",

    # HuBERT models
    "hubert-base": "facebook/hubert-base-ls960",
    "hubert-large": "facebook/hubert-large-ll60k",
    "hubert-xlarge": "facebook/hubert-xlarge-ls960-ft",
    
    # Whisper models
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
    "whisper-large": "openai/whisper-large",
    "whisper-large-v2": "openai/whisper-large-v2",
    "whisper-large-v3": "openai/whisper-large-v3",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Download and test speech models")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to download. Choose from: " + ", ".join(MODEL_CONFIGS.keys())
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the model (default: ~/github/MM-ser/bin/models/{model_name})"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip comprehensive testing after download"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for testing (default: auto-detect)"
    )
    return parser.parse_args()

def is_whisper_model(model_name):
    """Check if the model is a Whisper model"""
    return "whisper" in model_name

def download_model(model_key, save_dir):
    """Download model to specified directory"""
    model_name = MODEL_CONFIGS[model_key]
    
    print(f"Downloading {model_name} to {save_dir}...")
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Download complete snapshot
    print("Downloading complete model snapshot...")
    snapshot_download(
        repo_id=model_name,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.safetensors"]
    )
    
    # Load and save model
    print("\nLoading model...")
    model = AutoModel.from_pretrained(model_name)
    
    # Use appropriate processor/feature extractor
    if is_whisper_model(model_key):
        processor = AutoProcessor.from_pretrained(model_name)
        processor.save_pretrained(save_dir)
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        feature_extractor.save_pretrained(save_dir)
    
    # Save model
    print("Saving model...")
    model.save_pretrained(save_dir)
    
    return model, model_name

def test_model(model_key, save_dir, device=None):
    """Comprehensive model testing"""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL TESTING")
    print("="*70)
    
    # Load model from local path
    print("\n1. Loading model from local path...")
    model = AutoModel.from_pretrained(save_dir, local_files_only=True)
    model.eval()
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    # Model-specific configurations
    is_whisper = is_whisper_model(model_key)
    
    # Test 1: Simple forward pass
    print("\n2. Testing simple forward pass (no attention mask)...")
    with torch.no_grad():
        # Create realistic audio input (1 second at 16kHz)
        test_input = torch.randn(2, 16000).to(device)
        
        if is_whisper:
            # Whisper needs decoder_input_ids
            decoder_input_ids = torch.tensor([[1, 1]] * 2).to(device)
            output = model(test_input, decoder_input_ids=decoder_input_ids)
            output_tensor = output.last_hidden_state
        else:
            output = model(test_input)
            output_tensor = output.last_hidden_state
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output_tensor.shape}")
        print(f"  Has NaN: {torch.isnan(output_tensor).any()}")
        print(f"  Output range: [{output_tensor.min():.4f}, {output_tensor.max():.4f}]")
        print(f"  First 5 values of output[0,0,:]: {output_tensor[0,0,:5]}")
    
    # Test 2: Forward pass with attention mask (not for Whisper)
    if not is_whisper:
        print("\n3. Testing forward pass WITH attention mask...")
        with torch.no_grad():
            # Create input and mask
            test_input = torch.randn(2, 16000).to(device)
            attention_mask = torch.ones(2, 16000).to(device)
            
            # Test with full mask (all ones)
            output_full_mask = model(test_input, attention_mask=attention_mask)
            print(f"✓ Forward pass with full mask successful")
            print(f"  Has NaN: {torch.isnan(output_full_mask.last_hidden_state).any()}")
            
            # Test with partial mask
            attention_mask[0, 8000:] = 0  # Mask out second half of first sample
            attention_mask[1, 12000:] = 0  # Mask out last quarter of second sample
            
            output_partial_mask = model(test_input, attention_mask=attention_mask)
            print(f"✓ Forward pass with partial mask successful")
            print(f"  Has NaN: {torch.isnan(output_partial_mask.last_hidden_state).any()}")
    
    # Test 3: Different input lengths
    print("\n4. Testing different input lengths...")
    test_lengths = [8000, 16000, 32000, 48000]  # 0.5s, 1s, 2s, 3s
    for length in test_lengths:
        with torch.no_grad():
            test_input = torch.randn(1, length).to(device)
            
            if is_whisper:
                decoder_input_ids = torch.tensor([[1, 1]]).to(device)
                output = model(test_input, decoder_input_ids=decoder_input_ids)
                output_tensor = output.last_hidden_state
            else:
                test_mask = torch.ones(1, length).to(device)
                output = model(test_input, attention_mask=test_mask)
                output_tensor = output.last_hidden_state
            
            print(f"  Length {length} ({length/16000:.1f}s): "
                  f"Output shape: {output_tensor.shape}, "
                  f"Has NaN: {torch.isnan(output_tensor).any()}")
    
    # Test 4: Freeze feature encoder (not for Whisper)
    if not is_whisper and hasattr(model, 'freeze_feature_encoder'):
        print("\n5. Testing with frozen feature encoder...")
        model.freeze_feature_encoder()
        with torch.no_grad():
            test_input = torch.randn(2, 16000).to(device)
            output = model(test_input)
            print(f"✓ Forward pass with frozen feature encoder")
            print(f"  Has NaN: {torch.isnan(output.last_hidden_state).any()}")
    
    # Test 5: Training mode
    print("\n6. Testing in training mode...")
    model.train()
    with torch.no_grad():
        test_input = torch.randn(2, 16000).to(device)
        
        if is_whisper:
            decoder_input_ids = torch.tensor([[1, 1]] * 2).to(device)
            output = model(test_input, decoder_input_ids=decoder_input_ids)
            output_tensor = output.last_hidden_state
        else:
            output = model(test_input)
            output_tensor = output.last_hidden_state
        
        print(f"✓ Forward pass in training mode")
        print(f"  Has NaN: {torch.isnan(output_tensor).any()}")
    
    # Visualization
    print("\n7. Creating visualization...")
    model.eval()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_key.upper()} Model Analysis', fontsize=16)
    
    # Generate test data
    with torch.no_grad():
        # Create input waveform
        time = np.linspace(0, 1, 16000)
        # Mix of frequencies to simulate speech
        waveform = (0.5 * np.sin(2 * np.pi * 440 * time) + 
                   0.3 * np.sin(2 * np.pi * 880 * time) + 
                   0.2 * np.random.randn(16000))
        
        test_input = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
        
        if is_whisper:
            decoder_input_ids = torch.tensor([[1, 1]]).to(device)
            output = model(test_input, decoder_input_ids=decoder_input_ids)
            hidden_states = output.last_hidden_state.cpu().numpy()
        else:
            attention_mask = torch.ones_like(test_input).to(device)
            output = model(test_input, attention_mask=attention_mask)
            hidden_states = output.last_hidden_state.cpu().numpy()
    
    # Plot 1: Input waveform
    ax1 = axes[0, 0]
    ax1.plot(time[:1000], waveform[:1000])
    ax1.set_title('Input Waveform (first 1000 samples)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Plot 2: First two hidden dimensions
    ax2 = axes[0, 1]
    ax2.plot(hidden_states[0, :, 0], label='Dim 0', alpha=0.7)
    ax2.plot(hidden_states[0, :, 1], label='Dim 1', alpha=0.7)
    ax2.set_title('First Two Hidden Dimensions')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Hidden state statistics
    ax3 = axes[1, 0]
    mean_per_timestep = hidden_states[0].mean(axis=1)
    std_per_timestep = hidden_states[0].std(axis=1)
    timesteps = np.arange(len(mean_per_timestep))
    ax3.plot(timesteps, mean_per_timestep, label='Mean')
    ax3.fill_between(timesteps, 
                     mean_per_timestep - std_per_timestep,
                     mean_per_timestep + std_per_timestep,
                     alpha=0.3, label='±1 STD')
    ax3.set_title('Hidden State Statistics Over Time')
    ax3.set_xlabel('Time steps')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Hidden state heatmap (first 50 timesteps, first 50 dims)
    ax4 = axes[1, 1]
    max_timesteps = min(50, hidden_states.shape[1])
    max_dims = min(50, hidden_states.shape[2])
    im = ax4.imshow(hidden_states[0, :max_timesteps, :max_dims].T, aspect='auto', cmap='RdBu_r')
    ax4.set_title(f'Hidden States Heatmap (first {max_timesteps} timesteps × {max_dims} dims)')
    ax4.set_xlabel('Time steps')
    ax4.set_ylabel('Hidden dimensions')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(save_dir), f'{model_key}_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_path}")
    
    # Print summary statistics
    print("\n8. Summary Statistics:")
    print(f"  Hidden state mean: {hidden_states.mean():.4f}")
    print(f"  Hidden state std: {hidden_states.std():.4f}")
    print(f"  Hidden state min: {hidden_states.min():.4f}")
    print(f"  Hidden state max: {hidden_states.max():.4f}")
    print(f"  Contains NaN: {np.isnan(hidden_states).any()}")
    print(f"  Contains Inf: {np.isinf(hidden_states).any()}")
    
    # Test edge cases
    print("\n9. Testing edge cases...")
    
    # Very short input
    with torch.no_grad():
        short_input = torch.randn(1, 400).to(device)  # 0.025 seconds
        try:
            if is_whisper:
                decoder_input_ids = torch.tensor([[1, 1]]).to(device)
                output = model(short_input, decoder_input_ids=decoder_input_ids)
            else:
                output = model(short_input)
            print(f"  Very short input (400 samples): Success, has NaN: {torch.isnan(output.last_hidden_state).any()}")
        except Exception as e:
            print(f"  Very short input (400 samples): Failed - {e}")
    
    # Zero input
    with torch.no_grad():
        zero_input = torch.zeros(1, 16000).to(device)
        if is_whisper:
            decoder_input_ids = torch.tensor([[1, 1]]).to(device)
            output = model(zero_input, decoder_input_ids=decoder_input_ids)
        else:
            output = model(zero_input)
        print(f"  Zero input: Success, has NaN: {torch.isnan(output.last_hidden_state).any()}")
    
    # Very large input
    with torch.no_grad():
        large_input = torch.randn(1, 16000).to(device) * 100
        if is_whisper:
            decoder_input_ids = torch.tensor([[1, 1]]).to(device)
            output = model(large_input, decoder_input_ids=decoder_input_ids)
        else:
            output = model(large_input)
        print(f"  Large magnitude input: Success, has NaN: {torch.isnan(output.last_hidden_state).any()}")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)

def main():
    args = parse_args()
    
    # Set save directory
    if args.save_dir is None:
        save_dir = os.path.expanduser(f"~/github/MM-ser/bin/models/{args.model}")
    else:
        save_dir = os.path.expanduser(args.save_dir)
    
    try:
        # Download model
        model, model_name = download_model(args.model, save_dir)
        
        # Run tests unless skipped
        if not args.skip_tests:
            test_model(args.model, save_dir, args.device)
        
        # Print usage instructions
        print("\n" + "="*70)
        print("USAGE INSTRUCTIONS")
        print("="*70)
        print(f"1. Use in your training script:")
        print(f"   ssl_model = AutoModel.from_pretrained('{save_dir}')")
        print(f"\n2. For offline mode, add to your script:")
        print(f"   export TRANSFORMERS_OFFLINE=1")
        print(f"   export HF_HOME='{os.path.dirname(os.path.dirname(save_dir))}'")
        
        if not is_whisper_model(args.model):
            print(f"\n3. If you get NaN, check:")
            print(f"   - Your input audio normalization")
            print(f"   - The attention mask dimensions")
            print(f"   - Try without freeze_feature_encoder()")
        else:
            print(f"\n3. Note: Whisper models require decoder_input_ids")
            print(f"   Example: decoder_input_ids = torch.tensor([[1, 1]])")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()