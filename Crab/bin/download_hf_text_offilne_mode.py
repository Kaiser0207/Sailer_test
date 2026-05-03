#!/usr/bin/env python
"""
Download text models (RoBERTa, BERT, DistilBERT, ALBERT, DeBERTa) with comprehensive testing
"""

import os
import argparse
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Model configurations
MODEL_CONFIGS = {
    # RoBERTa models
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "distilroberta-base": "distilroberta-base",
    
    # BERT models
    "bert-base-uncased": "bert-base-uncased",
    "bert-base-cased": "bert-base-cased",
    "bert-large-uncased": "bert-large-uncased",
    "bert-large-cased": "bert-large-cased",
    
    # DistilBERT models
    "distilbert-base-uncased": "distilbert-base-uncased",
    "distilbert-base-cased": "distilbert-base-cased",
    
    # ALBERT models
    "albert-base-v2": "albert-base-v2",
    "albert-large-v2": "albert-large-v2",
    "albert-xlarge-v2": "albert-xlarge-v2",
    "albert-xxlarge-v2": "albert-xxlarge-v2",
    
    # DeBERTa models
    "deberta-base": "microsoft/deberta-base",
    "deberta-large": "microsoft/deberta-large",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    
    # Sentence transformers
    "sentence-bert": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-roberta": "sentence-transformers/all-roberta-large-v1",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Download and test text models")
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
    
    # Load and save tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save both
    print("Saving model and tokenizer...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    return model, tokenizer, model_name

def test_model(model_key, save_dir, device=None):
    """Comprehensive model testing"""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEXT MODEL TESTING")
    print("="*70)
    
    # Load model and tokenizer from local path
    print("\n1. Loading model and tokenizer from local path...")
    model = AutoModel.from_pretrained(save_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
    model.eval()
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on device: {device}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models process natural language efficiently.",
        "This is a test.",
        "A" * 512,  # Very long sentence
        "😊 Emojis and special characters: @#$%^&*()",
        "",  # Empty string
    ]
    
    # Test 1: Simple forward pass
    print("\n2. Testing simple forward pass...")
    with torch.no_grad():
        inputs = tokenizer(
            test_sentences[:2], 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {inputs['input_ids'].shape}")
        print(f"  Last hidden state shape: {outputs.last_hidden_state.shape}")
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            print(f"  Pooler output shape: {outputs.pooler_output.shape}")
        print(f"  Has NaN: {torch.isnan(outputs.last_hidden_state).any()}")
        print(f"  Output range: [{outputs.last_hidden_state.min():.4f}, {outputs.last_hidden_state.max():.4f}]")
    
    # Test 2: Different sequence lengths
    print("\n3. Testing different sequence lengths...")
    test_lengths = [8, 16, 32, 64, 128, 256, 512]
    for max_length in test_lengths:
        with torch.no_grad():
            inputs = tokenizer(
                test_sentences[0], 
                padding='max_length',
                max_length=max_length,
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            
            print(f"  Max length {max_length}: "
                  f"Output shape: {outputs.last_hidden_state.shape}, "
                  f"Has NaN: {torch.isnan(outputs.last_hidden_state).any()}")
    
    # Test 3: Batch processing
    print("\n4. Testing batch processing...")
    batch_sizes = [1, 4, 8, 16]
    for batch_size in batch_sizes:
        with torch.no_grad():
            batch_sentences = test_sentences[:batch_size] if batch_size <= len(test_sentences) else test_sentences * (batch_size // len(test_sentences) + 1)
            batch_sentences = batch_sentences[:batch_size]
            
            inputs = tokenizer(
                batch_sentences, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            
            print(f"  Batch size {batch_size}: "
                  f"Output shape: {outputs.last_hidden_state.shape}, "
                  f"Has NaN: {torch.isnan(outputs.last_hidden_state).any()}")
    
    # Test 4: Special tokens
    print("\n5. Testing special tokens...")
    special_tests = [
        ("[CLS] Test [SEP]", "with special tokens"),
        (tokenizer.pad_token * 10, "only padding tokens"),
        (tokenizer.unk_token * 5, "only unknown tokens"),
    ]
    
    for test_text, desc in special_tests:
        with torch.no_grad():
            inputs = tokenizer(
                test_text, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            print(f"  {desc}: Has NaN: {torch.isnan(outputs.last_hidden_state).any()}")
    
    # Test 5: Training mode
    print("\n6. Testing in training mode...")
    model.train()
    with torch.no_grad():
        inputs = tokenizer(
            test_sentences[:2], 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)
        print(f"✓ Forward pass in training mode")
        print(f"  Has NaN: {torch.isnan(outputs.last_hidden_state).any()}")
    
    # Visualization
    print("\n7. Creating visualization...")
    model.eval()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_key.upper()} Model Analysis', fontsize=16)
    
    # Generate embeddings for visualization
    viz_sentences = [
        "I love this movie!",
        "I hate this movie!",
        "The weather is nice today.",
        "The weather is terrible today.",
        "This is a neutral statement.",
    ]
    
    with torch.no_grad():
        inputs = tokenizer(
            viz_sentences, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state.cpu().numpy()
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooler_outputs = outputs.pooler_output.cpu().numpy()
        else:
            # Use mean pooling if no pooler output
            mask = inputs['attention_mask'].cpu().numpy()
            pooler_outputs = (hidden_states * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    
    # Plot 1: Token embeddings for first sentence
    ax1 = axes[0, 0]
    tokens = tokenizer.tokenize(viz_sentences[0])
    token_embeddings = hidden_states[0, 1:len(tokens)+1, :5]  # First 5 dims, skip CLS
    for i in range(min(5, token_embeddings.shape[1])):
        ax1.plot(token_embeddings[:, i], label=f'Dim {i}', alpha=0.7)
    ax1.set_title('Token Embeddings (First 5 Dimensions)')
    ax1.set_xlabel('Token Position')
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=45)
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Sentence embeddings similarity
    ax2 = axes[0, 1]
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(pooler_outputs)
    im = ax2.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_title('Sentence Similarity Matrix')
    ax2.set_xticks(range(len(viz_sentences)))
    ax2.set_yticks(range(len(viz_sentences)))
    ax2.set_xticklabels([f"S{i+1}" for i in range(len(viz_sentences))])
    ax2.set_yticklabels([f"S{i+1}" for i in range(len(viz_sentences))])
    
    # Add similarity values
    for i in range(len(viz_sentences)):
        for j in range(len(viz_sentences)):
            ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=10)
    
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Attention mask visualization
    ax3 = axes[1, 0]
    attention_masks = inputs['attention_mask'].cpu().numpy()
    ax3.imshow(attention_masks, aspect='auto', cmap='Blues')
    ax3.set_title('Attention Masks')
    ax3.set_xlabel('Token Position')
    ax3.set_ylabel('Sentence')
    ax3.set_yticks(range(len(viz_sentences)))
    ax3.set_yticklabels([f"S{i+1}: {s[:20]}..." for i, s in enumerate(viz_sentences)])
    
    # Plot 4: Hidden state statistics
    ax4 = axes[1, 1]
    mean_per_token = hidden_states.mean(axis=2)  # Average across hidden dims
    for i in range(min(3, len(viz_sentences))):
        ax4.plot(mean_per_token[i], label=f'Sentence {i+1}', alpha=0.7)
    ax4.set_title('Mean Hidden State Values per Token')
    ax4.set_xlabel('Token Position')
    ax4.set_ylabel('Mean Value')
    ax4.legend()
    ax4.grid(True)
    
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
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        print(f"  Pooler output mean: {pooler_outputs.mean():.4f}")
        print(f"  Pooler output std: {pooler_outputs.std():.4f}")
    print(f"  Contains NaN: {np.isnan(hidden_states).any()}")
    print(f"  Contains Inf: {np.isinf(hidden_states).any()}")
    
    # Test edge cases
    print("\n9. Testing edge cases...")
    
    # Very long input
    with torch.no_grad():
        long_text = "word " * 1000  # Very long text
        inputs = tokenizer(
            long_text, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        print(f"  Very long input: Success, has NaN: {torch.isnan(outputs.last_hidden_state).any()}")
    
    # Empty input
    with torch.no_grad():
        try:
            inputs = tokenizer(
                "", 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            print(f"  Empty input: Success, has NaN: {torch.isnan(outputs.last_hidden_state).any()}")
        except Exception as e:
            print(f"  Empty input: Failed - {e}")
    
    # Only special characters
    with torch.no_grad():
        special_text = "!@#$%^&*()"
        inputs = tokenizer(
            special_text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        print(f"  Special characters only: Success, has NaN: {torch.isnan(outputs.last_hidden_state).any()}")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Print sentences used for visualization
    print("\nSentences used in visualization:")
    for i, sent in enumerate(viz_sentences):
        print(f"  S{i+1}: {sent}")

def main():
    args = parse_args()
    
    # Set save directory
    if args.save_dir is None:
        save_dir = os.path.expanduser(f"~/github/MM-ser/bin/models/{args.model}")
    else:
        save_dir = os.path.expanduser(args.save_dir)
    
    try:
        # Download model
        model, tokenizer, model_name = download_model(args.model, save_dir)
        
        # Run tests unless skipped
        if not args.skip_tests:
            test_model(args.model, save_dir, args.device)
        
        # Print usage instructions
        print("\n" + "="*70)
        print("USAGE INSTRUCTIONS")
        print("="*70)
        print(f"1. Use in your training script:")
        print(f"   text_model = AutoModel.from_pretrained('{save_dir}')")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{save_dir}')")
        print(f"\n2. For offline mode, add to your script:")
        print(f"   export TRANSFORMERS_OFFLINE=1")
        print(f"   export HF_HOME='{os.path.dirname(os.path.dirname(save_dir))}'")
        
        print(f"\n3. Example usage:")
        print(f"   inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')")
        print(f"   outputs = text_model(**inputs)")
        print(f"   # Use outputs.pooler_output or outputs.last_hidden_state")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()