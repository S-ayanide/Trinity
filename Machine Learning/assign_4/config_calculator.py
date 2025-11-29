"""
Configuration Calculator for GPT Model
Calculates the number of parameters for different model configurations
"""

def calculate_parameters(n_embd, n_head, n_layer, block_size, vocab_size, use_bias=False):
    """
    Calculate total parameters in the GPT model
    
    Parameters:
    - n_embd: embedding dimension
    - n_head: number of attention heads
    - n_layer: number of transformer layers
    - block_size: maximum context length
    - vocab_size: vocabulary size
    - use_bias: whether to use bias in attention layers
    """
    head_size = n_embd // n_head
    
    # Token embedding table
    token_emb = vocab_size * n_embd
    
    # Position embedding table  
    pos_emb = block_size * n_embd
    
    # Per transformer block:
    # Multi-head attention
    per_head = 3 * n_embd * head_size  # key, query, value projections
    if use_bias:
        per_head += 3 * head_size  # biases for k, q, v
    
    multihead = n_head * per_head
    multihead += n_embd * n_embd  # projection layer
    if use_bias:
        multihead += n_embd  # projection bias
    
    # Feed forward
    ffwd = n_embd * (4 * n_embd) + (4 * n_embd)  # first linear + bias
    ffwd += (4 * n_embd) * n_embd + n_embd  # second linear + bias
    
    # Layer norms (2 per block)
    ln = 2 * (2 * n_embd)  # gamma and beta for each LayerNorm
    
    per_block = multihead + ffwd + ln
    
    # All blocks
    all_blocks = n_layer * per_block
    
    # Final layer norm
    final_ln = 2 * n_embd
    
    # Language model head
    lm_head = n_embd * vocab_size
    
    total = token_emb + pos_emb + all_blocks + final_ln + lm_head
    
    breakdown = {
        'token_embedding': token_emb,
        'position_embedding': pos_emb,
        'per_block_params': per_block,
        'all_blocks': all_blocks,
        'final_layer_norm': final_ln,
        'lm_head': lm_head,
        'total': total
    }
    
    return breakdown

def print_config(config_name, n_embd, n_head, n_layer, block_size, vocab_size, use_bias=False):
    """Print configuration and parameter count"""
    params = calculate_parameters(n_embd, n_head, n_layer, block_size, vocab_size, use_bias)
    
    print(f"\n{'='*80}")
    print(f"Configuration: {config_name}")
    print(f"{'='*80}")
    print(f"Hyperparameters:")
    print(f"  - n_embd: {n_embd}")
    print(f"  - n_head: {n_head}")
    print(f"  - n_layer: {n_layer}")
    print(f"  - block_size: {block_size}")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - use_bias: {use_bias}")
    print(f"\nParameter Breakdown:")
    print(f"  - Token embedding: {params['token_embedding']:,} ({params['token_embedding']/1e6:.3f}M)")
    print(f"  - Position embedding: {params['position_embedding']:,} ({params['position_embedding']/1e6:.3f}M)")
    print(f"  - Per block: {params['per_block_params']:,} ({params['per_block_params']/1e6:.3f}M)")
    print(f"  - All {n_layer} blocks: {params['all_blocks']:,} ({params['all_blocks']/1e6:.3f}M)")
    print(f"  - Final layer norm: {params['final_layer_norm']:,} ({params['final_layer_norm']/1e6:.3f}M)")
    print(f"  - LM head: {params['lm_head']:,} ({params['lm_head']/1e6:.3f}M)")
    print(f"\n  TOTAL: {params['total']:,} ({params['total']/1e6:.4f}M)")
    print(f"{'='*80}\n")
    
    return params['total']

if __name__ == "__main__":
    # Original configuration (too large)
    print("\n" + "="*80)
    print("ORIGINAL CONFIGURATION (from gpt.py)")
    print("="*80)
    print_config("Original", n_embd=384, n_head=6, n_layer=6, block_size=256, vocab_size=40)
    
    print("\n" + "="*80)
    print("PROPOSED CONFIGURATIONS (< 1M parameters)")
    print("="*80)
    
    # Configuration 1: Reduce embedding dimension significantly
    print("\n" + "*"*80)
    print("CONFIG 1: Reduce embedding dimension (primary reduction)")
    print("Rationale: Embedding dimension affects every component multiplicatively.")
    print("Reducing from 384->96 drastically cuts params while maintaining 4 heads and 4 layers.")
    print("*"*80)
    print_config("Config 1", n_embd=96, n_head=4, n_layer=4, block_size=256, vocab_size=40)
    
    # Configuration 2: Reduce number of layers
    print("\n" + "*"*80)
    print("CONFIG 2: Reduce number of layers (shallow network)")
    print("Rationale: Fewer layers means less depth. Using just 2 layers makes a shallow")
    print("network that's faster to train. Wider embeddings (128) to compensate.")
    print("*"*80)
    print_config("Config 2", n_embd=128, n_head=4, n_layer=2, block_size=256, vocab_size=40)
    
    # Configuration 3: Reduce context window and layers
    print("\n" + "*"*80)
    print("CONFIG 3: Balanced reduction (context + layers)")
    print("Rationale: Smaller context (64) for child speech + 3 layers + moderate embedding.")
    print("Balanced approach that maintains reasonable capacity across all dimensions.")
    print("*"*80)
    print_config("Config 3", n_embd=128, n_head=4, n_layer=3, block_size=64, vocab_size=40)
    
    print("\n" + "="*80)
    print("SUMMARY: All three configs are under 1M parameters")
    print("="*80)

