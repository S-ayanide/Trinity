"""
Model Evaluation Script
Evaluates trained GPT models on test datasets and calculates baseline comparisons
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, 256, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

def load_model(model_path):
    """Load a saved model"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = GPTLanguageModel(
        vocab_size=checkpoint['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint['stoi'], checkpoint['itos'], config

@torch.no_grad()
def evaluate_on_dataset(model, data, block_size, batch_size=64, eval_iters=200):
    """Evaluate model on a dataset"""
    model.eval()
    losses = []
    
    max_batches = min(eval_iters, (len(data) - block_size) // batch_size)
    
    for _ in range(max_batches):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        
        logits, loss = model(x, y)
        losses.append(loss.item())
    
    mean_loss = np.mean(losses)
    perplexity = np.exp(mean_loss)
    
    return {
        'loss': mean_loss,
        'perplexity': perplexity,
        'num_batches': len(losses)
    }

def calculate_baseline_loss(data, vocab_size):
    """Calculate baseline loss for uniform distribution"""
    # Uniform distribution: all tokens equally likely
    uniform_loss = -np.log(1.0 / vocab_size)
    uniform_perplexity = vocab_size
    
    # Character frequency baseline
    char_counts = np.bincount(data.numpy())
    char_probs = char_counts / char_counts.sum()
    char_probs = char_probs[char_probs > 0]  # Remove zeros
    entropy = -np.sum(char_probs * np.log(char_probs))
    
    return {
        'uniform_loss': uniform_loss,
        'uniform_perplexity': uniform_perplexity,
        'frequency_entropy': entropy,
        'frequency_perplexity': np.exp(entropy)
    }

def main():
    print("\n" + "="*80)
    print("MODEL EVALUATION ON TEST SETS")
    print("="*80 + "\n")
    
    # List available models
    models_dir = 'results'
    if not os.path.exists(models_dir):
        print("No models found! Please train models first.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pt')]
    
    if not model_files:
        print("No model files found in results/")
        return
    
    print("Available models:")
    for i, mf in enumerate(model_files, 1):
        print(f"  {i}. {mf}")
    
    # For automation, evaluate all models
    test_datasets = {
        'childSpeech_test': 'Week 9 Assignment/input_childSpeech_testSet.txt',
        'shakespeare': 'Week 9 Assignment/input_shakespeare.txt'
    }
    
    results_summary = {}
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = model_file.replace('_model.pt', '')
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}\n")
        
        try:
            model, stoi, itos, config = load_model(model_path)
            print(f"Model configuration:")
            print(f"  - Parameters: {config['parameters']:.4f}M")
            print(f"  - n_embd: {config['n_embd']}")
            print(f"  - n_head: {config['n_head']}")
            print(f"  - n_layer: {config['n_layer']}")
            print(f"  - block_size: {config['block_size']}")
            
            results_summary[model_name] = {
                'config': config,
                'evaluations': {}
            }
            
            for test_name, test_path in test_datasets.items():
                print(f"\n{'-'*80}")
                print(f"Testing on: {test_name}")
                print(f"{'-'*80}")
                
                # Load test data
                with open(test_path, 'r', encoding='utf-8') as f:
                    test_text = f.read()
                
                # Encode using model's vocabulary
                try:
                    encoded = [stoi[c] for c in test_text if c in stoi]
                    if len(encoded) < config['block_size']:
                        print(f"Warning: Test data too short for block size {config['block_size']}")
                        continue
                    
                    test_data = torch.tensor(encoded, dtype=torch.long)
                    
                    print(f"Test data size: {len(test_data):,} characters")
                    print(f"Vocabulary overlap: {len(encoded)}/{len(test_text)} characters")
                    
                    # Evaluate model
                    eval_results = evaluate_on_dataset(model, test_data, config['block_size'])
                    
                    print(f"\nModel Performance:")
                    print(f"  - Test Loss: {eval_results['loss']:.4f}")
                    print(f"  - Test Perplexity: {eval_results['perplexity']:.4f}")
                    
                    # Calculate baseline
                    baseline = calculate_baseline_loss(test_data, config['vocab_size'])
                    
                    print(f"\nBaseline Comparison:")
                    print(f"  - Uniform random loss: {baseline['uniform_loss']:.4f}")
                    print(f"  - Uniform random perplexity: {baseline['uniform_perplexity']:.2f}")
                    print(f"  - Character frequency entropy: {baseline['frequency_entropy']:.4f}")
                    print(f"  - Character frequency perplexity: {baseline['frequency_perplexity']:.4f}")
                    
                    improvement_over_uniform = (baseline['uniform_loss'] - eval_results['loss']) / baseline['uniform_loss'] * 100
                    improvement_over_freq = (baseline['frequency_entropy'] - eval_results['loss']) / baseline['frequency_entropy'] * 100
                    
                    print(f"\nImprovement:")
                    print(f"  - vs Uniform: {improvement_over_uniform:.2f}%")
                    print(f"  - vs Frequency baseline: {improvement_over_freq:.2f}%")
                    
                    results_summary[model_name]['evaluations'][test_name] = {
                        'test_loss': eval_results['loss'],
                        'test_perplexity': eval_results['perplexity'],
                        'baseline_uniform_loss': baseline['uniform_loss'],
                        'baseline_frequency_loss': baseline['frequency_entropy'],
                        'improvement_over_uniform': improvement_over_uniform,
                        'improvement_over_frequency': improvement_over_freq
                    }
                    
                except KeyError as e:
                    print(f"Error: Test data contains characters not in training vocabulary")
                    print(f"Missing character: {e}")
            
        except Exception as e:
            print(f"Error loading or evaluating model: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    with open('results/evaluation_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print("\nResults saved to: results/evaluation_summary.json")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    print(f"{'Model':<20} {'Test Set':<20} {'Loss':<10} {'Perplexity':<12}")
    print(f"{'-'*80}")
    for model_name, results in results_summary.items():
        for test_name, eval_data in results['evaluations'].items():
            print(f"{model_name:<20} {test_name:<20} {eval_data['test_loss']:<10.4f} {eval_data['test_perplexity']:<12.2f}")
    print()

if __name__ == "__main__":
    main()

