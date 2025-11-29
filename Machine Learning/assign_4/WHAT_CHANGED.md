# What Changed from Original gpt.py

This document details all modifications made to the original `gpt.py` for each assignment part.

## File Naming Convention

Files are named according to assignment structure:
- `gpt-i-c-config[1-3].py` - Part (i)c: Three configurations
- `gpt-i-d-bias.py` - Part (i)d: Bias exploration  
- `gpt-i-e-noskip.py` - Part (i)e: Skip connections
- `gpt-ii-evaluate.py` - Part (ii): Evaluation

## Common Changes (All Training Scripts)

Every training script includes these modifications from the original:

### 1. Import Statement (Line 7)
```python
# ADDED:
from tqdm import tqdm
```
**Purpose:** Progress bar visualization

### 2. Dataset Path (Line 22)
```python
# ORIGINAL:
with open('input.txt', 'r', encoding='utf-8') as f:

# CHANGED TO:
with open('Week 9 Assignment/input_childSpeech_trainingSet.txt', 'r', encoding='utf-8') as f:
```
**Purpose:** Load child speech training data

### 3. Training Iterations (Line 8)
```python
# ORIGINAL:
max_iters = 5000

# CHANGED TO:
max_iters = 1000
```
**Purpose:** Faster execution for testing (increase to 5000 for final results)

### 4. Evaluation Interval (Line 9)
```python
# ORIGINAL:
eval_interval = 500

# CHANGED TO:
eval_interval = 100
```
**Purpose:** More frequent loss reporting

### 5. Training Loop (Lines 206-211)
```python
# ORIGINAL:
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# CHANGED TO:
for iter in tqdm(range(max_iters), desc="Training [Config Name]"):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        tqdm.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
```
**Purpose:** Progress bar with clean output

### 6. History Tracking (Added after line 204)
```python
# ADDED:
import json
import os

history = {
    'train_loss': [],
    'val_loss': [],
    'iterations': [],
    'config': {
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
        'vocab_size': vocab_size,
        'parameters': num_params
    }
}

# IN LOOP:
history['iterations'].append(iter)
history['train_loss'].append(losses['train'].item())
history['val_loss'].append(losses['val'].item())
```
**Purpose:** Save training curves for analysis

### 7. Save Results (Added after training loop)
```python
# ADDED:
os.makedirs('results', exist_ok=True)

with open('results/config[X]_history.json', 'w') as f:
    json.dump(history, f, indent=2)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': history['config'],
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos
}, 'results/config[X]_model.pt')

with open('results/config[X]_generated.txt', 'w') as f:
    f.write(generated)
```
**Purpose:** Persist results for later analysis

## Configuration-Specific Changes

### gpt-i-c-config1.py (Reduced Embedding)

**Lines 13-16:**
```python
# ORIGINAL:
n_embd = 384
n_head = 6
n_layer = 6

# CHANGED TO:
n_embd = 96   # Reduced from 384
n_head = 4    # Reduced from 6
n_layer = 4   # Reduced from 6
```

**Result:** 0.478M parameters (from 10.77M)

**Rationale:** Embedding dimension affects all components multiplicatively. Reducing it provides the largest parameter reduction while maintaining reasonable depth.

---

### gpt-i-c-config2.py (Shallow Network)

**Lines 13-16:**
```python
# ORIGINAL:
n_embd = 384
n_head = 6
n_layer = 6

# CHANGED TO:
n_embd = 128  # Moderately reduced
n_head = 4    # Reduced from 6
n_layer = 2   # Drastically reduced - only 2 layers!
```

**Result:** 0.439M parameters

**Rationale:** Tests whether depth matters more than width. Shallow network with wider embeddings to compensate for lost layers.

---

### gpt-i-c-config3.py (Balanced Reduction)

**Lines 13-17:**
```python
# ORIGINAL:
n_embd = 384
n_head = 6
n_layer = 6
block_size = 256

# CHANGED TO:
n_embd = 128        # Moderately reduced
n_head = 4          # Reduced from 6
n_layer = 3         # Moderately reduced
block_size = 64     # Reduced from 256
```

**Result:** 0.612M parameters

**Rationale:** Balanced approach reducing both context window and depth. Child speech has short phrases, so 64 tokens should suffice.

---

### gpt-i-d-bias.py (Bias Exploration)

**Same hyperparameters as Config 3, BUT:**

**Lines 69-71 in Head class:**
```python
# ORIGINAL:
self.key = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)

# CHANGED TO:
self.key = nn.Linear(n_embd, head_size, bias=True)    # ← bias=True
self.query = nn.Linear(n_embd, head_size, bias=True)  # ← bias=True
self.value = nn.Linear(n_embd, head_size, bias=True)  # ← bias=True
```

**Result:** Slightly more parameters (~9K additional)

**Purpose:** Evaluate whether bias terms in attention projections improve performance.

---

### gpt-i-e-noskip.py (Skip Connection Removal)

**Same hyperparameters as Config 3, BUT:**

**Lines 134-135 in Block class forward():**
```python
# ORIGINAL:
def forward(self, x):
    x = x + self.sa(self.ln1(x))     # Skip connection via "x +"
    x = x + self.ffwd(self.ln2(x))   # Skip connection via "x +"
    return x

# CHANGED TO:
def forward(self, x):
    x = self.sa(self.ln1(x))     # ← Removed "x +"
    x = self.ffwd(self.ln2(x))   # ← Removed "x +"
    return x
```

**Result:** Same parameter count, different training dynamics

**Purpose:** Demonstrate importance of residual connections for gradient flow.

---

## gpt-ii-evaluate.py (New File)

This is a completely new file, not a modification of gpt.py.

**Purpose:** Evaluate all trained models on test sets.

**Key Functions:**
1. `load_model()` - Loads saved checkpoints
2. `evaluate_on_dataset()` - Computes test loss
3. `calculate_baseline_loss()` - Uniform and frequency baselines
4. Evaluates on both child speech test set and Shakespeare

**Output:** `results/evaluation_summary.json` with:
- Test losses
- Perplexity values
- Baseline comparisons
- Improvement percentages

---

## Helper Files (New)

### config_calculator.py
- Calculates parameters for different configurations
- Validates parameter counts
- Not a modification of gpt.py - standalone utility

### visualize_results.py
- Loads JSON training histories
- Generates comparison plots
- Creates individual detailed plots
- Not a modification of gpt.py - standalone utility

### run_all_experiments.py
- Master script to run all experiments sequentially
- Tracks execution and reports results
- Not a modification of gpt.py - standalone utility

---

## Summary of Line Changes

For each training script:
- **~250 lines total** (same as original)
- **~15 lines modified** (hyperparameters, imports, loop)
- **~30 lines added** (history tracking, saving)
- **0-2 lines changed** for architectural modifications (bias, skip)

**Total code modification: ~20% of file**
**Core model architecture: >95% unchanged**

The scripts are nearly identical to the original `gpt.py`, with only:
1. Different hyperparameter values (5 numbers)
2. Progress bar wrapper (1 line + 1 import)
3. Result saving (20 lines)
4. Architecture tweaks (0-2 lines for bias/skip experiments)

This demonstrates that the assignment is about understanding hyperparameter effects and architectural components, not writing a transformer from scratch.

