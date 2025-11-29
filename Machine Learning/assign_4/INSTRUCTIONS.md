# Week 9 Assignment - Execution Guide

This document explains what each script does and how to run them in order.

## File Structure (Mapped to Assignment Questions)

```
Part (i)a - Dataset Analysis:
  - config_calculator.py       → Calculate model parameters
  - gpt-i-a-analysis.txt       → Dataset analysis results

Part (i)b - Downsize model to <1M parameters:
  - gpt-i-c-config1.py         → Config 1 (0.478M params)

Part (i)c - Three downsizing configurations:
  - gpt-i-c-config1.py         → Config 1: Reduced embedding (0.478M)
  - gpt-i-c-config2.py         → Config 2: Shallow network (0.439M)
  - gpt-i-c-config3.py         → Config 3: Balanced reduction (0.612M)

Part (i)d - Bias terms exploration:
  - gpt-i-d-bias.py            → Model WITH bias in attention

Part (i)e - Skip connections exploration:
  - gpt-i-e-noskip.py          → Model WITHOUT skip connections

Part (ii) - Test set evaluation:
  - gpt-ii-evaluate.py         → Evaluate all models on test sets

Visualization:
  - visualize_results.py       → Generate comparison plots
```

## Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Run All Experiments
```bash
python3 run_all_experiments.py
```
This runs everything automatically (~15-20 minutes).

## Step-by-Step Execution

### Part (i)a: Dataset Analysis
```bash
# View pre-computed dataset statistics
cat gpt-i-a-analysis.txt

# Or calculate parameter counts for different configs
python3 config_calculator.py
```

**Output:** Shows vocabulary size, dataset length, and characteristics for all three datasets.

### Part (i)b: Create Model with <1M Parameters
```bash
python3 gpt-i-c-config1.py
```

**What changed from original gpt.py:**
- `n_embd = 96` (was 384)
- `n_head = 4` (was 6)
- `n_layer = 4` (was 6)
- `max_iters = 1000` (was 5000, for faster execution)
- Added progress bar with tqdm
- Added result saving to JSON

**Output:** 
- `results/config1_model.pt` - trained model
- `results/config1_history.json` - loss curves
- `results/config1_generated.txt` - sample text

### Part (i)c: Three Downsizing Approaches

**Config 1: Reduced Embedding**
```bash
python3 gpt-i-c-config1.py
```
- Strategy: Drastically reduce embedding dimension
- Params: 0.478M (n_embd=96, n_head=4, n_layer=4)

**Config 2: Shallow Network**
```bash
python3 gpt-i-c-config2.py
```
- Strategy: Minimize network depth
- Params: 0.439M (n_embd=128, n_head=4, n_layer=2)

**Config 3: Balanced Reduction**
```bash
python3 gpt-i-c-config3.py
```
- Strategy: Reduce both context window and layers
- Params: 0.612M (n_embd=128, n_head=4, n_layer=3, block_size=64)

**What to compare:** Training/validation loss curves, final losses, overfitting gap, generated text quality.

### Part (i)d: Bias Terms in Self-Attention

```bash
python3 gpt-i-d-bias.py
```

**What changed:**
```python
# Lines 69-71 in Head class:
self.key = nn.Linear(n_embd, head_size, bias=True)    # was False
self.query = nn.Linear(n_embd, head_size, bias=True)  # was False
self.value = nn.Linear(n_embd, head_size, bias=True)  # was False
```

**Output:** Compare with Config 3 (same hyperparameters) to isolate bias impact.

### Part (i)e: Skip Connections

```bash
python3 gpt-i-e-noskip.py
```

**What changed:**
```python
# Lines 134-135 in Block class forward():
x = self.sa(self.ln1(x))      # removed "x +"
x = self.ffwd(self.ln2(x))    # removed "x +"
```

**Output:** Compare with Config 3 to see impact of removing residual connections.

### Part (ii): Test Set Evaluation

```bash
python3 gpt-ii-evaluate.py
```

**What it does:**
- Loads all trained models
- Evaluates on `input_childSpeech_testSet.txt`
- Evaluates on `input_shakespeare.txt`
- Calculates baseline comparisons (uniform, frequency)
- Computes perplexity and improvements

**Output:** `results/evaluation_summary.json`

### Generate Plots

```bash
python3 visualize_results.py
```

**Output:**
- `results/training_comparison.png` - All configs compared
- `results/config[1-3]_detailed.png` - Individual analysis
- `results/evaluation_comparison.png` - Test set results

## Results Location

All results saved to `results/` directory:
```
results/
├── config1_model.pt              # Trained model weights
├── config1_history.json          # Training curves
├── config1_generated.txt         # Generated text sample
├── config2_model.pt
├── config2_history.json
├── config2_generated.txt
├── config3_model.pt
├── config3_history.json
├── config3_generated.txt
├── with_bias_model.pt
├── with_bias_history.json
├── with_bias_generated.txt
├── no_skip_model.pt
├── no_skip_history.json
├── no_skip_generated.txt
├── evaluation_summary.json       # Test set results
├── training_comparison.png       # Comparison plots
├── config1_detailed.png
├── config2_detailed.png
├── config3_detailed.png
└── evaluation_comparison.png
```

## What Changed from Original gpt.py

### All Training Scripts
1. **Progress bar:** Added `from tqdm import tqdm` and wrapped training loop
2. **Dataset path:** Changed to load `input_childSpeech_trainingSet.txt`
3. **Result saving:** Added JSON export of training history
4. **Model saving:** Save model checkpoint with config
5. **Reduced iterations:** Changed `max_iters` from 5000 to 1000 for faster execution

### Config-Specific Changes

**gpt-i-c-config1.py:**
- n_embd: 384 → 96
- n_head: 6 → 4
- n_layer: 6 → 4

**gpt-i-c-config2.py:**
- n_embd: 384 → 128
- n_layer: 6 → 2

**gpt-i-c-config3.py:**
- n_embd: 384 → 128
- n_layer: 6 → 3
- block_size: 256 → 64

**gpt-i-d-bias.py:**
- bias=False → bias=True in attention layers (3 lines)

**gpt-i-e-noskip.py:**
- Removed `x +` from skip connections (2 lines)

## Troubleshooting

**Training too slow?**
- Reduce `max_iters` from 1000 to 500
- Reduce `batch_size` from 64 to 32

**Out of memory?**
- Reduce `batch_size`
- Use smaller model (Config 2)

**Want better results?**
- Increase `max_iters` to 5000 (like original)
- Will take 5x longer but give better convergence

## Report Writing Tips

### Part (i)b: Rationale
Explain WHY you chose to reduce specific parameters:
- Embedding dimension affects all components multiplicatively
- Layers control network depth
- Context window affects positional embeddings

### Part (i)c: Discussion Points
- Which config achieved lowest validation loss?
- Evidence of overfitting? (train vs val gap)
- Quality of generated text
- Trade-off between parameters and performance

### Part (i)d: Bias Analysis
- Compare loss curves with/without bias
- Does bias improve convergence?
- Parameter count increase
- Impact on generated text

### Part (i)e: Skip Connections
- Training stability differences
- Convergence speed
- Final loss comparison
- Why are skip connections important?

### Part (ii)a: Child Speech Test
- Test loss vs validation loss
- Is model generalizing?
- Comparison with baseline
- Perplexity interpretation

### Part (ii)b: Shakespeare
- Why is loss higher?
- Vocabulary mismatch
- Domain shift explanation
- Practical use case: domain adaptation, transfer learning

## Citation

Based on nanoGPT by Andrej Karpathy.
Modified for CS7CS4/CSU44061 Machine Learning Assignment.

