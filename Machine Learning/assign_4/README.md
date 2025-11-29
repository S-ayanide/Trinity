# Week 9 Assignment: Transformer Analysis
## CS7CS4/CSU44061 Machine Learning - Trinity College Dublin

This repository contains a comprehensive solution for the Week 9 Assignment on GPT transformers.

---

## 📋 Assignment Overview

The assignment explores transformers by:
1. Analyzing datasets (child speech and Shakespeare)
2. Downsizing GPT models to <1M parameters
3. Exploring architectural components (bias terms, skip connections)
4. Evaluating models on test sets with baseline comparisons

---

## 📁 File Structure

```
assign_4/
├── Week 9 Assignment/           # Original assignment files
│   ├── gpt.py                   # Original GPT script
│   ├── input_childSpeech_trainingSet.txt
│   ├── input_childSpeech_testSet.txt
│   └── input_shakespeare.txt
│
├── dataset_analysis.py          # Part (i)a: Dataset analysis
├── config_calculator.py         # Part (i)b: Parameter calculations
│
├── gpt_config1.py              # Config 1: Reduced embedding (0.478M params)
├── gpt_config2.py              # Config 2: Shallow network (0.439M params)
├── gpt_config3.py              # Config 3: Balanced reduction (0.612M params)
│
├── gpt_with_bias.py            # Part (i)d: With bias in attention
├── gpt_no_skip.py              # Part (i)e: Without skip connections
│
├── evaluate_model.py           # Part (ii): Test set evaluation
├── visualize_results.py        # Plotting and visualization
│
├── run_all_experiments.py      # Master script to run everything
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch numpy matplotlib
```

### Run All Experiments
```bash
python3 run_all_experiments.py
```

This will:
1. Analyze all three datasets
2. Calculate parameters for different configurations
3. Train all model variants (takes 10-15 minutes)
4. Evaluate models on test sets
5. Generate comparison plots

### Run Individual Components

```bash
# Dataset analysis only
python3 dataset_analysis.py

# Train specific configuration
python3 gpt_config1.py

# Evaluate trained models
python3 evaluate_model.py

# Generate plots
python3 visualize_results.py
```

---

## 📊 Part (i): Model Configurations

### Part (i)a: Dataset Analysis

**Script:** `dataset_analysis.py`

Analyzes all three datasets:

| Dataset | Size | Vocabulary | Lines | Words | Unique Words |
|---------|------|------------|-------|-------|--------------|
| Child Speech (train) | 247K chars | 40 chars | 10,001 | 55,076 | 88 |
| Child Speech (test) | 24K chars | 40 chars | 1,001 | 5,347 | 88 |
| Shakespeare | 1.1M chars | 65 chars | 40,001 | 202,651 | 25,670 |

**Key Findings:**
- Child speech has simple vocabulary (40 characters, 88 unique words)
- Shakespeare is 4.5× larger with more complex language
- Both datasets have identical vocabulary (40 chars)

### Part (i)b & (i)c: Model Configurations (<1M Parameters)

**Script:** `config_calculator.py`

Three downsizing strategies:

#### Configuration 1: Reduced Embedding Dimension
```python
n_embd = 96    # Reduced from 384
n_head = 4     # Reduced from 6
n_layer = 4    # Reduced from 6
block_size = 256
Total: 0.478M parameters
```

**Rationale:** Embedding dimension affects every component multiplicatively. Reducing from 384→96 drastically cuts parameters while maintaining reasonable depth (4 layers) and multiple attention heads (4).

#### Configuration 2: Shallow Network
```python
n_embd = 128
n_head = 4
n_layer = 2    # Only 2 layers
block_size = 256
Total: 0.439M parameters
```

**Rationale:** Minimizes network depth. Using just 2 layers makes a shallow network that's faster to train. Slightly wider embeddings (128) compensate for lost depth.

#### Configuration 3: Balanced Reduction
```python
n_embd = 128
n_head = 4
n_layer = 3
block_size = 64    # Reduced context
Total: 0.612M parameters
```

**Rationale:** Balanced approach reducing both context window (256→64) and layers (6→3). For simple child speech patterns, 64 tokens may be sufficient context.

### Part (i)d: Bias Terms in Self-Attention

**Script:** `gpt_with_bias.py`

Explores impact of adding bias terms to key, query, and value projections in self-attention:

```python
self.key = nn.Linear(n_embd, head_size, bias=True)    # Changed
self.query = nn.Linear(n_embd, head_size, bias=True)  # Changed
self.value = nn.Linear(n_embd, head_size, bias=True)  # Changed
```

**What to observe:**
- Slightly more parameters (~9,216 additional for 3 layers × 4 heads)
- Training stability and convergence speed
- Final loss comparison
- Whether bias helps capture dataset-specific patterns

### Part (i)e: Skip Connections

**Script:** `gpt_no_skip.py`

Explores impact of removing residual/skip connections:

```python
# Original (with skip connections):
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))

# Modified (without skip connections):
x = self.sa(self.ln1(x))
x = self.ffwd(self.ln2(x))
```

**What to observe:**
- Training instability (vanishing/exploding gradients)
- Slower convergence
- Higher final losses
- Demonstrates importance of residual connections

---

## 📈 Part (ii): Test Set Evaluation

**Script:** `evaluate_model.py`

Evaluates all trained models on:
1. **Child Speech Test Set** (in-distribution)
2. **Shakespeare Dataset** (out-of-distribution)

### Baseline Comparisons

1. **Uniform Random Baseline**
   - Loss = -log(1/vocab_size)
   - For vocab_size=40: loss ≈ 3.69

2. **Character Frequency Baseline**
   - Based on empirical character distribution
   - Loss = entropy of character frequencies
   - Better baseline as it captures data statistics

### Metrics Calculated

- **Test Loss:** Cross-entropy loss on test data
- **Perplexity:** exp(loss), interpretable as "effective vocabulary size"
- **Improvement over baselines:** Percentage reduction in loss

### Expected Observations

**Child Speech Test Set:**
- Should achieve low loss (in-distribution)
- Much better than baseline
- Best config should have loss ~1.5-2.0

**Shakespeare Dataset:**
- Higher loss (out-of-distribution, different vocabulary)
- May not beat baseline if vocabularies don't overlap
- Demonstrates domain shift/generalization challenges

---

## 📊 Visualizations

**Script:** `visualize_results.py`

Generates:

1. **training_comparison.png**
   - Training loss curves (all configs)
   - Validation loss curves (all configs)
   - Final loss bar charts
   - Parameters vs performance scatter plot

2. **config[1-3]_detailed.png**
   - Individual loss curves
   - Overfitting gap analysis
   - Configuration details

3. **evaluation_comparison.png**
   - Test loss by dataset
   - Improvement over baseline

4. **[config]_generated.txt**
   - Sample generated text from each model
   - Qualitative assessment material

---

## 📝 Results Interpretation Guide

### Validation Loss Analysis

- **Lower = Better:** Model predicts next character accurately
- **Train vs Val Gap:** Large gap = overfitting
- **Convergence:** Should decrease and stabilize

### Overfitting Indicators

- Training loss keeps decreasing, validation plateaus/increases
- Large positive overfitting gap
- Generated text memorizes training data

### Configuration Comparison

**Expected Ranking:**
1. Config 3 (balanced) - best performance
2. Config 1 (reduced embedding) - good
3. Config 2 (shallow) - may underfit

### Bias Terms Impact

- **With bias:** Slightly more flexible, may help or hurt
- **Without bias:** Cleaner, fewer parameters
- Difference should be small on simple data

### Skip Connections Impact

- **With skip:** Stable training, good convergence
- **Without skip:** Unstable, poor results
- **Demonstrates:** Critical importance of residual connections

### Test Set Performance

**Good model indicators:**
- Child test loss ≈ validation loss (consistency)
- Perplexity < 10 on child speech
- 40-60% improvement over baseline
- Generated text is coherent

**Shakespeare test:**
- Higher loss expected (domain shift)
- May be worse than baseline if vocab mismatch
- Demonstrates need for domain-specific training

---

## 🎯 Key Insights for Report

### Model Downsizing

1. **Embedding dimension** has largest impact (multiplicative effect)
2. **Layer reduction** trades depth for speed
3. **Context window** matters less for simple patterns
4. **Sweet spot:** Balance between all hyperparameters

### Architectural Components

1. **Bias terms:** Minor impact on simple data
2. **Skip connections:** Critical for training deep networks
   - Enables gradient flow
   - Prevents vanishing gradients
   - Essential for convergence

### Generalization

1. **In-distribution:** Models perform well
2. **Out-of-distribution:** Performance degrades significantly
3. **Vocabulary mismatch:** Major challenge
4. **Domain adaptation:** Needed for cross-domain transfer

---

## 📚 Report Writing Guide

### Structure

1. **Introduction:** Brief overview of transformer architecture
2. **Part (i)a:** Dataset descriptions with statistics
3. **Part (i)b:** Configuration rationale and parameter calculations
4. **Part (i)c:** Training results, loss curves, comparisons
5. **Part (i)d:** Bias terms analysis
6. **Part (i)e:** Skip connections analysis
7. **Part (ii)a:** Child speech test evaluation
8. **Part (ii)b:** Shakespeare test evaluation and domain shift
9. **Conclusion:** Key findings and insights

### Plots to Include

- Training comparison plot (all configs)
- Best model detailed plot
- Evaluation comparison plot
- One example of generated text per config

### Discussion Points

1. Why did you choose your specific downsizing strategies?
2. Which configuration performed best? Why?
3. What does the overfitting gap tell you?
4. How do bias terms affect the model?
5. Why are skip connections important?
6. Why is Shakespeare loss higher?
7. What's the practical use case for this pipeline?

---

## 🔧 Customization

### Adjust Training Iterations

Edit `max_iters` in any `gpt_*.py` file:
```python
max_iters = 1000  # Increase for better results (5000 recommended)
```

### Change Hyperparameters

Modify hyperparameters at the top of each config file:
```python
batch_size = 64
learning_rate = 3e-4
dropout = 0.2
```

### Add New Configurations

Copy a config file and modify hyperparameters:
```bash
cp gpt_config1.py gpt_config4.py
# Edit gpt_config4.py with new hyperparameters
```

---

## ⚠️ Common Issues

### Out of Memory
- Reduce `batch_size`
- Reduce `block_size`
- Reduce model size

### Slow Training
- Reduce `max_iters`
- Reduce `eval_iters`
- Use GPU if available

### Poor Results
- Increase `max_iters` (try 5000)
- Adjust `learning_rate`
- Check for bugs in modifications

---

## 📊 Expected Runtime

On CPU (MacBook Pro M1):
- Dataset analysis: <1 second
- Config calculation: <1 second
- Training (1000 iters): 2-3 minutes per config
- Evaluation: <30 seconds
- Visualization: <10 seconds

**Total: ~15-20 minutes for all experiments**

---

## ✅ Checklist for Assignment Submission

- [ ] Run all experiments successfully
- [ ] Generate all plots
- [ ] Review generated text samples
- [ ] Analyze loss curves
- [ ] Compare configurations
- [ ] Write report with explanations
- [ ] Include plots in report
- [ ] Add code as appendix
- [ ] Create zip file with code and data
- [ ] Submit PDF and ZIP separately

---

## 📞 Troubleshooting

If you encounter issues:

1. **Check Python version:** Python 3.7+
2. **Check dependencies:** `pip install torch numpy matplotlib`
3. **Check file paths:** Ensure Week 9 Assignment folder exists
4. **Check disk space:** Need ~100MB for results
5. **Check permissions:** Need write access to create results/

---

## 🎓 Learning Objectives

After completing this assignment, you should understand:

1. How transformer parameters scale with hyperparameters
2. Trade-offs in model downsizing
3. Role of architectural components (bias, skip connections)
4. Model evaluation and baseline comparisons
5. Domain shift and generalization challenges
6. Practical transformer deployment considerations

---

## 📖 References

- Original GPT paper: "Improving Language Understanding by Generative Pre-Training"
- Attention is All You Need (Vaswani et al., 2017)
- Andrej Karpathy's "Neural Networks: Zero to Hero" series
- nanoGPT repository

---

## 📝 Notes

- Set random seed (1337) for reproducibility
- Models automatically save to `results/` directory
- JSON files contain exact numerical results
- Plots are high-resolution (300 DPI) for reports
- Generated text demonstrates learned patterns

---

**Good luck with your assignment! 🚀**

