# 🚀 START HERE - Week 9 Assignment Quick Guide

## What You Have

All assignment files are ready to run! Files are named according to the assignment structure (Part i-a, i-b, etc.).

## Quick Start (5 minutes to get results)

### 1. Install Dependencies
```bash
cd "/Users/sayanide/Documents/Assignments/Machine Learning/assign_4"
pip3 install -r requirements.txt
```

### 2. Run Everything
```bash
python3 run_all_experiments.py
```

Type `y` when prompted. Go get coffee ☕ for 15-20 minutes.

### 3. Check Results
```bash
# View plots
open results/training_comparison.png
open results/evaluation_comparison.png

# View generated text
cat results/config1_generated.txt
```

### 4. Fill in Report
Open `report.tex` and search for `[FILL FROM RESULTS]` - replace with your actual numbers from `results/` folder.

## File Structure

```
📁 assign_4/
├── 📄 README_START_HERE.md          ← You are here!
├── 📄 INSTRUCTIONS.md               ← Detailed execution guide
├── 📄 WHAT_CHANGED.md               ← What changed from original gpt.py
├── 📄 report.tex                    ← LaTeX report template
│
├── 🔧 requirements.txt              ← Dependencies
├── 🔧 run_all_experiments.py        ← Master script (runs everything)
│
├── 📊 config_calculator.py          ← Part (i)a: Parameter calculations
├── 🤖 gpt-i-c-config1.py           ← Part (i)b & (i)c: Config 1 (0.478M)
├── 🤖 gpt-i-c-config2.py           ← Part (i)c: Config 2 (0.439M)
├── 🤖 gpt-i-c-config3.py           ← Part (i)c: Config 3 (0.612M)
├── 🤖 gpt-i-d-bias.py              ← Part (i)d: With bias terms
├── 🤖 gpt-i-e-noskip.py            ← Part (i)e: Without skip connections
├── 📈 gpt-ii-evaluate.py            ← Part (ii): Test set evaluation
├── 📊 visualize_results.py          ← Generate plots
│
└── 📁 results/                      ← All outputs go here
    ├── config1_model.pt
    ├── config1_history.json
    ├── config1_generated.txt
    ├── training_comparison.png
    └── evaluation_summary.json
```

## What Each File Does

| File | Assignment Part | What It Does | Runtime |
|------|----------------|--------------|---------|
| `config_calculator.py` | (i)a | Show dataset stats & parameter calculations | 1 sec |
| `gpt-i-c-config1.py` | (i)b & (i)c | Train reduced embedding model | 2-3 min |
| `gpt-i-c-config2.py` | (i)c | Train shallow network model | 2-3 min |
| `gpt-i-c-config3.py` | (i)c | Train balanced reduction model | 2-3 min |
| `gpt-i-d-bias.py` | (i)d | Train with bias in attention | 2-3 min |
| `gpt-i-e-noskip.py` | (i)e | Train without skip connections | 2-3 min |
| `gpt-ii-evaluate.py` | (ii) | Evaluate on test sets | 30 sec |
| `visualize_results.py` | - | Generate comparison plots | 10 sec |

**Total time:** ~15-20 minutes for everything

## Quick Commands

### Run Individual Parts

```bash
# Part (i)a: Dataset analysis
python3 config_calculator.py

# Part (i)b: One config <1M params  
python3 gpt-i-c-config1.py

# Part (i)c: All three configs
python3 gpt-i-c-config1.py
python3 gpt-i-c-config2.py
python3 gpt-i-c-config3.py

# Part (i)d: Bias exploration
python3 gpt-i-d-bias.py

# Part (i)e: Skip connections
python3 gpt-i-e-noskip.py

# Part (ii): Evaluation
python3 gpt-ii-evaluate.py

# Generate plots
python3 visualize_results.py
```

### View Results

```bash
# Training loss curves
cat results/config1_history.json | grep "val_loss"

# Generated text
cat results/config1_generated.txt
cat results/config2_generated.txt
cat results/config3_generated.txt

# Test results
cat results/evaluation_summary.json

# Plots
open results/training_comparison.png
open results/config1_detailed.png
open results/evaluation_comparison.png
```

## Filling Out the Report

1. **Run all experiments** to get results
2. **Open `report.tex`**
3. **Search for** `[FILL FROM RESULTS]` and `[VALUE]`
4. **Replace with actual numbers** from your results
5. **Add generated text samples** where marked
6. **Include plots** in the figures section
7. **Compile:** `pdflatex report.tex`

### Where to Find Each Value

| Report Field | Location |
|--------------|----------|
| Final losses | `results/config[X]_history.json` - last values in arrays |
| Test losses | `results/evaluation_summary.json` |
| Generated text | `results/config[X]_generated.txt` |
| Plots | `results/*.png` |

### Example: Finding Final Loss

```bash
# Get last validation loss for config 1
python3 -c "import json; data=json.load(open('results/config1_history.json')); print(f\"Final val loss: {data['val_loss'][-1]:.4f}\")"
```

## What Changed from Original gpt.py?

Very little! Each file only changes:
1. **5 numbers** (hyperparameters: n_embd, n_head, n_layer, block_size, max_iters)
2. **1 import** (tqdm for progress bar)
3. **1 line** (dataset path)
4. **1 line** (training loop wrapper)
5. **~20 lines** (save results)

For bias/skip experiments: **0-2 additional lines** changed.

See `WHAT_CHANGED.md` for detailed line-by-line changes.

## Troubleshooting

### "No module named 'tqdm'"
```bash
pip3 install tqdm
```

### "File not found: input_childSpeech_trainingSet.txt"
Make sure `Week 9 Assignment/` folder exists with data files.

### Training too slow?
Edit any `gpt-*.py` file:
- Change `max_iters = 1000` to `max_iters = 500`
- Or run on GPU if available

### Out of memory?
Edit any `gpt-*.py` file:
- Change `batch_size = 64` to `batch_size = 32`

### Want better results?
Edit any `gpt-*.py` file:
- Change `max_iters = 1000` to `max_iters = 5000`
- Takes 5× longer but better convergence

## For Report Writing

### Key Points to Address

**Part (i)b - Rationale:**
- Why reduce embedding dimension? (affects everything multiplicatively)
- Why keep 4 layers? (reasonable depth for simple data)
- Why 96 dimensions? (sufficient for 40-character vocabulary)

**Part (i)c - Comparison:**
- Which config has lowest validation loss?
- Evidence of overfitting? (train vs val gap)
- How does generated text quality differ?

**Part (i)d - Bias Terms:**
- Do bias terms help or hurt?
- Why or why not for this dataset?
- Training stability differences?

**Part (i)e - Skip Connections:**
- Impact on training stability
- Impact on final loss
- Why are they important? (gradient flow)

**Part (ii)a - Child Test:**
- Is test loss close to validation loss? (generalization)
- How much better than baseline?
- Is the model good enough to use?

**Part (ii)b - Shakespeare:**
- Why is loss higher?
- Vocabulary mismatch (65 vs 40 characters)
- Domain shift (complex vs simple language)
- Practical use: domain adaptation, transfer learning

## Submission Checklist

- [ ] Ran all experiments
- [ ] Filled in all `[FILL]` and `[VALUE]` placeholders in report.tex
- [ ] Added generated text samples to report
- [ ] Included plots in report
- [ ] Compiled report to PDF: `pdflatex report.tex`
- [ ] Code runs without errors
- [ ] Created zip with all `.py` files and data
- [ ] Report PDF is under 10 pages (excluding appendix)

## Need Help?

1. **Read:** `INSTRUCTIONS.md` for detailed execution guide
2. **Read:** `WHAT_CHANGED.md` for code changes
3. **Check:** Results in `results/` folder
4. **Look at:** Previous assignment style in `../assign_3/report.md`

---

**You're all set!** Just run `python3 run_all_experiments.py` and wait 15 minutes. Then fill in the report.tex template with your results. Good luck! 🎓

