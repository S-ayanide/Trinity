# Assignment 3 - Classification

## Files

- `assignment3.py` - Main Python script with all analysis
- `report.md` - Comprehensive report answering all questions
- `week4.csv` - Dataset file
- `figures/` - Generated plots and figures
- `requirements.txt` - Python package dependencies
- `assignment3_env/` - Virtual environment

## Running the Code

### Option 1: Run Python Script
```bash
# Activate virtual environment
source assignment3_env/bin/activate

# Run the script (generates all figures and prints results)
python assignment3.py
```

### Option 2: Convert to Jupyter Notebook
The Python script can be opened directly in Jupyter Lab/Notebook:

```bash
# Activate virtual environment
source assignment3_env/bin/activate

# Start Jupyter
jupyter notebook

# Then open assignment3.py in Jupyter - it will display as a notebook
```

Or convert explicitly:
```bash
jupytext --to notebook assignment3.py
```

## Results Summary

**Dataset 1** (Hard):
- Best Logistic Regression: q=1, C=0.001, F1=0.8000, AUC=0.5045
- Best kNN: k=51, F1=0.8000, AUC=0.5686
- Conclusion: Dataset is too difficult to predict (near-random performance)

**Dataset 2** (Easy):
- Best Logistic Regression: q=2, C=100, F1=0.9508, AUC=0.9982
- Best kNN: k=7, F1=0.9498, AUC=0.9977
- Conclusion: Excellent prediction performance

See `report.md` for detailed analysis.

