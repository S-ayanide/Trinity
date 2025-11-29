#!/bin/bash

echo "=========================================="
echo "Week 9 Assignment - Quick Start Script"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -f "run_all_experiments.py" ]; then
    echo "❌ Error: Please run this script from the assign_4 directory"
    exit 1
fi

echo "📦 Step 1: Installing dependencies..."
pip3 install -r requirements.txt

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "🚀 Step 2: Running all experiments..."
echo "This will take approximately 15-20 minutes."
echo ""

python3 run_all_experiments.py

echo ""
echo "=========================================="
echo "✨ All Done!"
echo "=========================================="
echo ""
echo "📊 Results are in the results/ folder"
echo ""
echo "Next steps:"
echo "  1. View plots: open results/training_comparison.png"
echo "  2. Read generated text: cat results/config1_generated.txt"
echo "  3. Fill in report.tex with your results"
echo "  4. Compile: pdflatex report.tex"
echo ""
echo "📚 For help, see README_START_HERE.md"
echo ""

