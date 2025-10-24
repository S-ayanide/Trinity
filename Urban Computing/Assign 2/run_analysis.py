#!/usr/bin/env python3
"""
Script to run the coffee shop data analysis
"""
import subprocess
import sys

def main():
    print("=" * 80)
    print("COFFEE SHOP DATA ANALYSIS")
    print("Costa Coffee (Evening) vs Two Boys Cafe (Afternoon)")
    print("=" * 80)
    print("\nExecuting Jupyter notebook...")
    
    try:
        # Run the notebook using jupyter nbconvert
        result = subprocess.run([
            'jupyter', 'nbconvert', '--to', 'notebook', 
            '--execute', '--inplace',
            'coffee_shop_analysis.ipynb'
        ], capture_output=True, text=True, check=True)
        
        print("✅ Analysis completed successfully!")
        print("\nTo view the results:")
        print("1. Open 'coffee_shop_analysis.ipynb' in Jupyter")
        print("2. Or run: jupyter notebook coffee_shop_analysis.ipynb")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing notebook: {e}")
        print("\nAlternative: Open the notebook manually in Jupyter and run all cells")
        print("Command: jupyter notebook coffee_shop_analysis.ipynb")
    except FileNotFoundError:
        print("❌ Jupyter not found. Please install it with: pip install jupyter")
        print("\nAlternative: Open coffee_shop_analysis.ipynb in your IDE")

if __name__ == "__main__":
    main()

