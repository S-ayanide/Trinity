"""
Master Script to Run All Experiments
Runs all model configurations and generates comprehensive results for the assignment
"""
import subprocess
import sys
import os
import time

def run_script(script_name, description):
    """Run a Python script and track its execution"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     Week 9 Assignment - Experiment Runner                     ║
║                           GPT Transformer Analysis                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    experiments = [
        ("config_calculator.py", "Part (i)a: Dataset & Parameter Analysis"),
        ("gpt-i-c-config1.py", "Part (i)b & (i)c: Config 1 - Reduced Embedding (0.478M)"),
        ("gpt-i-c-config2.py", "Part (i)c: Config 2 - Shallow Network (0.439M)"),
        ("gpt-i-c-config3.py", "Part (i)c: Config 3 - Balanced Reduction (0.612M)"),
        ("gpt-i-d-bias.py", "Part (i)d: Exploring Bias Terms in Attention"),
        ("gpt-i-e-noskip.py", "Part (i)e: Exploring Skip Connections"),
        ("gpt-ii-evaluate.py", "Part (ii): Model Evaluation on Test Sets"),
        ("visualize_results.py", "Generating Visualizations and Plots"),
    ]
    
    results = []
    total_start = time.time()
    
    print("\nExperiment Plan:")
    for i, (script, desc) in enumerate(experiments, 1):
        print(f"  {i}. {desc}")
    
    print(f"\n{'='*80}")
    response = input("Run all experiments? This will take several minutes. [y/n]: ")
    
    if response.lower() != 'y':
        print("\nExperiments cancelled.")
        return
    
    print(f"\n{'='*80}")
    print("Starting experiments...")
    print(f"{'='*80}\n")
    
    for i, (script, desc) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        success = run_script(script, desc)
        results.append((desc, success))
        
        if not success:
            print(f"\n⚠ Warning: {script} failed!")
            response = input("Continue with remaining experiments? [y/n]: ")
            if response.lower() != 'y':
                break
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    
    successful = sum(1 for _, success in results if success)
    
    for desc, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status:12} | {desc}")
    
    print(f"\n{'='*80}")
    print(f"Total: {successful}/{len(results)} experiments completed successfully")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*80}\n")
    
    print("\nResults Location:")
    print(f"  - Training histories: results/*_history.json")
    print(f"  - Trained models: results/*_model.pt")
    print(f"  - Generated text: results/*_generated.txt")
    print(f"  - Evaluation summary: results/evaluation_summary.json")
    print(f"  - Plots: results/*.png")
    print(f"  - Dataset analysis: dataset_analysis_results.txt")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("""
1. Review the generated plots in results/*.png
2. Check evaluation_summary.json for detailed test results
3. Read generated text samples in results/*_generated.txt
4. Review dataset_analysis_results.txt for dataset descriptions

For the report, use:
- Training comparison plots (results/training_comparison.png)
- Individual detailed plots for each config (results/*_detailed.png)
- Evaluation comparison (results/evaluation_comparison.png)
- Generated text samples to discuss qualitative results
- JSON files for exact numerical results
""")

if __name__ == "__main__":
    main()

