"""
Visualization Script for Training Results
Creates plots comparing different model configurations
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_history(filename):
    """Load training history from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_training_curves():
    """Plot training and validation loss curves for all configurations"""
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        print("No results directory found!")
        return
    
    # Find all history files
    history_files = [f for f in os.listdir(results_dir) if f.endswith('_history.json')]
    
    if not history_files:
        print("No history files found!")
        return
    
    print(f"Found {len(history_files)} training history files")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GPT Model Training Comparison', fontsize=16, fontweight='bold')
    
    # All curves together
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    all_configs = []
    
    for i, history_file in enumerate(sorted(history_files)):
        filepath = os.path.join(results_dir, history_file)
        history = load_history(filepath)
        
        name = history_file.replace('_history.json', '')
        config = history['config']
        
        all_configs.append({
            'name': name,
            'history': history,
            'config': config,
            'color': colors[i % len(colors)]
        })
        
        iterations = history['iterations']
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        
        # Plot 1: All training losses
        ax1.plot(iterations, train_loss, label=name, color=colors[i % len(colors)], linewidth=2)
        
        # Plot 2: All validation losses
        ax2.plot(iterations, val_loss, label=name, color=colors[i % len(colors)], linewidth=2)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final losses bar chart
    names = [c['name'] for c in all_configs]
    final_train = [c['history']['train_loss'][-1] for c in all_configs]
    final_val = [c['history']['val_loss'][-1] for c in all_configs]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax3.bar(x - width/2, final_train, width, label='Train', color='skyblue')
    ax3.bar(x + width/2, final_val, width, label='Validation', color='lightcoral')
    ax3.set_xlabel('Configuration', fontsize=12)
    ax3.set_ylabel('Final Loss', fontsize=12)
    ax3.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Parameter count vs final validation loss
    params = [c['config']['parameters'] for c in all_configs]
    final_val_loss = [c['history']['val_loss'][-1] for c in all_configs]
    
    for i, c in enumerate(all_configs):
        ax4.scatter(c['config']['parameters'], c['history']['val_loss'][-1], 
                   s=200, color=c['color'], alpha=0.6, edgecolors='black', linewidth=2)
        ax4.annotate(c['name'], 
                    (c['config']['parameters'], c['history']['val_loss'][-1]),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    ax4.set_xlabel('Model Parameters (M)', fontsize=12)
    ax4.set_ylabel('Final Validation Loss', fontsize=12)
    ax4.set_title('Model Size vs Performance', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/training_comparison.png")
    
    # Create individual detailed plots for each configuration
    for config_data in all_configs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        name = config_data['name']
        history = config_data['history']
        config = config_data['config']
        
        iterations = history['iterations']
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        
        # Training and validation on same plot
        ax1.plot(iterations, train_loss, label='Training Loss', color='blue', linewidth=2)
        ax1.plot(iterations, val_loss, label='Validation Loss', color='red', linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{name} - Loss Curves', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Overfitting gap
        gap = [v - t for v, t in zip(val_loss, train_loss)]
        ax2.plot(iterations, gap, color='purple', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Validation Loss - Training Loss', fontsize=12)
        ax2.set_title(f'{name} - Overfitting Gap', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add config info
        config_text = f"Parameters: {config['parameters']:.3f}M\n"
        config_text += f"n_embd: {config['n_embd']}, "
        config_text += f"n_head: {config['n_head']}, "
        config_text += f"n_layer: {config['n_layer']}\n"
        config_text += f"block_size: {config['block_size']}"
        
        fig.text(0.5, 0.02, config_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(f'results/{name}_detailed.png', dpi=300, bbox_inches='tight')
        print(f"Saved: results/{name}_detailed.png")
    
    plt.show()

def plot_evaluation_results():
    """Plot evaluation results on test sets"""
    eval_file = 'results/evaluation_summary.json'
    
    if not os.path.exists(eval_file):
        print("No evaluation results found! Run evaluate_model.py first.")
        return
    
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)
    
    # Extract data for plotting
    models = []
    child_loss = []
    shakespeare_loss = []
    
    for model_name, data in eval_data.items():
        if 'childSpeech_test' in data['evaluations']:
            models.append(model_name)
            child_loss.append(data['evaluations']['childSpeech_test']['test_loss'])
            
            if 'shakespeare' in data['evaluations']:
                shakespeare_loss.append(data['evaluations']['shakespeare']['test_loss'])
            else:
                shakespeare_loss.append(None)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Test Set Evaluation Results', fontsize=16, fontweight='bold')
    
    x = np.arange(len(models))
    width = 0.35
    
    # Bar chart for losses
    bars1 = ax1.bar(x - width/2, child_loss, width, label='Child Speech Test', color='skyblue')
    bars2 = ax1.bar(x + width/2, shakespeare_loss, width, label='Shakespeare', color='salmon')
    
    ax1.set_xlabel('Model Configuration', fontsize=12)
    ax1.set_ylabel('Test Loss', fontsize=12)
    ax1.set_title('Test Loss by Dataset', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_height() is not None:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Improvement over baseline
    improvements_child = []
    improvements_shakespeare = []
    
    for model_name in models:
        if 'childSpeech_test' in eval_data[model_name]['evaluations']:
            improvements_child.append(
                eval_data[model_name]['evaluations']['childSpeech_test']['improvement_over_frequency']
            )
            if 'shakespeare' in eval_data[model_name]['evaluations']:
                improvements_shakespeare.append(
                    eval_data[model_name]['evaluations']['shakespeare']['improvement_over_frequency']
                )
            else:
                improvements_shakespeare.append(None)
    
    bars1 = ax2.bar(x - width/2, improvements_child, width, label='Child Speech Test', color='lightgreen')
    bars2 = ax2.bar(x + width/2, improvements_shakespeare, width, label='Shakespeare', color='lightcoral')
    
    ax2.set_xlabel('Model Configuration', fontsize=12)
    ax2.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax2.set_title('Improvement over Frequency Baseline', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/evaluation_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/evaluation_comparison.png")
    plt.show()

def main():
    print("\n" + "="*80)
    print("VISUALIZATION SCRIPT")
    print("="*80 + "\n")
    
    print("Generating training comparison plots...")
    plot_training_curves()
    
    print("\nGenerating evaluation comparison plots...")
    plot_evaluation_results()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nAll plots saved to results/ directory")

if __name__ == "__main__":
    main()

