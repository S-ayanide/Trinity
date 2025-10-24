#!/usr/bin/env python3
"""
Generate a simple comparison chart for quick visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    print("Generating comparison chart...")
    
    try:
        # Define paths
        costa_path = Path('costa_data')
        tbc_path = Path('tbc_data')
        
        # Load data
        costa_accel = pd.read_csv(costa_path / 'BE Accelerometer.csv')
        tbc_accel = pd.read_csv(tbc_path / 'Accelerometer.csv')
        costa_light = pd.read_csv(costa_path / 'BE Light.csv')
        tbc_light = pd.read_csv(tbc_path / 'Light.csv')
        costa_audio = pd.read_csv(costa_path / 'Audio Raw Data.csv', skiprows=3)
        tbc_audio = pd.read_csv(tbc_path / 'TBC Audio Raw Data.csv', skiprows=3)
        
        # Calculate metrics
        costa_accel_mag = np.sqrt(
            costa_accel['Acceleration x (m/s^2)']**2 + 
            costa_accel['Acceleration y (m/s^2)']**2 + 
            costa_accel['Acceleration z (m/s^2)']**2
        ).mean()
        
        tbc_accel_mag = np.sqrt(
            tbc_accel['Acceleration x (m/s^2)']**2 + 
            tbc_accel['Acceleration y (m/s^2)']**2 + 
            tbc_accel['Acceleration z (m/s^2)']**2
        ).mean()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Coffee Shop Data Comparison: Costa (Evening) vs Two Boys Cafe (Afternoon)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Light levels comparison
        axes[0, 0].plot(costa_light['Time (s)'], costa_light['Illuminance (lx)'], 
                       color='#FF6B35', linewidth=2, label='Costa', alpha=0.8)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Illuminance (lux)')
        axes[0, 0].set_title('Light Levels - Costa Coffee')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].plot(tbc_light['Time (s)'], tbc_light['Illuminance (lx)'], 
                       color='#4ECDC4', linewidth=2, label='Two Boys Cafe', alpha=0.8)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Illuminance (lux)')
        axes[0, 1].set_title('Light Levels - Two Boys Cafe')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 2. Light comparison box plot
        box_data = [costa_light['Illuminance (lx)'], tbc_light['Illuminance (lx)']]
        bp = axes[0, 2].boxplot(box_data, labels=['Costa\n(Evening)', 'TBC\n(Afternoon)'],
                               patch_artist=True, showmeans=True)
        colors = ['#FF6B35', '#4ECDC4']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 2].set_ylabel('Illuminance (lux)')
        axes[0, 2].set_title('Light Distribution Comparison')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 3. Acceleration comparison
        axes[1, 0].plot(costa_accel['Time (s)'][:5000], 
                       np.sqrt(costa_accel['Acceleration x (m/s^2)'][:5000]**2 + 
                              costa_accel['Acceleration y (m/s^2)'][:5000]**2 + 
                              costa_accel['Acceleration z (m/s^2)'][:5000]**2),
                       color='#FF6B35', linewidth=1, alpha=0.8)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Acceleration (m/s²)')
        axes[1, 0].set_title('Acceleration - Costa Coffee')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(tbc_accel['Time (s)'][:5000],
                       np.sqrt(tbc_accel['Acceleration x (m/s^2)'][:5000]**2 + 
                              tbc_accel['Acceleration y (m/s^2)'][:5000]**2 + 
                              tbc_accel['Acceleration z (m/s^2)'][:5000]**2),
                       color='#4ECDC4', linewidth=1, alpha=0.8)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Acceleration (m/s²)')
        axes[1, 1].set_title('Acceleration - Two Boys Cafe')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 4. Summary bar chart
        metrics = ['Avg Light\n(lux)', 'Avg Accel\n(m/s²)', 'Duration\n(sec)']
        costa_values = [
            costa_light['Illuminance (lx)'].mean(),
            costa_accel_mag,
            costa_accel['Time (s)'].max()
        ]
        tbc_values = [
            tbc_light['Illuminance (lx)'].mean(),
            tbc_accel_mag,
            tbc_accel['Time (s)'].max()
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        axes[1, 2].bar(x - width/2, costa_values, width, label='Costa', 
                      color='#FF6B35', alpha=0.8)
        axes[1, 2].bar(x + width/2, tbc_values, width, label='TBC', 
                      color='#4ECDC4', alpha=0.8)
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_title('Key Metrics Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save the figure
        output_file = 'coffee_shop_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Comparison chart saved as: {output_file}")
        
        # Also show the plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

