#!/usr/bin/env python3
"""
Quick summary script to generate basic comparison statistics
"""
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("\n" + "="*80)
    print("COFFEE SHOP DATA COMPARISON - QUICK SUMMARY")
    print("="*80)
    
    try:
        # Define paths
        costa_path = Path('costa_data')
        tbc_path = Path('tbc_data')
        
        # Check if data exists
        if not costa_path.exists() or not tbc_path.exists():
            print("\n❌ Data folders not found. Please run the extraction first.")
            return
        
        # Load key data files
        print("\n📁 Loading data files...")
        costa_accel = pd.read_csv(costa_path / 'BE Accelerometer.csv')
        tbc_accel = pd.read_csv(tbc_path / 'Accelerometer.csv')
        costa_light = pd.read_csv(costa_path / 'BE Light.csv')
        tbc_light = pd.read_csv(tbc_path / 'Light.csv')
        costa_audio = pd.read_csv(costa_path / 'Audio Raw Data.csv', skiprows=3)
        tbc_audio = pd.read_csv(tbc_path / 'TBC Audio Raw Data.csv', skiprows=3)
        
        print("✅ Data loaded successfully!\n")
        
        # Calculate statistics
        print("="*80)
        print("COMPARISON TABLE")
        print("="*80)
        
        # Create comparison data
        comparison = {
            'Metric': [
                'Duration (seconds)',
                'Accelerometer Samples',
                'Avg Illuminance (lux)',
                'Max Illuminance (lux)',
                'Audio Samples',
                'Avg Audio Level (abs)',
            ],
            'Costa Coffee (Evening)': [
                f"{costa_accel['Time (s)'].max():.2f}",
                f"{len(costa_accel):,}",
                f"{costa_light['Illuminance (lx)'].mean():.2f}",
                f"{costa_light['Illuminance (lx)'].max():.2f}",
                f"{len(costa_audio):,}",
                f"{abs(costa_audio['Recording (a.u.)']).mean():.6f}",
            ],
            'Two Boys Cafe (Afternoon)': [
                f"{tbc_accel['Time (s)'].max():.2f}",
                f"{len(tbc_accel):,}",
                f"{tbc_light['Illuminance (lx)'].mean():.2f}",
                f"{tbc_light['Illuminance (lx)'].max():.2f}",
                f"{len(tbc_audio):,}",
                f"{abs(tbc_audio['Recording (a.u.)']).mean():.6f}",
            ]
        }
        
        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))
        print("="*80)
        
        # Key insights
        costa_avg_lux = costa_light['Illuminance (lx)'].mean()
        tbc_avg_lux = tbc_light['Illuminance (lx)'].mean()
        
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        if tbc_avg_lux > costa_avg_lux:
            diff_percent = ((tbc_avg_lux / costa_avg_lux - 1) * 100)
            print(f"\n🔆 Two Boys Cafe (afternoon) was {diff_percent:.1f}% BRIGHTER")
            print("   This aligns with expectations - afternoon has more natural light!")
        else:
            diff_percent = ((costa_avg_lux / tbc_avg_lux - 1) * 100)
            print(f"\n🔆 Costa Coffee (evening) was {diff_percent:.1f}% BRIGHTER")
            print("   This is unexpected - possibly due to artificial lighting!")
        
        # Calculate acceleration magnitudes
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
        
        if costa_accel_mag > tbc_accel_mag:
            diff_percent = ((costa_accel_mag / tbc_accel_mag - 1) * 100)
            print(f"\n📱 Costa Coffee had {diff_percent:.1f}% MORE MOVEMENT")
            print("   Evening might have more device handling or activity!")
        else:
            diff_percent = ((tbc_accel_mag / costa_accel_mag - 1) * 100)
            print(f"\n📱 Two Boys Cafe had {diff_percent:.1f}% MORE MOVEMENT")
            print("   Afternoon might have more device handling or activity!")
        
        costa_duration = costa_accel['Time (s)'].max()
        tbc_duration = tbc_accel['Time (s)'].max()
        
        print(f"\n⏱️  Data Collection:")
        print(f"   • Costa Coffee: {costa_duration:.1f}s ({costa_duration/60:.1f} min)")
        print(f"   • Two Boys Cafe: {tbc_duration:.1f}s ({tbc_duration/60:.1f} min)")
        
        print("\n" + "="*80)
        print("\n✅ For detailed visualizations, run: jupyter notebook coffee_shop_analysis.ipynb")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the data has been extracted properly.")

if __name__ == "__main__":
    main()

