#!/usr/bin/env python3
"""
Data Challenges Analysis Script
Validates and reports on data quality issues in the coffee shop sensor datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)

def print_subsection(title):
    """Print a formatted subsection"""
    print(f"\n{title}")
    print('-'*60)

def analyze_location_data():
    """Analyze location data quality issues"""
    print_section("📍 LOCATION DATA QUALITY ISSUES")
    
    # Costa Location
    print_subsection("Costa Coffee Location Data")
    costa_loc = pd.read_csv('costa_data/BE Location.csv')
    print(f"Total samples: {len(costa_loc)}")
    print(f"Collection duration: {costa_loc['Time (s)'].max():.2f} seconds")
    print(f"Sampling rate: 1 sample per {costa_loc['Time (s)'].max()/len(costa_loc):.1f} seconds")
    print(f"\nMissing Data:")
    print(f"  - Height (m): {costa_loc['Height (m)'].isna().sum()}/{len(costa_loc)} ({costa_loc['Height (m)'].isna().sum()/len(costa_loc)*100:.1f}%)")
    print(f"  - Velocity (m/s): {costa_loc['Velocity (m/s)'].isna().sum()}/{len(costa_loc)} ({costa_loc['Velocity (m/s)'].isna().sum()/len(costa_loc)*100:.1f}%)")
    print(f"  - Direction (°): {costa_loc['Direction (°)'].isna().sum()}/{len(costa_loc)} ({costa_loc['Direction (°)'].isna().sum()/len(costa_loc)*100:.1f}%)")
    print(f"\nGPS Accuracy:")
    print(f"  - Horizontal Accuracy: {costa_loc['Horizontal Accuracy (m)'].mean():.2f}m (avg)")
    print(f"  - Vertical Accuracy: {costa_loc['Vertical Accuracy (m)'].mean():.2f}m (avg)")
    
    # Check for GPS drift (stationary device)
    lat_std = costa_loc['Latitude (°)'].std()
    lon_std = costa_loc['Longitude (°)'].std()
    print(f"\nGPS Stability (should be ~0 for stationary device):")
    print(f"  - Latitude std dev: {lat_std:.8f}°")
    print(f"  - Longitude std dev: {lon_std:.8f}°")
    print(f"  - Assessment: {'✓ Stable (device was stationary)' if lat_std < 0.0001 else '⚠ Drifting'}")
    
    # TBC Location
    print_subsection("Two Boys Cafe Location Data")
    tbc_loc = pd.read_csv('tbc_data/Location.csv')
    print(f"Total samples: {len(tbc_loc)}")
    print(f"Collection duration: {tbc_loc['Time (s)'].max():.2f} seconds")
    print(f"Sampling rate: 1 sample per {tbc_loc['Time (s)'].max()/len(tbc_loc):.1f} seconds")
    print(f"\nMissing Data:")
    print(f"  - Height (m): {tbc_loc['Height (m)'].isna().sum()}/{len(tbc_loc)} ({tbc_loc['Height (m)'].isna().sum()/len(tbc_loc)*100:.1f}%)")
    print(f"  - Velocity (m/s): {tbc_loc['Velocity (m/s)'].isna().sum()}/{len(tbc_loc)} ({tbc_loc['Velocity (m/s)'].isna().sum()/len(tbc_loc)*100:.1f}%)")
    print(f"  - Direction (°): {tbc_loc['Direction (°)'].isna().sum()}/{len(tbc_loc)} ({tbc_loc['Direction (°)'].isna().sum()/len(tbc_loc)*100:.1f}%)")
    print(f"\nGPS Accuracy:")
    print(f"  - Horizontal Accuracy: {tbc_loc['Horizontal Accuracy (m)'].mean():.2f}m (avg)")
    print(f"  - Vertical Accuracy: {tbc_loc['Vertical Accuracy (m)'].mean():.2f}m (avg)")
    
    # Check for GPS drift
    lat_std_tbc = tbc_loc['Latitude (°)'].std()
    lon_std_tbc = tbc_loc['Longitude (°)'].std()
    print(f"\nGPS Stability:")
    print(f"  - Latitude std dev: {lat_std_tbc:.8f}°")
    print(f"  - Longitude std dev: {lon_std_tbc:.8f}°")
    print(f"  - Assessment: {'✓ Stable' if lat_std_tbc < 0.0001 else '⚠ Significant drift (device moving or GPS error)'}")
    
    print(f"\n💡 Conclusion: Location data is {'SEVERELY LIMITED' if len(costa_loc) < 50 else 'LIMITED'} and cannot be used for spatial analysis")

def analyze_duration_mismatch():
    """Analyze data collection duration inconsistencies"""
    print_section("⏱️  DATA COLLECTION DURATION MISMATCH")
    
    costa_accel = pd.read_csv('costa_data/BE Accelerometer.csv')
    tbc_accel = pd.read_csv('tbc_data/Accelerometer.csv')
    
    costa_gyro = pd.read_csv('costa_data/BE Gyroscope.csv')
    tbc_gyro = pd.read_csv('tbc_data/Gyroscope.csv')
    
    costa_light = pd.read_csv('costa_data/BE Light.csv')
    tbc_light = pd.read_csv('tbc_data/Light.csv')
    
    print("\n┌─────────────────────┬──────────────┬──────────────┬──────────┐")
    print("│ Sensor              │ Costa (Eve)  │ TBC (Aft)    │ Ratio    │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────┤")
    
    costa_dur = costa_accel['Time (s)'].max()
    tbc_dur = tbc_accel['Time (s)'].max()
    print(f"│ Duration (seconds)  │ {costa_dur:>12.2f} │ {tbc_dur:>12.2f} │ {tbc_dur/costa_dur:>8.2f}x │")
    
    costa_accel_cnt = len(costa_accel)
    tbc_accel_cnt = len(tbc_accel)
    print(f"│ Accelerometer       │ {costa_accel_cnt:>12,} │ {tbc_accel_cnt:>12,} │ {tbc_accel_cnt/costa_accel_cnt:>8.2f}x │")
    
    costa_gyro_cnt = len(costa_gyro)
    tbc_gyro_cnt = len(tbc_gyro)
    print(f"│ Gyroscope           │ {costa_gyro_cnt:>12,} │ {tbc_gyro_cnt:>12,} │ {tbc_gyro_cnt/costa_gyro_cnt:>8.2f}x │")
    
    costa_light_cnt = len(costa_light)
    tbc_light_cnt = len(tbc_light)
    print(f"│ Light Sensor        │ {costa_light_cnt:>12,} │ {tbc_light_cnt:>12,} │ {tbc_light_cnt/costa_light_cnt:>8.2f}x │")
    
    print("└─────────────────────┴──────────────┴──────────────┴──────────┘")
    
    print(f"\n💡 Impact: TBC has ~2.15x more data, creating statistical imbalance")
    print(f"   - Averages may be more stable for TBC")
    print(f"   - Time-series comparisons require normalization")
    print(f"   - Suggests inconsistent data collection protocol")

def analyze_sampling_rates():
    """Analyze sampling rate consistency"""
    print_section("📊 SAMPLING RATE IRREGULARITIES")
    
    costa_accel = pd.read_csv('costa_data/BE Accelerometer.csv')
    tbc_accel = pd.read_csv('tbc_data/Accelerometer.csv')
    
    print_subsection("Accelerometer Sampling Rates")
    
    # Costa
    costa_diffs = costa_accel['Time (s)'].diff().dropna()
    print("\nCosta Coffee:")
    print(f"  - Mean interval: {costa_diffs.mean():.6f}s ({1/costa_diffs.mean():.1f} Hz)")
    print(f"  - Std deviation: {costa_diffs.std():.6f}s")
    print(f"  - Min interval: {costa_diffs.min():.6f}s")
    print(f"  - Max interval: {costa_diffs.max():.6f}s")
    print(f"  - Consistency: {'✓ Excellent' if costa_diffs.std() < 0.0001 else '⚠ Variable'}")
    
    # TBC
    tbc_diffs = tbc_accel['Time (s)'].diff().dropna()
    print("\nTwo Boys Cafe:")
    print(f"  - Mean interval: {tbc_diffs.mean():.6f}s ({1/tbc_diffs.mean():.1f} Hz)")
    print(f"  - Std deviation: {tbc_diffs.std():.6f}s")
    print(f"  - Min interval: {tbc_diffs.min():.6f}s")
    print(f"  - Max interval: {tbc_diffs.max():.6f}s")
    print(f"  - Consistency: {'✓ Good' if tbc_diffs.std() < 0.001 else '⚠ Significant gaps detected'}")
    
    # Find large gaps
    large_gaps = tbc_diffs[tbc_diffs > 0.01]
    if len(large_gaps) > 0:
        print(f"\n⚠️  Found {len(large_gaps)} sampling gaps > 10ms in TBC data")
        print(f"   - Largest gap: {large_gaps.max():.4f}s ({large_gaps.max()/tbc_diffs.mean():.1f}x normal)")
        print(f"   - Possible causes: App backgrounding, system interruptions, battery saving")

def analyze_file_naming():
    """Analyze file naming inconsistencies"""
    print_section("📁 FILE NAMING INCONSISTENCIES")
    
    costa_files = sorted([f.name for f in Path('costa_data').glob('*.csv')])
    tbc_files = sorted([f.name for f in Path('tbc_data').glob('*.csv')])
    
    print("\nCosta Coffee Files:")
    for f in costa_files:
        print(f"  • {f}")
    
    print("\nTwo Boys Cafe Files:")
    for f in tbc_files:
        print(f"  • {f}")
    
    be_count = sum(1 for f in costa_files if f.startswith('BE'))
    print(f"\n💡 Observations:")
    print(f"   - {be_count}/{len(costa_files)} Costa files have 'BE' prefix")
    print(f"   - 0/{len(tbc_files)} TBC files have 'BE' prefix")
    print(f"   - 'BE' likely indicates 'Bluetooth Enabled' or specific sensor app")
    print(f"   - Suggests different data collection apps/devices used")
    print(f"   - Required manual mapping in code to load corresponding files")

def analyze_unused_sensors():
    """Analyze which sensors were collected but not used"""
    print_section("🚫 SENSORS COLLECTED BUT NOT USED IN ANALYSIS")
    
    print("\n1. Magnetometer (Magnetic Field Sensor)")
    costa_mag = pd.read_csv('costa_data/Magnetometer.csv', skiprows=3)
    tbc_mag = pd.read_csv('tbc_data/Magnetometer.csv')
    print(f"   - Costa: {len(costa_mag):,} samples")
    print(f"   - TBC: {len(tbc_mag):,} samples")
    print(f"   - Reason: Not relevant for coffee shop environment comparison")
    print(f"   - Measures magnetic fields (compass), not ambient conditions")
    
    print("\n2. Orientation (Quaternion + Euler Angles)")
    costa_orient = pd.read_csv('costa_data/BE Orientation.csv')
    tbc_orient = pd.read_csv('tbc_data/Orientation.csv')
    print(f"   - Costa: {len(costa_orient):,} samples")
    print(f"   - TBC: {len(tbc_orient):,} samples")
    print(f"   - Reason: Measures device orientation, not environment")
    print(f"   - Dependent on how phone was held/placed (confounding variable)")
    
    print("\n3. Linear Acceleration")
    costa_linear = pd.read_csv('costa_data/BE Linear Acceleration.csv')
    tbc_linear = pd.read_csv('tbc_data/Linear Acceleration.csv')
    print(f"   - Costa: {len(costa_linear):,} samples")
    print(f"   - TBC: {len(tbc_linear):,} samples")
    print(f"   - Reason: Redundant with accelerometer data")
    print(f"   - Linear accel = Total accel - Gravity (derived metric)")
    
    print("\n4. Audio FFT Spectrum")
    costa_fft = pd.read_csv('costa_data/Audio FFT Spectrum.csv', skiprows=3)
    tbc_fft = pd.read_csv('tbc_data/TBC FFT Spectrum.csv', skiprows=3)
    print(f"   - Costa: {len(costa_fft):,} samples")
    print(f"   - TBC: {len(tbc_fft):,} samples")
    print(f"   - Reason: Raw audio waveform sufficient for ambient noise comparison")
    print(f"   - FFT analysis too complex for this use case")
    
    print("\n5. Audio Peak History")
    costa_peaks = pd.read_csv('costa_data/Audio Peak History.csv', skiprows=3)
    tbc_peaks = pd.read_csv('tbc_data/TBC Peak History.csv', skiprows=3)
    print(f"   - Costa: {len(costa_peaks):,} peak events")
    print(f"   - TBC: {len(tbc_peaks):,} peak events")
    print(f"   - Reason: Too granular for environment comparison")
    print(f"   - Average amplitude more interpretable than peak detection")
    
    # Calculate total unused data
    total_unused = (len(costa_mag) + len(tbc_mag) + len(costa_orient) + len(tbc_orient) + 
                    len(costa_linear) + len(tbc_linear) + len(costa_fft) + len(tbc_fft) + 
                    len(costa_peaks) + len(tbc_peaks))
    print(f"\n💡 Total unused sensor readings: {total_unused:,} samples")
    print(f"   - These sensors were collected but provided no value for the use case")
    print(f"   - Future collections should focus on: Light, Accelerometer, Audio, Gyroscope")

def generate_summary():
    """Generate overall summary"""
    print_section("📋 SUMMARY OF DATA CHALLENGES")
    
    print("\n✅ Usable Data Sources:")
    print("   • Light Sensor (Illuminance)")
    print("   • Accelerometer (3-axis motion)")
    print("   • Gyroscope (angular velocity)")
    print("   • Audio (raw waveform)")
    
    print("\n❌ Problematic/Unusable Data:")
    print("   • Location Data (GPS) - 100% missing critical fields in Costa, drift in TBC")
    print("   • Magnetometer - Not relevant to use case")
    print("   • Orientation - Device-specific, not environmental")
    print("   • Linear Acceleration - Redundant")
    print("   • Audio FFT/Peaks - Too complex for simple comparison")
    
    print("\n⚠️  Key Issues Found:")
    print("   1. Duration mismatch (2.15x difference)")
    print("   2. Location data almost completely unusable")
    print("   3. Inconsistent file naming (BE prefix)")
    print("   4. Sampling rate gaps in TBC data")
    print("   5. Large amount of collected but unused sensor data")
    
    print("\n💡 Lessons for Future Data Collection:")
    print("   • Standardize collection protocol (same app, device, settings)")
    print("   • Set equal duration timers for all locations")
    print("   • Validate GPS fix quality before starting")
    print("   • Focus on sensors relevant to research question")
    print("   • Add metadata (device info, environmental conditions)")
    print("   • Monitor data quality during collection")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COFFEE SHOP SENSOR DATA - QUALITY CHALLENGES ANALYSIS")
    print("="*80)
    print("\nAnalyzing data from:")
    print("  • Costa Coffee (Evening)")
    print("  • Two Boys Cafe (Afternoon)")
    print("\nThis script validates the data quality issues documented in DATA_CHALLENGES.md")
    
    try:
        analyze_location_data()
        analyze_duration_mismatch()
        analyze_sampling_rates()
        analyze_file_naming()
        analyze_unused_sensors()
        generate_summary()
        
        print("\n" + "="*80)
        print("✓ Analysis Complete!")
        print("="*80)
        print("\nFor detailed explanations of each issue, see: DATA_CHALLENGES.md")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

