#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)

costa_path = Path('costa_data')
tbc_path = Path('tbc_data')

costa_accel = pd.read_csv(costa_path / 'BE Accelerometer.csv')
tbc_accel = pd.read_csv(tbc_path / 'Accelerometer.csv')
costa_light = pd.read_csv(costa_path / 'BE Light.csv')
tbc_light = pd.read_csv(tbc_path / 'Light.csv')
costa_audio = pd.read_csv(costa_path / 'Audio Raw Data.csv', skiprows=3)
tbc_audio = pd.read_csv(tbc_path / 'TBC Audio Raw Data.csv', skiprows=3)
costa_gyro = pd.read_csv(costa_path / 'BE Gyroscope.csv')
tbc_gyro = pd.read_csv(tbc_path / 'Gyroscope.csv')

costa_accel['magnitude'] = np.sqrt(costa_accel['Acceleration x (m/s^2)']**2 + 
                                    costa_accel['Acceleration y (m/s^2)']**2 + 
                                    costa_accel['Acceleration z (m/s^2)']**2)
tbc_accel['magnitude'] = np.sqrt(tbc_accel['Acceleration x (m/s^2)']**2 + 
                                  tbc_accel['Acceleration y (m/s^2)']**2 + 
                                  tbc_accel['Acceleration z (m/s^2)']**2)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Light Sensor Comparison: Costa Coffee (Evening) vs Two Boys Cafe (Afternoon)', 
             fontsize=16, fontweight='bold')

axes[0, 0].plot(costa_light['Time (s)'], costa_light['Illuminance (lx)'], 
                color='#FF6B35', linewidth=1.5, label='Costa Coffee')
axes[0, 0].set_xlabel('Time (s)', fontsize=11)
axes[0, 0].set_ylabel('Illuminance (lux)', fontsize=11)
axes[0, 0].set_title('Costa Coffee - Evening Light Levels', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(tbc_light['Time (s)'], tbc_light['Illuminance (lx)'], 
                color='#4ECDC4', linewidth=1.5, label='Two Boys Cafe')
axes[0, 1].set_xlabel('Time (s)', fontsize=11)
axes[0, 1].set_ylabel('Illuminance (lux)', fontsize=11)
axes[0, 1].set_title('Two Boys Cafe - Afternoon Light Levels', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[1, 0].hist(costa_light['Illuminance (lx)'], bins=30, alpha=0.6, 
                color='#FF6B35', label='Costa Coffee', edgecolor='black')
axes[1, 0].hist(tbc_light['Illuminance (lx)'], bins=30, alpha=0.6, 
                color='#4ECDC4', label='Two Boys Cafe', edgecolor='black')
axes[1, 0].set_xlabel('Illuminance (lux)', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

box_data = [costa_light['Illuminance (lx)'], tbc_light['Illuminance (lx)']]
bp = axes[1, 1].boxplot(box_data, labels=['Costa Coffee\n(Evening)', 'Two Boys Cafe\n(Afternoon)'],
                         patch_artist=True, showmeans=True)
colors = ['#FF6B35', '#4ECDC4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 1].set_ylabel('Illuminance (lux)', fontsize=11)
axes[1, 1].set_title('Statistical Comparison', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('light_sensor_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Accelerometer Comparison: Costa Coffee vs Two Boys Cafe', 
             fontsize=16, fontweight='bold')

axes[0, 0].plot(costa_accel['Time (s)'], costa_accel['Acceleration x (m/s^2)'], 
                label='X-axis', alpha=0.7, linewidth=0.8)
axes[0, 0].plot(costa_accel['Time (s)'], costa_accel['Acceleration y (m/s^2)'], 
                label='Y-axis', alpha=0.7, linewidth=0.8)
axes[0, 0].plot(costa_accel['Time (s)'], costa_accel['Acceleration z (m/s^2)'], 
                label='Z-axis', alpha=0.7, linewidth=0.8)
axes[0, 0].set_xlabel('Time (s)', fontsize=11)
axes[0, 0].set_ylabel('Acceleration (m/s²)', fontsize=11)
axes[0, 0].set_title('Costa Coffee - 3-Axis Acceleration', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(tbc_accel['Time (s)'], tbc_accel['Acceleration x (m/s^2)'], 
                label='X-axis', alpha=0.7, linewidth=0.8)
axes[0, 1].plot(tbc_accel['Time (s)'], tbc_accel['Acceleration y (m/s^2)'], 
                label='Y-axis', alpha=0.7, linewidth=0.8)
axes[0, 1].plot(tbc_accel['Time (s)'], tbc_accel['Acceleration z (m/s^2)'], 
                label='Z-axis', alpha=0.7, linewidth=0.8)
axes[0, 1].set_xlabel('Time (s)', fontsize=11)
axes[0, 1].set_ylabel('Acceleration (m/s²)', fontsize=11)
axes[0, 1].set_title('Two Boys Cafe - 3-Axis Acceleration', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(costa_accel['Time (s)'], costa_accel['magnitude'], 
                color='#FF6B35', label='Costa Coffee', linewidth=1, alpha=0.8)
axes[1, 0].set_xlabel('Time (s)', fontsize=11)
axes[1, 0].set_ylabel('Acceleration Magnitude (m/s²)', fontsize=11)
axes[1, 0].set_title('Costa Coffee - Acceleration Magnitude', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(tbc_accel['Time (s)'], tbc_accel['magnitude'], 
                color='#4ECDC4', label='Two Boys Cafe', linewidth=1, alpha=0.8)
axes[1, 1].set_xlabel('Time (s)', fontsize=11)
axes[1, 1].set_ylabel('Acceleration Magnitude (m/s²)', fontsize=11)
axes[1, 1].set_title('Two Boys Cafe - Acceleration Magnitude', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('accelerometer_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Audio Recording Comparison: Costa Coffee vs Two Boys Cafe', 
             fontsize=16, fontweight='bold')

axes[0, 0].plot(costa_audio['Time (s)'], costa_audio['Recording (a.u.)'], 
                color='#FF6B35', linewidth=0.5, alpha=0.8)
axes[0, 0].set_xlabel('Time (s)', fontsize=11)
axes[0, 0].set_ylabel('Audio Level (a.u.)', fontsize=11)
axes[0, 0].set_title('Costa Coffee - Raw Audio Signal', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(tbc_audio['Time (s)'], tbc_audio['Recording (a.u.)'], 
                color='#4ECDC4', linewidth=0.5, alpha=0.8)
axes[0, 1].set_xlabel('Time (s)', fontsize=11)
axes[0, 1].set_ylabel('Audio Level (a.u.)', fontsize=11)
axes[0, 1].set_title('Two Boys Cafe - Raw Audio Signal', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(costa_audio['Recording (a.u.)'], bins=50, color='#FF6B35', 
                alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Audio Level (a.u.)', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('Costa Coffee - Audio Distribution', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(tbc_audio['Recording (a.u.)'], bins=50, color='#4ECDC4', 
                alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Audio Level (a.u.)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Two Boys Cafe - Audio Distribution', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('audio_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('Comprehensive Sensor Data Comparison', fontsize=18, fontweight='bold')

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(costa_light['Time (s)'], costa_light['Illuminance (lx)'], 
         color='#FF6B35', label='Costa Coffee (Evening)', linewidth=2, alpha=0.8)
ax1.plot(tbc_light['Time (s)'], tbc_light['Illuminance (lx)'], 
         color='#4ECDC4', label='Two Boys Cafe (Afternoon)', linewidth=2, alpha=0.8)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Illuminance (lux)', fontsize=12)
ax1.set_title('Light Levels Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(costa_accel['Time (s)'][:5000], costa_accel['magnitude'][:5000], 
         color='#FF6B35', linewidth=1, alpha=0.8)
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_ylabel('Accel. (m/s²)', fontsize=10)
ax2.set_title('Costa - Acceleration', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(tbc_accel['Time (s)'][:5000], tbc_accel['magnitude'][:5000], 
         color='#4ECDC4', linewidth=1, alpha=0.8)
ax3.set_xlabel('Time (s)', fontsize=10)
ax3.set_ylabel('Accel. (m/s²)', fontsize=10)
ax3.set_title('TBC - Acceleration', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 2])
box_data = [costa_accel['magnitude'], tbc_accel['magnitude']]
bp = ax4.boxplot(box_data, labels=['Costa', 'TBC'], patch_artist=True)
colors = ['#FF6B35', '#4ECDC4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('Accel. (m/s²)', fontsize=10)
ax4.set_title('Accel. Distribution', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(costa_audio['Time (s)'][:2000], costa_audio['Recording (a.u.)'][:2000], 
         color='#FF6B35', linewidth=0.5, alpha=0.8)
ax5.set_xlabel('Time (s)', fontsize=10)
ax5.set_ylabel('Audio (a.u.)', fontsize=10)
ax5.set_title('Costa - Audio', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(tbc_audio['Time (s)'][:2000], tbc_audio['Recording (a.u.)'][:2000], 
         color='#4ECDC4', linewidth=0.5, alpha=0.8)
ax6.set_xlabel('Time (s)', fontsize=10)
ax6.set_ylabel('Audio (a.u.)', fontsize=10)
ax6.set_title('TBC - Audio', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(gs[2, 2])
metrics = ['Avg Light\n(lux)', 'Avg Accel\n(m/s²)', 'Duration\n(s)']
costa_values = [
    costa_light['Illuminance (lx)'].mean(),
    costa_accel['magnitude'].mean(),
    costa_accel['Time (s)'].max()
]
tbc_values = [
    tbc_light['Illuminance (lx)'].mean(),
    tbc_accel['magnitude'].mean(),
    tbc_accel['Time (s)'].max()
]

x = np.arange(len(metrics))
width = 0.35
ax7.bar(x - width/2, costa_values, width, label='Costa', color='#FF6B35', alpha=0.8)
ax7.bar(x + width/2, tbc_values, width, label='TBC', color='#4ECDC4', alpha=0.8)
ax7.set_ylabel('Value', fontsize=10)
ax7.set_title('Key Metrics', fontsize=11, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(metrics, fontsize=9)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')

plt.savefig('comprehensive_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Gyroscope Comparison: Angular Velocity', fontsize=16, fontweight='bold')

axes[0].plot(costa_gyro['Time (s)'][:5000], costa_gyro['Gyroscope x (rad/s)'][:5000], 
             label='X-axis', alpha=0.7, linewidth=0.8)
axes[0].plot(costa_gyro['Time (s)'][:5000], costa_gyro['Gyroscope y (rad/s)'][:5000], 
             label='Y-axis', alpha=0.7, linewidth=0.8)
axes[0].plot(costa_gyro['Time (s)'][:5000], costa_gyro['Gyroscope z (rad/s)'][:5000], 
             label='Z-axis', alpha=0.7, linewidth=0.8)
axes[0].set_xlabel('Time (s)', fontsize=11)
axes[0].set_ylabel('Angular Velocity (rad/s)', fontsize=11)
axes[0].set_title('Costa Coffee - Gyroscope', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(tbc_gyro['Time (s)'][:5000], tbc_gyro['Gyroscope x (rad/s)'][:5000], 
             label='X-axis', alpha=0.7, linewidth=0.8)
axes[1].plot(tbc_gyro['Time (s)'][:5000], tbc_gyro['Gyroscope y (rad/s)'][:5000], 
             label='Y-axis', alpha=0.7, linewidth=0.8)
axes[1].plot(tbc_gyro['Time (s)'][:5000], tbc_gyro['Gyroscope z (rad/s)'][:5000], 
             label='Z-axis', alpha=0.7, linewidth=0.8)
axes[1].set_xlabel('Time (s)', fontsize=11)
axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=11)
axes[1].set_title('Two Boys Cafe - Gyroscope', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gyroscope_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
