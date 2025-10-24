#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

color_app = '#4A90E2'
color_sensors = '#50C878'
color_location = '#FF6B35'
color_export = '#9B59B6'
color_analysis = '#E74C3C'

ax.text(5, 11.5, 'Phyphox Data Collection Flow', 
        ha='center', va='center', fontsize=20, fontweight='bold')
ax.text(5, 11, 'Assignment 2: Coffee Shop Environmental Analysis', 
        ha='center', va='center', fontsize=12, style='italic', color='gray')

app_box = FancyBboxPatch((2, 9.5), 6, 1, 
                         boxstyle="round,pad=0.1", 
                         edgecolor=color_app, facecolor=color_app, 
                         linewidth=3, alpha=0.3)
ax.add_patch(app_box)
ax.text(5, 10, 'Phyphox Mobile App', 
        ha='center', va='center', fontsize=14, fontweight='bold')

arrow1 = FancyArrowPatch((5, 9.5), (5, 8.8),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow1)

sensors_box = FancyBboxPatch((1.5, 6.5), 7, 2, 
                            boxstyle="round,pad=0.1", 
                            edgecolor=color_sensors, facecolor=color_sensors, 
                            linewidth=3, alpha=0.3)
ax.add_patch(sensors_box)
ax.text(5, 8.2, '8 Sensors Collected', 
        ha='center', va='center', fontsize=13, fontweight='bold')
ax.text(2.5, 7.6, '✓ Accelerometer', ha='left', fontsize=10)
ax.text(2.5, 7.3, '✓ Gyroscope', ha='left', fontsize=10)
ax.text(2.5, 7.0, '✓ Light Sensor', ha='left', fontsize=10)
ax.text(2.5, 6.7, '✓ Audio', ha='left', fontsize=10)

ax.text(5.5, 7.6, '✗ GPS (poor)', ha='left', fontsize=10, color='red')
ax.text(5.5, 7.3, '✗ Magnetometer', ha='left', fontsize=10, color='gray')
ax.text(5.5, 7.0, '✗ Orientation', ha='left', fontsize=10, color='gray')
ax.text(5.5, 6.7, '✗ Linear Accel', ha='left', fontsize=10, color='gray')

arrow2 = FancyArrowPatch((5, 6.5), (5, 5.8),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow2)

costa_box = FancyBboxPatch((0.5, 4.3), 4, 1.3, 
                          boxstyle="round,pad=0.08", 
                          edgecolor=color_location, facecolor=color_location, 
                          linewidth=2, alpha=0.3)
ax.add_patch(costa_box)
ax.text(2.5, 5.4, 'Costa Coffee (Evening)', 
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(2.5, 5.0, '⏱ 152 seconds', ha='center', fontsize=9)
ax.text(2.5, 4.7, '📊 75,624 samples', ha='center', fontsize=9)
ax.text(2.5, 4.4, '🏷 "BE" prefix files', ha='center', fontsize=9)

tbc_box = FancyBboxPatch((5.5, 4.3), 4, 1.3, 
                        boxstyle="round,pad=0.08", 
                        edgecolor=color_location, facecolor=color_location, 
                        linewidth=2, alpha=0.3)
ax.add_patch(tbc_box)
ax.text(7.5, 5.4, 'Two Boys Cafe (Afternoon)', 
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(7.5, 5.0, '⏱ 327 seconds', ha='center', fontsize=9)
ax.text(7.5, 4.7, '📊 162,771 samples', ha='center', fontsize=9)
ax.text(7.5, 4.4, '🏷 No prefix files', ha='center', fontsize=9)

arrow3a = FancyArrowPatch((2.5, 4.3), (4, 3.8),
                         arrowstyle='->', mutation_scale=25, 
                         linewidth=2, color='black')
ax.add_patch(arrow3a)
arrow3b = FancyArrowPatch((7.5, 4.3), (6, 3.8),
                         arrowstyle='->', mutation_scale=25, 
                         linewidth=2, color='black')
ax.add_patch(arrow3b)

export_box = FancyBboxPatch((2, 2.5), 6, 1, 
                           boxstyle="round,pad=0.1", 
                           edgecolor=color_export, facecolor=color_export, 
                           linewidth=3, alpha=0.3)
ax.add_patch(export_box)
ax.text(5, 3, 'Export to CSV Files', 
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(5, 2.65, '(10 files per location)', 
        ha='center', va='center', fontsize=9, style='italic')

arrow4 = FancyArrowPatch((5, 2.5), (5, 1.8),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow4)

analysis_box = FancyBboxPatch((1, 0.2), 8, 1.4, 
                             boxstyle="round,pad=0.1", 
                             edgecolor=color_analysis, facecolor=color_analysis, 
                             linewidth=3, alpha=0.3)
ax.add_patch(analysis_box)
ax.text(5, 1.4, 'Data Analysis Results', 
        ha='center', va='center', fontsize=13, fontweight='bold')
ax.text(2.5, 1.05, '✓ Used: 62.9%', ha='center', fontsize=10, fontweight='bold', color='green')
ax.text(2.5, 0.75, '(487,331 samples)', ha='center', fontsize=8)
ax.text(2.5, 0.45, 'Light, Accel, Gyro, Audio', ha='center', fontsize=8)

ax.text(7.5, 1.05, '✗ Unused: 37.1%', ha='center', fontsize=10, fontweight='bold', color='red')
ax.text(7.5, 0.75, '(287,751 samples)', ha='center', fontsize=8)
ax.text(7.5, 0.45, 'GPS, Mag, Orient, etc.', ha='center', fontsize=8)

key_box = FancyBboxPatch((0.3, 10.2), 2.5, 0.7, 
                        boxstyle="round,pad=0.05", 
                        edgecolor='gray', facecolor='lightyellow', 
                        linewidth=1, alpha=0.8)
ax.add_patch(key_box)
ax.text(1.55, 10.75, 'Key Findings:', ha='center', fontsize=9, fontweight='bold')
ax.text(1.55, 10.5, 'TBC 108% brighter', ha='center', fontsize=7)
ax.text(1.55, 10.3, 'TBC 97% louder', ha='center', fontsize=7)

issue_box = FancyBboxPatch((7.2, 10.2), 2.5, 0.7, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='gray', facecolor='#FFE6E6', 
                          linewidth=1, alpha=0.8)
ax.add_patch(issue_box)
ax.text(8.45, 10.75, 'Data Issues:', ha='center', fontsize=9, fontweight='bold')
ax.text(8.45, 10.5, 'GPS 100% missing', ha='center', fontsize=7)
ax.text(8.45, 10.3, 'Duration mismatch 2.15x', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig('phyphox_data_flow.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
