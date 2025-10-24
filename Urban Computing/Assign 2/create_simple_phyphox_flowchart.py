#!/usr/bin/env python3
"""
Create a simple Phyphox data collection flowchart
Similar to a login/signup flow diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')

# Colors
color_start = '#D6EAF8'      # Light blue
color_process = '#D5F4E6'    # Light green
color_decision = '#FADBD8'   # Light pink
color_end = '#E8DAEF'        # Light purple

def draw_process_box(ax, x, y, width, height, text, fontsize=11):
    """Draw a rounded rectangle process box"""
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.15", 
                         edgecolor='black', facecolor=color_process, 
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_start_box(ax, x, y, width, height, text, fontsize=11):
    """Draw a start/end box"""
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.2", 
                         edgecolor='black', facecolor=color_start, 
                         linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize, fontweight='bold')

def draw_decision_diamond(ax, x, y, width, height, text, fontsize=10):
    """Draw a diamond decision box"""
    points = [
        [x + width/2, y + height],      # top
        [x + width, y + height/2],      # right
        [x + width/2, y],                # bottom
        [x, y + height/2]                # left
    ]
    diamond = Polygon(points, closed=True, 
                     edgecolor='black', facecolor=color_decision, 
                     linewidth=2)
    ax.add_patch(diamond)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Title
ax.text(7, 10.5, 'Phyphox Data Collection Process', 
        ha='center', va='center', fontsize=18, fontweight='bold')

# 1. Start
draw_start_box(ax, 0.5, 9.3, 2.5, 0.8, 'Open Phyphox App', 12)

# Arrow to sensors
draw_arrow(ax, 1.75, 9.3, 1.75, 8.8)

# 2. Configure sensors
draw_process_box(ax, 0.3, 7.5, 2.9, 1.1, 'Configure Sensors\n(Accelerometer, Gyroscope,\nLight, Audio, GPS, etc.)', 10)

# Arrow to decision
draw_arrow(ax, 1.75, 7.5, 1.75, 6.8)

# 3. Decision: Sensors ready?
draw_decision_diamond(ax, 0.5, 5.5, 2.5, 1.1, 'Sensors\nready?', 11)

# NO arrow - loop back
draw_arrow(ax, 0.5, 6.05, 0.2, 8.05)
ax.text(0.1, 7, 'NO', ha='right', va='center', fontsize=9, fontweight='bold')

# YES arrow
draw_arrow(ax, 3, 6.05, 4.5, 6.05)
ax.text(3.5, 6.25, 'YES', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Start recording
draw_process_box(ax, 4.5, 5.5, 2.5, 1.1, 'Start Recording\nAll Sensors', 11)

# Arrow down
draw_arrow(ax, 5.75, 5.5, 5.75, 4.8)

# 5. Data collection location 1
draw_process_box(ax, 4.5, 3.5, 2.5, 1.1, 'Collect Data at\nLocation 1\n(Costa Coffee)', 10)

# Arrow down
draw_arrow(ax, 5.75, 3.5, 5.75, 2.8)

# 6. Data collection location 2
draw_process_box(ax, 4.5, 1.5, 2.5, 1.1, 'Collect Data at\nLocation 2\n(Two Boys Cafe)', 10)

# Arrow to export
draw_arrow(ax, 7, 2.05, 8.5, 2.05)

# 7. Stop recording
draw_process_box(ax, 8.5, 1.5, 2.5, 1.1, 'Stop Recording\n& Save Data', 11)

# Arrow up
draw_arrow(ax, 9.75, 2.6, 9.75, 3.5)

# 8. Export decision
draw_decision_diamond(ax, 8.5, 3.5, 2.5, 1.1, 'Export\nformat?', 11)

# CSV arrow
draw_arrow(ax, 11, 4.05, 12, 4.05)
ax.text(11.5, 4.25, 'CSV', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 9. Export to CSV
draw_process_box(ax, 12, 3.5, 2, 1.1, 'Export to\nCSV Files', 11)

# Arrow down from CSV export
draw_arrow(ax, 13, 3.5, 13, 2.8)

# 10. Files saved
draw_process_box(ax, 11.8, 1.5, 2.4, 1.1, '10 CSV Files\nper Location\nSaved', 10)

# Arrow to analysis
draw_arrow(ax, 13, 1.5, 13, 0.8)

# 11. End - Data ready
draw_start_box(ax, 11.5, 0.1, 3, 0.6, 'Data Ready for Analysis', 11)

# Add sub-process boxes on the left for details
# Sensor configuration details
ax.text(0.2, 9, 'Sensor Setup:', ha='left', va='top', fontsize=8, 
        fontweight='bold', style='italic', color='gray')
ax.text(0.2, 8.7, '• 500 Hz sampling', ha='left', fontsize=7, color='gray')
ax.text(0.2, 8.5, '• Multi-sensor recording', ha='left', fontsize=7, color='gray')

# Collection details
ax.text(4.5, 4.8, 'Collection Details:', ha='left', va='top', fontsize=8, 
        fontweight='bold', style='italic', color='gray')
ax.text(4.5, 4.55, '• Fixed position', ha='left', fontsize=7, color='gray')
ax.text(4.5, 4.35, '• 2-5 minute sessions', ha='left', fontsize=7, color='gray')
ax.text(4.5, 4.15, '• Multiple time slots', ha='left', fontsize=7, color='gray')

# Export details
ax.text(8.5, 2.9, 'Export Includes:', ha='left', va='top', fontsize=8, 
        fontweight='bold', style='italic', color='gray')
ax.text(8.5, 2.65, '• Raw sensor data', ha='left', fontsize=7, color='gray')
ax.text(8.5, 2.45, '• Timestamps', ha='left', fontsize=7, color='gray')
ax.text(8.5, 2.25, '• Metadata', ha='left', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig('phyphox_flowchart.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Flowchart saved as 'phyphox_flowchart.png'")
plt.close()

print("\nFlowchart created successfully!")
print("File: phyphox_flowchart.png")

