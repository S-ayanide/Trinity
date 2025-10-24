# Coffee Shop Environmental Data Analysis

This analysis compares sensor data collected from two different coffee shops at different times:
- **Costa Coffee** - Data collected in the **evening**
- **Two Boys Cafe (TBC)** - Data collected in the **afternoon**

## 📊 Data Overview

Both datasets contain comprehensive sensor data collected from mobile devices:

### Sensors Included:
1. **Accelerometer** - 3-axis motion detection (X, Y, Z)
2. **Gyroscope** - Angular velocity measurements
3. **Light Sensor** - Ambient illuminance levels
4. **Audio** - Raw audio recordings
5. **Linear Acceleration** - Device movement
6. **Magnetometer** - Magnetic field detection
7. **Orientation** - Device orientation
8. **Location** - GPS coordinates

## 📁 Files Structure

```
Assign 2/
├── Costa Data.zip                # Original Costa Coffee data
├── TBC Data.zip                  # Original Two Boys Cafe data
├── costa_data/                   # Extracted Costa Coffee data
├── tbc_data/                     # Extracted Two Boys Cafe data
├── coffee_shop_analysis.ipynb    # Main analysis notebook
├── run_analysis.py               # Script to execute analysis
├── DATA_CHALLENGES.md            # ⚠️ Detailed data quality issues
├── analyze_data_challenges.py    # Script to validate data issues
└── README.md                     # This file
```

## 🚀 How to Run the Analysis

### Option 1: Jupyter Notebook (Recommended)
```bash
cd "/Users/sayanide/Documents/Assignments/Urban Computing/Assign 2"
jupyter notebook coffee_shop_analysis.ipynb
```
Then run all cells (Cell → Run All)

### Option 2: Using the Python Script
```bash
cd "/Users/sayanide/Documents/Assignments/Urban Computing/Assign 2"
python3 run_analysis.py
```

### Option 3: VS Code / Cursor
Simply open `coffee_shop_analysis.ipynb` in your IDE and run all cells.

## 📈 Analysis Components

### 1. Data Summary Comparison Table
A comprehensive comparison table showing:
- Data collection duration
- Number of samples collected
- Average illuminance levels
- Audio level statistics
- Acceleration magnitudes

### 2. Statistical Summary Tables
Detailed statistics for:
- **Light Sensor**: Mean, Median, Std Dev, Min, Max, Range
- **Accelerometer**: Statistical distribution of motion data

### 3. Visualizations

#### 4.1 Light Sensor Comparison
- Time series plots for both locations
- Distribution histograms
- Box plot comparisons
- Shows the difference in lighting between evening (Costa) and afternoon (TBC)

#### 4.2 Accelerometer Data
- 3-axis acceleration plots
- Magnitude calculations and comparisons
- Movement pattern analysis

#### 4.3 Audio Data
- Raw audio signal waveforms
- Distribution analysis
- Ambient noise level comparisons

#### 4.4 Combined Multi-Sensor Overview
- Comprehensive dashboard showing all sensors
- Side-by-side comparisons
- Key metrics bar chart

#### 4.5 Gyroscope Data
- Angular velocity measurements
- Rotation patterns in 3D space

### 4. Key Findings Summary
Automated analysis generating insights about:
- Light level differences between evening and afternoon
- Movement and activity patterns
- Audio/noise level comparisons
- Location information
- Duration of data collection

## 📊 Expected Findings

The analysis will reveal:

1. **Lighting Differences**: Afternoon (TBC) likely has higher illuminance due to natural daylight
2. **Activity Patterns**: Different movement patterns based on time of day
3. **Ambient Noise**: Comparison of noise levels between locations and times
4. **Environmental Context**: How the urban environment differs across times and locations

## 🔧 Requirements

Make sure you have these Python packages installed:

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## ⚠️ Data Quality Challenges

As mentioned in the assignment ("If you cannot find any problem with the data that you collected (unlikely!)"), several data quality issues were identified:

### Major Issues Found:
1. **Location Data Almost Unusable** - 100% missing values for Height, Velocity, Direction in Costa data
2. **Duration Mismatch** - TBC has 2.15x more data (327s vs 152s)
3. **Inconsistent File Naming** - "BE" prefix only in Costa files
4. **Sampling Rate Gaps** - TBC data has occasional large gaps (up to 177ms)
5. **Sensors Not Used** - Magnetometer, Orientation, Linear Accel, FFT, Peak History (287,420 unused samples!)

### View Details:
```bash
# Read comprehensive documentation
cat DATA_CHALLENGES.md

# Run validation script
python3 analyze_data_challenges.py
```

**See `DATA_CHALLENGES.md` for:**
- Detailed analysis of each issue
- Why certain sensors weren't used
- Impact on the analysis
- Lessons for future data collection

## 📝 Notes

- All data has been extracted from the ZIP files into `costa_data/` and `tbc_data/` folders
- The notebook contains interactive visualizations
- Color scheme: 
  - 🟠 Orange (#FF6B35) = Costa Coffee
  - 🔵 Teal (#4ECDC4) = Two Boys Cafe
- **287,420 sensor readings** were collected but not used in analysis (see DATA_CHALLENGES.md)

## 🎯 Purpose

This analysis is part of an Urban Computing assignment to understand how environmental sensor data varies across different urban locations and times of day. The insights can help in:
- Understanding urban activity patterns
- Environmental monitoring
- Location-based context awareness
- Urban planning and design decisions

## 📧 Questions?

If you encounter any issues running the analysis, make sure:
1. All required packages are installed
2. The data has been extracted (check for `costa_data/` and `tbc_data/` folders)
3. You're running Python 3.7 or later

---

**Created**: October 2025  
**Assignment**: Urban Computing - Assignment 2  
**Tools Used**: Python, Pandas, Matplotlib, Seaborn, Jupyter

