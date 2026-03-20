# Prediction of Fatigue Thresholds and Characteristic Crack Lengths

This repository contains the source code and datasets for the multiscale fatigue model described in the paper:
**"Prediction of fatigue thresholds and characteristic crack lengths by multiscale simulation of the Kitagawa–Takahashi diagram and cyclic R curve"**

## Repository Structure

### 1. `multiscale-fatigue-model/`
- **Purpose**: Forward analysis for fatigue life prediction and model validation.
- **Description**: This directory contains the code used to reproduce the small-crack growth behavior and S-N data discussed in **Section 3** of the manuscript.

### 2. `fatigue-thresholds-multiscale-model/`
- **Purpose**: Inverse analysis for fatigue threshold prediction.
- **Description**: This directory contains the reformulated framework used to predict the four fatigue threshold parameters ($\sigma_e$, $\Delta K_{th,LC}$, $d_1$, and $d_2$) by constructing the Kitagawa–Takahashi (K-T) diagram and cyclic R-curve, as discussed in **Sections 4 and 5**.

## Requirements
- Python 3.x
- Libraries: `numpy`, `pandas`, `scipy`

## Usage
1. Navigate to the desired folder.
2. Ensure all required `N50R_*.csv` data files are present in the same directory as the script.
3. Run the Python script to generate the results.
