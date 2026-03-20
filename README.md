# Prediction of Fatigue Thresholds and Characteristic Crack Lengths

This repository contains the source code and datasets for the multiscale fatigue model described in the paper:

> **"Prediction of fatigue thresholds and characteristic crack lengths by multiscale simulation of the Kitagawa–Takahashi diagram and cyclic R curve"** by Qingzhi Yao and Kazuki Shibanuma.

## Overview
The repository is divided into two independent, self-contained directories corresponding to the logical workflow of the manuscript. Each directory contains its respective scripts and required data files.

## Repository Structure

### 1. `multiscale-fatigue-model/` (Validation on Real Specimens)
- **Purpose**: Corresponds to **Section 3** of the manuscript. It validates the multiscale fatigue model's capability to reproduce small-crack growth behavior (non-monotonic da/dN curves) and predict fatigue life (S-N curves).
- **Contents**: 
  - Forward simulation scripts for real specimen geometries.
  - Required data files including finite element (FEM) mesh files (`.inp`), strain field data (`.dat`), weight functions, and microstructural information.
- **Usage**: Run the main life prediction script within this directory.

### 2. `fatigue-thresholds-multiscale-model/` (Threshold Prediction Framework)
- **Purpose**: Corresponds to **Sections 4 and 5** of the manuscript. It implements the threshold prediction framework on an idealized semi-infinite body. 
- **Contents**:
  - **Forward Analysis Script** (`forward_analysis.py`): Calculates the fatigue life analytically to generate the reference S-N curve (failure probability) and determine the reference applied stress amplitude ($\sigma_{ref}$).
  - **Inverse Analysis Script** (`fatigue_thresholds_prediction.py`): Implements the crack arrest condition (da/dN=0) to determine the critical driving force. It generates the Kitagawa–Takahashi (K-T) diagram and cyclic R-curve, identifying four key parameters: $\sigma_e$, $\Delta K_{th,LC}$, $d_1$, and $d_2$.
  - Microstructural information..
- **Usage**: Run the forward analysis script first to obtain the reference stress, followed by the inverse analysis script to predict the thresholds.

## Requirements
- Python 3.x
- Required libraries: `numpy`, `pandas`, `scipy`

Install the dependencies using:
```bash
pip install numpy pandas scipy
