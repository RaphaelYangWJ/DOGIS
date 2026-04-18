# DOGIS: Dynamic Operator-Guided Flow Matching

**A Generative Physical Inverse Solver for Arbitrary Sparse Observations**

[![Paper](https://img.shields.io/badge/Paper-MLST%20(Under%20Review)-blue)](#)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> This repository contains the official PyTorch implementation of **DOGIS**, a generative framework designed to solve severely ill-posed inverse problems under extreme data sparsity and high noise regimes.

## 📖 Overview

[cite_start]Solving ill-posed inverse problems—inversing high-dimensional, heterogeneous physical fields from limited observations—is a fundamental challenge in scientific engineering[cite: 153]. [cite_start]Traditional solvers are computationally prohibitive [cite: 191][cite_start], while recent deep learning approaches (e.g., Diffusion Posterior Sampling) suffer from catastrophic posterior collapse and significant inference latency when physical gradients become ill-conditioned under extreme sparsity[cite: 192].

[cite_start]**DOGIS** overcomes this bottleneck by paradigm-shifting the computational burden from test-time optimization to amortized training[cite: 319]. [cite_start]By explicitly embedding a pre-trained Fourier Neural Operator (FNO) into the continuous normalizing flow trajectory, DOGIS successfully internalizes complex PDE dynamics into the generative prior itself[cite: 320]. 

### 🔥 Key Highlights
* [cite_start]**Extreme Sparsity Robustness:** Exhibits graceful degradation under severe sparsity, successfully recovering $64 \times 64$ resolution fields from merely 16 random sensors (an extreme 0.78% sparsity)[cite: 199].
* [cite_start]**Ultra-Fast Inference:** Achieves an average 8x inference speedup over standard DPS frameworks[cite: 201].
* [cite_start]**Physics-Aware UQ:** Provides precise, physics-aware Uncertainty Quantification (UQ) that isolates high-variance regions to genuinely ambiguous topological boundaries, avoiding uninformative variance inflation[cite: 200, 411].
* [cite_start]**Continuous Data Assimilation:** The time-adaptive weighting scheduling $\lambda(t)$ is mathematically equivalent to the continuous-time limit of covariance inflation in Ensemble Smoother with Multiple Data Assimilation (ES-MDA)[cite: 97].

## ⚙️ Core Architecture & Inference Modes

[cite_start]DOGIS provides a highly flexible hybrid inference paradigm, allowing users to dynamically balance computational latency and physical precision[cite: 124]:

1. [cite_start]**Agile Mode (Amortized Zero-Shot Inference):** Executes an amortized inference path using Classifier-Free Guidance (CFG) for real-time decision-making and large-scale UQ sampling[cite: 125, 127].
2. [cite_start]**Guided Mode (Test-Time Physics Guidance):** Performs a continuous manifold projection[cite: 133]. [cite_start]During the later stages of integration (e.g., $t \ge 0.5$), the exact automatic differentiation of the FNO surrogate is leveraged to compute the steepest-descent direction, continuously pulling the trajectory towards the precise sparse data manifold[cite: 109, 110, 130].

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/RaphaelYangWJ/DOGIS.git](https://github.com/RaphaelYangWJ/DOGIS.git)
cd DOGIS
pip install -r requirements.txt
```
*(Ensure you have a CUDA-compatible PyTorch version installed).*

## 🚀 Quick Start & Evaluation

The repository evaluates DOGIS on two highly heterogeneous physical scenarios:
1. [cite_start]**Darcy Flow:** Subsurface fluid transport governed by second-order elliptic PDEs (permeability field inversion)[cite: 160].
2. [cite_start]**Structural Health Monitoring (SHM):** Mechanical behaviors governed by elastodynamic equations mapping spatially heterogeneous elastic properties (Young's modulus inversion)[cite: 209].

### Running the Inference (Guided Mode)
[cite_start]To run the evaluation script and save the inverted fields (GT, Inputs, and Inversions) into HDF5 format[cite: 8, 9]:

```python
# Example evaluation command (refer to specific scripts in /scripts)
python eval.py --scenario shm --obs_num 16 --mode guided --t_start 0.5
```

### Visualizing Results (3D & Animations)
We provide comprehensive Jupyter Notebooks for post-processing and visualizing the `h5` results:
* [cite_start]`Sparsity_Robustness_Analysis.ipynb`: Generates 3D surface comparisons and quantitative RMSE/SSIM tables[cite: 1, 21].
* [cite_start]`Sampling_dynamic.ipynb` / `Video_Generation.ipynb`: Generates IOP-compliant `.mp4` animations (H.264, 15fps, 480x360) of the step-by-step generative inversion process from Gaussian noise[cite: 441, 454].

## 📂 Repository Structure

```text
DOGIS/
├── data/                  # Data loaders and dataset handling
├── models/                # Core architectures (FlowMatching, FNO)
├── scripts/               # Training and evaluation scripts
├── notebooks/             # UQ, 3D visualizations, and video generation
├── requirements.txt       # Dependencies
└── README.md
```



