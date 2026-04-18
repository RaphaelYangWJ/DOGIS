# Elasticity Dataset Generator (based on Scikit-FEM)

This project generates synthetic datasets for 2D linear elasticity problems, including Young's Modulus fields (input), displacement fields (output), and stress fields (output). The dataset can be used to train deep learning-based surrogate models (such as FNO, UNet, etc.) for stress analysis or inverse problem research.

## 1. Project Introduction

The core code utilizes the `scikit-fem` finite element library to solve plane stress/strain problems.
- **Physical Model**: Linear elastic material, isotropic.
- **Geometric Model**: Unit square domain $[0, 1] \times [0, 1]$.
- **Boundary Conditions**:
  - Bottom ($y=0$) fixed ($u_y=0$).
  - Bottom-left corner ($0,0$) fixed horizontal displacement ($u_x=0$) to eliminate rigid body motion.
  - Top ($y=1$) subjected to vertical compressive displacement ($u_y = \text{const}$).
- **Randomness**:
  - Randomly generated Young's modulus distribution (simulating inclusions in a matrix).
  - Smooth random stiffness fields generated using bicubic spline interpolation.

## 2. Requirements

Please ensure the following Python libraries are installed:

```bash
pip install numpy matplotlib scipy scikit-fem
```

Recommended Python version >= 3.8.

## 3. File Structure

```
SHM/
├── DataGenerator_Skfem_Function Sampling Space_Stress.py  # Main generation script
├── dataset_skfem_output/                                # Output directory (automatically generated)
│   ├── TrainingDataset/                                   # Training dataset
│   └── TestingDataset/                                    # Testing dataset
└── README.md                                              # This documentation file
```

## 4. Usage Guide

Run the Python script directly to start generating data:

```bash
python "DataGenerator_Skfem_Function Sampling Space_Stress.py"
```

The program will automatically perform the following steps:
1. Create the output directory `dataset_skfem_output`.
2. Generate training samples (default 5000).
3. Generate testing samples (default 200).
4. Save data files in `.npy` format.
5. Generate visualization images (`.png`) for some samples.

## 5. Configuration

You can adjust generation parameters by modifying the `Config` class in the script:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `RES` | 64 | Grid resolution ($64 \times 64$) |
| `E_MATRIX` | 10.0 | Matrix Young's Modulus |
| `E_INCLUSION` | 50.0 | Inclusion Young's Modulus |
| `NU` | 0.4 | Poisson's Ratio |
| `COMPRESSION` | -0.01 | Compressive displacement applied at the top |
| `N_TRAIN` | 5000 | Number of training samples |
| `N_TEST` | 200 | Number of testing samples |
| `MAX_DISPLACEMENT` | 0.2 | Sample filtering threshold: Max displacement |
| `MAX_STRESS` | 2.0 | Sample filtering threshold: Max stress |

## 6. Output Data Format

The following files will be generated in the `TrainingDataset` and `TestingDataset` directories:

- **`dataset_labels.npy`**: Young's Modulus field (input), shape `(N, RES, RES)`.
- **`dataset_full_displacement.npy`**: Displacement field (output), shape `(N, RES, RES, 2)`, the last dimension is $(u_x, u_y)$.
- **`dataset_full_stress.npy`**: Stress field (output), shape `(N, RES, RES, 3)`, the last dimension is $(\sigma_{xx}, \sigma_{yy}, \sigma_{xy})$.

In addition, `vis_sample_stress_{id}.png` image files will be generated to visually display the modulus distribution, displacement magnitude, and Von Mises stress distribution.
