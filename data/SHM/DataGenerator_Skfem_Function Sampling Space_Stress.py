import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RectBivariateSpline
from skfem import *
from skfem.helpers import div, ddot, sym_grad


# ------------------------------------------------------------------------------
# Utility Function: LAME Parameter Conversion
# ------------------------------------------------------------------------------
def lam_lame(E, nu):
    """
    Calculate the Lame parameter lambda.
    Parameters:
        E: Young's Modulus
        nu: Poisson's Ratio
    Returns:
        Lame parameter lambda
    """
    return E * nu / ((1 + nu) * (1 - 2 * nu))


def mu_lame(E, nu):
    """
    Calculate the Lame's second parameter mu (shear modulus).
    Parameters:
        E: Young's Modulus
        nu: Poisson's Ratio
    Returns:
        Lame's mu parameter
    """
    return E / (2 * (1 + nu))


# ------------------------------------------------------------------------------
# 弱形式定义 (Weak Form)
# ------------------------------------------------------------------------------
@BilinearForm
def Lame_Elasticity(u, v, w):
    """
    Define the bilinear form of linear elastic mechanics.
    Based on the Lame parameters lambda (w.lam) and mu (w.mu).

    Formula corresponds to:
    ∫ (lambda * (div u) * (div v) + 2 * mu * (epsilon(u) : epsilon(v))) dx
    where epsilon(u) = sym_grad(u)
    """
    return w.lam * div(u) * div(v) + 2 * w.mu * ddot(sym_grad(u), sym_grad(v))


# ------------------------------------------------------------------------------
# Configuration Class
# ------------------------------------------------------------------------------
class Config:
    L = 1.0               # The side length of the domain (square domain [0, L] x [0, L])
    RES = 64              # Grid Resolution (RES x RES)
    E_MATRIX = 10.0       # Young's modulus of the matrix material
    E_INCLUSION = 50.0    # Young's modulus in inclusion/high-stiffness regions
    NU = 0.4              # Poisson ratio (Note: Values close to 0.5 may cause volume locking or numerical instability)
    COMPRESSION = -0.01   # Displacement boundary conditions applied at the top (compression amount)
    N_TRAIN = 5000        # Number of training set samples
    N_TEST = 200          # Number of samples in the test set
    SEED_TRAIN = 42       # Training set random seed
    SEED_TEST = 4242      # Test set random seed
    VIS_SAMPLES = 5       # Number of Visualization Samples
    MAX_DISPLACEMENT = 0.2  # Threshold: Maximum allowable displacement modulus (used to filter out divergent or unreasonable solutions)
    MAX_STRESS = 2.0        # Threshold: Maximum allowable value of stress component
    MAX_RESAMPLE_TRIES = 20 # Maximum number of resampling attempts (if generated samples do not meet the threshold)
    SAVE_DIR = Path("dataset_skfem_output") # Data Storage Root Directory
    TRAIN_SUBDIR = "TrainingDataset"        # Training Set Subdirectory
    TEST_SUBDIR = "TestingDataset"          # Test Set Subdirectory


# Initialize configuration and create directory
cfg = Config()
cfg.SAVE_DIR.mkdir(parents=True, exist_ok=True)
cfg.TRAIN_DIR = cfg.SAVE_DIR / cfg.TRAIN_SUBDIR
cfg.TEST_DIR = cfg.SAVE_DIR / cfg.TEST_SUBDIR
cfg.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
cfg.TEST_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Stiffness Field Generating Function
# ------------------------------------------------------------------------------
def generate_stiffness_map(rng: np.random.Generator, res=64):
    """
    Generate a random distribution field for Young's modulus (E).

    Method:
    1. Randomly generate 0/1 states on a coarse grid.
    2. Smoothly up-sample to a fine grid using bicubic spline interpolation.
    3. Map to the range [E_MATRIX, E_INCLUSION].
    4. Randomly apply flip enhancement.

    Parameters:
        rng: numpy random number generator
        res: target resolution
    Returns:
        E_field: Young's modulus field with shape (res, res)
    """
    n = 4  # Number of internal nodes in the coarse mesh
    grid_size = n + 2
    grid_values = np.zeros((grid_size, grid_size))
    
    # Randomly generate an internal value (0 or 1)
    inner_values = rng.integers(0, 2, size=(n, n))
    grid_values[1:-1, 1:-1] = inner_values
    
    # Define coarse grid coordinates
    x_coarse = np.linspace(0, cfg.L, grid_size)
    y_coarse = np.linspace(0, cfg.L, grid_size)
    
    # Interpolation
    spline = RectBivariateSpline(y_coarse, x_coarse, grid_values, kx=3, ky=3)
    x_fine = np.linspace(0, cfg.L, res)
    y_fine = np.linspace(0, cfg.L, res)
    g_field = spline(y_fine, x_fine)
    
    # Truncate and map to physical values
    g_field = np.clip(g_field, 0.0, 1.0)
    E_field = cfg.E_MATRIX + (cfg.E_INCLUSION - cfg.E_MATRIX) * g_field
    
    # Data Augmentation: Random Flip
    if rng.random() > 0.5:
        E_field = np.flipud(E_field)
    if rng.random() > 0.5:
        E_field = np.fliplr(E_field)
        
    return E_field


# ------------------------------------------------------------------------------
# Finite Element Solver
# ------------------------------------------------------------------------------
def solve_elasticity_skfem(E_img, query_coords):
    """
    Solve linear elastic equations using scikit-fem.

    Parameters:
        E_img: Young's modulus distribution map (RES x RES)
        query_coords: Query point coordinates for interpolating output results
    Returns:
        u_at_query: Displacement vector at query points (N x 2)
    """

    mesh = MeshTri.init_tensor(
        np.linspace(0, cfg.L, cfg.RES + 1),
        np.linspace(0, cfg.L, cfg.RES + 1)
    )
    element = ElementVector(ElementTriP1())
    basis = Basis(mesh, element)
    

    gpts = basis.global_coordinates()
    ix = np.clip((gpts[0] / cfg.L * cfg.RES).astype(int), 0, cfg.RES - 1)
    iy = np.clip((gpts[1] / cfg.L * cfg.RES).astype(int), 0, cfg.RES - 1)
    

    E_at_quads = E_img[cfg.RES - 1 - iy, ix]
    

    mu_vals = mu_lame(E_at_quads, cfg.NU)
    lam_vals = lam_lame(E_at_quads, cfg.NU)
    

    K = asm(Lame_Elasticity, basis, mu=mu_vals, lam=lam_vals)
    

    dofs_bot = basis.get_dofs(lambda x: np.isclose(x[1], 0.0))

    dofs_top = basis.get_dofs(lambda x: np.isclose(x[1], cfg.L))

    dofs_corner_x = basis.get_dofs(lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))

    D = np.concatenate([
        dofs_bot.nodal['u^2'],      # 底部 y 方向固定
        dofs_top.nodal['u^2'],      # 顶部 y 方向受控
        dofs_corner_x.nodal['u^1']  # 角点 x 方向固定
    ])

    x_full = np.zeros(basis.N)
    top_dofs_indices = dofs_top.nodal['u^2']
    x_full[top_dofs_indices] = cfg.COMPRESSION

    # condense 用于消除 Dirichlet 边界条件的自由度
    u = solve(*condense(K, np.zeros(basis.N), D=D, x=x_full))

    ux = u[basis.nodal_dofs[0]]
    uy = u[basis.nodal_dofs[1]]
    
    basis_scalar = Basis(mesh, ElementTriP1())
    query_points_T = query_coords.T
    M = basis_scalar.probes(query_points_T) # 探针矩阵
    
    vals_x = M @ ux
    vals_y = M @ uy
    
    u_at_query = np.column_stack((vals_x, vals_y))
    return u_at_query


# ------------------------------------------------------------------------------
# 应力场计算
# ------------------------------------------------------------------------------
def compute_stress_field(u_field, E_map):
    """
    Calculate the stress field based on the displacement field and Young's modulus field.
    Compute strain using the finite difference method, then calculate stress via the constitutive equation.

    Parameters:
        u_field: Displacement field (RES x RES x 2)
        E_map: Young's modulus field (RES x RES)
    Returns:
        stress_tensor: Stress tensor field (RES x RES x 3), containing [sigma_xx, sigma_yy, sigma_xy]
    """
    y_coords = np.linspace(cfg.L, 0.0, cfg.RES)
    x_coords = np.linspace(0.0, cfg.L, cfg.RES)
    
    ux = u_field[:, :, 0]
    uy = u_field[:, :, 1]

    dux_dy, dux_dx = np.gradient(ux, y_coords, x_coords, edge_order=2)
    duy_dy, duy_dx = np.gradient(uy, y_coords, x_coords, edge_order=2)
    
    eps_xx = dux_dx
    eps_yy = duy_dy
    eps_xy = 0.5 * (dux_dy + duy_dx)

    mu = mu_lame(E_map, cfg.NU)
    lam = lam_lame(E_map, cfg.NU)

    trace_eps = eps_xx + eps_yy
    sigma_xx = lam * trace_eps + 2 * mu * eps_xx
    sigma_yy = lam * trace_eps + 2 * mu * eps_yy
    sigma_xy = 2 * mu * eps_xy
    
    stress_tensor = np.stack((sigma_xx, sigma_yy, sigma_xy), axis=-1)
    return stress_tensor


# ------------------------------------------------------------------------------
# Results Verification
# ------------------------------------------------------------------------------
def validate_field(u_field: np.ndarray, stress_field: np.ndarray):
    """
    Verify whether the calculation results are reasonable to prevent numerically unstable or physically unreasonable solutions.
    """
    disp_max = float(np.max(np.abs(u_field)))
    stress_max = float(np.max(np.abs(stress_field)))

    valid = (disp_max <= cfg.MAX_DISPLACEMENT) and (stress_max <= cfg.MAX_STRESS)
    return valid, disp_max, stress_max


# ------------------------------------------------------------------------------
# Main Loop for Dataset Generation
# ------------------------------------------------------------------------------
def generate_dataset(num_samples, split_name, split_dir: Path,
                     rng: np.random.Generator, full_field_coords):
    """
    Generate and save the specified number of samples.

    Parameters:
        num_samples: Number of samples
        split_name: Dataset name (e.g., “Training”)
        split_dir: Save directory
        rng: Random number generator
        full_field_coords: Full-field grid coordinates
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Gen {split_name} dataset ({num_samples} samples) ---")
    
    disp_fields = []
    stress_fields = []
    modulus_fields = []

    i = 0
    while i < num_samples:
        if i % 5 == 0:
            print(f"{split_name}: Processing {i}/{num_samples}...")

        for attempt in range(1, cfg.MAX_RESAMPLE_TRIES + 1):

            E_map = generate_stiffness_map(rng, cfg.RES)

            u_flat = solve_elasticity_skfem(E_map, full_field_coords)
            u_field = u_flat.reshape(cfg.RES, cfg.RES, 2)
            
            stress_field = compute_stress_field(u_field, E_map)
            
            valid, disp_max, stress_max = validate_field(u_field, stress_field)

            if valid:
                disp_fields.append(u_field)
                stress_fields.append(stress_field)
                modulus_fields.append(E_map)
                i += 1
                break
            

            print(f"{split_name}: Sample {i} No. {attempt}/{cfg.MAX_RESAMPLE_TRIES} The simulation was discarded. "
                  f"(max|u|={disp_max:.3e}, max|σ|={stress_max:.3e})")

        else:

            raise RuntimeError(
                f"{split_name}: cannot {cfg.MAX_RESAMPLE_TRIES} Generate samples that meet the threshold，"
                f"Please relax the MAX_DISPLACEMENT / MAX_STRESS settings or check the boundary conditions."
            )


    X_data = np.array(disp_fields)
    S_data = np.array(stress_fields)
    Y_data = np.array(modulus_fields)


    np.save(split_dir / "dataset_full_displacement.npy", X_data)
    np.save(split_dir / "dataset_full_stress.npy", S_data)
    np.save(split_dir / "dataset_labels.npy", Y_data)
    print(f"{split_name}: Displacement/stress/modulus data has been saved to {split_dir}")


    vis_count = min(cfg.VIS_SAMPLES, num_samples)
    if vis_count > 0:
        vis_indices = rng.choice(num_samples, size=vis_count, replace=False)
        for idx in np.atleast_1d(vis_indices):
            idx = int(idx)
            E_sample = Y_data[idx]
            U_sample = X_data[idx]
            S_sample = S_data[idx]

            U_mag = np.linalg.norm(U_sample, axis=-1)

            S_vm = np.sqrt(
                S_sample[:, :, 0] ** 2 +
                S_sample[:, :, 1] ** 2 -
                S_sample[:, :, 0] * S_sample[:, :, 1] +
                3 * S_sample[:, :, 2] ** 2
            )
            

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            
            im1 = axes[0].imshow(E_sample, origin='upper', cmap='viridis')
            axes[0].set_title(f"{split_name} #{idx}: E (Modulus)")
            axes[0].axis('off')
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            
            im2 = axes[1].imshow(U_mag, origin='upper', cmap='magma')
            axes[1].set_title(f"{split_name} #{idx}: |u| (Disp Mag)")
            axes[1].axis('off')
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            
            im3 = axes[2].imshow(S_vm, origin='upper', cmap='inferno')
            axes[2].set_title(f"{split_name} #{idx}: σ_vm (Von Mises)")
            axes[2].axis('off')
            fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(split_dir / f"vis_sample_stress_{idx}.png")
            plt.close(fig)
    print(f"{split_name}: completed for viz")


# ------------------------------------------------------------------------------
# Main Program Entry Point
# ------------------------------------------------------------------------------
def main():
    print(f"--- Launch Scikit-FEM Generator (Stress + Displacement) ---")
    print(f"trainset: {cfg.N_TRAIN} samples, testset: {cfg.N_TEST} samples, resolutions {cfg.RES}x{cfg.RES}")

    # Precompute the coordinate grid for the entire field for subsequent interpolation.
    x_linspace = np.linspace(0, cfg.L, cfg.RES)
    y_linspace = np.linspace(cfg.L, 0, cfg.RES) # Note the y-axis order to match the image coordinate system.
    X_grid, Y_grid = np.meshgrid(x_linspace, y_linspace)
    full_field_coords = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

    # Generate training set
    generate_dataset(cfg.N_TRAIN, "Training", cfg.TRAIN_DIR,
                     np.random.default_rng(cfg.SEED_TRAIN), full_field_coords)
    
    # Generate the test set
    generate_dataset(cfg.N_TEST, "Testing", cfg.TEST_DIR,
                     np.random.default_rng(cfg.SEED_TEST), full_field_coords)

    print("\nCompleted")


if __name__ == "__main__":
    main()
