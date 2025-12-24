#!/usr/bin/env python3
"""
Shape classification vs sensing-matrix noise
--------------------------------------------

This script is similar in spirit to shape_identification_vs_noise.py, but:

  - It adds noise to the *sensing matrices* (Green's matrices) instead of
    to the receiver data.
  - It compares three different "conditions", each represented by its own
    snapshot_generation_data_*.npz file (e.g. different frequencies,
    different random media, different initialization choices, etc.).
  - For each condition and each noise level, it measures how often random
    shapes are correctly identified using noisy sensing matrices and
    plots the percentage of correct classifications vs sensing-matrix
    noise level.

Expected input:
  - Three files like snapshot_generation_data_cond1.npz, each with the
    same structure as your existing snapshot_generation_data.npz:
      * grid_points, receivers, freqs, meta_json
      * a Green's matrix named either "G_true" or "G"

All shapes, GPTs, and invariants are built per condition using inclusions.py,
just like in the previous script.
"""

import json
import math
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from inclusions import random_inclusion_mask


# ============================================================================
# Tunable parameters (edit these as needed)
# ============================================================================

# Three different conditions, each with its own snapshot_generation_data_*.npz
SNAPSHOT_FILES = [
    "snapshot_generation_data_cond1.npz",
    "snapshot_generation_data_cond2.npz",
    "snapshot_generation_data_cond3.npz",
]

# Labels for the three conditions (used in the legend)
CONDITION_LABELS = [
    "# frequencies 2",
    "# frequencies 5",
    "# frequencies 10",
]

# Number of random shapes to draw per condition
N_SHAPES = 100

# Number of synthetic dipole sources (illuminations)
N_SOURCES = 3

# Approximate number of pixels per inclusion (controls shape size)
N_PIXELS_PER_INCLUSION = 200

# Noise levels (in PERCENT of the fluctuations of G) to test
# Noise model: for each frequency block G_f,
#   G_noisy = G + eps * W,   eps = (level/100)*(max|G_f| - min|G_f|)
#   W has i.i.d. complex Gaussian entries of variance 1.
NOISE_LEVELS = [0.0,10, 20.0,35, 50.0, 75,100.0, 125,150.0,175, 200.0]

# Random seed for reproducibility
RANDOM_SEED = 2025


# ============================================================================
# Utility functions for loading and coercing data
# ============================================================================

def load_snapshot_and_meta(npz_path: str):
    """Load snapshot_generation_data_*.npz and return (data, meta_dict)."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find '{npz_path}' in current directory.")
    data = np.load(npz_path, allow_pickle=True)
    if "meta_json" not in data.files:
        raise KeyError(
            f"'{npz_path}' does not contain 'meta_json'. "
            f"Available keys: {data.files}"
        )
    meta = json.loads(str(data["meta_json"]))
    return data, meta


def find_true_G(snapshot_data) -> np.ndarray:
    """
    Extract the "true" random-medium Green's matrix.

    Accepts multiple possible key names for robustness:
      - G_true
      - G
    """
    files = snapshot_data.files
    if "G_true" in files:
        G_true_raw = snapshot_data["G_true"]
        name = "G_true"
    elif "G" in files:
        G_true_raw = snapshot_data["G"]
        name = "G"
    else:
        raise KeyError(
            "Could not find a true/random-medium Green's matrix in "
            "snapshot_generation_data_*.npz. "
            f"Looked for 'G_true' or 'G'. Available keys: {files}"
        )
    print(f"  Using '{name}' as G_true with shape {G_true_raw.shape}")
    return G_true_raw


def coerce_G_stack(G_raw: np.ndarray, meta) -> np.ndarray:
    """
    Ensure G_stack has shape (Nf*Nr, K) where K = Nx*Nz.

    Accepts:
      - shape (Nf*Nr, K)
      - shape (Nf, Nr, K)
      - shape (Nf, K, Nr)
    """
    Nx = int(meta["Nx"])
    Nz = int(meta["Nz"])
    Nf = int(meta["Nf"])
    Nr = int(meta["Nr"])
    K = Nx * Nz

    G_raw = np.asarray(G_raw)

    if G_raw.ndim == 2:
        nrows, ncols = G_raw.shape
        if nrows == Nf * Nr and ncols == K:
            return G_raw
        raise ValueError(
            f"2D Green's matrix has shape {G_raw.shape}, "
            f"but expected (Nf*Nr, K)=({Nf*Nr}, {K})."
        )
    elif G_raw.ndim == 3:
        s0, s1, s2 = G_raw.shape
        if s0 == Nf and s1 == Nr and s2 == K:
            return G_raw.reshape(Nf * Nr, K)
        if s0 == Nf and s2 == Nr and s1 == K:
            return G_raw.transpose(0, 2, 1).reshape(Nf * Nr, K)
        raise ValueError(
            f"3D Green's matrix has shape {G_raw.shape}, which is not "
            f"compatible with Nf={Nf}, Nr={Nr}, K={K}."
        )
    else:
        raise ValueError(
            f"Green's matrix has ndim={G_raw.ndim}, expected 2 or 3."
        )


# ============================================================================
# Grid, inclusions, GPT construction
# ============================================================================

def build_grid(grid_points: np.ndarray, meta):
    """
    Reconstruct the 2D grid from the flat list of grid_points.

        X = grid_points[:,0].reshape(Nz, Nx)
        Z = grid_points[:,1].reshape(Nz, Nx)
    """
    Nx = int(meta["Nx"])
    Nz = int(meta["Nz"])
    X = grid_points[:, 0].reshape(Nz, Nx)
    Z = grid_points[:, 1].reshape(Nz, Nx)
    return X, Z


def compute_shape_moment_tensor(X: np.ndarray,
                                Z: np.ndarray,
                                mask: np.ndarray,
                                meta) -> np.ndarray:
    """
    Build a simple 2x2 *geometric* GPT (shape moment tensor) from the
    inclusion mask, using second moments around the centroid:

        S_11 = ∫ (x - x_c)^2 dA
        S_22 = ∫ (z - z_c)^2 dA
        S_12 = S_21 = ∫ (x - x_c)(z - z_c) dA
    """
    dx = float(meta["dx"])
    dz = float(meta["dz"])

    x = X[mask]
    z = Z[mask]
    xc = x.mean()
    zc = z.mean()

    xr = x - xc
    zr = z - zc

    area_el = dx * dz
    m_xx = np.sum(xr * xr) * area_el
    m_xz = np.sum(xr * zr) * area_el
    m_zz = np.sum(zr * zr) * area_el

    S = np.array([[m_xx, m_xz],
                  [m_xz, m_zz]], dtype=np.complex128)
    return S


def build_frequency_dependent_GPTs(S_shape: np.ndarray,
                                   freqs: np.ndarray,
                                   sigma: float = 2.0,
                                   sigma0: float = 1.0,
                                   epsilon: float = 1.0) -> List[np.ndarray]:
    """
    Build a list of 2x2 GPTs M_f(ω) from the geometric tensor S_shape.

    Frequency dependence (Ammari-style contrast parameter):

        k(ω)   = σ + i ε ω
        λ(ω)   = (k(ω) + σ0) / (2 (k(ω) - σ0))
        M_f(ω) = λ(ω) * S_shape
    """
    Ms: List[np.ndarray] = []
    for f in freqs:
        omega = 2.0 * math.pi * float(f)
        k = sigma + 1j * epsilon * omega
        lam = (k + sigma0) / (2.0 * (k - sigma0))
        Ms.append(lam * S_shape)
    return Ms


# ============================================================================
# Dipoles, receiver gradients, forward model, GPT inversion
# ============================================================================

def build_dipole_gradients(num_sources: int = 3) -> np.ndarray:
    """
    Build a small set of synthetic dipole moments.

    Each row of B is a unit dipole direction in 2D.
    """
    S = int(num_sources)
    angles = np.linspace(0.0, 2.0 * np.pi, S, endpoint=False)
    B = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (S,2)
    return B


def compute_receiver_gradients(G_stack: np.ndarray,
                               k_idx: int,
                               meta) -> List[np.ndarray]:
    """
    For a given target grid index k_idx, compute the receiver-side gradients

        C_f(r,:) ≈ ∇_z G_b(y_r, z; ω_f)   (r=1..Nr, f=1..Nf)

    using central finite differences on the provided Green's matrix G_stack.

    G_stack is shape (Nf*Nr, K), with rows grouped by frequency.
    """
    Nx = int(meta["Nx"])
    Nz = int(meta["Nz"])
    Nf = int(meta["Nf"])
    Nr = int(meta["Nr"])
    dx = float(meta["dx"])
    dz = float(meta["dz"])

    iz, ix = divmod(int(k_idx), Nx)
    if not (1 <= ix <= Nx - 2 and 1 <= iz <= Nz - 2):
        raise ValueError(
            f"Target grid index {k_idx} -> (iz,ix)=({iz},{ix}) is too "
            "close to the boundary for central differences."
        )

    k_x_plus  = iz * Nx + (ix + 1)
    k_x_minus = iz * Nx + (ix - 1)
    k_z_plus  = (iz + 1) * Nx + ix
    k_z_minus = (iz - 1) * Nx + ix

    C_list: List[np.ndarray] = []
    for f_idx in range(Nf):
        row0 = f_idx * Nr
        rows = slice(row0, row0 + Nr)

        G_xp = G_stack[rows, k_x_plus]
        G_xm = G_stack[rows, k_x_minus]
        G_zp = G_stack[rows, k_z_plus]
        G_zm = G_stack[rows, k_z_minus]

        dG_dx = (G_xp - G_xm) / (2.0 * dx)
        dG_dz = (G_zp - G_zm) / (2.0 * dz)

        C_f = np.stack([dG_dx, dG_dz], axis=1)  # (Nr,2)
        C_list.append(C_f)

    return C_list  # list length Nf, each (Nr,2)


def simulate_Q_list(B: np.ndarray,
                    Ms_true: List[np.ndarray],
                    C_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    For each frequency, build the SxR data matrix

        Q_f = B @ M_true_f @ C_f.T

    where
        B     : (S,2) dipole gradients
        M_f   : (2,2) GPT at that frequency
        C_f   : (Nr,2) receiver gradients
    """
    Nf = len(Ms_true)
    Q_list: List[np.ndarray] = []
    for f_idx in range(Nf):
        M_f = Ms_true[f_idx]
        C_f = C_list[f_idx]
        Q_f = B @ M_f @ C_f.T
        Q_list.append(Q_f)
    return Q_list


def reconstruct_M_from_data(B: np.ndarray,
                            C_f: np.ndarray,
                            Q_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Reconstruct a 2x2 GPT M_f from one frequency block Q_f by least squares.

    B:   (S,2)   source/dipole gradients
    C_f: (R,2)   receiver gradients at frequency f
    Q_f: (S,R)   synthetic data at frequency f

    We build an explicit design matrix A with 4 columns:

        Q_sr = b_s^T M c_r
             = M_11 (b1 c1) + M_12 (b1 c2) + M_21 (b2 c1) + M_22 (b2 c2)

    so that vec(Q) = A m, with m = [M11, M12, M21, M22]^T in row-major order.
    This avoids any Kronecker/reshape order confusion.
    """
    S_sources, R = Q_f.shape

    # Flatten data row-wise to match the loop over (s,r)
    q_vec = Q_f.reshape(S_sources * R)

    # Design matrix A: (S*R, 4)
    A = np.zeros((S_sources * R, 4), dtype=np.complex128)
    row = 0
    for s in range(S_sources):
        b1, b2 = B[s]
        for r in range(R):
            c1, c2 = C_f[r]
            A[row, 0] = b1 * c1  # multiplies M11
            A[row, 1] = b1 * c2  # multiplies M12
            A[row, 2] = b2 * c1  # multiplies M21
            A[row, 3] = b2 * c2  # multiplies M22
            row += 1

    m_hat, residuals, rank, svals = np.linalg.lstsq(A, q_vec, rcond=None)

    # Put back into 2x2, row-major: [[M11, M12], [M21, M22]]
    M_hat = m_hat.reshape((2, 2))

    # The first-order GPT should be symmetric; symmetrize to clean up tiny noise.
    M_hat = 0.5 * (M_hat + M_hat.T)

    return M_hat, residuals, rank, svals


# ============================================================================
# GPT invariants & shape dictionary
# ============================================================================

def compute_spectral_invariants(M_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a list of 2x2 complex matrices M_list (one per frequency), compute:

      - singular values s_j^{(f)} (j=1,2) at each frequency;
      - scale-invariant ratios μ_j^{(f)} = s_j^{(f)} / s_j^{(F)}.

    Returns:
      singvals : (Nf,2) array of singular values
      ratios   : (Nf,2) array of μ_j^{(f)}.
    """
    Nf = len(M_list)
    singvals = np.zeros((Nf, 2))
    for f_idx, M in enumerate(M_list):
        _, s, _ = np.linalg.svd(M)
        s = np.sort(s)[::-1]  # descending
        singvals[f_idx, :] = s

    ratios = singvals / singvals[-1, :][None, :]
    return singvals, ratios


def flatten_singular_values(singvals: np.ndarray) -> np.ndarray:
    """
    Flatten singular values s_j^{(f)} into a 1D real feature vector.

    Using *all singular values* across frequencies (scale-dependent)
    is more robust than using ratios.
    """
    return np.real(singvals).ravel()


def build_shape_dictionary(
    X: np.ndarray,
    Z: np.ndarray,
    meta,
    freqs: np.ndarray,
    B: np.ndarray,
    C_true_list: List[np.ndarray],
    n_shapes: int,
    n_pixels_per_inclusion: int,
    random_seed: int,
):
    """
    Generate a dictionary of 'n_shapes' random inclusions (all at the same
    physical target position, modeled via C_true_list), and for each shape:

      - compute its geometric moment tensor S_shape,
      - build its frequency-dependent GPTs M_true_f,
      - simulate noise-free data Q_true_f with the *true* sensing matrix,
      - compute invariants (singular values) from M_true_f.

    Returns:
      Ms_true_shapes   : list length n_shapes, each is list length Nf of 2x2 matrices
      Q_true_shapes    : list length n_shapes, each is list length Nf of (SxR) matrices
      invariants_dict  : list length n_shapes, each is (Nf,2) array of singular values
    """
    rng = np.random.default_rng(random_seed)
    Nf = len(freqs)

    Ms_true_shapes: List[List[np.ndarray]] = []
    Q_true_shapes: List[List[np.ndarray]] = []
    invariants_dict: List[np.ndarray] = []

    for j in range(n_shapes):
        shape_seed = int(random_seed + j)

        mask, boundary_mask, center_xy, a_equiv = random_inclusion_mask(
            X,
            Z,
            n_pixels=n_pixels_per_inclusion,
            connectivity=4,
            seed=shape_seed,
        )

        S_shape = compute_shape_moment_tensor(X, Z, mask, meta)
        Ms_true = build_frequency_dependent_GPTs(S_shape, freqs)

        # simulate noise-free data using *true* C_list
        Q_true = simulate_Q_list(B, Ms_true, C_true_list)

        # invariants directly from the true GPTs (dictionary)
        singvals_true, _ = compute_spectral_invariants(Ms_true)

        Ms_true_shapes.append(Ms_true)
        Q_true_shapes.append(Q_true)
        invariants_dict.append(singvals_true)

        print(
            f"    Shape {j+1}/{n_shapes}: mask_pixels={int(mask.sum())}, "
            f"centroid≈({center_xy[0]:.3f},{center_xy[1]:.3f}), "
            f"a_equiv≈{a_equiv:.3f}"
        )

    return Ms_true_shapes, Q_true_shapes, invariants_dict


def identify_from_invariants(singvals_candidate: np.ndarray,
                             dict_flat: List[np.ndarray]) -> Tuple[int, np.ndarray]:
    """
    Nearest-neighbor classification in invariant space using singular values.

    singvals_candidate : (Nf,2) array of singular values
    dict_flat          : list of 1D vectors (one per shape in dictionary)

    Returns:
       best_index : index of closest shape in the dictionary
       distances  : array of L2 distances to all dictionary entries
    """
    v = flatten_singular_values(singvals_candidate)
    D = np.vstack(dict_flat)  # (n_shapes, dim)
    diffs = D - v[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    best_idx = int(np.argmin(dists))
    return best_idx, dists


# ============================================================================
# Noise on sensing matrix
# ============================================================================

def add_noise_to_G(G_true_stack: np.ndarray,
                   meta,
                   noise_level_percent: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Add complex Gaussian noise to the sensing matrix G (Green's matrix),
    in a way analogous to the receiver-noise model used earlier.

    For each frequency block (Nr x K) of G_true_stack we do:
        G̃_f = G_f + ε_n W_f,
    where:
      - W_f has i.i.d. complex Gaussian entries with variance 1,
      - ε_n = (level/100) * (max|G_f| - min|G_f|).

    G_true_stack is shape (Nf*Nr, K), and we preserve that shape.
    """
    if noise_level_percent == 0.0:
        # No noise: just return a copy
        return G_true_stack.copy()

    Nx = int(meta["Nx"])
    Nz = int(meta["Nz"])
    Nf = int(meta["Nf"])
    Nr = int(meta["Nr"])
    K = Nx * Nz

    if G_true_stack.shape != (Nf * Nr, K):
        raise ValueError(
            f"G_true_stack has shape {G_true_stack.shape}, expected {(Nf*Nr, K)}."
        )

    G_noisy = np.empty_like(G_true_stack)

    for f_idx in range(Nf):
        row0 = f_idx * Nr
        rows = slice(row0, row0 + Nr)

        G_sub = G_true_stack[rows, :]  # (Nr, K)

        mag = np.abs(G_sub)
        fluct = mag.max() - mag.min()
        if fluct == 0:
            epsilon_n = 0.0
        else:
            epsilon_n = (noise_level_percent / 100.0) * fluct

        if epsilon_n == 0.0:
            G_noisy[rows, :] = G_sub
            continue

        # Complex Gaussian noise with variance 1
        noise_real = rng.normal(scale=1.0 / math.sqrt(2.0), size=G_sub.shape)
        noise_imag = rng.normal(scale=1.0 / math.sqrt(2.0), size=G_sub.shape)
        W = noise_real + 1j * noise_imag

        G_noisy[rows, :] = G_sub + epsilon_n * W

    return G_noisy


# ============================================================================
# Per-condition experiment
# ============================================================================

def run_condition(snapshot_file: str,
                  label: str,
                  noise_levels: List[float],
                  n_shapes: int,
                  n_sources: int,
                  n_pixels_per_inclusion: int,
                  random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the shape-classification-vs-G-noise experiment for a single condition
    (one snapshot_generation_data_*.npz file).

    Returns:
      noise_levels_array : np.array of noise levels
      acc                : np.array of accuracies (same length)
    """
    print(f"\n=== Condition '{label}' ===")
    print(f"  Loading '{snapshot_file}'")
    snapshot_data, meta = load_snapshot_and_meta(snapshot_file)
    grid_points = snapshot_data["grid_points"]
    receivers = snapshot_data["receivers"]
    freqs = snapshot_data["freqs"]
    print(f"  Grid: Nx={meta['Nx']}, Nz={meta['Nz']}, Nr={meta['Nr']}, Nf={meta['Nf']}")
    print(f"  Number of receivers: {receivers.shape[0]}")
    print(f"  Frequencies: {freqs}")

    # True G and its coerced form
    G_true_raw = find_true_G(snapshot_data)
    G_true_stack = coerce_G_stack(G_true_raw, meta)

    # Rebuild grid and set up dipoles
    X, Z = build_grid(grid_points, meta)
    B = build_dipole_gradients(n_sources)
    print(f"  Using {n_sources} synthetic dipole sources. B =")
    print(B)

    # Choose interior target index for central differences
    Nx_int = int(meta["Nx"])
    Nz_int = int(meta["Nz"])
    iz = Nz_int // 2
    ix = Nx_int // 2
    # ensure interior
    if iz <= 0:
        iz = 1
    if iz >= Nz_int - 1:
        iz = Nz_int - 2
    if ix <= 0:
        ix = 1
    if ix >= Nx_int - 1:
        ix = Nx_int - 2
    k_center = iz * Nx_int + ix
    print(f"  Target grid index: k_center={k_center} -> (ix={ix}, iz={iz})")
    print(f"  Physical coordinates z≈({X[iz, ix]:.3f}, {Z[iz, ix]:.3f})")

    # Receiver gradients for the *true* sensing matrix
    print("  Computing receiver gradients C_true_list for the true G...")
    C_true_list = compute_receiver_gradients(G_true_stack, k_center, meta)
    print("  Done.")

    # Build shape dictionary
    print("  Building shape dictionary (ground truth GPTs and data) ...")
    Ms_true_shapes, Q_true_shapes, invariants_dict = build_shape_dictionary(
        X,
        Z,
        meta,
        freqs,
        B,
        C_true_list,
        n_shapes=n_shapes,
        n_pixels_per_inclusion=n_pixels_per_inclusion,
        random_seed=random_seed,
    )
    print("  Dictionary built.")

    # Flatten invariants for classification
    dict_flat = [flatten_singular_values(sv) for sv in invariants_dict]
    inv_dim = dict_flat[0].shape[0]
    print(f"  Invariant feature dimension: {inv_dim}")

    # Accuracy vs noise
    Nf = len(freqs)
    noise_levels_array = np.asarray(noise_levels, dtype=float)
    acc = np.zeros_like(noise_levels_array, dtype=float)

    rng = np.random.default_rng(random_seed + 1000)

    print("  Running classification vs sensing-matrix noise level ...")
    print("  Noise levels are in PERCENT of the fluctuations of G (per frequency).")

    for i, noise_level in enumerate(noise_levels_array):
        correct = 0

        # Build noisy sensing matrix and its receiver gradients
        G_noisy_stack = add_noise_to_G(G_true_stack, meta, noise_level, rng)
        C_noisy_list = compute_receiver_gradients(G_noisy_stack, k_center, meta)

        for j in range(n_shapes):
            Q_true = Q_true_shapes[j]  # list length Nf

            # Reconstruct GPTs using *noisy* receiver gradients
            Mhat_list: List[np.ndarray] = []
            for f_idx in range(Nf):
                M_hat_f, _, _, _ = reconstruct_M_from_data(
                    B, C_noisy_list[f_idx], Q_true[f_idx]
                )
                Mhat_list.append(M_hat_f)

            sv_hat, _ = compute_spectral_invariants(Mhat_list)
            idx_hat, _ = identify_from_invariants(sv_hat, dict_flat)
            if idx_hat == j:
                correct += 1

        acc[i] = correct / float(n_shapes)
        print(
            f"    Noise level {noise_level:.1f}%: "
            f"accuracy={acc[i]*100:.1f}% "
            f"({correct}/{n_shapes})"
        )

    print(f"  Finished condition '{label}'.\n")
    return noise_levels_array, acc


# ============================================================================
# Main: run all conditions and make a plot
# ============================================================================

def main():
    if len(SNAPSHOT_FILES) != len(CONDITION_LABELS):
        raise ValueError(
            "SNAPSHOT_FILES and CONDITION_LABELS must have the same length "
            f"(got {len(SNAPSHOT_FILES)} and {len(CONDITION_LABELS)})."
        )

    all_noise_levels = None
    all_accs = []

    for snapshot_file, label in zip(SNAPSHOT_FILES, CONDITION_LABELS):
        noise_arr, acc = run_condition(
            snapshot_file=snapshot_file,
            label=label,
            noise_levels=NOISE_LEVELS,
            n_shapes=N_SHAPES,
            n_sources=N_SOURCES,
            n_pixels_per_inclusion=N_PIXELS_PER_INCLUSION,
            random_seed=RANDOM_SEED,
        )
        all_accs.append(acc)
        if all_noise_levels is None:
            all_noise_levels = noise_arr
        else:
            # sanity check: noise grids must match
            if not np.allclose(all_noise_levels, noise_arr):
                raise ValueError(
                    "Noise level grids differ between conditions. "
                    "Make sure NOISE_LEVELS is the same for all."
                )

    # Plot
    plt.figure()
    for label, acc in zip(CONDITION_LABELS, all_accs):
        plt.plot(all_noise_levels, acc * 100.0, "-o", label=label)

    plt.xlabel("Sensing-matrix noise (% of fluctuations of G)")
    plt.ylabel("Correctly identified shapes (%)")
    plt.title("Shape classification vs sensing-matrix noise")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png = "shape_identification_vs_G_noise.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to '{out_png}'.")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
