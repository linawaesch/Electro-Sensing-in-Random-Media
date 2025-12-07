#!/usr/bin/env python3
"""
Fixed-size dictionaries with different diversity levels.

This script:
  - Uses snapshot_generation_data.npz and inclusions.py.
  - Generates a pool of random shapes and their invariants.
  - From this pool, constructs several dictionaries of the SAME size
    but with different diversity levels in invariant space:
       * very diverse (farthest-point sampling),
       * random (typical),
       * medium similar (cluster),
       * very similar (tight cluster).
  - For each dictionary, runs nearest-neighbour classification vs
    receiver noise (Ammari-style complex Gaussian noise on Q).
  - Computes dictionary diversity metrics (d_min, d_avg) from Section 3.7.
  - Outputs two plots:
       1) accuracy vs receiver noise for each diversity level;
       2) d_min and d_avg vs diversity level.
"""

import json
import math
import os
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from inclusions import random_inclusion_mask


# ============================================================================
# Tunable parameters
# ============================================================================

# Snapshot file (single condition)
SNAPSHOT_NPZ = "snapshot_generation_data.npz"

# Size of each dictionary (same for all diversity levels)
DICT_SIZE = 50

# Number of shapes in the initial pool (must be >= DICT_SIZE)
N_POOL_SHAPES = 1000

# Diversity levels we want to build (labels are used in legends & plots)
DIVERSITY_LEVELS = [
    "very_diverse",
    "random",
    "medium_similar",
    "very_similar",
]

# Number of synthetic dipole sources (illuminations)
N_SOURCES = 3

# Approximate number of pixels per inclusion (controls shape size)
N_PIXELS_PER_INCLUSION = 200

# Noise levels on receiver data (in PERCENT of the fluctuations of Q)
# Noise model: for each frequency and shape,
#   Q̃ = Q + ε_n W,  ε_n = (level/100) * (max|Q| - min|Q|),
#   with W having i.i.d. complex Gaussian entries of variance 1.
NOISE_LEVELS = [0.0, 20.0, 50.0, 100.0, 150.0, 200.0]

# Random seed for reproducibility
RANDOM_SEED = 2025


# ============================================================================
# Utility functions for loading and coercing data
# ============================================================================

def load_snapshot_and_meta(npz_path: str):
    """Load snapshot_generation_data.npz and return (data, meta_dict)."""
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

    Accepts:
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
            "snapshot_generation_data.npz. "
            f"Looked for 'G_true' or 'G'. Available keys: {files}"
        )
    print(f"Using '{name}' as G_true with shape {G_true_raw.shape}")
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
    """Reconstruct the 2D grid from the flat list of grid_points."""
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
    Build a simple 2x2 geometric moment tensor from the inclusion mask,
    using second moments around the centroid.
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
    """Build a small set of synthetic dipole moments."""
    S = int(num_sources)
    angles = np.linspace(0.0, 2.0 * np.pi, S, endpoint=False)
    B = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (S,2)
    return B


def compute_receiver_gradients(G_stack: np.ndarray,
                               k_idx: int,
                               meta) -> List[np.ndarray]:
    """
    For a given target index, compute receiver-side gradients
        C_f(r,:) ≈ ∇_z G_b(y_r, z; ω_f)
    using central finite differences on G_stack (shape (Nf*Nr, K)).
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

    return C_list


def simulate_Q_list(B: np.ndarray,
                    Ms_true: List[np.ndarray],
                    C_list: List[np.ndarray]) -> List[np.ndarray]:
    """For each frequency, build Q_f = B @ M_f @ C_f.T."""
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
    Reconstruct a 2x2 GPT M_f from Q_f via least squares:

        Q_sr = b_s^T M c_r
             = M_11 (b1 c1) + M_12 (b1 c2) + M_21 (b2 c1) + M_22 (b2 c2).
    """
    S_sources, R = Q_f.shape
    q_vec = Q_f.reshape(S_sources * R)

    A = np.zeros((S_sources * R, 4), dtype=np.complex128)
    row = 0
    for s in range(S_sources):
        b1, b2 = B[s]
        for r in range(R):
            c1, c2 = C_f[r]
            A[row, 0] = b1 * c1
            A[row, 1] = b1 * c2
            A[row, 2] = b2 * c1
            A[row, 3] = b2 * c2
            row += 1

    m_hat, residuals, rank, svals = np.linalg.lstsq(A, q_vec, rcond=None)
    M_hat = m_hat.reshape((2, 2))
    M_hat = 0.5 * (M_hat + M_hat.T)  # enforce symmetry
    return M_hat, residuals, rank, svals


# ============================================================================
# GPT invariants & dictionary diversity
# ============================================================================

def compute_spectral_invariants(M_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute singular values and scale-free ratios for each frequency."""
    Nf = len(M_list)
    singvals = np.zeros((Nf, 2))
    for f_idx, M in enumerate(M_list):
        _, s, _ = np.linalg.svd(M)
        s = np.sort(s)[::-1]
        singvals[f_idx, :] = s
    ratios = singvals / singvals[-1, :][None, :]
    return singvals, ratios


def flatten_singular_values(singvals: np.ndarray) -> np.ndarray:
    """Flatten singular values into a 1D real feature vector."""
    return np.real(singvals).ravel()


def compute_dictionary_diversity(q_list: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compute diversity metrics from Section 3.7:

      - d_min: minimal pairwise Euclidean distance between invariant vectors;
      - d_avg: sqrt of average squared pairwise distance.
    """
    N = len(q_list)
    if N < 2:
        return 0.0, 0.0

    Q = np.vstack(q_list)
    d2_list = []

    for n in range(N):
        diff = Q[n+1:, :] - Q[n, :][None, :]
        if diff.size == 0:
            continue
        d2 = np.sum(diff * np.conjugate(diff), axis=1).real
        d2_list.append(d2)

    if not d2_list:
        return 0.0, 0.0

    d2_all = np.concatenate(d2_list)
    d_min = float(np.sqrt(np.min(d2_all)))
    d2_mean = float(np.mean(d2_all))
    d_avg = float(np.sqrt(d2_mean))
    return d_min, d_avg


def compute_distance_matrix(q_list: List[np.ndarray]) -> np.ndarray:
    """Compute full pairwise distance matrix between invariant vectors."""
    Q = np.vstack(q_list)  # (N, d)
    diff = Q[:, None, :] - Q[None, :, :]
    D2 = np.sum(diff * np.conjugate(diff), axis=-1).real
    return np.sqrt(D2)


# ============================================================================
# Shape generation (pool) and invariant bank
# ============================================================================

def build_shape_bank(
    X: np.ndarray,
    Z: np.ndarray,
    meta,
    freqs: np.ndarray,
    B: np.ndarray,
    C_true_list: List[np.ndarray],
    max_shapes: int,
    n_pixels_per_inclusion: int,
    random_seed: int,
):
    """
    Generate a pool of 'max_shapes' random inclusions and, for each shape:

      - compute geometric moment tensor,
      - build frequency-dependent GPTs,
      - simulate noise-free data,
      - compute singular values of GPTs.

    Returns:
      Ms_true_all    : list length max_shapes, each is list length Nf of 2x2 matrices
      Q_true_all     : list length max_shapes, each is list length Nf of (SxR) matrices
      singvals_all   : list length max_shapes, each is (Nf,2) array of singular values
    """
    rng = np.random.default_rng(random_seed)
    Ms_true_all: List[List[np.ndarray]] = []
    Q_true_all: List[List[np.ndarray]] = []
    singvals_all: List[np.ndarray] = []

    print(f"Generating a pool of {max_shapes} shapes...")
    for j in range(max_shapes):
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
        Q_true = simulate_Q_list(B, Ms_true, C_true_list)
        singvals_true, _ = compute_spectral_invariants(Ms_true)

        Ms_true_all.append(Ms_true)
        Q_true_all.append(Q_true)
        singvals_all.append(singvals_true)

        if (j + 1) % max(1, max_shapes // 10) == 0:
            print(f"  Generated {j+1}/{max_shapes} shapes.")

    print("Done generating shape pool.")
    return Ms_true_all, Q_true_all, singvals_all


# ============================================================================
# Building dictionaries with different diversity levels
# ============================================================================

def farthest_point_dictionary(D: np.ndarray,
                              dict_size: int,
                              rng: np.random.Generator) -> List[int]:
    """
    Greedy farthest-point sampling for high diversity.
    D is the distance matrix of shape (N, N).
    """
    N = D.shape[0]
    seed = int(rng.integers(low=0, high=N))
    selected = [seed]
    candidates = set(range(N)) - {seed}

    while len(selected) < dict_size and candidates:
        # For each candidate, find distance to closest selected point
        d_min_to_sel = {}
        for j in candidates:
            dmin = np.min(D[j, selected])
            d_min_to_sel[j] = dmin
        # pick candidate with largest such distance
        next_idx = max(d_min_to_sel, key=d_min_to_sel.get)
        selected.append(next_idx)
        candidates.remove(next_idx)

    return selected


def clustered_dictionary(D: np.ndarray,
                         dict_size: int,
                         rng: np.random.Generator,
                         radius_factor: float = 1.0) -> List[int]:
    """
    Build a clustered dictionary around a random center.

    radius_factor controls how tight the cluster is:
      - radius_factor=1.0 : pick closest 'dict_size' neighbours (very similar)
      - radius_factor>1.0 : pick from a larger neighbourhood and subsample.
    """
    N = D.shape[0]
    center = int(rng.integers(low=0, high=N))
    distances = D[center, :]

    # sort all points by distance from the center
    order = np.argsort(distances)
    # at least dict_size + some margin depending on radius_factor
    k_max = min(N, int(radius_factor * dict_size))
    if k_max < dict_size:
        k_max = dict_size
    candidates = order[:k_max]

    # always include the center
    if center not in candidates:
        candidates = np.concatenate(([center], candidates[:-1]))
    # randomly pick dict_size indices from candidates
    rng.shuffle(candidates)
    selected = list(candidates[:dict_size])
    return selected


def random_dictionary(N_pool: int,
                      dict_size: int,
                      rng: np.random.Generator) -> List[int]:
    """Random subset of given size."""
    return list(rng.choice(N_pool, size=dict_size, replace=False))


def build_diversity_dictionaries(D: np.ndarray,
                                 dict_size: int,
                                 rng: np.random.Generator) -> Dict[str, List[int]]:
    """
    Build dictionaries (index sets) for each diversity level.

    D: distance matrix over the entire pool.
    """
    N_pool = D.shape[0]
    dicts: Dict[str, List[int]] = {}

    # Very diverse: farthest-point sampling
    dicts["very_diverse"] = farthest_point_dictionary(D, dict_size, rng)

    # Random: typical diversity
    dicts["random"] = random_dictionary(N_pool, dict_size, rng)

    # Medium similar: cluster with moderate radius factor
    dicts["medium_similar"] = clustered_dictionary(D, dict_size, rng, radius_factor=3.0)

    # Very similar: tight cluster
    dicts["very_similar"] = clustered_dictionary(D, dict_size, rng, radius_factor=1.0)

    return dicts


# ============================================================================
# Classification vs noise
# ============================================================================

def identify_from_invariants(singvals_candidate: np.ndarray,
                             dict_flat: List[np.ndarray]) -> Tuple[int, np.ndarray]:
    """Nearest-neighbour classification in invariant space using singular values."""
    v = flatten_singular_values(singvals_candidate)
    D = np.vstack(dict_flat)
    diffs = D - v[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    best_idx = int(np.argmin(dists))
    return best_idx, dists


def run_experiment():
    # ---------------------------------------------------------------
    # Load snapshot and set up geometry / Green's function
    # ---------------------------------------------------------------
    print("=== Loading snapshot_generation_data ===")
    snapshot_data, meta = load_snapshot_and_meta(SNAPSHOT_NPZ)
    grid_points = snapshot_data["grid_points"]
    receivers = snapshot_data["receivers"]
    freqs = snapshot_data["freqs"]
    G_true_raw = find_true_G(snapshot_data)
    G_true_stack = coerce_G_stack(G_true_raw, meta)

    print(f"Grid: Nx={meta['Nx']}, Nz={meta['Nz']}, Nr={meta['Nr']}, Nf={meta['Nf']}")
    print(f"Number of receivers: {receivers.shape[0]}")
    print(f"Frequencies: {freqs}\n")

    # Rebuild grid and dipoles
    X, Z = build_grid(grid_points, meta)
    B = build_dipole_gradients(N_SOURCES)
    print(f"Using {N_SOURCES} synthetic dipole sources. B =")
    print(B)
    print()

    # Choose interior target index
    Nx_int = int(meta["Nx"])
    Nz_int = int(meta["Nz"])
    iz = Nz_int // 2
    ix = Nx_int // 2
    if iz <= 0:
        iz = 1
    if iz >= Nz_int - 1:
        iz = Nz_int - 2
    if ix <= 0:
        ix = 1
    if ix >= Nx_int - 1:
        ix = Nx_int - 2
    k_center = iz * Nx_int + ix
    print(f"Target grid index: k_center={k_center} -> (ix={ix}, iz={iz})")
    print(f"Physical coordinates z≈({X[iz, ix]:.3f}, {Z[iz, ix]:.3f})\n")

    print("Computing receiver gradients C_true_list for the true G...")
    C_true_list = compute_receiver_gradients(G_true_stack, k_center, meta)
    print("Done.\n")

    # ---------------------------------------------------------------
    # Build shape pool
    # ---------------------------------------------------------------
    if N_POOL_SHAPES < DICT_SIZE:
        raise ValueError("N_POOL_SHAPES must be >= DICT_SIZE.")

    Ms_true_all, Q_true_all, singvals_all = build_shape_bank(
        X,
        Z,
        meta,
        freqs,
        B,
        C_true_list,
        max_shapes=N_POOL_SHAPES,
        n_pixels_per_inclusion=N_PIXELS_PER_INCLUSION,
        random_seed=RANDOM_SEED,
    )

    # Precompute feature vectors and distance matrix over pool
    q_all = [flatten_singular_values(sv) for sv in singvals_all]
    print("Computing pairwise distance matrix in invariant space...")
    D_pool = compute_distance_matrix(q_all)
    print("Done.\n")

    rng = np.random.default_rng(RANDOM_SEED + 123)

    # ---------------------------------------------------------------
    # Build dictionaries for different diversity levels
    # ---------------------------------------------------------------
    dict_indices_by_level = build_diversity_dictionaries(D_pool, DICT_SIZE, rng)

    # Compute diversity metrics
    d_min_level: Dict[str, float] = {}
    d_avg_level: Dict[str, float] = {}
    for level, idx_list in dict_indices_by_level.items():
        q_dict = [q_all[j] for j in idx_list]
        d_min, d_avg = compute_dictionary_diversity(q_dict)
        d_min_level[level] = d_min
        d_avg_level[level] = d_avg
        print(
            f"Dictionary '{level}': size={len(idx_list)}, "
            f"d_min={d_min:.3e}, d_avg={d_avg:.3e}"
        )
    print()

    # ---------------------------------------------------------------
    # Classification vs receiver noise for each diversity level
    # ---------------------------------------------------------------
    noise_levels = np.asarray(NOISE_LEVELS, dtype=float)
    Nf = len(freqs)
    accuracies: Dict[str, np.ndarray] = {
        level: np.zeros_like(noise_levels, dtype=float) for level in DIVERSITY_LEVELS
    }

    print("=== Running classification vs receiver noise for different diversity levels ===")
    print("Noise model: Q̃ = Q + ε_n W,  ε_n = (level/100) * (max|Q| - min|Q|), "
          "W ~ CN(0, I).\n")

    for level in DIVERSITY_LEVELS:
        idx_list = dict_indices_by_level[level]
        dict_q = [q_all[j] for j in idx_list]
        dict_flat = dict_q  # already flattened

        print(f"--- Diversity level '{level}' ---")
        for i, noise_level in enumerate(noise_levels):
            correct = 0
            for j_local, j_global in enumerate(idx_list):
                Q_true = Q_true_all[j_global]

                # Add noise per frequency
                Q_noisy_list: List[np.ndarray] = []
                for f_idx in range(Nf):
                    Q0 = Q_true[f_idx]
                    mag_Q = np.abs(Q0)
                    fluct = mag_Q.max() - mag_Q.min()
                    if fluct == 0:
                        fluct = 1.0
                    epsilon_n = (noise_level / 100.0) * fluct

                    noise_real = rng.normal(scale=1.0 / math.sqrt(2.0), size=Q0.shape)
                    noise_imag = rng.normal(scale=1.0 / math.sqrt(2.0), size=Q0.shape)
                    W = noise_real + 1j * noise_imag

                    Q_noisy = Q0 + epsilon_n * W
                    Q_noisy_list.append(Q_noisy)

                # Reconstruct GPTs
                Mhat_list: List[np.ndarray] = []
                for f_idx in range(Nf):
                    M_hat_f, _, _, _ = reconstruct_M_from_data(
                        B, C_true_list[f_idx], Q_noisy_list[f_idx]
                    )
                    Mhat_list.append(M_hat_f)

                sv_hat, _ = compute_spectral_invariants(Mhat_list)
                idx_hat_local, _ = identify_from_invariants(sv_hat, dict_flat)
                if idx_hat_local == j_local:
                    correct += 1

            acc = correct / float(DICT_SIZE)
            accuracies[level][i] = acc
            print(
                f"  Noise level {noise_level:.1f}%: "
                f"accuracy={acc*100:.1f}% ({correct}/{DICT_SIZE})"
            )
        print()

    # ---------------------------------------------------------------
    # Plot 1: accuracy vs noise for each diversity level
    # ---------------------------------------------------------------
    plt.figure()
    for level in DIVERSITY_LEVELS:
        plt.plot(
            noise_levels,
            accuracies[level] * 100.0,
            "-o",
            label=level.replace("_", " "),
        )
    plt.xlabel("Receiver noise level (% of fluctuations of Q)")
    plt.ylabel("Correctly identified shapes (%)")
    plt.title("Shape classification vs receiver noise for different dictionary diversities")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png1 = "classification_vs_noise_diversity_levels.png"
    plt.savefig(out_png1, dpi=150)
    print(f"Saved plot to '{out_png1}'.")

        # ---------------------------------------------------------------
    # Plot 2: diversity metrics vs diversity level (only d_avg)
    # ---------------------------------------------------------------
    labels = [lvl.replace("_", " ") for lvl in DIVERSITY_LEVELS]
    x = np.arange(len(labels))
    width = 0.6

    d_avg_vals = [d_avg_level[lvl] for lvl in DIVERSITY_LEVELS]

    plt.figure()
    plt.bar(x, d_avg_vals, width, label=r"$d_{\mathrm{avg}}$")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Dictionary diversity (invariant-space distance)")
    plt.title("Dictionary diversity (average spread) for different diversity levels")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    out_png2 = "dictionary_diversity_by_level.png"
    plt.savefig(out_png2, dpi=150)
    print(f"Saved plot to '{out_png2}'.")


    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    run_experiment()
