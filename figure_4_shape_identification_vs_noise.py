#!/usr/bin/env python3
"""
Shape identification vs noise level in a random medium.

Compared to the previous version, this script is modified to be
significantly more robust to noise by:

  - Using *scale-dependent* invariants (all singular values of M_f(ω))
    across frequencies, instead of scale-free ratios τ_j^(f)/τ_j^(F),
    which are much less stable under noise.
  - Using multiple illuminations (n_sources ≥ 3) so the least-squares
    GPT inversion is better conditioned.
  - Adding noise in the same way as Ammari et al.: for each frequency,
    Q̃ = Q + ε_n W, where W has i.i.d. CN(0,1) entries and
    ε_n = (level/100) · (max |Q| − min |Q|), i.e. noise level is given
    in percent of the fluctuations of Q.

This version computes the first-order polarization tensor M(λ(ω),B) according to
Ammari et al. (PNAS 2014) using a boundary integral discretization of the
Neumann–Poincaré adjoint operator K_B^* on ∂B:

  (λ I - K_B^*) φ_α = ν_α,    α=1,2
  M_{αβ}(λ,B) = ∫_{∂B} φ_α(y) y_β dσ(y)

The boundary curve is extracted from the binary inclusion mask via marching squares.
"""

import json
import math
import os
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import find_contours

from inclusions import random_inclusion_mask


# -------------------------------------------------------------------
# Green's matrix loading / coercion
# -------------------------------------------------------------------

def load_snapshot_and_meta(npz_path: str = "snapshot_generation_data.npz"):
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


def find_true_and_homogeneous_G(snapshot_data, meta):
    """
    Extract the "true" random-medium Green's matrix and the homogeneous one.

    Accepts multiple possible key names for robustness:
      - G_true or G        : true / random medium
      - g_homogenous or G0 : homogeneous medium
    """
    files = snapshot_data.files

    # True / random-medium G
    if "G_true" in files:
        G_true_raw = snapshot_data["G_true"]
        true_name = "G_true"
    elif "G" in files:
        G_true_raw = snapshot_data["G"]
        true_name = "G"
    else:
        raise KeyError(
            "Could not find a true/random-medium Green's matrix in "
            "snapshot_generation_data.npz. "
            f"Looked for 'G_true' or 'G'. Available keys: {files}"
        )

    # Homogeneous G
    if "g_homogenous" in files:
        G_hom_raw = snapshot_data["g_homogenous"]
        hom_name = "g_homogenous"
    elif "G0" in files:
        G_hom_raw = snapshot_data["G0"]
        hom_name = "G0"
    else:
        raise KeyError(
            "Could not find a homogeneous Green's matrix in "
            "snapshot_generation_data.npz. "
            f"Looked for 'g_homogenous' or 'G0'. Available keys: {files}"
        )

    print(f"Using '{true_name}' as G_true with shape {G_true_raw.shape}")
    print(f"Using '{hom_name}' as G_hom with shape {G_hom_raw.shape}")
    return G_true_raw, G_hom_raw


def load_recovered_G(baseline_npz: str = "order_out.npz"):
    """
    Load the recovered Green's matrix from order_out_baseline.npz.

    We prefer 'G_ordered', whose columns are aligned with grid_points.
    """
    if not os.path.exists(baseline_npz):
        raise FileNotFoundError(
            f"Could not find '{baseline_npz}' in current directory. "
            "Put it next to this script or adjust the path."
        )
    data = np.load(baseline_npz, allow_pickle=True)
    files = data.files
    candidates = ["G_ordered", "G_hat", "Ghat", "G_recovered"]
    for key in candidates:
        if key in files:
            print(f"Using '{key}' as recovered G with shape {data[key].shape}")
            return data[key]
    raise KeyError(
        "Could not find a recovered Green's matrix in order_out_baseline.npz. "
        f"Tried keys {candidates}. Available keys: {files}"
    )


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


# -------------------------------------------------------------------
# Grid, inclusions, and polarization tensor (Ammari et al.)
# -------------------------------------------------------------------

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


def _bilinear_interp(grid: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on a 2D grid for (row=r, col=c) float coordinates."""
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    r1 = np.clip(r0 + 1, 0, grid.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, grid.shape[1] - 1)
    r0 = np.clip(r0, 0, grid.shape[0] - 1)
    c0 = np.clip(c0, 0, grid.shape[1] - 1)

    fr = r - r0
    fc = c - c0

    v00 = grid[r0, c0]
    v01 = grid[r0, c1]
    v10 = grid[r1, c0]
    v11 = grid[r1, c1]
    return (1 - fr) * (1 - fc) * v00 + (1 - fr) * fc * v01 + fr * (1 - fc) * v10 + fr * fc * v11


def _resample_closed_curve(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline to n points approximately uniformly in arc length."""
    if points.shape[0] < 3:
        raise ValueError("Boundary extraction produced too few points.")
    pts = np.asarray(points, dtype=float)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0:
        raise ValueError("Degenerate boundary curve (zero length).")
    t = np.linspace(0.0, total, n + 1)[:-1]
    x = np.interp(t, s, pts[:, 0])
    z = np.interp(t, s, pts[:, 1])
    return np.stack([x, z], axis=1)


def extract_boundary_geometry(
    X: np.ndarray,
    Z: np.ndarray,
    mask: np.ndarray,
    n_boundary: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a single closed boundary curve from a binary mask and return
    (points, normals, weights, centroid).

    points   : (N,2) boundary coordinates centered at the inclusion centroid
    normals  : (N,2) outward unit normals at boundary points
    weights  : (N,)  quadrature weights approximating d\\sigma
    centroid : (2,)  centroid in physical coordinates
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.sum() < 10:
        raise ValueError("Inclusion mask is too small to extract a stable boundary.")

    # Centroid from area pixels (robust on a grid)
    xc = float(np.mean(X[mask]))
    zc = float(np.mean(Z[mask]))
    centroid = np.array([xc, zc], dtype=float)

    # Extract boundary in pixel coordinates (row, col)
    contours = find_contours(mask.astype(float), level=0.5)
    if not contours:
        raise ValueError("Could not extract a boundary contour from the mask.")
    contour = max(contours, key=lambda a: a.shape[0])  # (m,2) with (row, col)
    rr = contour[:, 0]
    cc = contour[:, 1]

    # Map to physical coordinates via bilinear interpolation on X and Z
    x_phys = _bilinear_interp(X, rr, cc)
    z_phys = _bilinear_interp(Z, rr, cc)
    pts = np.stack([x_phys, z_phys], axis=1)

    # Resample and center
    pts = _resample_closed_curve(pts, n_boundary)
    pts = pts - centroid[None, :]

    # Tangents and orientation
    pts_fwd = np.roll(pts, -1, axis=0)
    pts_bwd = np.roll(pts, 1, axis=0)
    tvec = pts_fwd - pts_bwd
    tnorm = np.linalg.norm(tvec, axis=1, keepdims=True)
    tnorm[tnorm == 0] = 1.0
    tvec = tvec / tnorm

    # Signed area to determine CCW/CW orientation
    area2 = float(np.sum(pts[:, 0] * pts_fwd[:, 1] - pts_fwd[:, 0] * pts[:, 1]))
    ccw = area2 > 0

    # Outward normal: rotate tangent by -90 for CCW, +90 for CW
    if ccw:
        nvec = np.stack([tvec[:, 1], -tvec[:, 0]], axis=1)
    else:
        nvec = np.stack([-tvec[:, 1], tvec[:, 0]], axis=1)
    nnorm = np.linalg.norm(nvec, axis=1, keepdims=True)
    nnorm[nnorm == 0] = 1.0
    nvec = nvec / nnorm

    # Quadrature weights based on segment lengths
    seg = np.linalg.norm(pts_fwd - pts, axis=1)
    weights = seg.astype(float)
    return pts, nvec, weights, centroid


def build_Kstar(points: np.ndarray, normals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Build a Nyström discretization of the Neumann–Poincaré adjoint operator K^* for
    the 2D Laplace kernel on ∂B:

        (K^*φ)(x) = p.v. (1/(2π)) ∫_{∂B} ((x-y)·ν(x))/|x-y|^2 φ(y) dσ(y).

    The discretization uses observation normals ν(x_i) and quadrature weights weights[j].
    """
    pts = np.asarray(points, dtype=float)
    nu = np.asarray(normals, dtype=float)
    w = np.asarray(weights, dtype=float)
    N = pts.shape[0]

    # Pairwise differences: diff[i,j,:] = x_i - x_j
    diff = pts[:, None, :] - pts[None, :, :]  # (N,N,2)
    r2 = np.sum(diff * diff, axis=2)  # (N,N)
    np.fill_diagonal(r2, np.inf)

    # dot with observation normal nu_i
    dot = diff[:, :, 0] * nu[:, None, 0] + diff[:, :, 1] * nu[:, None, 1]
    kernel = (1.0 / (2.0 * math.pi)) * (dot / r2)  # (N,N)
    K = kernel * w[None, :]
    np.fill_diagonal(K, 0.0)
    return K.astype(np.complex128)


def polarization_tensor_from_mask(
    X: np.ndarray,
    Z: np.ndarray,
    mask: np.ndarray,
    freqs: np.ndarray,
    sigma: float = 2.0,
    sigma0: float = 1.0,
    epsilon: float = 1.0,
    n_boundary: int = 200,
) -> List[np.ndarray]:
    """
    Compute the first-order polarization tensor M(λ(ω),B) as in Ammari et al.
    for a 2D inclusion described by a binary mask.

    For each frequency f (Hz), set ω=2π f and k(ω)=σ + i ε ω, then
    λ(ω) = (k(ω)+σ0) / (2 (k(ω)-σ0)).

    Using a boundary discretization and the Neumann–Poincaré adjoint K^*, solve
        (λ I - K^*) φ_α = ν_α,  α=1,2,
    and compute
        M_{αβ}(λ,B) = ∫_{∂B} φ_α(y) y_β dσ(y).

    Notes:
      - The boundary points are centered at the inclusion centroid, so translation is removed.
      - The inclusion size is taken from the physical grid coordinates, so the correct scaling is implicit.
    """
    pts, normals, weights, _ = extract_boundary_geometry(X, Z, mask, n_boundary=n_boundary)
    K = build_Kstar(pts, normals, weights)
    N = pts.shape[0]

    Ms: List[np.ndarray] = []
    I = np.eye(N, dtype=np.complex128)
    nu_x = normals[:, 0].astype(np.complex128)
    nu_z = normals[:, 1].astype(np.complex128)

    for f in freqs:
        omega = 2.0 * math.pi * float(f)
        k = sigma + 1j * epsilon * omega
        lam = (k + sigma0) / (2.0 * (k - sigma0))

        A = lam * I - K
        phi_x = np.linalg.solve(A, nu_x)
        phi_z = np.linalg.solve(A, nu_z)

        # Quadrature for M_{alpha,beta}
        w = weights.astype(np.complex128)
        yx = pts[:, 0].astype(np.complex128)
        yz = pts[:, 1].astype(np.complex128)

        M11 = np.sum(phi_x * yx * w)
        M12 = np.sum(phi_x * yz * w)
        M21 = np.sum(phi_z * yx * w)
        M22 = np.sum(phi_z * yz * w)
        M = np.array([[M11, M12], [M21, M22]], dtype=np.complex128)

        # Ammari's first-order PT is symmetric; enforce symmetry numerically
        M = 0.5 * (M + M.T)
        Ms.append(M)

    return Ms


# -------------------------------------------------------------------
# Dipoles, receiver gradients, forward model, GPT inversion
# -------------------------------------------------------------------

def build_dipole_gradients(num_sources: int = 3) -> np.ndarray:
    """
    Build a small set of synthetic dipole moments.

    Each row of B is a unit dipole direction in 2D.
    """
    S = int(num_sources)
    angles = np.linspace(0.0, 2.0 * np.pi, S, endpoint=False)
    B = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (S,2)
    return B


def compute_receiver_gradients(G_raw: np.ndarray,
                               k_idx: int,
                               meta) -> List[np.ndarray]:
    """
    For a given target grid index k_idx, compute the receiver-side gradients

        C_f(r,:) ≈ ∇_z G_b(y_r, z; ω_f)   (r=1..Nr, f=1..Nf)

    using central finite differences on the provided Green's matrix G_raw.

    G_raw is coerced to shape (Nf*Nr, K), with rows grouped by frequency.
    """
    Nx = int(meta["Nx"])
    Nz = int(meta["Nz"])
    Nf = int(meta["Nf"])
    Nr = int(meta["Nr"])
    dx = float(meta["dx"])
    dz = float(meta["dz"])

    G_stack = coerce_G_stack(G_raw, meta)  # (Nf*Nr, K)

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


# -------------------------------------------------------------------
# GPT invariants & shape dictionary
# -------------------------------------------------------------------

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

    Using *all singular values* across frequencies (scale-dependent) is much
    more stable than using ratios of singular values.
    """
    return np.real(singvals).ravel()


def build_shape_dictionary(
    X: np.ndarray,
    Z: np.ndarray,
    meta,
    freqs: np.ndarray,
    B: np.ndarray,
    C_true_list: List[np.ndarray],
    n_shapes: int = 20,
    n_pixels_per_inclusion: int = 80,
    random_seed: int = 12345,
):
    """
    Generate a dictionary of 'n_shapes' random inclusions (all at the same
    physical target position, modeled via C_true_list), and for each shape:

      - compute its first-order polarization tensors M(λ(ω_f),B)
        from a boundary integral discretization (Ammari et al.),
      - simulate noise-free data Q_true_f with G_true,
      - compute ground-truth invariants (singular values) from M_true_f.

    Returns:
      Ms_true_shapes   : list of length n_shapes, each is list of length Nf of 2x2 matrices
      Q_true_shapes    : list of length n_shapes, each is list of length Nf of (SxR) matrices
      invariants_dict  : list of length n_shapes, each is (Nf,2) array of singular values
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

        # Compute Ammari-style first-order polarization tensors from the mask
        Ms_true = polarization_tensor_from_mask(
            X,
            Z,
            mask,
            freqs,
            sigma=2.0,
            sigma0=1.0,
            epsilon=1.0,
            n_boundary=200,
        )

        # simulate noise-free data using C_true_list (true medium at target location)
        Q_true = simulate_Q_list(B, Ms_true, C_true_list)

        # invariants directly from the true GPTs (dictionary = precomputed shapes)
        singvals_true, _ = compute_spectral_invariants(Ms_true)

        Ms_true_shapes.append(Ms_true)
        Q_true_shapes.append(Q_true)
        invariants_dict.append(singvals_true)

        print(
            f"  Shape {j+1}/{n_shapes}: mask_pixels={int(mask.sum())}, "
            f"centroid≈({center_xy[0]:.3f},{center_xy[1]:.3f}), "
            f"a_equiv≈{a_equiv:.3f}"
        )

    return Ms_true_shapes, Q_true_shapes, invariants_dict


# -------------------------------------------------------------------
# Shape identification experiment
# -------------------------------------------------------------------

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


def run_shape_identification_experiment(
    snapshot_npz: str = "snapshot_generation_data.npz",
    baseline_npz: str = "order_out.npz",
    n_shapes: int = 100,
    n_sources: int = 3,
    n_pixels_per_inclusion: int = 200,
    noise_levels=None,
    random_seed: int = 2025,
):
    """
    Full experiment:
      - builds a shape dictionary (ground truth invariants from G_true),
      - for each noise level and each G version (true, recovered, homogeneous),
        reconstructs GPTs from noisy data and classifies shapes from invariants,
      - plots percentage of correctly identified shapes vs noise level.
    """
    if noise_levels is None:
        # Noise levels in PERCENT of the fluctuations of Q, as in Ammari et al.:
        #   Q̃ = Q + ε_n W, ε_n = (level/100) * (max|Q| - min|Q|)
        noise_levels = np.array([0.0, 20.0, 50.0, 100.0, 150.0, 200.0])

    print("=== Loading snapshot_generation_data ===")
    snapshot_data, meta = load_snapshot_and_meta(snapshot_npz)
    grid_points = snapshot_data["grid_points"]
    receivers = snapshot_data["receivers"]
    freqs = snapshot_data["freqs"]
    G_true_raw, G_hom_raw = find_true_and_homogeneous_G(snapshot_data, meta)

    print(f"Grid: Nx={meta['Nx']}, Nz={meta['Nz']}, Nr={meta['Nr']}, Nf={meta['Nf']}")
    print(f"Number of receivers: {receivers.shape[0]}")
    print(f"Frequencies: {freqs}")
    print()

    print("=== Loading recovered G ===")
    G_rec_raw = load_recovered_G(baseline_npz)
    print()

    # Coerce all G matrices to a consistent 2D (Nf*Nr,K) shape
    G_true_stack = coerce_G_stack(G_true_raw, meta)
    G_hom_stack  = coerce_G_stack(G_hom_raw, meta)
    G_rec_stack  = coerce_G_stack(G_rec_raw, meta)

    # Rebuild grid
    X, Z = build_grid(grid_points, meta)

    # Build dipole sources
    B = build_dipole_gradients(n_sources)
    print(f"Using {n_sources} synthetic dipole sources. B =")
    print(B)
    print()

    # Pick a fixed interior target position (same for all shapes)
    Nx = int(meta["Nx"])
    Nz = int(meta["Nz"])
    k_center = (Nz // 2) * Nx + (Nx // 2)
    iz, ix = divmod(k_center, Nx)
    print(f"Target grid index: k_center={k_center} -> (ix={ix}, iz={iz})")
    print(f"Physical coordinates z≈({X[iz, ix]:.3f}, {Z[iz, ix]:.3f})")
    print()

    print("Computing receiver gradients (C_f) for each Green's matrix...")
    C_true_list = compute_receiver_gradients(G_true_stack, k_center, meta)
    C_rec_list  = compute_receiver_gradients(G_rec_stack,  k_center, meta)
    C_hom_list  = compute_receiver_gradients(G_hom_stack,  k_center, meta)
    print("Done.\n")

    # Build dictionary of shapes and their ground-truth invariants (using G_true).
    print("=== Building shape dictionary (ground truth invariants using G_true) ===")
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
    n_shapes = len(Ms_true_shapes)
    print(f"Built dictionary with {n_shapes} shapes.\n")

    # Flatten invariants for nearest-neighbor classification
    dict_flat = [flatten_singular_values(sv) for sv in invariants_dict]
    inv_dim = dict_flat[0].shape[0]
    print(f"Invariant feature dimension (singular values across freqs): {inv_dim}")
    print()

    # Accuracy arrays
    num_noise = len(noise_levels)
    acc_true = np.zeros(num_noise)
    acc_rec  = np.zeros(num_noise)
    acc_hom  = np.zeros(num_noise)

    rng = np.random.default_rng(random_seed + 999)

    print("=== Running classification vs noise level ===")
    print("Noise levels are given in PERCENT of the fluctuations of Q,")
    print("i.e. for each shape and frequency we set")
    print("    Q̃ = Q + ε_n W,   ε_n = (level/100) * (max|Q| - min|Q|),")
    print("with W having i.i.d. complex Gaussian entries of variance 1.")
    print(f"Noise levels (% of fluctuations of Q): {noise_levels}\n")

    Nf = len(freqs)

    for i, noise_level in enumerate(noise_levels):
        correct_true = 0
        correct_rec  = 0
        correct_hom  = 0

        for j in range(n_shapes):
            Q_true = Q_true_shapes[j]

            # Build noisy data Q_noisy_f for each frequency
            Q_noisy_list: List[np.ndarray] = []
            for f_idx in range(Nf):
                Q0 = Q_true[f_idx]

                # Add complex Gaussian noise as in Ammari et al.:
                #  Q̃ = Q + ε_n W,  W_ij ~ CN(0,1)
                mag_Q = np.abs(Q0)
                fluct = mag_Q.max() - mag_Q.min()
                if fluct == 0:
                    fluct = 1.0  # avoid division by zero for degenerate data
                epsilon_n = (noise_level / 100.0) * fluct

                noise_real = rng.normal(scale=1.0 / math.sqrt(2.0), size=Q0.shape)
                noise_imag = rng.normal(scale=1.0 / math.sqrt(2.0), size=Q0.shape)
                W = noise_real + 1j * noise_imag

                Q_noisy = Q0 + epsilon_n * W
                Q_noisy_list.append(Q_noisy)

            # --- Reconstruct & classify using each Green's version ---

            # 1) G_true
            Mhat_true_list: List[np.ndarray] = []
            for f_idx in range(Nf):
                M_hat_f, _, _, _ = reconstruct_M_from_data(
                    B, C_true_list[f_idx], Q_noisy_list[f_idx]
                )
                Mhat_true_list.append(M_hat_f)
            sv_true_hat, _ = compute_spectral_invariants(Mhat_true_list)
            idx_true, _ = identify_from_invariants(sv_true_hat, dict_flat)
            if idx_true == j:
                correct_true += 1

            # 2) G_rec
            Mhat_rec_list: List[np.ndarray] = []
            for f_idx in range(Nf):
                M_hat_f, _, _, _ = reconstruct_M_from_data(
                    B, C_rec_list[f_idx], Q_noisy_list[f_idx]
                )
                Mhat_rec_list.append(M_hat_f)
            sv_rec_hat, _ = compute_spectral_invariants(Mhat_rec_list)
            idx_rec, _ = identify_from_invariants(sv_rec_hat, dict_flat)
            if idx_rec == j:
                correct_rec += 1

            # 3) G_hom
            Mhat_hom_list: List[np.ndarray] = []
            for f_idx in range(Nf):
                M_hat_f, _, _, _ = reconstruct_M_from_data(
                    B, C_hom_list[f_idx], Q_noisy_list[f_idx]
                )
                Mhat_hom_list.append(M_hat_f)
            sv_hom_hat, _ = compute_spectral_invariants(Mhat_hom_list)
            idx_hom, _ = identify_from_invariants(sv_hom_hat, dict_flat)
            if idx_hom == j:
                correct_hom += 1

        acc_true[i] = correct_true / n_shapes
        acc_rec[i]  = correct_rec  / n_shapes
        acc_hom[i]  = correct_hom  / n_shapes

        print(
            f"Noise level {noise_level:.1f}%: "
            f"acc(G_true)={acc_true[i]*100:.1f}%, "
            f"acc(G_rec)={acc_rec[i]*100:.1f}%, "
            f"acc(G_hom)={acc_hom[i]*100:.1f}%"
        )

    print("\n=== Summary ===")
    for i, noise_level in enumerate(noise_levels):
        print(
            f"Noise={noise_level:.1f}% -> "
            f"G_true: {acc_true[i]*100:.1f}% | "
            f"G_rec: {acc_rec[i]*100:.1f}% | "
            f"G_hom: {acc_hom[i]*100:.1f}%"
        )
    print()

    # ----------------------------------------------------------------
    # Plot: percentage of correctly identified shapes vs noise
    # ----------------------------------------------------------------
    plt.figure()
    plt.plot(noise_levels, acc_true * 100.0, "-o", label="Using G_true")
    plt.plot(noise_levels, acc_rec  * 100.0, "-s", label="Using G_rec")
    plt.plot(noise_levels, acc_hom  * 100.0, "-^", label="Using G_homogeneous")

    plt.xlabel("Noise strength (% of fluctuations of Q)")
    plt.ylabel("Correctly identified shapes (%)")
    plt.title("Shape identification vs noise level")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png = "shape_identification_vs_noise.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to '{out_png}'.")
    try:
        plt.show()
    except Exception:
        # In non-interactive environments, just skip showing.
        pass


if __name__ == "__main__":
    # You can tweak these parameters as needed.
    run_shape_identification_experiment(
        snapshot_npz="snapshot_generation_data.npz",
        baseline_npz="order_out.npz",
        n_shapes=40,
        n_sources=3,
        n_pixels_per_inclusion=80,
        # Noise levels in PERCENT of fluctuations of Q:
        noise_levels=[0.0, 50.0,100, 149.0,200, 250,350.0,450, 550],
        random_seed=2025,
    )
