#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave-informed dictionary learning (with homogeneous+random mixture for G)
==========================================================================================

This version first builds the *stacked homogeneous* Green's matrix G0 across all frequencies
(Eq. (2) of the paper), then builds the *stacked random-medium* matrix Gb exactly as before,
and finally uses G_used = G0 + mix_beta * Gb for data-generation: Y = G_used @ X.

Multi-frequency stacking and all dimensions remain exactly as in the paper's Sec. 4.
"""

from dataclasses import dataclass
import json, math
import numpy as np
from typing import Dict, Tuple

# ---------------------- Config ----------------------

@dataclass
class Config:
    # Fundamental normalized constants
    c0: float = 1.0
    lam: float = 1.0
    # Medium / geometry scales (Sec. 4)
    ell_over_lam: float = 100.0
    L_over_ell: float = 100.0
    a_over_ell: float = 48.0
    Nr: int = 100
    # Frequency band (Sec. 4)
    Nf: int = 2
    band_low: float = 0.64
    # Grid size (20x20 as in examples)
    Nx: int = 20 #TUNE
    Nz: int = 20 #TUNE
    # Sparsity & snapshots
    s: int = 5
    M: int = None  # set to ceil(K log K) if None
    # Random medium parameters (Sec. 4)
    sigma_tilde: float = 0.6
    RFF_Q: int = 128
    seed: int = 1
    # NEW: mixture factor beta (small, tunable)
    mix_beta: float = 0.2  # G_used = G0 + mix_beta * Gb
    factor: int=1

    def finalize(self):
        if self.M is None:
            K = self.Nx * self.Nz
            self.M = int(math.ceil(self.factor* K * math.log(K)))
        return self

# ---------------------- Geometry & utilities ----------------------

def build_geometry(cfg: Config) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    lam = cfg.lam
    c0  = cfg.c0
    ell = cfg.ell_over_lam * lam
    L   = cfg.L_over_ell * ell
    a   = cfg.a_over_ell * ell
    Nr  = cfg.Nr

    f0 = c0 / lam
    fmin, fmax = cfg.band_low * f0, f0
    freqs = np.linspace(fmin, fmax, cfg.Nf)

    B  = fmax - fmin
    dx = lam * L / a
    dz = c0 / B

    xs = np.linspace(-a/2, a/2, Nr)
    receivers = np.stack([xs, np.zeros_like(xs)], axis=1)

    gx = (np.arange(cfg.Nx) - (cfg.Nx-1)/2.0) * dx
    gz = L + (np.arange(cfg.Nz) - (cfg.Nz-1)/2.0) * dz
    grid_x, grid_z = np.meshgrid(gx, gz, indexing='xy')
    grid_points = np.stack([grid_x.ravel(), grid_z.ravel()], axis=1)

    a_sub = a / 2.0
    sub_mask = np.abs(xs) <= (a_sub / 2.0)

    sigma = cfg.sigma_tilde * (lam / math.sqrt(ell * L))

    meta = {
        "lam": lam, "c0": c0, "ell": ell, "L": L, "a": a, "Nr": Nr,
        "f0": f0, "fmin": fmin, "fmax": fmax, "Nf": cfg.Nf,
        "dx": dx, "dz": dz, "B": B,
        "sigma_tilde": cfg.sigma_tilde, "sigma": sigma,
        "Nx": cfg.Nx, "Nz": cfg.Nz, "K": cfg.Nx * cfg.Nz,
        "a_sub": a_sub, "sub_mask_count": int(np.sum(sub_mask)),
        "seed": cfg.seed,
        "mix_beta": cfg.mix_beta,
    }
    return {
        "receivers": receivers,
        "grid_points": grid_points,
        "freqs": freqs,
        "sub_mask": sub_mask,
        "meta": meta,
        "rng": rng,
    }

def _random_fourier_features(cfg: Config, geo: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rng = geo["rng"]
    Q = cfg.RFF_Q
    W = rng.normal(0.0, 1.0, size=(Q, 2))
    Phi = rng.uniform(0.0, 2*np.pi, size=(Q,))
    return W, Phi

def _compute_line_integral_I(cfg: Config, geo: Dict[str, np.ndarray], W: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    receivers = geo["receivers"]
    grid_points = geo["grid_points"]
    ell = geo["meta"]["ell"]

    x_scaled = (grid_points / ell).T      # (2,K)
    y_scaled = (receivers / ell).T        # (2,Nr)

    W_dot_x = W @ x_scaled                # (Q,K)
    W_dot_y = W @ y_scaled                # (Q,Nr)

    Nr = receivers.shape[0]
    K  = grid_points.shape[0]
    I_accum = np.zeros((Nr, K), dtype=np.float64)
    sqrt2_over_Q = math.sqrt(2.0 / W.shape[0])

    for q in range(W.shape[0]):
        Ay = W_dot_y[q, :] + Phi[q]      # (Nr,)
        Bx = W_dot_x[q, :] + Phi[q]      # (K,)
        denom = Ay[:, None] - Bx[None, :]
        num = np.sin(Ay)[:, None] - np.sin(Bx)[None, :]
        ratio = np.empty_like(denom)
        small = np.isclose(denom, 0.0, atol=1e-8)
        ratio[~small] = num[~small] / denom[~small]
        ratio[small]  = np.cos(0.5 * (Ay[:, None] + Bx[None, :]))[small]
        I_accum += ratio

    return sqrt2_over_Q * I_accum  # (Nr,K)

# ---------------------- Green's matrices ----------------------

def build_G0_stack(cfg: Config, geo: Dict[str, np.ndarray]) -> np.ndarray:
    """Stacked homogeneous Green's matrix G0 over all frequencies (Eq. (2))."""
    receivers   = geo["receivers"]
    grid_points = geo["grid_points"]
    freqs       = geo["freqs"]
    c0          = geo["meta"]["c0"]

    diffs = receivers[:, None, :] - grid_points[None, :, :]   # (Nr,K,2)
    r = np.linalg.norm(diffs, axis=2)                         # (Nr,K)
    amp = 1.0 / (4.0 * np.pi * r)                             # (Nr,K)

    blocks = []
    for f in freqs:
        kappa = 2.0 * np.pi * f / c0
        blocks.append(amp * np.exp(1j * kappa * r))
    return np.vstack(blocks)                                  # (Nr*Nf, K)

def build_G_random_stack(cfg: Config, geo: Dict[str, np.ndarray]) -> np.ndarray:
    """Stacked random-medium Green's matrix Gb using the random travel-time model (Eq. (14))."""
    receivers   = geo["receivers"]
    grid_points = geo["grid_points"]
    freqs       = geo["freqs"]

    diffs = receivers[:, None, :] - grid_points[None, :, :]   # (Nr,K,2)
    r = np.linalg.norm(diffs, axis=2)                         # (Nr,K)

    # line integral of μ along the segment (Eq. 14)
    W, Phi = _random_fourier_features(cfg, geo)
    I = _compute_line_integral_I(cfg, geo, W, Phi)            # (Nr,K)

    amp = 1.0 / (4.0 * np.pi * r)
    lam  = cfg.lam
    c0   = cfg.c0
    sigma = geo["meta"]["sigma"]

    blocks = []
    for f in freqs:
        kappa = 2.0 * np.pi * f / c0
        phase = kappa * r
        rand_phase = sigma * kappa * r * I
        G_f = amp * np.exp(1j * (phase + rand_phase))
        blocks.append(G_f)
    return np.vstack(blocks)                                   # (Nr*Nf, K)

# ---------------------- Data generation & misc ----------------------

def draw_sparse_X_and_Y(cfg: Config, G_stack: np.ndarray, rng: np.random.Generator):
    N, K = G_stack.shape
    M, s = cfg.M, cfg.s
    X = np.zeros((K, M), dtype=np.complex128)
    Y = np.zeros((N, M), dtype=np.complex128)
    for m in range(M):
        idx = rng.choice(K, size=s, replace=False)
        mags = rng.uniform(1.0, 2.0, size=s)
        phases = rng.uniform(0.0, 2*np.pi, size=s)
        coeffs = mags * np.exp(1j * phases)
        X[idx, m] = coeffs
        Y[:, m] = G_stack[:, idx] @ coeffs
    return X, Y

def coherence(G: np.ndarray) -> float:
    col_norms = np.linalg.norm(G, axis=0); col_norms[col_norms == 0] = 1.0
    Gram = G.conj().T @ G
    C = np.abs(Gram) / np.outer(col_norms, col_norms)
    np.fill_diagonal(C, 0.0)
    return float(np.max(C))

def build_subarray_stack(G_stack: np.ndarray, cfg: Config, geo: Dict[str, np.ndarray]) -> np.ndarray:
    Nr, Nf = cfg.Nr, cfg.Nf
    sub_mask = geo["sub_mask"]
    rows = []
    for m in range(Nf):
        base = m * Nr
        rows_m = np.arange(base, base + Nr)[sub_mask]
        rows.append(rows_m)
    rows = np.concatenate(rows, axis=0)
    return G_stack[rows, :]

# ---------------------- Main ----------------------

def main(save_path: str = "snapshot_generation_data.npz"):
    cfg = Config().finalize()
    geo = build_geometry(cfg)

    # Build G0 (homogeneous, Eq. 2) and Gb (random-medium, Eq. 14)
    G0_stack   = build_G0_stack(cfg, geo)
    G_rand_stack = build_G_random_stack(cfg, geo)

    # MIXTURE: G_used = G0 + mix_beta * Gb  (tunable)
    G_stack = G0_stack + cfg.mix_beta * G_rand_stack

    # Generate data Y = G_used X
    rng = geo["rng"]
    X, Y = draw_sparse_X_and_Y(cfg, G_stack, rng)

    # Diagnostics: coherence (full & subarray)
    nu_full = coherence(G_stack)
    G_sub   = build_subarray_stack(G_stack, cfg, geo)
    nu_sub  = coherence(G_sub)

    # Sample-complexity check
    K = cfg.Nx * cfg.Nz
    M_req = K * math.log(K)
    M_ok  = cfg.M > M_req

    # Save results
    meta = geo["meta"].copy()
    meta.update({
        "Nx": cfg.Nx, "Nz": cfg.Nz, "K": K, "Nr": cfg.Nr, "Nf": cfg.Nf,
        "s": cfg.s, "M": cfg.M,
        "nu_full": nu_full, "nu_sub": nu_sub,
        "M_req_K_log_K": M_req, "M_ok": bool(M_ok),
    })

    meta_json = json.dumps(meta, indent=2)
    np.savez(
        save_path,
        Y=Y,
        G=G_stack,        # mixed dictionary: G0 + beta * Gb
        G0=G0_stack,      # NEW: homogeneous Green's function (stacked over freq)
        X=X,
        receivers=geo["receivers"],
        grid_points=geo["grid_points"],
        freqs=geo["freqs"],
        sub_mask=geo["sub_mask"].astype(np.uint8),
        meta_json=np.array(meta_json, dtype=object)
    )

    # Console report
    print("=== Wave-informed DL (A–C with G_mix = G0 + beta*Gb) ===")
    print(f"K={K}, Nr={cfg.Nr}, Nf={cfg.Nf}, M={cfg.M}, s={cfg.s}, beta={cfg.mix_beta}")
    print(f"Band: [{geo['meta']['fmin']:.3f}, {geo['meta']['fmax']:.3f}] with Nf={cfg.Nf}")
    print(f"PSF-guided spacings: Δx ≈ {geo['meta']['dx']:.4f}, Δz ≈ {geo['meta']['dz']:.4f}")
    print(f"Coherence ν (full stacked array): {nu_full:.4f}")
    print(f"Coherence ν (half-aperture subarray): {nu_sub:.4f}")
    print(f"Sample complexity: M > K log K ? {'YES' if M_ok else 'NO'} (M={cfg.M}, K log K≈{M_req:.1f})")
    print(f"Saved Y, G (G0 + beta*Gb), X to: {save_path}")

if __name__ == "__main__":
    main()
