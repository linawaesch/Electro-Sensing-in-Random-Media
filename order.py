#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ordering an unordered stacked sensing matrix (Step E) — updated to match the proven implementation
=================================================================================================

Two modes (default = debug):
  - debug : take TRUE stacked G from A–C, apply a random column permutation (optional tiny complex noise),
            then order it. Expect near-perfect recovery => validates algorithm wiring.
  - normal: load an unordered matrix (e.g., A_hat from Step-D) and order it.

Algorithm (as in the working reference you pasted):
  1) Column ℓ2-normalize; select a subarray of stacked rows.
     (Now supports --subarray-mode receivers (default, paper-faithful) or rows (central stacked block).)
  2) Build k-NN (union-symmetrized) from magnitude correlations on that subarray.
     If disconnected, increase k by +2 repeatedly until connected (or k=K-1).
  3) Weighted geodesics on the k-NN graph using Dijkstra with edge weight w_ij = arccos(clip(C_ij, 0, 1)).
  4) Classical MDS on *squared* distances: B = -1/2 J D^2 J (Eq. 13), take top r dims.
  5) Choose n_anchors by the largest cross-correlations between G_true and unordered; fit similarity (s,R,t) on anchors;
     align Z_hat to Z_true.
  6) Solve 1–1 assignment (Hungarian if available; greedy fallback). Return perm with A_ordered = A_unordered[:, perm].

Inputs
------
A–C NPZ (e.g.data.npz) must contain:
  receivers (Nr,2), grid_points (K,2), freqs (Nf,), sub_mask (Nr,), meta_json, G (true stacked), Y (N,M), X (K,M).

Normal-mode unordered NPZ (e.g., _out.npz) should contain one of:
  - A_hat (preferred), or
  - G_unordered, or
  - A

Outputs
-------
NPZ with:
  G_unordered, G_ordered, perm, Z_hat, Z_true, Z_hat_aligned, metrics (JSON).
  (In debug mode also permutation accuracy.)
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Tuple, Optional, Sequence

import numpy as np

# -------- Tunables (override via CLI) --------
DEFAULT_MODE = "normal"             # 'debug' or 'normal'  (default debug)
DEFAULT_R = 2                      # embed dimension (2 for 2D)
DEFAULT_K = 6                      # initial k for k-NN (≈ 2r..3r typical). Will auto-increase for connectivity.
DEFAULT_SUB_FRAC = 0.4             # fraction of stacked rows to keep (central contiguous block)
DEFAULT_N_ANCHORS = 4              # anchors >= r+1
DEFAULT_PERM_SEED = 7              # debug permutation seed
DEFAULT_DEBUG_NOISE = 0.001          # relative column-wise complex noise in debug mode
DEFAULT_DTYPE = "complex128"
ABC_DEFAULT = "snapshot_generation_data.npz"
UNORDERED_DEFAULT = "recover_unordered_g_out.npz"
SAVE_DEFAULT = "order_out.npz"

# NEW: subarray selection mode (paper-faithful default: 'receivers')
DEFAULT_SUBARRAY_MODE = "receivers"  # 'receivers' or 'rows'

# SciPy only for the Hungarian assignment (optional)
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ---------------- I/O helpers & normalizations ----------------

def normalize_cols(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(A, axis=0)
    n_safe = np.where(n > 0, n, 1.0)
    return A / n_safe, n

def best_phase_align(a: np.ndarray, b: np.ndarray) -> complex:
    """Return unit-modulus factor that rotates b toward a: exp(-i arg(<a,b>))."""
    c = np.vdot(a, b)  # a^H b
    if c == 0:
        return 1.0 + 0j
    return np.conj(c) / np.abs(c)


# ---------------- k-NN graph (with auto-connectivity) ----------------

def _build_knn_union(C: np.ndarray, k: int) -> np.ndarray:
    """Union-symmetrized k-NN from similarity matrix C (diag zero)."""
    K = C.shape[0]
    A = np.zeros((K, K), dtype=bool)
    topk = np.argsort(-C, axis=1)[:, :min(k, K-1)]
    rows = np.arange(K)[:, None]
    A[rows, topk] = True
    A = np.logical_or(A, A.T)
    np.fill_diagonal(A, False)
    return A

def _n_components(A: np.ndarray) -> int:
    K = A.shape[0]
    seen = np.zeros(K, dtype=bool)
    comps = 0
    for s in range(K):
        if not seen[s]:
            comps += 1
            q = deque([s])
            seen[s] = True
            while q:
                u = q.popleft()
                for v in np.where(A[u])[0]:
                    if not seen[v]:
                        seen[v] = True
                        q.append(v)
    return comps

def ensure_connected_adj(C: np.ndarray, k_start: int, step: int = 2) -> Tuple[np.ndarray, int]:
    """Increase k (by step) until the union-symmetrized k-NN graph is connected (or k = K-1)."""
    K = C.shape[0]
    k = max(1, min(k_start, K-1))
    while True:
        A = _build_knn_union(C, k)
        if _n_components(A) == 1 or k >= K-1:
            return A, k
        k = min(k + step, K-1)


# ---------------- Weighted geodesics (Dijkstra) ----------------

def weighted_geodesic_distances(C: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """
    All-pairs shortest paths on weighted k-NN graph with angular edge costs:
        w_ij = arccos( clip(C_ij, 0, 1) ),  where C_ij = |<g_i, g_j>| on subarray.
    """
    import heapq
    K = C.shape[0]
    W = np.arccos(np.clip(C, 0.0, 1.0))
    np.fill_diagonal(W, 0.0)
    neighbors = [np.where(adj[i])[0] for i in range(K)]
    D = np.full((K, K), np.inf, dtype=float)
    for s in range(K):
        dist = np.full(K, np.inf, dtype=float)
        dist[s] = 0.0
        h = [(0.0, s)]
        while h:
            du, u = heapq.heappop(h)
            if du > dist[u]:
                continue
            for v in neighbors[u]:
                nd = du + W[u, v]
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(h, (nd, v))
        D[s, :] = dist
    D = 0.5 * (D + D.T)
    return D


# ---------------- Classical MDS (on squared distances) ----------------

def classical_mds(D_proxy: np.ndarray, r: int = 2) -> np.ndarray:
    """Classical metric MDS via double-centering of squared distances."""
    K = D_proxy.shape[0]
    D2 = D_proxy ** 2
    J = np.eye(K) - np.ones((K, K)) / K
    B = -0.5 * (J @ D2 @ J)
    B = 0.5 * (B + B.T)
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    w_r = np.clip(w[:r], 0.0, None)
    Z = V[:, :r] * np.sqrt(w_r)
    return Z


# ---------------- Similarity alignment on anchors ----------------

def fit_similarity(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Minimize || s X R + t - Y ||_F over scale s>0, rotation/reflection R, translation t.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    s = np.trace(np.diag(S)) / max(1e-12, np.sum(Xc * Xc))
    t = Y.mean(axis=0, keepdims=True) - (s * X.mean(axis=0, keepdims=True)) @ R
    return float(s), R, t

def apply_similarity(Z: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * Z) @ R + t


# ---------------- Anchors & assignment ----------------

def choose_anchor_pairs_by_corr(G_true_n: np.ndarray, A_un_n: np.ndarray, n_anchors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (I_hat, J_true) indices of size >= max(2, n_anchors) from largest cross-correlations."""
    K = G_true_n.shape[1]
    CC = np.abs(G_true_n.conj().T @ A_un_n)  # (K x K): true vs unordered
    pairs: Sequence[Tuple[int, int]] = []
    used_i, used_j = set(), set()
    flat = np.argsort(-CC, axis=None)  # descending
    for idx in flat:
        i = idx % K   # column in A_unordered
        j = idx // K  # column in G_true
        if i in used_i or j in used_j:
            continue
        pairs.append((i, j))
        used_i.add(i); used_j.add(j)
        if len(pairs) >= max(2, n_anchors):
            break
    if len(pairs) < 2:
        raise RuntimeError("Could not find at least 2 unique anchors by correlation.")
    I_hat = np.array([p[0] for p in pairs], dtype=int)
    J_true = np.array([p[1] for p in pairs], dtype=int)
    return I_hat, J_true

def hungarian(cost: np.ndarray) -> np.ndarray:
    """Hungarian if SciPy available; else greedy fallback."""
    K = cost.shape[0]
    if SCIPY_AVAILABLE:
        r, c = linear_sum_assignment(cost)
        perm = np.empty(K, dtype=int)
        perm[c] = r
        return perm
    # Greedy fallback (okay if embeddings are close)
    perm = -np.ones(K, dtype=int)
    taken = np.zeros(K, dtype=bool)
    order = np.argsort(cost.min(axis=1))
    for i in order:
        j = int(np.argmin(cost[i, :] + 1e9 * taken.astype(float)))
        perm[j] = i
        taken[j] = True
    if (perm == -1).any():
        # final fill by cheapest available
        avail = np.where(~taken)[0]
        free_i = np.where(perm == -1)[0]
        for j in free_i:
            i = int(np.argmin(cost[:, j]))
            perm[j] = i
    return perm


# ---------------- NEW: receiver-subarray helpers (paper-correct) ----------------

def build_receiver_mask(Nr: int, frac: float) -> np.ndarray:
    """Central contiguous receiver subset (length Nr)."""
    frac = max(0.0, min(1.0, float(frac)))
    sub_nr = max(2, int(round(frac * Nr)))
    sub_nr = min(sub_nr, Nr)
    c = Nr // 2
    half = sub_nr // 2
    lo = max(0, c - half)
    hi = min(Nr, lo + sub_nr)
    mask = np.zeros(Nr, dtype=bool)
    mask[lo:hi] = True
    return mask

def stacked_rows_from_mask(Nr: int, Nf: int, recv_mask: np.ndarray) -> np.ndarray:
    """Repeat the same receiver mask across all frequency blocks (frequency-major stacking)."""
    idx = np.flatnonzero(recv_mask)
    rows = []
    for f in range(Nf):
        base = f * Nr
        rows.extend(base + idx)
    return np.asarray(rows, dtype=np.int64)


# ---------------- Core ordering (as in the proven script) ----------------

def order_via_mds(
    A_unordered: np.ndarray,
    G_true: np.ndarray,
    *,
    r: int,
    k_start: int,
    subarray_fraction: float,
    n_anchors: int,
    grid_points: Optional[np.ndarray] = None,
    rows: Optional[np.ndarray] = None,  # NEW: allow passing precomputed subarray rows
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Return (perm, Z_hat_aligned, Z_true, A_ordered, info).
      perm has length K with A_ordered = A_unordered[:, perm]
    """
    N, K = A_unordered.shape
    assert G_true.shape == (N, K), "Shape mismatch between A_unordered and G_true."

    # Normalize full columns (used later for anchors), unchanged:
    A_un_n, _ = normalize_cols(A_unordered.astype(np.complex128, copy=False))
    G_true_n, _ = normalize_cols(G_true.astype(np.complex128, copy=False))

    # --- Subarray rows ---
    if rows is None:
        # Fallback: central contiguous subarray (previous behavior)
        m = max(2, int(round(subarray_fraction * N)))
        m = min(m, N)
        start = (N - m) // 2
        rows = np.arange(start, start + m)
    rows = np.asarray(rows, dtype=int)

    # --- Correlations on subarray with SUBARRAY-NORMALIZED columns (fix) ---
    A_sub = A_unordered[rows, :]
    G_sub = G_true[rows, :]
    A_sub_n = A_sub / (np.linalg.norm(A_sub, axis=0, keepdims=True) + 1e-15)
    G_sub_n = G_sub / (np.linalg.norm(G_sub, axis=0, keepdims=True) + 1e-15)

    C_hat = np.abs((A_sub_n.conj().T @ A_sub_n))  # (K,K)
    C_true = np.abs((G_sub_n.conj().T @ G_sub_n))
    np.fill_diagonal(C_hat, 0.0)
    np.fill_diagonal(C_true, 0.0)

    # Build connected k-NN (auto-increase k)
    adj_hat, k_used_hat = ensure_connected_adj(C_hat, k_start=k_start, step=2)
    adj_true, k_used_true = ensure_connected_adj(C_true, k_start=k_start, step=2)

    # Weighted geodesics
    D_hat = weighted_geodesic_distances(C_hat, adj_hat)
    D_true = weighted_geodesic_distances(C_true, adj_true)
    if not np.isfinite(D_hat).all() or not np.isfinite(D_true).all():
        raise RuntimeError("Geodesic distances contain inf/NaN; check connectivity or subarray selection.")

    # Embeddings (classical MDS on squared distances)
    Z_hat = classical_mds(D_hat, r=r)
    Z_true = classical_mds(D_true, r=r)

    # Anchors from cross-correlation (using full-column normalization); fit similarity on anchors
    I_hat, J_true = choose_anchor_pairs_by_corr(G_true_n, A_un_n, n_anchors=n_anchors)
    s, R, t = fit_similarity(Z_hat[I_hat, :], Z_true[J_true, :])
    Z_hat_aligned = apply_similarity(Z_hat, s, R, t)

    # Assignment between aligned Z_hat and Z_true
    diff = Z_hat_aligned[:, None, :] - Z_true[None, :, :]
    Ccost = np.sum(diff * diff, axis=2)
    perm = hungarian(Ccost)  # A_ordered = A_unordered[:, perm]
    A_ordered = A_unordered[:, perm]

    info = {
        "k_used_hat": int(k_used_hat),
        "k_used_true": int(k_used_true),
        "rows_kept": int(rows.size),
        "adj_hat": adj_hat,
        "adj_true": adj_true,
        "D_hat": D_hat,
        "D_true": D_true,
        "Z_hat": Z_hat,
        "Z_true": Z_true,
        "Z_hat_aligned": Z_hat_aligned,
        "anchors_hat": I_hat,
        "anchors_true": J_true,
    }
    return perm, Z_hat_aligned, Z_true, A_ordered, info


# ---------------- Metrics (residuals + correlations) ----------------

def evaluate_against_truth(G_true: np.ndarray, G_est: np.ndarray) -> dict:
    """Column-wise |corr| and Frobenius relative error after per-column phase alignment."""
    Gt_n, _ = normalize_cols(G_true)
    Ge_n, _ = normalize_cols(G_est)
    K = Ge_n.shape[1]
    Ge_aligned = np.empty_like(Ge_n)
    corrs = np.zeros(K, dtype=float)
    for j in range(K):
        phi = best_phase_align(Gt_n[:, j], Ge_n[:, j])
        Ge_aligned[:, j] = Ge_n[:, j] * phi
        corrs[j] = np.abs(np.vdot(Gt_n[:, j], Ge_n[:, j]))
    relF = np.linalg.norm(Gt_n - Ge_aligned, 'fro') / max(1e-12, np.linalg.norm(Gt_n, 'fro'))
    return {
        "corr_mean": float(np.mean(corrs)),
        "corr_median": float(np.median(corrs)),
        "corr_min": float(np.min(corrs)),
        "corr_p05": float(np.quantile(corrs, 0.05)),
        "relF_error_after_phase": float(relF),
    }


# ---------------- Debug helpers ----------------

def permute_true_G(G_true: np.ndarray, seed: int, noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (G_perm, perm) where G_perm[:, i] = G_true[:, perm[i]]; optional complex noise at relative column norm."""
    rng = np.random.default_rng(seed)
    K = G_true.shape[1]
    perm = rng.permutation(K)
    Gp = G_true[:, perm].astype(np.complex128, copy=True)
    if noise > 0:
        E = rng.standard_normal(Gp.shape) + 1j * rng.standard_normal(Gp.shape)
        for j in range(K):
            nE = np.linalg.norm(E[:, j])
            if nE > 0:
                E[:, j] *= (noise * np.linalg.norm(Gp[:, j]) / nE)
        Gp = Gp + E
    return Gp, perm


# ---------------- Main ----------------

def build_subarray_rows_receivers(Nr: int, Nf: int, sub_mask: np.ndarray) -> np.ndarray:
    """Replicate the receiver mask at every frequency block (paper-faithful)."""
    sub_mask = np.asarray(sub_mask, bool)
    idx = np.where(sub_mask)[0]
    rows = []
    for m in range(Nf):
        rows.append(m * Nr + idx)
    return np.concatenate(rows, axis=0).astype(int)

def build_subarray_rows_rows(N: int, fraction: float) -> np.ndarray:
    """Central contiguous block of stacked rows (legacy behavior)."""
    m = max(2, int(round(fraction * N)))
    m = min(m, N)
    start = (N - m) // 2
    return np.arange(start, start + m, dtype=int)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["debug", "normal"],
                   help="debug: permute true G; normal: order unordered file")
    p.add_argument("--abc", type=str, default=ABC_DEFAULT, help="A–C npz (must include G, Y, X, receivers, grid_points, freqs, sub_mask, meta_json)")
    p.add_argument("--unordered", type=str, default=UNORDERED_DEFAULT, help="unordered npz (A_hat or G_unordered or A) for mode=normal")
    p.add_argument("--save", type=str, default=SAVE_DEFAULT, help="output npz path")

    # algorithm knobs
    p.add_argument("--r", type=int, default=DEFAULT_R, help="embedding dimension (2)")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="initial k for k-NN (auto-increases if disconnected)")
    p.add_argument("--subarray-fraction", type=float, default=DEFAULT_SUB_FRAC, help="fraction (for --subarray-mode rows)")
    p.add_argument("--n-anchors", type=int, default=DEFAULT_N_ANCHORS, help="number of anchors (>= r+1)")
    p.add_argument("--subarray-mode", type=str, default=DEFAULT_SUBARRAY_MODE,
                   choices=["receivers", "rows"],
                   help="receivers: use receiver sub_mask at every frequency (paper default). "
                        "rows: use a central contiguous block of stacked rows (legacy).")

    # debug options
    p.add_argument("--perm-seed", type=int, default=DEFAULT_PERM_SEED, help="permutation seed in debug mode")
    p.add_argument("--debug-noise", type=float, default=DEFAULT_DEBUG_NOISE, help="relative column-wise complex noise in debug mode")
    p.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, choices=["complex128", "complex64"])
    args = p.parse_args()

    # Load A–C
    ABC = np.load(args.abc, allow_pickle=True)
    G_true = ABC["G"].astype(np.complex128, copy=False)
    Y = ABC["Y"].astype(np.complex128, copy=False)
    X_true = ABC["X"].astype(np.complex128, copy=False)
    receivers = ABC["receivers"]
    grid_points = ABC["grid_points"]
    freqs = ABC["freqs"]
    sub_mask = ABC["sub_mask"].astype(bool)
    meta = json.loads(ABC["meta_json"].item()) if "meta_json" in ABC.files else {}
    Nr = int(meta.get("Nr", receivers.shape[0]))
    Nf = int(meta.get("Nf", len(freqs)))
    N, K = G_true.shape

    # --- build subarray rows (paper-faithful default) ---
    if args.subarray_mode == "receivers":
        rows = build_subarray_rows_receivers(Nr, Nf, sub_mask)
    else:
        rows = build_subarray_rows_rows(N, args.subarray_fraction)

    # Build unordered input depending on mode
    dtype = np.complex128 if args.dtype == "complex128" else np.complex64
    debug_perm = None
    if args.mode == "debug":
        A_unordered, debug_perm = permute_true_G(G_true, seed=args.perm_seed, noise=args.debug_noise)
        A_unordered = A_unordered.astype(dtype, copy=False)
        print(f"[DEBUG] Using permuted true G (seed={args.perm_seed}, noise={args.debug_noise}).")
        # In debug, also build the X compatible with A_unordered so Y = A_unordered @ X_permuted
        X_permuted = X_true[debug_perm, :]
    else:
        U = np.load(args.unordered, allow_pickle=True)
        if "A_hat" in U.files:
            A_unordered = U["A_hat"].astype(dtype, copy=False)
            print("[INFO] Using 'A_hat' from unordered file.")
        elif "G_unordered" in U.files:
            A_unordered = U["G_unordered"].astype(dtype, copy=False)
            print("[INFO] Using 'G_unordered' from unordered file.")
        elif "A" in U.files:
            A_unordered = U["A"].astype(dtype, copy=False)
            print("[INFO] Using 'A' from unordered file.")
        else:
            print("[ERROR] Unordered file must contain 'A_hat' or 'G_unordered' or 'A'.", file=sys.stderr)
            sys.exit(2)
        X_permuted = None

    # Sanity
    if A_unordered.shape != (N, K):
        print(f"[ERROR] Row/col mismatch: unordered has shape {A_unordered.shape}, but G_true is {G_true.shape}.", file=sys.stderr)
        sys.exit(2)

    # Run ordering
    perm, Z_hat_aligned, Z_true_mds, G_ordered, info = order_via_mds(
        A_unordered=A_unordered,
        G_true=G_true,
        r=int(args.r),
        k_start=int(args.k),
        subarray_fraction=float(args.subarray_fraction),
        n_anchors=int(args.n_anchors),
        grid_points=grid_points,
        rows=rows,  # <-- use the paper-faithful subarray rows
    )

    # Metrics vs ground truth
    metrics = {
        "mode": args.mode,
        "r": int(args.r),
        "k_init": int(args.k),
        "k_used_hat": info["k_used_hat"],
        "k_used_true": info["k_used_true"],
        "subarray_rows_kept": info["rows_kept"],
        "subarray_mode": args.subarray_mode,
    }
    metrics.update(evaluate_against_truth(G_true, G_ordered))

    # Residuals:
    try:
        # Before ordering
        if args.mode == "debug":
            R0 = Y - (A_unordered @ X_permuted)
            r0_abs = float(np.linalg.norm(R0, "fro"))
            r0_rel = r0_abs / max(1e-16, float(np.linalg.norm(Y, "fro")))
        else:
            # compute LS residual so it's finite (no NaN in normal mode)
            try:
                X_ls, *_ = np.linalg.lstsq(A_unordered, Y, rcond=None)
                R0 = Y - (A_unordered @ X_ls)
                r0_abs = float(np.linalg.norm(R0, "fro"))
                r0_rel = r0_abs / max(1e-16, float(np.linalg.norm(Y, "fro")))
            except Exception:
                r0_abs, r0_rel = np.nan, np.nan

        # After ordering: Y ≈ G_ordered @ X_true (grid order)
        R1 = Y - (G_ordered @ X_true)
        r1_abs = float(np.linalg.norm(R1, "fro"))
        r1_rel = r1_abs / max(1e-16, float(np.linalg.norm(Y, "fro")))

        metrics.update({
            "residual_unordered_abs": r0_abs,
            "residual_unordered_rel": r0_rel,
            "residual_ordered_abs": r1_abs,
            "residual_ordered_rel": r1_rel,
        })
    except Exception:
        pass

    # Debug: permutation accuracy
    if debug_perm is not None:
        inv_debug = np.empty_like(debug_perm)
        inv_debug[debug_perm] = np.arange(K)
        acc = float(np.mean(perm == inv_debug))
        metrics.update({
            "debug_perm_accuracy": acc,
            "debug_perm_num_correct": int(np.sum(perm == inv_debug)),
            "debug_perm_num_total": int(K),
            "perm_seed": int(args.perm_seed),
            "debug_noise": float(args.debug_noise),
        })

    # >>>>>>>>>>>>>>> NEW: normal-mode correctness count (how many “correct” out of K) <<<<<<<<<<<<<<
    if debug_perm is None:  # i.e., normal mode
        # "Correct" if the diagonal correlation is the maximum in both its row and its column
        Gt_n, _ = normalize_cols(G_true)
        Go_n, _ = normalize_cols(G_ordered)
        C = np.abs(Gt_n.conj().T @ Go_n)  # (K x K)
        diag = np.diag(C)
        row_max = C.max(axis=1)
        col_max = C.max(axis=0)
        tol = 1e-8
        correct_mask = (diag >= row_max - tol) & (diag >= col_max - tol)
        normal_correct_count = int(np.sum(correct_mask))
        metrics.update({
            "normal_correct_count": normal_correct_count,
            "normal_correct_pct": float(normal_correct_count) / float(K),
        })
    # >>>>>>>>>>>>>>> END addition <<<<<<<<<<<<<<

    # Save
    np.savez(
        args.save,
        G_unordered=A_unordered,
        G_ordered=G_ordered,
        perm=perm,
        Z_hat=info["Z_hat"],
        Z_true=info["Z_true"],
        Z_hat_aligned=info["Z_hat_aligned"],
        adj_hat=info["adj_hat"].astype(np.uint8),
        adj_true=info["adj_true"].astype(np.uint8),
        D_hat=info["D_hat"],
        D_true=info["D_true"],
        anchors_hat=info["anchors_hat"],
        anchors_true=info["anchors_true"],
        metrics=np.array(json.dumps(metrics), dtype=object),
    )

    # Console
    print("=== Ordering (weighted geodesics + auto-k connectivity + MDS + anchors + Hungarian) ===")
    print(f"Mode={args.mode} | K={K}, N={N}, r={args.r}, k_init={args.k}, k_used_hat={info['k_used_hat']}, k_used_true={info['k_used_true']}")
    print(f"Subarray rows kept: {info['rows_kept']} (fraction={info['rows_kept']/N:.2f}) [mode={args.subarray_mode}]")
    print(f"Diag corr (ordered vs true): mean={metrics['corr_mean']:.3f}, median={metrics['corr_median']:.3f}, min={metrics['corr_min']:.3f}")
    print(f"Rel Fro error (after phase): {metrics['relF_error_after_phase']:.4e}")
    if "residual_ordered_rel" in metrics and not np.isnan(metrics["residual_ordered_rel"]):
        print(f"Residuals: unordered rel={metrics['residual_unordered_rel']:.3e}, ordered rel={metrics['residual_ordered_rel']:.3e}")
    if debug_perm is None and "normal_correct_count" in metrics:
        print(f"Normal-mode 'correct' columns: {metrics['normal_correct_count']}/{K} "
              f"({metrics['normal_correct_pct']*100:.1f}%)")
    if "debug_perm_accuracy" in metrics:
        print(f"[DEBUG] Perm accuracy: {metrics['debug_perm_accuracy']:.4f} "
              f"({metrics['debug_perm_num_correct']}/{metrics['debug_perm_num_total']})")
    print("Saved:", args.save)

        # Return (num_correct, K) so it can be used when imported
    if args.mode == "normal" and "normal_correct_count" in metrics:
        return metrics["normal_correct_count"], K
    return None, K


if __name__ == "__main__":
    main()
