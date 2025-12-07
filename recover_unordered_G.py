#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
initialize A0 as the stacked homogeneous Green's matrix G0 (all frequencies),
then run GeLMA (Eq. 10) + MOD (Eq. 12) unchanged on the stacked data.
"""

import argparse, json, time
import numpy as np

# ---------- helpers (unchanged) ----------

def normalize_columns(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=0); n[n==0] = 1.0
    return A / n

def spectral_norm(A: np.ndarray) -> float:
    try:
        return float(np.linalg.norm(A, 2))
    except Exception:
        B = A.conj().T @ A
        try:
            ev = np.linalg.eigvalsh(B).real
            return float(np.sqrt(max(ev.max(), 0.0)))
        except Exception:
            return float(np.linalg.norm(A, "fro"))

def complex_soft_threshold(Z: np.ndarray, T) -> np.ndarray:
    mag = np.abs(Z)
    denom = np.maximum(mag, 1e-30)
    T_arr = np.asarray(T)
    scale = np.maximum(0.0, 1.0 - T_arr / denom)
    return scale * Z

def residual_ratio(A: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    num = np.linalg.norm(A @ X - Y, "fro")
    den = np.linalg.norm(Y, "fro") + 1e-30
    return float(num / den)

def greedy_assignment_max(C: np.ndarray) -> np.ndarray:
    K = C.shape[0]
    flat = np.argsort(C.ravel())[::-1]
    used_i = np.zeros(K, bool); used_j = np.zeros(K, bool)
    perm = -np.ones(K, int)
    cnt = 0
    for idx in flat:
        i = idx // K; j = idx % K
        if (not used_i[i]) and (not used_j[j]):
            perm[i] = j; used_i[i] = True; used_j[j] = True
            cnt += 1
            if cnt == K: break
    return perm

def two_opt_improve(C: np.ndarray, perm: np.ndarray, max_passes=2) -> np.ndarray:
    K = C.shape[0]
    for _ in range(max_passes):
        improved = False
        for i1 in range(K-1):
            j1 = perm[i1]
            for i2 in range(i1+1, K):
                j2 = perm[i2]
                gain = (C[i1,j2] + C[i2,j1]) - (C[i1,j1] + C[i2,j2])
                if gain > 0:
                    perm[i1], perm[i2] = j2, j1
                    improved = True
        if not improved: break
    return perm

def best_permutation(G_true: np.ndarray, A_hat: np.ndarray):
    Gt = normalize_columns(G_true.astype(np.complex128, copy=False))
    Ah = normalize_columns(A_hat.astype(np.complex128, copy=False))
    C  = np.abs(Gt.conj().T @ Ah)  # (K,K)
    try:
        from scipy.optimize import linear_sum_assignment
        r,c = linear_sum_assignment(-C)
        perm = np.zeros(C.shape[0], dtype=int); perm[r] = c
    except Exception:
        perm = greedy_assignment_max(C)
        perm = two_opt_improve(C, perm)
    matched = C[np.arange(C.shape[0]), perm]
    metrics  = {
        "corr_mean": float(np.mean(matched)),
        "corr_median": float(np.median(matched)),
        "corr_min": float(np.min(matched)),
        "corr_max": float(np.max(matched)),
        "corr_p05": float(np.quantile(matched, 0.05)),
        "corr_p95": float(np.quantile(matched, 0.95)),
    }
    return perm, matched, metrics, C

def gelma_matrix(A: np.ndarray, Y: np.ndarray, lam_base, dt: float, iters: int, X0=None):
    N,K = A.shape
    N2,M = Y.shape
    assert N == N2
    dtype = np.complex128 if (A.dtype==np.complex128 or Y.dtype==np.complex128) else np.complex64
    X = np.zeros((K,M), dtype=dtype) if X0 is None else X0.astype(dtype, copy=False)
    Z = np.zeros((K,M), dtype=dtype)
    if np.isscalar(lam_base):
        T = lam_base * dt
    else:
        lam_vec = np.asarray(lam_base).reshape(1,-1)
        T = lam_vec * dt
    for _ in range(iters):
        R = Y - A @ X
        Z += dt * (A.conj().T @ R)
        X = complex_soft_threshold(Z, T)
    return X

def mod_update(Y: np.ndarray, X: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    XH = X.conj().T
    G1 = Y @ XH
    S  = X @ XH
    diag = np.arange(S.shape[0]); S[diag, diag] += ridge
    A = np.linalg.solve(S.T, G1.T).T
    return normalize_columns(A)

# ---------- NEW: homogeneous G0 builder (Eq. (2), stacked across frequencies) ----------

def build_G0_stack(receivers: np.ndarray, grid_points: np.ndarray, freqs: np.ndarray, c0: float) -> np.ndarray:
    Nr = receivers.shape[0]
    K  = grid_points.shape[0]
    diffs = receivers[:, None, :] - grid_points[None, :, :]    # (Nr,K,2)
    r = np.linalg.norm(diffs, axis=2)                          # (Nr,K)
    amp = 1.0 / (4.0 * np.pi * r)
    blocks = []
    for f in freqs:
        kappa = 2.0 * np.pi * f / c0
        blocks.append(amp * np.exp(1j * kappa * r))
    return np.vstack(blocks)                                    # (Nr*Nf, K)

# ---------- main ----------

def run(data_path: str, save_path: str, outer_iters: int, gelma_iters: int,
        alpha: float, batch_cols: int, dtype: str, seed: int = 123, ridge: float = 1e-8):

    rng  = np.random.default_rng(seed)
    data = np.load(data_path, allow_pickle=True)

    Y_full      = data["Y"]
    G_true      = data["G"]   # this is G0 + beta*Gb from ABC
    receivers   = data["receivers"]
    grid_points = data["grid_points"]
    freqs       = data["freqs"]
    meta = json.loads(data["meta_json"].item()) if "meta_json" in data.files else {}
    c0  = float(meta.get("c0", 1.0))

    N,M = Y_full.shape
    K   = G_true.shape[1]
    assert G_true.shape[0] == N

    # batch selection
    if (batch_cols is None) or (batch_cols >= M):
        cols = np.arange(M)
    else:
        cols = rng.choice(M, size=max(batch_cols, min(M, K)), replace=False)
    Y = Y_full[:, cols]

    cast = np.complex128 if dtype == "complex128" else np.complex64
    Y      = Y.astype(cast, copy=False)
    G_true = G_true.astype(cast, copy=False)

    # ---------- CHANGED INIT: A0 := stacked homogeneous G0 (Eq. (2), all frequencies) ----------
    G0_stack = build_G0_stack(receivers, grid_points, freqs, c0=c0).astype(cast)
    A = normalize_columns(G0_stack)
    # --------------------------------------------------------------------------------------------

    # step-size & thresholds
    sn = spectral_norm(A)
    dt = 0.99 / max(sn**2, 1e-12)

    s_target = meta.get("s", None)
    AtY = A.conj().T @ Y
    if s_target is not None and 0 < s_target < K:
        q = 1.0 - float(s_target) / float(K)
        lam_base = alpha * np.quantile(np.abs(AtY), q, axis=0)
    else:
        lam_base = alpha * float(np.median(np.abs(AtY)))

    logs, X = [], None
    t0 = time.time()
    for it in range(outer_iters):
        X = gelma_matrix(A, Y, lam_base=lam_base, dt=dt, iters=gelma_iters, X0=X)
        A_old = A
        A = mod_update(Y, X, ridge=ridge)

        res = residual_ratio(A, X, Y)
        dA  = float(np.linalg.norm(A - A_old, "fro")/(np.linalg.norm(A_old, "fro")+1e-30))
        logs.append({"iter": it+1, "residual_ratio": res, "deltaA": dA, "dt": dt, "sn": sn,
                     "lam_base_median": float(np.median(lam_base)) if not np.isscalar(lam_base) else float(lam_base)})
        if it == 0:
            sn = spectral_norm(A)
            dt = 0.99 / max(sn**2, 1e-12)
    elapsed = time.time() - t0

    # alignment
    perm, matched, align_metrics, C = best_permutation(G_true, A)
    Cmax = C.max(axis=0)

    Gn = normalize_columns(G_true.astype(np.complex128))
    An = normalize_columns(A.astype(np.complex128))
    Gp = Gn[:, perm]
    ph = np.sum(np.conj(Gp) * An, axis=0); ph = ph / (np.abs(ph) + 1e-30)
    frob_rel = float(np.linalg.norm(An - Gp * ph, "fro")/(np.linalg.norm(Gn, "fro")+1e-30))

    metrics = {
        "train_residual_ratio": float(residual_ratio(A, X, Y)),
        "align_corr_mean": align_metrics["corr_mean"],
        "align_corr_median": align_metrics["corr_median"],
        "align_corr_min": align_metrics["corr_min"],
        "align_corr_max": align_metrics["corr_max"],
        "align_corr_p05": align_metrics["corr_p05"],
        "align_corr_p95": align_metrics["corr_p95"],
        "frob_rel_error_phase_aligned": frob_rel,
        "Cmax_mean": float(np.mean(Cmax)),
        "Cmax_median": float(np.median(Cmax)),
        "Cmax_min": float(np.min(Cmax)),
        "elapsed_sec": elapsed,
        "batch_M": int(Y.shape[1]),
        "N": int(N), "K": int(K),
    }

    np.savez(save_path,
        A_hat=A, X_hat=X, perm=perm, matched_corr=matched,
        logs=np.array(json.dumps(logs), dtype=object),
        metrics=np.array(json.dumps(metrics), dtype=object),
    )

    print("=== Step D (GeLMA+MOD) — A0 := stacked homogeneous G0 ===")
    print(f"Data: {data_path}")
    print(f"N={N}, K={K}, batch M={Y.shape[1]}, outer iters={outer_iters}, GeLMA iters={gelma_iters}")
    print("Last iteration:", logs[-1])
    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {save_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="snapshot_generation_data.npz")
    ap.add_argument("--save", type=str, default="recover_unordered_g_out.npz")
    ap.add_argument("--outer-iters", type=int, default=8)
    ap.add_argument("--gelma-iters", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--batch-cols", type=int, default=None)
    ap.add_argument("--dtype", type=str, default="complex128", choices=["complex128","complex64"])
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    run(args.data, args.save, args.outer_iters, args.gelma_iters,
        args.alpha, args.batch_cols, args.dtype, args.seed)

if __name__ == "__main__":
    main()
