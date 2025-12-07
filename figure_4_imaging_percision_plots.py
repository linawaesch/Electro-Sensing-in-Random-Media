#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imaging_precision_stats.py
==========================

Compute statistics of imaging precision for:
  - HOMOGENEOUS dictionary G0
  - RECOVERED dictionary G_rec
  - TRUE random-medium dictionary G_true

Based on the imaging functional (Moscoso et al., Eq. (16)):
    I(x_i; x_j0) = | g_hat(x_i)^* g_true(x_j0) |

For each grid index j0:
  - Use g_true(x_j0) as the data vector
  - Build I_hom, I_rec, I_true
  - Normalize each image to max = 1
  - Precision metric = intensity at the correct pixel j0:
        p_hom(j0) = I_hom[j0]
        p_rec(j0) = I_rec[j0]
        p_true(j0) = I_true[j0]  (should be 1 by definition)
  - Also compute MSE vs true image:
        mse_hom(j0) = mean( (I_hom - I_true)^2 )
        mse_rec(j0) = mean( (I_rec - I_true)^2 )

Outputs:
  - A PNG figure with histograms and scatter plot
  - A NPZ file with all precision / MSE arrays

Assumptions about input NPZ files:
  - ABC file (default: snapshot_generation_data.npz) contains:
        G      : stacked mixture dictionary (G0 + beta * Gb)
        G0     : stacked homogeneous dictionary
        meta_json : JSON string with key "mix_beta" and grid sizes (Nx, Nz, K)
  - ORDER file (default: order_g_out.npz) contains:
        G_ordered : final recovered dictionary (same shape as G0)
    If G_ordered is missing, we fall back to G_unordered.
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------- helpers ----------------------------

def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path, allow_pickle=True)


def parse_meta(abc_npz):
    meta = {}
    if "meta_json" in abc_npz:
        raw = abc_npz["meta_json"]
        try:
            s = raw.item() if hasattr(raw, "item") else str(raw)
            meta = json.loads(s)
        except Exception:
            pass
    return meta


def normalize_columns(G):
    """Normalize columns of G to unit l2 norm."""
    G = np.array(G)
    norms = np.linalg.norm(G, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return G / norms


# ------------------------ main computation ------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute distribution of imaging precision for homogeneous vs recovered dictionaries."
    )
    parser.add_argument("--abc", default="snapshot_generation_data.npz",
                        help="ABC data NPZ (from snapshot_generation.py). Default: snapshot_generation_data.npz")
    parser.add_argument("--order", default="order_out.npz",
                        help="Ordering/recovered NPZ (with G_ordered). Default: order_g_out.npz")
    parser.add_argument("--out-prefix", default="imaging_precision",
                        help="Prefix for output files (PNG + NPZ).")
    args = parser.parse_args()

    # ---- Load ABC data ----
    abc_npz = load_npz(args.abc)
    meta = parse_meta(abc_npz)

    if "G0" not in abc_npz or "G" not in abc_npz:
        raise KeyError("ABC file must contain 'G0' (homogeneous) and 'G' (mixture) arrays.")

    G0 = np.array(abc_npz["G0"])
    G_mix = np.array(abc_npz["G"])

    beta = float(meta.get("mix_beta", 0.0))
    if abs(beta) < 1e-14:
        raise ValueError("meta_json['mix_beta'] is zero or missing – cannot reconstruct random-medium dictionary.")

    # True random-medium dictionary Gb
    G_true = (G_mix - G0) / beta

    # Grid sizes
    K = int(meta.get("K", G0.shape[1]))
    Nx = int(meta.get("Nx", int(round(np.sqrt(K)))))
    Nz = int(meta.get("Nz", K // Nx))

    # ---- Load recovered/ordered dictionary ----
    order_npz = load_npz(args.order)
    if "G_ordered" in order_npz:
        G_rec = np.array(order_npz["G_ordered"])
        dict_key = "G_ordered"
    elif "G_unordered" in order_npz:
        G_rec = np.array(order_npz["G_unordered"])
        dict_key = "G_unordered"
    else:
        raise KeyError("ORDER file must contain 'G_ordered' or 'G_unordered' array.")

    # Sanity checks on shapes
    if G0.shape != G_true.shape or G0.shape != G_rec.shape:
        raise ValueError(
            f"Shape mismatch: G0{G0.shape}, G_true{G_true.shape}, G_rec{G_rec.shape} "
            "must all have the same shape."
        )

    N_rows, K_cols = G0.shape
    if K_cols != K:
        print(f"[Warning] meta K={K} but dictionary has {K_cols} columns; using K={K_cols} from data.")
        K = K_cols

    # ---- Normalize columns (recommended for imaging) ----
    G0 = normalize_columns(G0)
    G_true = normalize_columns(G_true)
    G_rec = normalize_columns(G_rec)

    # ---- Precompute Gram-like matrices ----
    # W_true[:, j] = <g_true_i, g_true_j>  (i along rows, j along cols)
    # W_hom[:,  j] = <g0_i,    g_true_j>
    # W_rec[:,  j] = <grec_i,  g_true_j>
    print("Computing Gram matrices ...")
    W_true = G_true.conj().T @ G_true
    W_hom = G0.conj().T @ G_true
    W_rec = G_rec.conj().T @ G_true

    # ---- Loop over all grid points j0 ----
    p_true = np.zeros(K, dtype=float)
    p_hom = np.zeros(K, dtype=float)
    p_rec = np.zeros(K, dtype=float)
    mse_hom = np.zeros(K, dtype=float)
    mse_rec = np.zeros(K, dtype=float)

    for j0 in range(K):
        # Imaging vectors (absolute value, normalized to max = 1)
        I_true = np.abs(W_true[:, j0])
        I_hom = np.abs(W_hom[:, j0])
        I_rec = np.abs(W_rec[:, j0])

        # Avoid division by zero
        if I_true.max() > 0:
            I_true /= I_true.max()
        if I_hom.max() > 0:
            I_hom /= I_hom.max()
        if I_rec.max() > 0:
            I_rec /= I_rec.max()

        # Precision = value at correct pixel j0
        p_true[j0] = I_true[j0]
        p_hom[j0] = I_hom[j0]
        p_rec[j0] = I_rec[j0]

        # Full-image MSE vs true
        mse_hom[j0] = np.mean((I_hom - I_true) ** 2)
        mse_rec[j0] = np.mean((I_rec - I_true) ** 2)

    # ---- Some summary numbers ----
    improvement_mask = p_rec > p_hom
    frac_improve = float(np.mean(improvement_mask))
    mean_delta = float(np.mean(p_rec - p_hom))
    mean_p_hom = float(np.mean(p_hom))
    mean_p_rec = float(np.mean(p_rec))

    print("=== Imaging precision statistics ===")
    print(f"Grid: Nx={Nx}, Nz={Nz}, K={K}")
    print(f"Dictionaries shape: N_rows={N_rows}, K={K}")
    print(f"Recovered dictionary key: {dict_key}")
    print(f"Mean precision (homogeneous): {mean_p_hom:.3f}")
    print(f"Mean precision (recovered) : {mean_p_rec:.3f}")
    print(f"Mean delta p_rec - p_hom   : {mean_delta:.3f}")
    print(f"Fraction of points where recovered > homogeneous: {frac_improve:.3f}")

    # ---- Save raw stats to NPZ ----
    out_npz = f"{args.out_prefix}_stats.npz"
    np.savez(
        out_npz,
        p_true=p_true,
        p_hom=p_hom,
        p_rec=p_rec,
        mse_hom=mse_hom,
        mse_rec=mse_rec,
        meta=np.array(meta, dtype=object),
    )
    print(f"Saved precision / MSE arrays to: {out_npz}")

    # ---- Make plots ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Histogram of precision values
    ax = axes[0, 0]
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(p_hom, bins=bins, alpha=0.6, label="homogeneous")
    ax.hist(p_rec, bins=bins, alpha=0.6, label="recovered")
    ax.set_xlabel("precision at true pixel (I(j0) after norm)")
    ax.set_ylabel("count")
    ax.set_title("Precision distribution")
    ax.legend()

    # Histogram of delta precision
    ax = axes[0, 1]
    delta_p = p_rec - p_hom
    bins_delta = np.linspace(delta_p.min(), delta_p.max(), 25)
    ax.hist(delta_p, bins=bins_delta, alpha=0.8)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Δ precision = p_rec - p_hom")
    ax.set_ylabel("count")
    ax.set_title("Improvement distribution (recovered - homogeneous)")

    # Scatter plot p_hom vs p_rec
    ax = axes[1, 0]
    ax.scatter(p_hom, p_rec, s=10, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("p_hom")
    ax.set_ylabel("p_rec")
    ax.set_title("Per-point precision (hom vs rec)")

    # Histogram of MSE differences
    ax = axes[1, 1]
    delta_mse = mse_hom - mse_rec   # positive => recovered has lower MSE
    bins_mse = np.linspace(delta_mse.min(), delta_mse.max(), 25)
    ax.hist(delta_mse, bins=bins_mse, alpha=0.8)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Δ MSE = mse_hom - mse_rec")
    ax.set_ylabel("count")
    ax.set_title("Image MSE improvement (positive = recovered better)")

    plt.tight_layout()
    out_png = f"{args.out_prefix}_plots.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved plots to: {out_png}")

    return mean_delta , mean_p_rec , mean_p_hom



if __name__ == "__main__":
    main()
