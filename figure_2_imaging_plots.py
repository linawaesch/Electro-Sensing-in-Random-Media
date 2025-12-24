#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_imaging_plots.py
=====================

Create the three imaging plots (homogeneous, recovered, true) from NPZ outputs.

Imaging functional (paper Eq. (16)):
    I(x_i; x_j0) = | g_hat(x_i)^* g_true(x_j0) |

- g_true(x_j0) is the TRUE random-medium Green's vector at grid index j0
- g_hat is taken from: (1) G0 (homogeneous), (2) recovered/ordered dictionary, (3) G_true itself

Inputs (default names work out-of-the-box if you ran snapshot_generation.py + ordering):
    --abc snapshot_generation_data.npz
    --recovered widl_order_out.npz  (falls back to widl_stepE_out.npz if not given)
    --j0    source index (default: center of grid)
    --out   output directory (default: .)
    --triptych  also save 3-panel figure

"""

import argparse, os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------- utils --------------------------------

def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path, allow_pickle=True)

def parse_meta(npz_dict):
    meta = {}
    if "meta_json" in npz_dict:
        raw = npz_dict["meta_json"]
        try:
            s = raw.item() if hasattr(raw, "item") else str(raw)
            meta = json.loads(s)
        except Exception:
            pass
    return meta

def normalize_columns(G):
    G = np.array(G)
    norms = np.linalg.norm(G, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return G / norms

def choose_source_index(nx, nz, j0=None):
    if j0 is None:
        # default: grid center
        j0 = (nz // 2) * nx + (nx // 2)
    if not (0 <= j0 < nx * nz):
        raise ValueError(f"j0={j0} is out of range [0, {nx*nz-1}]")
    return int(j0)

def try_get_true_random(abc_npz, meta):
    # Prefer explicit keys first
    for key in ["Gb", "G_true", "G_rand", "G_random", "Gb_stack", "G_rand_stack", "G_b", "Gtrue"]:
        if key in abc_npz:
            return np.array(abc_npz[key]), key

    # Else reconstruct from mixture G and G0
    if "G0" in abc_npz and "G" in abc_npz:
        beta = float(meta.get("mix_beta", 0.0))
        if abs(beta) < 1e-14:
            raise KeyError("mix_beta is zero or missing; cannot reconstruct Gb from G and G0.")
        Gb = (np.array(abc_npz["G"]) - np.array(abc_npz["G0"])) / beta
        return Gb, "reconstructed_from_(G-G0)/beta"

    raise KeyError("Could not find TRUE random-medium dictionary in ABC npz.")

def unique_sizes_from_grid_points(grid_points):
    # Infer Nx, Nz from grid coordinates if present
    xs = np.unique(np.asarray(grid_points)[:, 0])
    zs = np.unique(np.asarray(grid_points)[:, 1])
    return len(xs), len(zs)

def match_rows_for_inner_product(G_im, G_true, sub_mask, Nf):
    """
    Ensure G_im and G_true have the same number of rows by restricting
    the full matrix to the subarray rows if needed.
    """
    N1, K = G_im.shape
    N2, _ = G_true.shape
    if N1 == N2:
        return G_im, G_true

    if sub_mask is None or Nf is None:
        raise ValueError("Row mismatch but no sub_mask/Nf available to reconcile.")

    Nr = int(len(sub_mask))
    sub_idx = np.where(np.asarray(sub_mask).astype(bool))[0]
    rows = []
    for m in range(Nf):
        base = m * Nr
        rows.extend(list(base + sub_idx))

    # If G_im is subarray and G_true is full, down-select true
    if N1 == len(rows) and N2 == Nf * Nr:
        return G_im, G_true[rows, :]
    # If G_true is subarray and G_im is full, down-select im
    if N2 == len(rows) and N1 == Nf * Nr:
        return G_im[rows, :], G_true

    raise ValueError(f"Cannot reconcile row counts: G_im rows={N1}, G_true rows={N2}, expected sub rows={len(rows)}.")

def find_recovered_dict(recovered_npz, N_expected=None, K_expected=None):
    """
    Heuristics to find the recovered/ordered dictionary in a Step-E or order NPZ.
    Returns (G_rec, perm_or_None, key_used)
    """
    perm = None
    for k in ["perm", "perm_to_grid", "order", "perm_vec", "pi"]:
        if k in recovered_npz:
            perm = np.asarray(recovered_npz[k]).astype(int)
            break

    # Candidate 2-D arrays that could be the dictionary
    skip = {"meta_json", "receivers", "grid_points", "freqs", "Y", "X", "G", "G0", "Gb",
            "sub_mask", "submask", "Zhat", "Z_hat", "coords", "positions"}
    candidates = []
    for k in recovered_npz.files:
        if k in skip:
            continue
        A = np.asarray(recovered_npz[k])
        if A.ndim == 2 and A.size > 0 and (np.iscomplexobj(A) or np.issubdtype(A.dtype, np.floating)):
            candidates.append((k, A))

    # Prefer exact (N,K) match
    for k, A in candidates:
        if N_expected is not None and K_expected is not None and A.shape == (N_expected, K_expected):
            return A, perm, k
    # Next: any with K columns
    for k, A in candidates:
        if K_expected is not None and A.shape[1] == K_expected:
            return A, perm, k
    # Otherwise: largest 2-D array
    if candidates:
        candidates.sort(key=lambda t: t[1].size, reverse=True)
        k, A = candidates[0]
        return A, perm, k

    raise KeyError("Could not find a recovered dictionary-like array in the recovered NPZ.")

def imaging_vector(G_im, g_data):
    # I(x_i; x_j0) = | G_im^H g_data |
    I = np.abs(G_im.conj().T @ g_data)
    m = I.max()
    if m > 0:
        I /= m
    return I

def save_single_image(I2d, nx, nz, j0, title, out_path):
    ix = int(j0 % nx)
    iy = int(j0 // nx)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    h = ax.imshow(I2d, origin="lower", vmin=0.0, vmax=1.0)
    ax.scatter(ix, iy, marker="x", s=60, linewidths=2)
    ax.set_title(title)
    fig.colorbar(h, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def save_triptych(images, titles, nx, nz, j0, out_path):
    ix = int(j0 % nx)
    iy = int(j0 // nx)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, I2d, title in zip(axes, images, titles):
        h = ax.imshow(I2d, origin="lower", vmin=0.0, vmax=1.0)
        ax.scatter(ix, iy, marker="x", s=60, linewidths=2)
        ax.set_title(title)
        fig.colorbar(h, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

# ------------------------------- main ---------------------------------

def main():
    p = argparse.ArgumentParser(description="Make imaging plots from NPZ outputs (Eq.16).")
    p.add_argument("--abc", default="snapshot_generation_data.npz", help="ABC data NPZ (from snapshot_generation.py).")
    p.add_argument("--recovered", default=None, help="Recovered/order NPZ (e.g., order_out.npz).")
    p.add_argument("--j0", type=int, default=60, help="Source index (default: center).")
    p.add_argument("--out", default=".", help="Output directory.")
    p.add_argument("--triptych", action="store_true", help="Also save a 3-panel figure.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Load ABC file ---
    abc_npz = load_npz(args.abc)
    meta = parse_meta(abc_npz)

    # Required matrices: G0 and TRUE random-medium Gb
    if "G0" not in abc_npz:
        raise KeyError("ABC file must contain 'G0' (homogeneous stacked dictionary).")

    G0 = np.array(abc_npz["G0"])
    Gb, gb_key = try_get_true_random(abc_npz, meta)

    # Shapes
    N, K = Gb.shape
    # Grid size
    if "grid_points" in abc_npz:
        nx, nz = unique_sizes_from_grid_points(abc_npz["grid_points"])
    else:
        # fallback: use K if it's a square grid
        root = int(round(np.sqrt(K)))
        nx = root
        nz = K // root
    nx = int(meta.get("Nx", nx))
    nz = int(meta.get("Nz", nz))

    # Frequencies/Subarray info for row-matching if needed
    Nf = int(meta.get("Nf", 1))
    sub_mask = None
    if "sub_mask" in abc_npz:
        sub_mask = np.asarray(abc_npz["sub_mask"]).astype(bool)

    # --- Load recovered dictionary ---
    rec_path = args.recovered
    if rec_path is None:
        # Choose a sensible default if not provided
        for cand in ["order_out.npz", "widl_stepE_out.npz"]:
            if os.path.exists(cand):
                rec_path = cand
                break
    if rec_path is None:
        raise FileNotFoundError("Recovered/order NPZ not provided and no default file found.")

    rec_npz = load_npz(rec_path)
    G_rec, perm, key_used = find_recovered_dict(rec_npz, N_expected=None, K_expected=K)

    # Apply permutation if provided (perm[i] is the grid index for recovered column i)
    if perm is not None and len(perm) == G_rec.shape[1]:
        # Reorder columns so they match the grid indexing
        P = np.argsort(perm)  # columns -> grid order
        G_rec = G_rec[:, P]

    # --- Normalize columns (recommended for Eq.16 comparisons) ---
    G0   = normalize_columns(G0)
    Gb   = normalize_columns(Gb)
    G_rec = normalize_columns(G_rec)

    # --- Ensure same number of rows for the inner product ---
    # We must have rows(G_im) == rows(Gb) for each imaging dictionary we use.
    # Handle subarray vs full-array mismatch via sub_mask.
    G0_m,  Gb_m  = match_rows_for_inner_product(G0,   Gb, sub_mask, Nf) if G0.shape[0]   != Gb.shape[0] else (G0, Gb)
    Grec_m, Gb_m = match_rows_for_inner_product(G_rec, Gb, sub_mask, Nf) if G_rec.shape[0] != Gb.shape[0] else (G_rec, Gb)

    # --- Choose source index and form data vector from TRUE random medium ---
    j0 = choose_source_index(nx, nz, args.j0)
    g_data = Gb_m[:, j0]  # column j0 of TRUE dictionary

    # --- Build the three images ---
    I_hom  = imaging_vector(G0_m,  g_data).reshape(nz, nx)
    I_rec  = imaging_vector(Grec_m, g_data).reshape(nz, nx)
    I_true = imaging_vector(Gb_m,  g_data).reshape(nz, nx)

    # --- Save figures ---
    out_hom  = os.path.join(args.out, "imaging_homogeneous.png")
    out_rec  = os.path.join(args.out, "imaging_recovered.png")
    out_true = os.path.join(args.out, "imaging_true_random.png")

    save_single_image(I_hom,  nx, nz, j0, "Imaging with HOMOGENEOUS dictionary", out_hom)
    save_single_image(I_rec,  nx, nz, j0, "Imaging with RECOVERED dictionary",   out_rec)
    save_single_image(I_true, nx, nz, j0, "Imaging with TRUE random-medium dictionary", out_true)

    if args.triptych:
        out_trip = os.path.join(args.out, "imaging_triptych.png")
        save_triptych(
            [I_hom, I_rec, I_true],
            ["Imaging with HOMOGENEOUS dictionary",
             "Imaging with RECOVERED dictionary",
             "Imaging with TRUE random-medium dictionary"],
            nx, nz, j0, out_trip
        )

    # --- Console summary ---
    print("=== Imaging plots saved ===")
    print(f"ABC file         : {args.abc}")
    print(f"Recovered file   : {rec_path}  (dict key used: {key_used}, perm: {'yes' if perm is not None else 'no'})")
    print(f"G0 shape         : {G0.shape}")
    print(f"Gb shape         : {Gb.shape}  (source key: {gb_key})")
    print(f"G_rec shape      : {G_rec.shape}")
    print(f"Grid (nx, nz)    : ({nx}, {nz}), K={nx*nz}, j0={j0} -> (ix={j0%nx}, iy={j0//nx})")
    print(f"Outputs          : {out_hom}, {out_rec}, {out_true}" + (f", and {out_trip}" if args.triptych else ""))

if __name__ == "__main__":
    main()
