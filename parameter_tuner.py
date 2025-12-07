#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Very simple hyperparameter loop for the 3-step pipeline + imaging precision.

- Assumes this file lives in the same folder as:
    snapshot_generation.py
    recover_unordered_G.py
    order.py
    imaging_percision_plots.py

- For each combination of parameters in the lists below, it will:
    1) run snapshot generation (with optional overrides)
    2) run recover_unordered_G
    3) run order.py (mode=normal)
    4) run imaging_percision_plots.main() to get mean_delta
    5) append one row to "simple_tuning_results.csv"

You mostly only need to touch the three *_PARAM_LISTS variables.
"""

import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

# Path where this script and the other scripts live
SCRIPT_DIR = Path(__file__).resolve().parent

# Make sure we can import the other scripts as modules
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import snapshot_generation as sg
import recover_unordered_G as rec
import imaging_percision_plots as ipp
import order  # imported just to make sure it's in the same folder


# ----------------------------------------------------------------------
# 1) Hyperparameters to try (EDIT THIS PART)
# ----------------------------------------------------------------------
# For each script you give a "list of lists".
# Each inner list has the form: ["param_name", [value1, value2, ...]]
# If the outer list is empty, defaults are used.
# ---------------------------------------------------------------------
# Snapshot generation hyperparameters (snapshot_generation.py / Config)
# ---------------------------------------------------------------------
# We vary only medium / frequency parameters that really change the
# effective sensing matrix, not factor or grid size.
SNAPSHOT_PARAM_LISTS = [

    # Lower edge of the frequency band (relative to central frequency).
    # This changes the overall frequency content while keeping Nf fixed.
    ["band_low", [0.50, 0.64, 0.80]],

    # Number of random Fourier features used to synthesise the medium.
    # Controls how well the random field statistics are approximated.
    ["RFF_Q", [64, 128, 256]],
]
# If you want to keep the search even smaller you can later drop one of
# these parameters (typically band_low or RFF_Q).


# ---------------------------------------------------------------------
# Dictionary learning / recovery hyperparameters (recover_unordered_G.py)
# ---------------------------------------------------------------------
# We tune the GeLMA regularisation/step parameter and the number of
# inner iterations; outer_iters is left fixed as discussed.
RECOVER_PARAM_LISTS = [
    # GeLMA step / regularisation parameter (default 1.2 in the script).
    # Smaller values → more conservative updates / stronger ℓ1 penalty,
    # larger values → more aggressive updates and sparser codes.
    ["alpha", [0.8, 1.2, 1.6]],

    # Number of GeLMA iterations per outer loop (default 300).
    # Too small: under–converged sparse codes; too large: expensive
    # and potentially overfitting to noise.
    ["gelma_iters", [150, 300, 450]],
]


# ---------------------------------------------------------------------
# Ordering / graph-embedding hyperparameters (order.py)
# ---------------------------------------------------------------------
# These are the main knobs for the adjacency graph and anchor scheme.
ORDER_PARAM_LISTS = [
    # k-nearest neighbours in the receiver-graph used for the embedding.
    # Too small → graph may be disconnected; too large → graph becomes
    # dense and distances less informative.
    ["k", [4, 6, 8]],

    # Fraction of receivers used in each random subarray when building
    # the adjacency information.
    ["subarray_fraction", [0.4, 0.6, 0.8]],

    # Number of anchor receivers used to fix the global permutation.
    ["n_anchors", [4, 6, 8]],
]

RESULTS_CSV_NAME = "simple_tuning_results.csv"


def expand_param_lists(param_lists):
    """
    Convert a list of lists of the form:
        [["name1", [v11, v12, ...]],
         ["name2", [v21, v22, ...]],
         ...]
    into a list of dicts over all combinations:
        [{"name1": v11, "name2": v21},
         {"name1": v11, "name2": v22},
         ...]
    If param_lists is empty, return [ {} ] (single 'no-change' config).
    """
    if not param_lists:
        return [{}]

    names = []
    values_lists = []
    for inner in param_lists:
        if len(inner) != 2:
            raise ValueError(
                f"Each inner list must look like ['param_name', [v1, v2, ...]], "
                f"but got {inner!r}"
            )
        name, vals = inner
        names.append(str(name))
        values_lists.append(list(vals))

    configs = []
    for combo in itertools.product(*values_lists):
        cfg = dict(zip(names, combo))
        configs.append(cfg)
    return configs


# ----------------------------------------------------------------------
# 2) Stage runners
# ----------------------------------------------------------------------

def run_snapshot_generation(snap_params, save_path="snapshot_generation_data.npz"):
    """
    Re-implement snapshot_generation.main() but with the ability to override
    Config fields via snap_params.

    snap_params is a dict like {"mix_beta": 0.2, "s": 8, ...}.
    """
    # Build config and apply overrides
    cfg = sg.Config()
    for name, value in snap_params.items():
        if not hasattr(cfg, name):
            raise AttributeError(
                f"snapshot_generation.Config has no attribute '{name}'. "
                f"Available: {', '.join(sorted(sg.Config.__dataclass_fields__.keys()))}"
            )
        setattr(cfg, name, value)
    cfg = cfg.finalize()

    # Geometry and matrices (this mirrors sg.main)
    geo = sg.build_geometry(cfg)

    G0_stack = sg.build_G0_stack(cfg, geo)
    G_rand_stack = sg.build_G_random_stack(cfg, geo)

    G_stack = G0_stack + cfg.mix_beta * G_rand_stack

    rng = geo["rng"]
    X, Y = sg.draw_sparse_X_and_Y(cfg, G_stack, rng)

    nu_full = sg.coherence(G_stack)
    G_sub = sg.build_subarray_stack(G_stack, cfg, geo)
    nu_sub = sg.coherence(G_sub)

    K = cfg.Nx * cfg.Nz
    M_req = K * math.log(K)
    M_ok = cfg.M > M_req

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
        G=G_stack,
        G0=G0_stack,
        X=X,
        receivers=geo["receivers"],
        grid_points=geo["grid_points"],
        freqs=geo["freqs"],
        sub_mask=geo["sub_mask"].astype(np.uint8),
        meta_json=np.array(meta_json, dtype=object),
    )

    print("=== snapshot_generation (custom) ===")
    print(f"K={K}, Nr={cfg.Nr}, Nf={cfg.Nf}, M={cfg.M}, s={cfg.s}, beta={cfg.mix_beta}")
    print(f"Band: [{geo['meta']['fmin']:.3f}, {geo['meta']['fmax']:.3f}] with Nf={cfg.Nf}")
    print(f"Coherence ν (full stacked array): {nu_full:.4f}")
    print(f"Coherence ν (half-aperture subarray): {nu_sub:.4f}")
    print(f"Sample complexity: M > K log K ? {'YES' if M_ok else 'NO'} (M={cfg.M}, K log K≈{M_req:.1f})")
    print(f"Saved ABC NPZ to: {save_path}")

    return save_path


def run_recover_unordered(abc_path, rec_params, save_path="recover_unordered_g_out.npz"):
    """
    Call recover_unordered_G.run(...) with possible overrides from rec_params.

    rec_params keys should match the argparse names, but with '-' removed, e.g.:
        'outer_iters' for --outer-iters
        'gelma_iters' for --gelma-iters
        'batch_cols' for --batch-cols
        'dtype' for --dtype
        'alpha', 'seed' as usual.
    """
    # Defaults from the script
    kwargs = {
        "outer_iters": rec_params.get("outer_iters", 8),
        "gelma_iters": rec_params.get("gelma_iters", 300),
        "alpha": rec_params.get("alpha", 1.2),
        "batch_cols": rec_params.get("batch_cols", None),
        "dtype": rec_params.get("dtype", "complex128"),
        "seed": rec_params.get("seed", 123),
    }

    print("=== recover_unordered_G ===")
    print(f"Using abc_path={abc_path}, save_path={save_path}")
    print(f"Params: {kwargs}")

    rec.run(
        data_path=str(abc_path),
        save_path=str(save_path),
        outer_iters=kwargs["outer_iters"],
        gelma_iters=kwargs["gelma_iters"],
        alpha=kwargs["alpha"],
        batch_cols=kwargs["batch_cols"],
        dtype=kwargs["dtype"],
        seed=kwargs["seed"],
    )

    return save_path


def run_order(abc_path, unordered_path, ord_params, save_path="order_out.npz"):
    """
    Run order.py via subprocess in mode='normal', passing hyperparameters
    using its command line flags.
    """
    order_script = SCRIPT_DIR / "order.py"

    cmd = [
        sys.executable,
        str(order_script),
        "--mode", "normal",
        "--abc", str(abc_path),
        "--unordered", str(unordered_path),
        "--save", str(save_path),
    ]

    # Map our parameter names to CLI flags
    mapping = {
        "r": "--r",
        "k": "--k",
        "subarray_fraction": "--subarray-fraction",
        "n_anchors": "--n-anchors",
        "subarray_mode": "--subarray-mode",
        "perm_seed": "--perm-seed",
        "debug_noise": "--debug-noise",
        "dtype": "--dtype",
    }
    for name, flag in mapping.items():
        if name in ord_params:
            cmd.extend([flag, str(ord_params[name])])

    print("=== order.py ===")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))

    return save_path


def run_imaging_precision(abc_path, order_path):
    """
    Call imaging_percision_plots.main() and return its mean_delta.

    We don't pass any CLI args here, so the script uses its defaults:
        --abc snapshot_generation_data.npz
        --order order_out.npz
        --out-prefix imaging_precision

    That matches the filenames we are using above.
    """
    print("=== imaging_percision_plots ===")
    mean_delta , mean_p_rec , mean_p_hom= ipp.main()
    print(f"mean_delta returned by imaging_percision_plots.main(): {mean_delta}")
    return float(mean_delta), float(mean_p_rec), float(mean_p_hom)


# ----------------------------------------------------------------------
# 3) CSV helper
# ----------------------------------------------------------------------

def append_result_to_csv(csv_path, row_dict):
    """
    Append a row to CSV, writing a header if the file does not yet exist.
    """
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    fieldnames = ["run_index", "snap_params", "rec_params", "ord_params", "mean_delta", "mean_p_rec", "mean_p_hom"]

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# ----------------------------------------------------------------------
# 4) Main loop
# ----------------------------------------------------------------------

def main():
    snap_configs = expand_param_lists(SNAPSHOT_PARAM_LISTS)
    rec_configs = expand_param_lists(RECOVER_PARAM_LISTS)
    ord_configs = expand_param_lists(ORDER_PARAM_LISTS)

    csv_path = SCRIPT_DIR / RESULTS_CSV_NAME

    run_index = 0
    for snap_params in snap_configs:
        for rec_params in rec_configs:
            for ord_params in ord_configs:
                run_index += 1
                print()
                print("=" * 80)
                print(f"RUN {run_index}")
                print("  snapshot params:", snap_params)
                print("  recover params :", rec_params)
                print("  order params   :", ord_params)

                # 1) Snapshot generation
                abc_path = SCRIPT_DIR / "snapshot_generation_data.npz"
                run_snapshot_generation(snap_params, save_path=abc_path)

                # 2) Recover unordered G
                recover_path = SCRIPT_DIR / "recover_unordered_g_out.npz"
                run_recover_unordered(abc_path, rec_params, save_path=recover_path)

                # 3) Order
                order_path = SCRIPT_DIR / "order_out.npz"
                run_order(abc_path, recover_path, ord_params, save_path=order_path)

                # 4) Imaging precision (returns mean_delta)
                mean_delta,  mean_p_rec , mean_p_hom = run_imaging_precision(abc_path, order_path)

                row = {
                    "run_index": run_index,
                    "snap_params": repr(snap_params),
                    "rec_params": repr(rec_params),
                    "ord_params": repr(ord_params),
                    "mean_delta": mean_delta,
                    "mean_p_rec": mean_p_rec,
                    "mean_p_hom": mean_p_hom
                }
                append_result_to_csv(csv_path, row)

    print()
    print("All runs finished.")
    print(f"Results written to: {csv_path}")


if __name__ == "__main__":
    main()
