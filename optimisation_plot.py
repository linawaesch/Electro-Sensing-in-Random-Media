#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter sweep driver for the WIDL pipeline.

For each value in SWEEP_VALUES:
  1. Regenerate snapshot_generation_data.npz with that parameter in snapshot_generation.Config.
  2. Run widl_stepD_final.main() to produce widl_stepD_out.npz.
  3. Run widl_step_E.main() (normal mode) to order G.
  4. Read "normal_correct_count / K" from the RETURN VALUE of widl_step_E.main().

Then plot: x = parameter values, y = percentage of correctly ordered columns.
"""

# ------------------------- TUNABLES -------------------------

# Name of the Config field in snapshot_generation.Config you want to sweep.
# Examples: "Nr", "Nf", "sigma_tilde", "mix_beta", ...
SWEEP_PARAM_NAME = "seed"

# Values to test on the x-axis.
SWEEP_VALUES = [2,3,4]

# Paths for the intermediate NPZ files (keep consistent with your other scripts)
ABC_SAVE_PATH = "snapshot_generation_data.npz"
D_SAVE_PATH   = "recover_unordered_g_out.npz"   # widl_stepD_final default
E_SAVE_PATH   = "order_out.npz"   # widl_step_E default

# -----------------------------------------------------------

import json
import math

import numpy as np
import matplotlib.pyplot as plt

import snapshot_generation as sg
import recover_unordered_G
import order


def run_abc_with_overrides(**overrides):
    """
    Recreate snapshot_generation.main() but with the ability to override Config fields.
    Writes snapshot_generation_data.npz with the same structure as snapshot_generation.main().
    """
    # Build config with overrides and finalize M, etc.
    cfg = sg.Config(**overrides).finalize()

    # Geometry
    geo = sg.build_geometry(cfg)

    # ----- BUILD G_stack EXACTLY AS IN snapshot_generation.main() -----
    # Homogeneous Green's stack (G0) and random-medium stack (Gb)
    G0_stack      = sg.build_G0_stack(cfg, geo)
    G_rand_stack  = sg.build_G_random_stack(cfg, geo)
    # Mixture used for data generation
    G_stack       = G0_stack + cfg.mix_beta * G_rand_stack
    # --------------------------------------------------------

    # Generate sparse X and Y
    rng = geo["rng"]
    X, Y = sg.draw_sparse_X_and_Y(cfg, G_stack, rng)

    # Coherence diagnostics
    nu_full = sg.coherence(G_stack)
    G_sub   = sg.build_subarray_stack(G_stack, cfg, geo)
    nu_sub  = sg.coherence(G_sub)

    # Sample-complexity check
    K = cfg.Nx * cfg.Nz
    M_req = K * math.log(K)
    M_ok  = cfg.M > M_req

    # Meta (same pattern as snapshot_generation.main)
    meta = geo["meta"].copy()
    meta.update({
        "Nx": cfg.Nx,
        "Nz": cfg.Nz,
        "K": K,
        "Nr": cfg.Nr,
        "Nf": cfg.Nf,
        "s": cfg.s,
        "M": cfg.M,
        "nu_full": nu_full,
        "nu_sub": nu_sub,
        "M_req_K_log_K": M_req,
        "M_ok": bool(M_ok),
    })
    meta_json = json.dumps(meta, indent=2)

    # Save NPZ in the exact format Step D/E expect
    np.savez(
        ABC_SAVE_PATH,
        Y=Y,
        G=G_stack,
        X=X,
        receivers=geo["receivers"],
        grid_points=geo["grid_points"],
        freqs=geo["freqs"],
        sub_mask=geo["sub_mask"].astype(np.uint8),
        meta_json=np.array(meta_json, dtype=object),
    )

    return cfg, meta


def run_full_pipeline_for_value(param_name, value):
    """
    Single experiment:
      - regenerate ABC data with param_name=value
      - run Step D
      - run Step E
      - get percentage of correctly ordered columns from widl_step_E.main()
    Returns (value, num_correct, K).
    """
    print(f"\n=== Running pipeline for {param_name} = {value} ===")

    # 1) Regenerate snapshot_generation_data.npz with the new config
    overrides = {param_name: value}
    cfg, meta = run_abc_with_overrides(**overrides)

    # 2) Run Step D (dictionary learning). Uses snapshot_generation_data.npz internally.
    recover_unordered_G.main()

    # 3) Run Step E (ordering). Uses snapshot_generation_data.npz + widl_stepD_out.npz.
    #    Its main() returns (num_correct, K) in normal mode.
    num_correct, K = order.main()

    if num_correct is None:
        raise RuntimeError(
            "widl_step_E.main() did not return a normal-mode correctness count. "
            "Make sure it's running in normal mode and has the return statement."
        )

    print(f"Result: {num_correct}/{K} correctly ordered columns "
          f"({100.0 * num_correct / K:.1f}%)")

    return value, num_correct, K


def main():
    x_vals = []
    y_pct_vals = []

    for val in SWEEP_VALUES:
        x, num_correct, K = run_full_pipeline_for_value(SWEEP_PARAM_NAME, val)
        x_vals.append(x)
        y_pct_vals.append(100.0 * num_correct / K)

    x_vals = np.array(x_vals, dtype=float)
    y_pct_vals = np.array(y_pct_vals, dtype=float)

    # --------- Plot ---------
    plt.figure()
    plt.plot(x_vals, y_pct_vals, marker="o")
    plt.xlabel(SWEEP_PARAM_NAME)
    plt.ylabel("Correctly ordered columns [%]")
    plt.title(f"Ordering performance vs {SWEEP_PARAM_NAME}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
