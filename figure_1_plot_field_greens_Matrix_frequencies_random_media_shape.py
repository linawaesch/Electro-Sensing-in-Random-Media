# Merged: multi-frequency sensing matrix + random-G toggle + external inclusion module
# -------------------------------------------------------------------------------
# - Builds three frequency-dependent sensing matrices G^{(f)} with 2D Helmholtz kernel.
# - Optional random-medium simulation: USE_RANDOM_G adds smooth complex row/column fluctuations
#   + small additive complex noise (separable approximation).
# - Inclusion is delegated to inclusions.py (either random connected shape or a circle).
# - Computes Δu via sensing-matrix dipole dictionary: Δy ≈ - D_j M(ω) ∇H(z).
# - Saves outputs in the folder where this script resides.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# External inclusion helpers
from inclusions import random_inclusion_mask, circle_inclusion_mask

TWOPI = 2.0 * np.pi
EPS = 1e-12

# SciPy Hankel for Helmholtz Green's function
try:
    from scipy.special import hankel1
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ------------------------
# Background utilities (quasi-static, unchanged)
# ------------------------
def grad_G_2d(rx, ry):
    r2 = rx*rx + ry*ry + EPS
    gx = (1.0 / TWOPI) * rx / r2
    gy = (1.0 / TWOPI) * ry / r2
    return gx, gy

def hess_G_2d_at_point(rx, ry):
    r2 = rx*rx + ry*ry + EPS
    factor = 1.0 / (TWOPI * r2*r2)
    Hxx = factor * (r2 - 2.0*rx*rx)
    Hxy = factor * (     - 2.0*rx*ry)
    Hyx = Hxy
    Hyy = factor * (r2 - 2.0*ry*ry)
    return np.array([[Hxx, Hxy],[Hyx, Hyy]])

def background_potential_dipole(X, Y, z0, p0, sigma0):
    rx = X - z0[0]
    ry = Y - z0[1]
    gx, gy = grad_G_2d(rx, ry)
    return (p0[0]*gx + p0[1]*gy) / sigma0

def grad_H_at_z(z, z0, p0, sigma0):
    dzx = z[0] - z0[0]
    dzy = z[1] - z0[1]
    Hess = hess_G_2d_at_point(dzx, dzy)
    return (Hess @ np.asarray(p0)) / sigma0

def polarization_tensor_disk(a, sigma0, k_complex):
    alpha = (k_complex - sigma0) / (k_complex + sigma0)
    return (TWOPI * (a**2) * alpha) * np.eye(2, dtype=complex)

# ------------------------
# Frequency-aware Green's functions (2D Helmholtz)
# ------------------------
def helmholtz_G_2d(r, k):
    if not SCIPY_OK:
        raise RuntimeError("scipy.special.hankel1 (SciPy) is required for 2D Helmholtz.")
    return 0.25j * hankel1(0, k * r)

def laplace_G_2d(r):
    return (1.0 / TWOPI) * np.log(r + EPS)

# ------------------------
# Sensing-matrix utilities (frequency-aware)
# ------------------------
def build_G(obs_pts, src_x, src_y, omega=None, c0=1500.0, use_helmholtz=True):
    """
    Return G (Nobs×K) and indexer dict. If use_helmholtz, G depends on omega via k=omega/c0.
    """
    Xs, Ys = np.meshgrid(src_x, src_y, indexing='xy')
    src_pts = np.column_stack([Xs.ravel(), Ys.ravel()])
    diff = obs_pts[:, None, :] - src_pts[None, :, :]
    r = np.sqrt(np.sum(diff**2, axis=2)) + EPS

    if use_helmholtz:
        if omega is None:
            raise ValueError("omega must be provided when use_helmholtz=True.")
        k = omega / c0
        G = helmholtz_G_2d(r, k)
    else:
        G = laplace_G_2d(r)

    def idx(ix, iy): return iy * len(src_x) + ix
    dx = (src_x[1] - src_x[0]) if len(src_x) > 1 else 1.0
    dy = (src_y[1] - src_y[0]) if len(src_y) > 1 else 1.0
    return G.astype(complex), {'idx': idx, 'nx': len(src_x), 'ny': len(src_y),
                               'dx': dx, 'dy': dy, 'src_pts': src_pts}

def D_block_from_G(G, indexer, j_star):
    """
    Central-difference dipole dictionary at source index j_star:
    D = [∂_{z_x} g, ∂_{z_y} g] ∈ C^{N×2}, using G's neighboring columns.
    """
    nx, ny = indexer['nx'], indexer['ny']
    dx, dy = indexer['dx'], indexer['dy']
    ix = j_star % nx
    iy = j_star // nx

    def col(i, j): return G[:, indexer['idx'](i, j)]
    ixm = max(ix-1, 0); ixp = min(ix+1, nx-1)
    iym = max(iy-1, 0); iyp = min(iy+1, ny-1)

    if ixp != ixm:
        d_x = (col(ixp, iy) - col(ixm, iy)) / ((ixp - ixm) * dx)
    else:
        d_x = (col(ixp, iy) - col(ix, iy)) / ((ixp - ix) * dx) if ixp > ix else \
              (col(ix, iy) - col(ixm, iy)) / ((ix - ixm) * dx)

    if iyp != iym:
        d_y = (col(ix, iyp) - col(ix, iym)) / ((iyp - iym) * dy)
    else:
        d_y = (col(ix, iyp) - col(ix, iy)) / ((iyp - iy) * dy) if iyp > iy else \
              (col(ix, iy) - col(ix, iym)) / ((iy - iym) * dy)

    return np.column_stack([d_x, d_y])

def G_columns_at(obs_pts, src_pts_subset, omega=None, c0=1500.0, use_helmholtz=True):
    diff = obs_pts[:, None, :] - src_pts_subset[None, :, :]
    r = np.sqrt(np.sum(diff**2, axis=2)) + EPS
    if use_helmholtz:
        if omega is None:
            raise ValueError("omega must be provided when use_helmholtz=True.")
        k = omega / c0
        return helmholtz_G_2d(r, k)
    else:
        return laplace_G_2d(r)

# ------------------------
# Random-medium utilities (restored)
# ------------------------
def _gaussian_kernel_1d(sigma_px):
    sigma = float(max(sigma_px, 1e-6))
    radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius+1, dtype=float)
    k = np.exp(-0.5 * (x / sigma)**2)
    k /= k.sum()
    return k

def _smooth1d_noise(n, sigma_px, rng):
    if sigma_px <= 0:
        v = rng.standard_normal(n)
        return (v - v.mean()) / (v.std() + 1e-12)
    k = _gaussian_kernel_1d(sigma_px)
    v = rng.standard_normal(n)
    sm = np.convolve(v, k, mode='same')
    sm -= sm.mean(); sm /= (sm.std() + 1e-12)
    return sm

def _smooth2d_noise(ny, nx, sigma_y_px, sigma_x_px, rng):
    if sigma_x_px <= 0 and sigma_y_px <= 0:
        Z = rng.standard_normal((ny, nx))
        Z -= Z.mean(); Z /= (Z.std() + 1e-12)
        return Z
    from numpy import apply_along_axis, convolve
    ky = _gaussian_kernel_1d(max(sigma_y_px, 0.0))
    kx = _gaussian_kernel_1d(max(sigma_x_px, 0.0))
    Z = rng.standard_normal((ny, nx))
    Zx = apply_along_axis(lambda m: np.convolve(m, kx, mode='same'), axis=1, arr=Z)
    Zy = apply_along_axis(lambda m: np.convolve(m, ky, mode='same'), axis=0, arr=Zx)
    Zy -= Zy.mean(); Zy /= (Zy.std() + 1e-12)
    return Zy

def build_random_G(obs_pts, src_x, src_y, omega, c0, use_helmholtz, rand_cfg):
    """
    1) Baseline G via build_G(...)
    2) Multiply by smooth complex fluctuations row/column-wise (separable)
    3) Add small complex additive noise
    Returns (G_rand, indexer, col_complex_mult)
    """
    G0, indexer = build_G(obs_pts, src_x, src_y, omega=omega, c0=c0, use_helmholtz=use_helmholtz)
    N, K = G0.shape
    nx, ny = indexer['nx'], indexer['ny']

    seed_f = int(rand_cfg['seed'] + (0 if omega is None else int(omega))) % (2**31-1)
    rng = np.random.default_rng(seed_f)

    amp_col_field = _smooth2d_noise(ny, nx, rand_cfg['col_corr_len_px'], rand_cfg['col_corr_len_px'], rng)
    phi_col_field = _smooth2d_noise(ny, nx, rand_cfg['col_corr_len_px'], rand_cfg['col_corr_len_px'], rng)

    amp_col = rand_cfg['col_amp_std']   * amp_col_field.ravel()
    phi_col = rand_cfg['col_phase_std'] * phi_col_field.ravel()
    col_complex_mult = (1.0 + amp_col) * np.exp(1j * phi_col)  # (K,)

    amp_row = rand_cfg['row_amp_std']   * _smooth1d_noise(N, rand_cfg['row_corr_len_px'], rng)
    phi_row = rand_cfg['row_phase_std'] * _smooth1d_noise(N, rand_cfg['row_corr_len_px'], rng)
    row_complex_mult = (1.0 + amp_row) * np.exp(1j * phi_row)  # (N,)

    G_rand = G0 * (row_complex_mult[:, None] * col_complex_mult[None, :])

    add_rel = float(rand_cfg['add_noise_rel'])
    if add_rel > 0:
        scale = np.median(np.abs(G0)) + 1e-12
        eta = (rng.standard_normal(G0.shape) + 1j * rng.standard_normal(G0.shape)) / np.sqrt(2.0)
        G_rand = G_rand + add_rel * scale * eta

    return G_rand.astype(complex), indexer, col_complex_mult.astype(complex)

def apply_random_to_subset(base_cols, col_indices, col_complex_mult, n_rows, rand_cfg, omega):
    """
    Apply the same column multipliers to a subset of columns and regenerate row multipliers
    for the observation grid in which 'base_cols' live.
    """
    seed_f = int(rand_cfg['seed'] + (0 if omega is None else int(omega)) + 7919) % (2**31-1)
    rng = np.random.default_rng(seed_f)

    amp_row = rand_cfg['row_amp_std']   * _smooth1d_noise(n_rows, rand_cfg['row_corr_len_px'], rng)
    phi_row = rand_cfg['row_phase_std'] * _smooth1d_noise(n_rows, rand_cfg['row_corr_len_px'], rng)
    row_complex_mult = (1.0 + amp_row) * np.exp(1j * phi_row)  # (n_rows,)

    col_mult_subset = col_complex_mult[np.asarray(col_indices)]  # (q,)
    return base_cols * (row_complex_mult[:, None] * col_mult_subset[None, :])

# ------------------------
# Configuration
# ------------------------
USE_HELMHOLTZ = True
c0 = 1500.0

# Toggle random G (random-medium simulation) — restored
USE_RANDOM_G = True
RAND_CFG = {
    'seed'            : 20251012,
    'col_amp_std'     : 0.10,
    'col_phase_std'   : 0.50,
    'col_corr_len_px' : 8,
    'row_amp_std'     : 0.05,
    'row_phase_std'   : 0.25,
    'row_corr_len_px' : 10,
    'add_noise_rel'   : 0.02
}

# Toggle inclusion type (external module)
USE_RANDOM_INCLUSION = True
RAND_INCL_PIXELS = 300
INCL_CONNECTIVITY = 4
INCL_SEED = 12345

# Frequencies (Hz)
freqs_hz = [500.0, 1000.0, 2000.0]
omegas = [TWOPI * f for f in freqs_hz]

# Material/admittivity
sigma0 = 1.0
sigma  = 5.0
epsilon = 5e-3

# Dipole source
z0 = np.array([-0.8, -0.6])
p0 = np.array([1.0, 0.0])

# Visualization grid
Nx, Ny = 260, 200
x_min, x_max = -1.2, 1.2
y_min, y_max = -1.0, 1.0
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
X, Y = np.meshgrid(x, y, indexing='xy')
vis_pts = np.column_stack([X.ravel(), Y.ravel()])

# Receivers
Nr_x, Nr_y = 22, 8
rx = np.linspace(-0.8, 0.8, Nr_x)
ry = np.linspace(-0.9, -0.6, Nr_y)
RX, RY = np.meshgrid(rx, ry, indexing='xy')
rec_pts = np.column_stack([RX.ravel(), RY.ravel()])

# Source-grid for G
Sx = np.linspace(x_min, x_max, 121)
Sy = np.linspace(y_min, y_max,  91)

# Output dir
try:
    OUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUT_DIR = Path.cwd()

# Equipotential-line density
N_LEVELS = 40

# ------------------------
# Inclusion from external module
# ------------------------
if USE_RANDOM_INCLUSION:
    mask_incl, boundary_mask, (xc, yc), a_equiv = random_inclusion_mask(
        X, Y, n_pixels=RAND_INCL_PIXELS, connectivity=INCL_CONNECTIVITY, seed=INCL_SEED
    )
    z = np.array([xc, yc])     # center from random shape
    a = float(a_equiv)         # equivalent disk radius
else:
    z = np.array([0.25, 0.10])
    a = 0.08
    mask_incl, boundary_mask = circle_inclusion_mask(X, Y, center=tuple(z), radius=a)

# ------------------------
# Background (unchanged)
# ------------------------
H_bg = background_potential_dipole(X, Y, z0, p0, sigma0)
gradH_z = grad_H_at_z(z, z0, p0, sigma0)

# ------------------------
# Multi-frequency loop: build G^{(f)}, compute Δu^{(f)}, and plot
# ------------------------
for f_hz, omega in zip(freqs_hz, omegas):

    # PT and equivalent dipole at this frequency
    k_complex = sigma + 1j * epsilon * omega
    M = polarization_tensor_disk(a, sigma0, k_complex)
    m_eff = M @ gradH_z

    # Sensing matrix at receivers (baseline or random)
    if USE_RANDOM_G:
        G_rec, indexer, col_complex_mult = build_random_G(
            rec_pts, Sx, Sy, omega=omega, c0=c0, use_helmholtz=USE_HELMHOLTZ, rand_cfg=RAND_CFG
        )
    else:
        G_rec, indexer = build_G(rec_pts, Sx, Sy, omega=omega, c0=c0, use_helmholtz=USE_HELMHOLTZ)
        col_complex_mult = None

        # --- SAVE the inhomogeneous/random sensing matrix G to CSVs (one per frequency) ---
    if USE_RANDOM_G:
        out_base = OUT_DIR / f"G_random_receivers_f{int(f_hz)}"
        # Save real and imaginary parts separately for robust CSV parsing
        np.savetxt(out_base.with_suffix(".real.csv"), G_rec.real, delimiter=",")
        np.savetxt(out_base.with_suffix(".imag.csv"), G_rec.imag, delimiter=",")

        # (Optional but handy) Save tiny metadata to help the reader script rebuild indices
        # nx, ny = number of source-grid points in x/y used to build G’s columns
        # dx, dy = grid spacings; rows = receivers in the same order used when building G
        with open(out_base.with_suffix(".meta.csv"), "w") as m:
            m.write("key,value\n")
            m.write(f"nx,{indexer['nx']}\n")
            m.write(f"ny,{indexer['ny']}\n")
            m.write(f"dx,{indexer['dx']}\n")
            m.write(f"dy,{indexer['dy']}\n")
            m.write(f"n_receivers,{G_rec.shape[0]}\n")
            m.write(f"n_columns,{G_rec.shape[1]}\n")


    # Inclusion index on source-grid + D at receivers
    def nearest_src_index(pt, Sx, Sy, indexer):
        ix = int(np.clip(np.searchsorted(Sx, pt[0]), 1, len(Sx)-1))
        iy = int(np.clip(np.searchsorted(Sy, pt[1]), 1, len(Sy)-1))
        if abs(Sx[ix]-pt[0]) > abs(Sx[ix-1]-pt[0]): ix -= 1
        if abs(Sy[iy]-pt[1]) > abs(Sy[iy-1]-pt[1]): iy -= 1
        return indexer['idx'](ix, iy), ix, iy

    j_star, ix_star, iy_star = nearest_src_index(z, Sx, Sy, indexer)
    D_rec = D_block_from_G(G_rec, indexer, j_star)
    DeltaU_rec = - (D_rec @ m_eff)   # (Nrec,) — not directly plotted, but available if needed

    # Visualization D: compute only the needed columns
    nx, ny = indexer['nx'], indexer['ny']; dx_s, dy_s = indexer['dx'], indexer['dy']
    def ij_from_j(j): return (j % nx, j // nx)
    ix, iy = ij_from_j(j_star)
    ixm = max(ix-1, 0); ixp = min(ix+1, nx-1)
    iym = max(iy-1, 0); iyp = min(iy+1, ny-1)

    src_pts_all = indexer['src_pts']
    subset_ids = [
        indexer['idx'](ixp, iy),
        indexer['idx'](ixm, iy),
        indexer['idx'](ix, iyp),
        indexer['idx'](ix, iym)
    ]
    subset_src = src_pts_all[subset_ids, :]

    G_vis_subset_base = G_columns_at(
        vis_pts, subset_src, omega=omega, c0=c0, use_helmholtz=USE_HELMHOLTZ
    )

    if USE_RANDOM_G:
        # Apply same column multipliers with a fresh row multiplier for the visualization grid
        G_vis_subset = apply_random_to_subset(
            G_vis_subset_base, subset_ids, col_complex_mult, n_rows=vis_pts.shape[0],
            rand_cfg=RAND_CFG, omega=omega
        )
    else:
        G_vis_subset = G_vis_subset_base

    # Central differences at visualization points
    if ixp != ixm:
        dvis_x = (G_vis_subset[:, 0] - G_vis_subset[:, 1]) / ((ixp - ixm) * dx_s)
    else:
        center = G_columns_at(vis_pts, src_pts_all[indexer['idx'](ix,iy)][None,:],
                              omega=omega, c0=c0, use_helmholtz=USE_HELMHOLTZ)[:,0]
        if USE_RANDOM_G:
            center = apply_random_to_subset(
                center.reshape(-1,1), [indexer['idx'](ix,iy)], col_complex_mult,
                n_rows=vis_pts.shape[0], rand_cfg=RAND_CFG, omega=omega
            )[:,0]
        dvis_x = (G_vis_subset[:, 0] - center) / ((ixp - ix) * dx_s) if ixp > ix else \
                 (center - G_vis_subset[:, 1]) / ((ix - ixm) * dx_s)

    if iyp != iym:
        dvis_y = (G_vis_subset[:, 2] - G_vis_subset[:, 3]) / ((iyp - iym) * dy_s)
    else:
        center = G_columns_at(vis_pts, src_pts_all[indexer['idx'](ix,iy)][None,:],
                              omega=omega, c0=c0, use_helmholtz=USE_HELMHOLTZ)[:,0]
        if USE_RANDOM_G:
            center = apply_random_to_subset(
                center.reshape(-1,1), [indexer['idx'](ix,iy)], col_complex_mult,
                n_rows=vis_pts.shape[0], rand_cfg=RAND_CFG, omega=omega
            )[:,0]
        dvis_y = (G_vis_subset[:, 2] - center) / ((iyp - iy) * dy_s) if iyp > iy else \
                 (center - G_vis_subset[:, 3]) / ((iy - iym) * dy_s)

    D_vis = np.column_stack([dvis_x, dvis_y])
    DeltaU = - (D_vis @ m_eff)
    DeltaU = DeltaU.reshape(Ny, Nx)

    # Total field & masking
    U = H_bg + DeltaU
    U_masked = np.where(mask_incl, np.nan, U)
    H_masked = np.where(mask_incl, np.nan, H_bg)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    levels = np.linspace(np.nanpercentile(np.real(U_masked), 2),
                         np.nanpercentile(np.real(U_masked), 98), N_LEVELS)
    plt.contour(X, Y, np.real(U_masked), levels=levels, linewidths=1.0)
    plt.contour(X, Y, np.real(H_masked), levels=levels, linestyles='dashed', linewidths=0.8)

    # Inclusion boundary (works for any shape)
    by, bx = np.nonzero(boundary_mask)
    if by.size > 0:
        plt.scatter(x[bx], y[by], s=2, marker='s', label='Inclusion boundary')

    # Dipole & receivers
    plt.plot(z0[0], z0[1], marker='o', markersize=6, label='Dipole source')
    plt.scatter(rec_pts[:,0], rec_pts[:,1], s=12, marker='s', label='Receivers')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    title_tag = "random G" if USE_RANDOM_G else "baseline G"
    plt.title(f'Equipotential lines Re(u), f = {f_hz:.0f} Hz  ({title_tag})')
    plt.legend(loc='upper right'); plt.tight_layout()

    # Save to script directory
    fig_path = OUT_DIR / f'equipotential_lines_f{int(f_hz)}.png'
    fig.savefig(fig_path, dpi=160)
    np.save(OUT_DIR / f'G_rec_f{int(f_hz)}.npy', G_rec)
    np.save(OUT_DIR / f'DeltaU_f{int(f_hz)}.npy', DeltaU)
    np.save(OUT_DIR / f'U_total_f{int(f_hz)}.npy', U)

    plt.show(block=False); plt.pause(0.2)

# Save frequencies used
np.save(OUT_DIR / 'frequencies_used_hz.npy', np.array(freqs_hz, dtype=float))
plt.show()
print(f"Saved outputs in: {OUT_DIR}")
print(f"USE_RANDOM_G = {USE_RANDOM_G} | USE_RANDOM_INCLUSION = {USE_RANDOM_INCLUSION}")
