# inclusions.py
# Utilities to build inclusions on a Cartesian grid.
# - circle_inclusion_mask: circle mask from a geometric center & radius
# - random_inclusion_mask: connected random shape with a given pixel count (4- or 8-neighborhood)
# Returns:
#   mask          : (Ny,Nx) bool, True inside inclusion
#   boundary_mask : (Ny,Nx) bool, True on boundary pixels (4-neighborhood)
#   center_xy     : (xc, yc) float tuple (centroid in physical coordinates)
#   a_equiv       : float, equivalent disk radius = sqrt(area/π)

import numpy as np

def _grid_steps(X, Y):
    Ny, Nx = X.shape
    dx = (X[0, -1] - X[0, 0]) / max(Nx - 1, 1)
    dy = (Y[-1, 0] - Y[0, 0]) / max(Ny - 1, 1)
    return float(dx), float(dy)

def circle_inclusion_mask(X, Y, center=(0.0, 0.0), radius=0.1):
    """Return circle mask & boundary on the (X,Y) grid."""
    cx, cy = center
    mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
    boundary_mask = _boundary_from_mask(mask)
    return mask, boundary_mask

def random_inclusion_mask(X, Y, n_pixels=300, connectivity=4, seed=None, start=None):
    """
    Build a connected random shape of exactly n_pixels using a random region-growing BFS.
    - connectivity: 4 (Von Neumann) or 8 (Moore)
    - start: optional (row, col) to seed the growth; if None, chosen randomly
    Returns (mask, boundary_mask, (xc, yc), a_equiv).
    """
    Ny, Nx = X.shape
    if n_pixels <= 0 or n_pixels > Ny * Nx:
        raise ValueError("n_pixels must be in [1, Ny*Nx].")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    rng = np.random.default_rng(seed)

    # Choose a random start if not provided
    if start is None:
        i0 = rng.integers(0, Ny)
        j0 = rng.integers(0, Nx)
    else:
        i0, j0 = map(int, start)
        if not (0 <= i0 < Ny and 0 <= j0 < Nx):
            raise ValueError("start out of grid bounds.")

    mask = np.zeros((Ny, Nx), dtype=bool)
    mask[i0, j0] = True

    frontier = [(i0, j0)]
    dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]
    dirs8 = dirs4 + [(1,1),(1,-1),(-1,1),(-1,-1)]
    dirs = dirs4 if connectivity == 4 else dirs8

    # Region grow until desired size
    while mask.sum() < n_pixels and frontier:
        # pick a random frontier pixel
        fi, fj = frontier[rng.integers(len(frontier))]
        # find neighbors not yet included
        candidates = []
        for di, dj in dirs:
            ni, nj = fi + di, fj + dj
            if 0 <= ni < Ny and 0 <= nj < Nx and not mask[ni, nj]:
                candidates.append((ni, nj))
        if not candidates:
            # remove exhausted frontier node
            frontier.remove((fi, fj))
            continue
        # add one random neighbor
        ni, nj = candidates[rng.integers(len(candidates))]
        mask[ni, nj] = True
        frontier.append((ni, nj))

    # If frontier got exhausted (rare), greedily add neighbors of boundary to hit target size
    if mask.sum() < n_pixels:
        boundary = np.argwhere(_boundary_from_mask(mask))
        # Flatten all neighbors of boundary pixels and add those not yet in mask
        for bi, bj in boundary[rng.permutation(len(boundary))]:
            for di, dj in dirs:
                ni, nj = bi + di, bj + dj
                if 0 <= ni < Ny and 0 <= nj < Nx and not mask[ni, nj]:
                    mask[ni, nj] = True
                    if mask.sum() >= n_pixels:
                        break
            if mask.sum() >= n_pixels:
                break

    # Final check: exact size
    if mask.sum() > n_pixels:
        # randomly drop extras from interior (rare)
        extras = np.argwhere(mask)
        drop = mask.sum() - n_pixels
        to_drop = extras[rng.permutation(len(extras))[:drop]]
        for (ii, jj) in to_drop:
            mask[ii, jj] = False

    # Boundary mask (4-neighborhood)
    boundary_mask = _boundary_from_mask(mask)

    # Centroid in physical coords
    xc = float(X[mask].mean()) if mask.any() else float(X.mean())
    yc = float(Y[mask].mean()) if mask.any() else float(Y.mean())

    # Equivalent disk radius from area = n_pixels * ΔxΔy
    dx, dy = _grid_steps(X, Y)
    area = n_pixels * dx * dy
    a_equiv = np.sqrt(area / np.pi)

    return mask, boundary_mask, (xc, yc), float(a_equiv)

def _boundary_from_mask(mask):
    """4-neighborhood boundary: pixels in 'mask' with at least one 4-neighbor outside."""
    Ny, Nx = mask.shape
    up    = np.zeros_like(mask); up[1: , :] = mask[:-1, :]
    down  = np.zeros_like(mask); down[:-1, :] = mask[1: , :]
    left  = np.zeros_like(mask); left[:, 1: ] = mask[:, :-1]
    right = np.zeros_like(mask); right[:, :-1] = mask[:, 1: ]
    interior_4 = mask & up & down & left & right
    boundary = mask & (~interior_4)
    return boundary
