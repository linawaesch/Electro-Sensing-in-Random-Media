# Electro-Sensing-in-Random-Media
The code base implements the full pipeline: it simulates multi frequency snapshots in random media, learns and reorders the sensing matrix, and then uses it for GPT reconstruction and shape classification experiments.

## File overview

- `snapshot_generation.py`  
  Builds the array and grid, constructs homogeneous and random sensing matrices, and generates multi frequency snapshots.

- `recover_unordered_G.py`  
  Learns an unordered sensing matrix from the snapshots using sparse coding and MOD style dictionary updates.

- `order.py`  
  Reorders the learned matrix using correlations on receiver subarrays and metric MDS, and returns an ordered sensing matrix.

- `inclusions.py`  
  Helper routines to define and place inclusions on the grid and to work with their masks.

- `parameter_tuner.py`  
  Runs parameter sweeps, stores results in CSV format, and is used for the tuning plots in the thesis.

- `figure_1_*.py`, `figure_2_*.py`, `figure_3_*.py`, `figure_4_*.py`,  
  `figure_5_*.py`, `figure_6_*.py`, `figure_10_*.py`, `figure_11_*.py`  
  Scripts that reproduce the figures from the thesis (field plots, optimisation curves, imaging quality, parameter studies, and classification accuracy).

## Typical workflow

1. Generate snapshots and sensing matrices with `snapshot_generation.py`.
2. Learn the unordered sensing matrix with `recover_unordered_G.py`.
3. Reorder the matrix with `order.py`.
4. Run the desired `figure_*.py` scripts to reproduce thesis plots or explore new parameter settings.
