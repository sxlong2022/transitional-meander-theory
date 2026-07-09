# Transitional Meander Theory: Nonlinear Mode Coupling in Variable-Width Channels

This repository contains the reproducibility package for the WRR manuscript on transitional meander theory in variable-width channels. It provides the linear stability solver, weakly nonlinear Landau coefficient calculator, 2D Landlab morphodynamic benchmark, and phase-plane visualization scripts used to generate the results and figures in the paper.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21273728.svg)](https://doi.org/10.5281/zenodo.21273728)

**GitHub Repository**: [https://github.com/sxlong2022/transitional-meander-theory](https://github.com/sxlong2022/transitional-meander-theory)  
**Zenodo Archive & DOI**: [https://doi.org/10.5281/zenodo.21273728](https://doi.org/10.5281/zenodo.21273728)

### Recommended Citation
> Song, X., Xu, H., & Bai, Y. (2026). Reproducibility Package for Asymmetric Mutual Enhancement of Central and Alternate Bars in Variable-Width Channels (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.21273728

## Mathematical Framework Overview

The theoretical core couples a depth-averaged 2D shallow-water equation (SWE) solver with a weakly nonlinear amplitude-equation framework to study bar instability in transitional meandering channels with width oscillations.

### Linear Stability Solver
The linear stability analysis (`code/linear_solver.py`) assembles a 4x4 block matrix for the depth-averaged state vector **X** = [u, v, eta, zb]^T, where:
- u, v = streamwise and transverse velocity perturbations
- eta = free-surface elevation perturbation
- zb = bed-elevation perturbation (Exner equation)

The solver uses Chebyshev spectral collocation in the transformed cross-stream coordinate zeta = n / b(s) in [-1, 1]. The eigenvalue problem A X = omega B X yields the linear growth rates and modal structures for alternate bars (m=1) and central bars (m=2).

Key parameters:
- alpha: streamwise wavenumber k
- beta: width-to-depth aspect ratio B/H
- Cf: friction coefficient
- Fr: Froude number
- N_curv: curvature parameter N = b * C
- sigma_width: width gradient parameter sigma = b'/b

### Weakly Nonlinear Landau Equations
The nonlinear calculator (`code/landau_calculator.py` and `code/landau_base.py`) computes the cubic Landau coefficients by adjoint projection onto the linear eigenmodes. The coupled amplitude equations for central-bar mode A and alternate-bar mode B read:

```
dA/dt = sigma_a * A - gamma_aa * |A|^2 * A - gamma_ab * |B|^2 * A
dB/dt = sigma_b * B - gamma_bb * |B|^2 * B - gamma_ba * |A|^2 * B
```

For subcritical central-bar modes (Re(gamma_AA) < 0), a fifth-order stabilizing term g5 * |A|^4 * A (g5 = 1e-6) is included to bound the growth. The resulting stabilized mixed-state amplitudes (|A_eq| approx 20.42, |B_eq| approx 10.06) represent a 5.8-fold catalytic amplification of B compared to its isolated state.

The adjoint projection requires solving:
1. The direct and adjoint eigenproblems for the linear operator
2. Mean-flow (k=0) and second-harmonic (k=2*alpha) forced responses via least-squares solves
3. Cubic nonlinear forcing terms projected onto the adjoint modes

### 2D Morphodynamic Benchmark
The Landlab benchmark (`code/landlab_sweep.py`) provides a physics-based cross-check of the weakly nonlinear theory. It solves 2D shallow-water hydraulics with Landlab's OverlandFlow component and updates the bed with a finite-difference Exner equation using the Meyer-Peter and Muller (MPM) bedload formula. Width varies along x as b(x) = B0 * (1 + sigma * cos(k_width * x)), with nodes outside the channel masked as closed boundaries.

The benchmark extracts equilibrium central-bar and alternate-bar amplitudes by standard deviation of the detrended centreline bed elevation and the left-minus-right asymmetry, respectively.

## Codebase Structure

```
transitional-meander-theory/
  code/
    linear_solver.py              # Depth-averaged 2D SWE + Exner stability solver
    landau_base.py                # Base LandauCalculator class (Chebyshev grid, adjoint solve)
    landau_calculator.py          # LandauCoupledCalculator: adjoint projection for coupled modes
    landlab_sweep.py              # 2D Landlab morphodynamic sweep for variable-width channels
    diagnostics.py                # Symmetry and diagnostic metric calculations
    figure_utils.py               # Figure styling and publication export utilities
    fig01_conceptual_sketch.py    # Plots schematic diagrams (Figure 1)
    fig02_phase_diagram.py        # Plots linear growth rates vs sigma (Figure 2)
    fig03_eigenmode_structures.py # Plots transverse eigenmode profiles (Figure 3)
    fig04_phase_plane.py          # Plots amplitude phase-plane trajectories (Figure 4)
    fig05_mixed_state.py          # Reconstructs 2D bed topography (Figure 5)
    fig06_gamma_BA_curve.py        # Plots cross-coupling coefficient vs sigma (Figure 6)
    fig07_mechanism_schematic.py  # Plots coupling mechanism schematic (Figure 7)
  data/
    landlab_sweep_results.csv              # Pre-computed Landlab benchmark results
    garcia_lugo_2015_parameters.csv        # Digitized experimental run parameters (Garcia Lugo et al., 2015)Locally
    compare_confinement.py                 # Script to map confinement ratios to theoretical sigma
  README.md
```

### File Descriptions

| File | Purpose |
|------|---------|
| `code/linear_solver.py` | Assembles the 4x4 block SWE + Exner operator on a Chebyshev grid. Returns (A, B) matrices for the generalized eigenvalue problem. |
| `code/landau_base.py` | Computes Chebyshev differentiation matrix, solves the direct and adjoint eigenproblems, and constructs quadratic/cubic nonlinear forcing vectors for mean-flow and second-harmonic modes. |
| `code/landau_calculator.py` | Extends `landau_base.py` to compute the full set of coupled Landau coefficients (gamma_aa, gamma_ab, gamma_ba, gamma_bb) via adjoint projection. |
| `code/landlab_sweep.py` | Runs the 2D morphodynamic benchmark for a list of width-gradient sigma values. Outputs equilibrium amplitudes to a CSV file. |
| `code/diagnostics.py` | Calculates symmetry metrics and profiles of alternate and central bars from 2D topography datasets. |
| `code/fig01_conceptual_sketch.py` | Script to generate the conceptual planform and cross-section profiles (Figure 1). |
| `code/fig02_phase_diagram.py` | Solves the linear stability matrices across a range of sigma and plots the linear growth rates (Figure 2). |
| `code/fig03_eigenmode_structures.py` | Extracts and plots the transverse profiles of central and alternate bar eigenmodes (Figure 3). |
| `code/fig04_phase_plane.py` | Integrates the coupled Landau amplitude equations with a fifth-order stabilizing term and plots trajectories from multiple initial conditions converging to the mixed-state equilibrium (Figure 4). |
| `code/fig05_mixed_state.py` | Reconstructs and plots the 2D bed topography at the mixed-state fixed point (Figure 5). |
| `code/fig06_gamma_BA_curve.py` | Computes the Landau coefficients across a range of sigma and plots the cross-coupling coefficient $\mathrm{Re}(\gamma_{BA})$ to highlight the cross-enhancement window (Figure 6). |
| `code/fig07_mechanism_schematic.py` | Script to plot the schematic of the mean flow distortion and cross-enhancement mechanism (Figure 7). |
| `data/landlab_sweep_results.csv` | Tabulated results from the Landlab sweep (sigma, simulated_A_eq, simulated_B_eq, simulated_ratio). |
| `data/garcia_lugo_2015_parameters.csv` | Digitized experimental parameters (run ID, valley width, braiding index, morphology) for comparison with the theoretical width-gradient mapping. |
| `data/compare_confinement.py` | Prints the experimental run table and maps valley-width confinement ratios to equivalent theoretical sigma values. |
## Installation Instructions

### Prerequisites
- Python 3.9 or later
- Conda environment manager (recommended)
- XeLaTeX (optional, for LaTeX compilation)

### Python Dependencies
Install the required packages using pip or conda:

```bash
# Using pip
pip install numpy scipy matplotlib pandas landlab

# Using conda (recommended for landlab)
conda create -n transitional-meander python=3.9
conda activate transitional-meander
pip install numpy scipy matplotlib pandas
conda install -c conda-forge landlab
```

### Verify Installation
After installation, verify that the linear solver loads correctly:

```bash
python -c "from code.linear_solver import assemble_swe_matrices; import numpy as np; print('Linear solver ready')"
```

Note: Because the scripts now live in a flat `code/` directory, they can be imported using the standard `from code.xxx import yyy` syntax when running from the repository root, or executed directly as scripts from within the `code/` directory.

## Run Guide

All commands should be executed from the repository root (`transitional-meander-theory/`).

### 1. Compute Coupled Landau Coefficients

The `LandauCoupledCalculator` in `code/landau_calculator.py` computes the cubic coefficients for a given physical parameter set and width gradient. Typical usage:

```bash
python -c "
from code.landau_calculator import LandauCoupledCalculator
import numpy as np

params = {
    'beta': 15.0,
    'Cf': 0.01,
    'Fr': 0.25,
    'Exner_Coeff': 0.01,
    'slope_coeff': 0.5,
    'transport_exponent_u': 3.0,
    'bed_slope': 0.005,
}

calc = LandauCoupledCalculator(params, N=64, L_domain=200.0, sigma_width=0.15)

# Solve linear eigenproblem for mode A (alternate bar, m=1)
omega_a, vec_a, _ = calc.solve_linear_adjoint(sigma_width=0.15, k_wavenumber=2*np.pi/200.0)
print(f'Mode omega: {omega_a}')

# Compute full coupled coefficients
gamma_aa, gamma_ab, gamma_ba, gamma_bb = calc.compute_full_coefficients(vec_a, omega_a, vec_a, omega_a, k=2*np.pi/200.0)
print(f'gamma_aa = {gamma_aa}')
print(f'gamma_ab = {gamma_ab}')
print(f'gamma_ba = {gamma_ba}')
print(f'gamma_bb = {gamma_bb}')
"
```

### 2. Run the Landlab Morphodynamic Sweep

The benchmark sweeps a set of width-gradient sigma values and writes equilibrium amplitudes to `code/landlab_sweep_results.csv`:

```bash
python code/landlab_sweep.py
```

Expected output after completion:
```
  Running sigma = 0.000 ...
    step 1: |A|=..., |B|=...
    ...
  Running sigma = 0.050 ...
  ...
Saved: code/landlab_sweep_results.csv
  sigma  simulated_A_eq  simulated_B_eq  simulated_ratio
  0.00         ...              ...               ...
  0.05         ...              ...               ...
  ...
```

The CSV is also pre-populated in `data/landlab_sweep_results.csv` for reference.

### 3. Compare with Garcia Lugo et al. (2015) Confinement Data

Map the experimental valley-width confinement ratios to theoretical width gradients:

```bash
python data/compare_confinement.py
```

This reads `data/garcia_lugo_2015_parameters.csv` and prints the mapped sigma values for the transitional (wandering) runs.

### 4. Generate Figure Plots

You can run the individual plotting scripts in `code/` to reproduce each figure in the paper. All generated PDF/PNG plots are output to the repository root directory by default:

```bash
# Figure 1: Conceptual planform and profile sketch
python code/fig01_conceptual_sketch.py

# Figure 2: Linear growth rates vs width gradient sigma
python code/fig02_phase_diagram.py

# Figure 3: Transverse eigenmode profile shapes
python code/fig03_eigenmode_structures.py

# Figure 4: Amplitude phase-plane trajectories and fixed point
python code/fig04_phase_plane.py

# Figure 5: Reconstructed 2D bed topography at mixed-state fixed point
python code/fig05_mixed_state.py

# Figure 6: Cross-coupling coefficient Re(gamma_BA) vs sigma
python code/fig06_gamma_BA_curve.py

# Figure 7: Physical coupling and mean flow distortion schematic
python code/fig07_mechanism_schematic.py
```

For each script, the vector PDF (suitable for LaTeX) and a raster PNG preview are generated.
## Expected Numerical Results

### Stabilized Mixed-State Amplitudes (Fifth-Order Stabilization)
| Quantity | Value |
|----------|-------|
| \|A_eq\| (central bar) | 20.42 |
| \|B_eq\| (alternate bar) | 10.06 |
| Catalytic amplification of B | 5.8-fold |

### Representative Landau Coefficients (Beta = 15, Sigma = 0.15)
| Coefficient | Value |
|-------------|-------|
| sigma_a | 0.3085 - 5.9616j |
| sigma_b | 0.1629 - 4.7062j |
| gamma_aa | -0.0030 + 0.0773j |
| gamma_bb | 0.0548 + 1.0238j |
| gamma_ab | 0.0137 + 0.4416j |
| gamma_ba | -0.0129 + 0.1475j |

These values should be kept synchronized across the Abstract, Results, Discussion, figures, and Conclusion sections of the manuscript.

## Theoretical Self-Review Notes

1. **Conservation**: The linear solver conserves mass (fluid and sediment) in the integrated sense; the block-structure boundary conditions (v = +/- sigma * u at banks, sediment flux matching) are implemented in `assemble_swe_matrices`.
2. **Scaling**: All lengths are non-dimensionalized by a reference depth D0 = 1.0 and half-width b0 = 1.0, so beta = B/H = 15 is dimensionless.
3. **Neutral Limit**: When sigma = 0 (straight channel, no width oscillation), the linear solver recovers distinct marginal thresholds for alternate bars (m=1) and central bars (m=2).
4. **Subcritical Stabilization**: The fifth-order term g5 * |A|^4 * A (g5 = 1e-6) bounds the growth of the subcritical central-bar mode. The phase-plane integrator (`fig04_phase_plane.py`) uses `solve_ivp` with tight tolerances (rtol=1e-8, atol=1e-10) to capture the slow approach to the mixed state.

## Reproducibility Checklist

- [ ] All Python dependencies installed in a fresh environment
- [ ] `code/linear_solver.py` imports without error
- [ ] `code/landau_calculator.py` completes an adjoint projection for the test case above
- [ ] `code/landlab_sweep.py` produces `code/landlab_sweep_results.csv`
- [ ] `code/fig04_phase_plane.py` produces `Fig4_phase_plane.pdf` and `Fig4_phase_plane.png`
- [ ] Numerical values in the CSV and printed coefficients match the manuscript tables within solver tolerance

## License

This reproducibility package is provided for academic peer review. Please cite the corresponding WRR manuscript when using these scripts.
