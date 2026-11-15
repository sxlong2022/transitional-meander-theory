# Transitional Meander Theory: Nonlinear Mode Coupling in Variable-Width Channels

Reproducibility package for the manuscript on asymmetric cross-enhancement of central
and alternate bars in variable-width channels. It contains the depth-averaged linear
stability solver, the weakly nonlinear Stuart--Landau coefficient calculator, a 2D
Landlab morphodynamic benchmark, and the scripts that generate every figure and
sensitivity table in the paper and its Supporting Information.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21273728.svg)](https://doi.org/10.5281/zenodo.21273728)

**GitHub Repository**: [https://github.com/sxlong2022/transitional-meander-theory](https://github.com/sxlong2022/transitional-meander-theory)
**Zenodo Archive & DOI**: [https://doi.org/10.5281/zenodo.21273728](https://doi.org/10.5281/zenodo.21273728)

### Recommended Citation

> Song, X., Xu, H., & Bai, Y. (2026). Reproducibility Package for Asymmetric Cross-Enhancement of Central and Alternate Bars in Variable-Width Channels (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.21273728

## Physical and Mathematical Framework

A transitional reach carries two bar modes at once: the symmetric central bar
(transverse mode number `m = 2`) and the antisymmetric alternate bar (`m = 1`). Linear
stability predicts that the central bar dominates the diverging part of a variable-width
channel, yet field observations show persistent alternate bars. The package quantifies
one mechanism behind that persistence: a finite-amplitude central bar actively promotes
the alternate bar through a negative cross-coupling coefficient.

### Dimensionless parameters

| Parameter | Symbol | Definition | Typical range |
|---|---|---|---|
| Aspect ratio | `beta` | `B_0 / D_r` | 8--20 |
| Froude number | `F` | `U_r / sqrt(g D_r)` | 0.3--0.8 |
| Friction coefficient | `Cf` | Chezy resistance | 0.005--0.02 |
| Width gradient | `sigma` | `(1/b) db/ds` | -0.1 to +0.2 |
| Sediment exponent | `p` | bedload power law | 2.5--3.5 |

Here `B_0` is the reference channel width, `D_r` the reference depth, `U_r` the
reference velocity, `b(s)` the local channel half-width, and `s` the streamwise
coordinate. The transverse coordinate is stretched as `zeta = n / b(s)` on `[-1, 1]`,
so a single Chebyshev collocation grid resolves a channel of varying width.

### Linear stability solver

`code/linear_solver.py` assembles the pencil `(A, B)` of the generalized eigenvalue
problem `A X = omega B X` for the perturbation state vector `X = [u, v, eta, zb]^T`:

- `u`, `v`: streamwise and transverse velocity perturbations,
- `eta`: free-surface elevation perturbation,
- `zb`: bed-elevation perturbation, advanced by the Exner equation.

The real part of `omega` is the linear growth rate. The four governing equations are
streamwise momentum, transverse momentum, fluid mass conservation, and sediment mass
conservation.

Bank boundary conditions are imposed at `zeta = +/- 1`. The flow condition is
`v' = +/- sigma u'`; the sediment condition follows from linearizing the vectorial
flux closure, which makes `dz_b'/dzeta = 0` at both banks. The parameter `r_s` is the
dimensionless transverse bed-slope deflection coefficient, approximately 0.5 for sand
and gravel beds.

### Weakly nonlinear amplitude equations

`code/landau_base.py` supplies the grid, the direct and adjoint eigenproblems, and the
second-order forcing assemblers. `code/landau_calculator.py` adds the coupled
projection. The solvability condition at third order yields

```
dA/dt = lambda_A A - gamma_AA |A|^2 A - gamma_AB |B|^2 A
dB/dt = lambda_B B - gamma_BB |B|^2 B - gamma_BA |A|^2 B
```

where `A` is the central-bar amplitude and `B` the alternate-bar amplitude. Each
coefficient is an adjoint projection of third-order forcing:

- **Self-saturation** (`gamma_AA`, `gamma_BB`): self mean-flow distortion plus the self
  second harmonic.
- **Cross-coupling** (`gamma_AB`, `gamma_BA`): the mean-flow distortion of one mode
  projected onto the other, plus the mixed sum harmonic at `k_A + k_B`.

The sign of `Re(gamma_BA)` carries the physics. Negative values mean the central bar
enhances alternate-bar growth; positive values mean competition.

The central-bar branch is subcritical (`Re(gamma_AA) < 0`), so the cubic truncation
alone does not bound it. A phenomenological fifth-order term `-g_5 |A|^4 A` with
`g_5 = 1e-6` closes the system. Its influence is quantified across four orders of
magnitude in Table S2.

### 2D morphodynamic benchmark

`code/landlab_sweep.py` solves 2D shallow-water hydraulics with Landlab's
`OverlandFlow` component and advances the bed with a finite-difference Exner equation
using the Meyer-Peter and Muller bedload formula. The half-width varies as

```
b(x) = B_0 [1 + a cos(k_w x)]
```

with modulation amplitude `a`, the direct counterpart of the theory's `sigma`. Nodes
outside the instantaneous channel are masked and treated as closed boundaries.

Two spatial proxies are extracted from the final detrended bed topography: `A_c`, the
centreline elevation standard deviation, and `A_a`, the cross-channel asymmetry
standard deviation. Both are bulk spatial standard deviations in metres, not projections
onto the theoretical eigenmodes, so the benchmark supports comparison of trends rather
than coefficient-by-coefficient agreement.

Two initialization protocols are available:

- `noise_only`: Gaussian random bed perturbation only, with no organised seed. This
  tests whether bar structure emerges spontaneously from a flat bed, following Crosato
  et al. (2012).
- `seeded`: antisymmetric alternate-bar seed superposed on Gaussian noise, used to
  confirm that the diagnostic proxies respond to a known transverse mode.

## Repository Layout

```
transitional-meander-theory/
  code/
    linear_solver.py                  # depth-averaged 2D SWE + Exner linear stability operator
    diagnostics.py                    # Chebyshev grid, mode-symmetry metrics, mode classifier
    landau_base.py                    # grid, adjoint eigensolve, second-order forcing
    landau_calculator.py              # coupled two-mode Landau coefficients
    landlab_sweep.py                  # 2D morphodynamic benchmark
    figure_utils.py                   # publication styling and figure export
    fig01_conceptual_sketch.py        # Figure 1
    fig02_phase_diagram.py            # Figure 2
    fig03_eigenmode_structures.py     # Figure 3
    fig04_phase_plane.py              # Figure 4
    fig05_mixed_state.py              # Figure 5
    fig06_gamma_BA_curve.py           # Figure 6
    fig07_mechanism_schematic.py      # Figure 7
    recalculate_all.py                # Landau coefficient table over (beta, sigma)
    run_sensitivity_sweeps.py         # Tables S1 to S3
    harmonic_frequency_sensitivity.py # harmonic operator convention check
    grid_convergence.py               # discretization sensitivity
    verify_equilibria.py              # closed-form equilibrium and stability audit
    verify_linear_tracking.py         # branch tracking and growth-rate ordering audit
  data/
    landau_sweep_results.csv          # 20 (beta, sigma) coefficient rows
    sensitivity_wavenumbers.csv       # Table S1
    sensitivity_g5.csv                # Table S2
    sensitivity_rs.csv                # Table S3
    grid_convergence_straight.csv     # straight-channel discretization residual
    grid_convergence_subharmonic.csv  # subharmonic-pair grid refinement
    harmonic_frequency_sensitivity.csv
    landlab_results_noise.csv         # Table 3 ensemble, 25 runs
    landlab_results_seeded.csv        # seeded protocol, 5 runs
    landlab_results_s20.csv           # a = 0.20 subset with modal decomposition
    landlab_results_legacy_coarse.csv # earlier coarse-grid sweep, dx = 2 m, 200 steps
    garcia_lugo_2015_parameters.csv   # digitized flume run table
    compare_confinement.py            # maps valley confinement to sigma
  figures/                            # generated output, created on demand
  requirements.txt
  README.md
```

### Module and script descriptions

| File | Purpose |
|---|---|
| `code/linear_solver.py` | Assembles the 4x4 block SWE plus Exner operator on a Chebyshev grid and returns `(A, B)`. Also provides `cheb_collocation`, the grid used by every other module. |
| `code/diagnostics.py` | Chebyshev grid helper, symmetry index, peak-axis locator, and a coarse mode classifier. |
| `code/landau_base.py` | Chebyshev grid, direct and adjoint eigensolves, mean-flow and second-harmonic forcing assemblers, second-order solves. |
| `code/landau_calculator.py` | Extends `landau_base` with the adjoint projection that yields `gamma_AA`, `gamma_AB`, `gamma_BA`, `gamma_BB`. |
| `code/landlab_sweep.py` | Runs the 2D benchmark over width modulations and writes scalar diagnostics to `data/`. |
| `code/fig01_conceptual_sketch.py` | Plan view and cross-sections of the problem configuration. |
| `code/fig02_phase_diagram.py` | Growth rates of the two bar branches across `sigma`. |
| `code/fig03_eigenmode_structures.py` | Transverse bed profiles of the two eigenmodes. |
| `code/fig04_phase_plane.py` | Phase-plane trajectories to the mixed-state node. |
| `code/fig05_mixed_state.py` | Reconstructed 2D bed topography of the mixed state. |
| `code/fig06_gamma_BA_curve.py` | `Re(gamma_BA)` across `sigma` with continuous branch tracking. |
| `code/fig07_mechanism_schematic.py` | Causal schematic of the cross-enhancement mechanism. |
| `code/recalculate_all.py` | Recomputes `data/landau_sweep_results.csv` over `beta` and `sigma`. |
| `code/run_sensitivity_sweeps.py` | Recomputes the three sensitivity tables. |
| `code/harmonic_frequency_sensitivity.py` | Compares the unshifted and frequency-shifted mixed harmonic operators. |
| `code/grid_convergence.py` | Recomputes both grid-convergence tables. |
| `code/verify_equilibria.py` | Closed-form equilibrium, radial Jacobian, and regularization scaling. |
| `code/verify_linear_tracking.py` | Overlap-tracked branch audit with eigenpair residuals and an equal-damping root solve. |
| `data/compare_confinement.py` | Prints the flume run table and maps valley confinement to an equivalent `sigma`. |

## Installation

Requires Python 3.9 or later.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux or macOS
source .venv/bin/activate

pip install -r requirements.txt
```

`landlab` installs more reliably from conda-forge:

```bash
conda create -n transitional-meander python=3.10
conda activate transitional-meander
conda install -c conda-forge numpy scipy matplotlib pandas landlab
```

The package was validated with Python 3.10.18, numpy 2.1.2, scipy 1.15.3,
matplotlib 3.10.3, pandas 2.3.1, and landlab 2.9.2.

### Verify the installation

```bash
python -c "import sys; sys.path.insert(0, 'code'); import linear_solver as L; D, y = L.cheb_collocation(16); A, B = L.assemble_swe_matrices(D, y, 2.0, 15.0, 0.01, 0.5, 0.0, 0.15, {'Exner_Coeff': 0.005}); print('solver OK, matrix', A.shape)"
```

Expected output:

```
solver OK, matrix (68, 68)
```

## Run Guide

All commands run from the repository root. Every script resolves its imports from
`code/` directly, so no package installation is required.

### 1. Reproduce every figure

Each script writes a vector PDF and a 300 dpi PNG into `figures/`, creating the
directory when absent.

```bash
python code/fig01_conceptual_sketch.py
python code/fig02_phase_diagram.py
python code/fig03_eigenmode_structures.py
python code/fig04_phase_plane.py
python code/fig05_mixed_state.py
python code/fig06_gamma_BA_curve.py
python code/fig07_mechanism_schematic.py
```

Figure 6 performs continuous eigenmode tracking across 55 width gradients and takes a
few seconds. The others finish in under a second each.

`fig04_phase_plane.py` also prints the equilibrium it plots:

```
Mixed state: |A_eq| = 20.42, |B_eq| = 10.06, |B_iso| = 1.72, amplification = 5.83x
```

### 2. Compute the coupled Landau coefficients

```bash
python -c "
import sys
sys.path.insert(0, 'code')
import numpy as np
import scipy.linalg as la
import linear_solver as L
from landau_calculator import LandauCoupledCalculator

params = {'beta': 15.0, 'Cf': 0.01, 'Fr': 0.5,
          'transport_exponent_u': 3.0, 'slope_coeff': 0.5, 'Exner_Coeff': 0.005}
D, y = L.cheb_collocation(16)
def mode(k, target):
    A, B = L.assemble_swe_matrices(D, y, k, 15.0, 0.01, 0.5, 0.0, 0.15, params)
    vals, vecs = la.eig(A, B)
    keep = np.isfinite(vals) & (np.real(vals) < 2.0)
    vals, vecs = vals[keep], vecs[:, keep]
    i = int(np.argmin(np.abs(vals - target)))
    return vecs[:, i], vals[i]
vec_a, om_a = mode(2.0, 0.3085 - 5.9616j)
vec_b, om_b = mode(1.0, 0.1629 - 4.7062j)
calc = LandauCoupledCalculator(params, 16, sigma_width=0.15)
g_aa, g_ab, g_ba, g_bb = calc.compute_full_coefficients(vec_a, om_a, vec_b, om_b, 2.0, 1.0)
print('gamma_AA =', np.round(g_aa, 4))
print('gamma_BB =', np.round(g_bb, 4))
print('gamma_AB =', np.round(g_ab, 4))
print('gamma_BA =', np.round(g_ba, 4))
"
```

Expected output:

```
gamma_AA = (-0.0724+0.1384j)
gamma_BB = (0.0548+1.0238j)
gamma_AB = (0.0029+0.8665j)
gamma_BA = (-0.0566+0.1593j)
```

These are the dual-wavenumber values, with independent direct and adjoint operators at
`k_A = 2.0` and `k_B = 1.0` and the mixed sum harmonic at `k_A + k_B = 3.0`. Omitting
the `k_b` argument switches to the illustrative single-carrier baseline of main-text
Table 2, where both adjoints use `k = k_B = 1.0` and the second harmonic sits at
`2 k_B = 2.0`.

### 3. Regenerate the coefficient and sensitivity tables

```bash
python code/recalculate_all.py
python code/run_sensitivity_sweeps.py
python code/grid_convergence.py
python code/harmonic_frequency_sensitivity.py
```

Each script rewrites the corresponding CSV in `data/`. `recalculate_all.py` covers
`beta in {15, 20, 25, 30}` against `sigma in {0, 0.05, 0.10, 0.15, 0.20}`.
`run_sensitivity_sweeps.py` covers `g_5`, `r_s`, and the wavenumber pairings.
`grid_convergence.py` covers `N in [16, 40]` for the straight channel and `N in [16, 64]`
for the long subharmonic pair.

### 4. Run the 2D morphodynamic benchmark

```bash
python code/landlab_sweep.py --mode noise_only --steps 2000 --dx 1.0 --out-dir data
```

The default configuration is the Table 3 ensemble: five width modulations and five
seeds, 25 runs of 2000 morphological steps on a 1 m grid. Each run prints its
diagnostics and writes a topography snapshot.

```bash
python code/landlab_sweep.py --mode seeded --tag _seeded --seeds 42 --out-dir data
```

reproduces the seeded protocol in `data/landlab_results_seeded.csv`.

The full ensemble takes about an hour on a laptop. For a quick check, reduce the step
count:

```bash
python code/landlab_sweep.py --steps 20 --sigmas 0.0 0.1 --seeds 1 --out-dir data
```

Inspect the full option set with:

```bash
python code/landlab_sweep.py --help
```

### 5. Compare with the Garcia Lugo et al. (2015) flume runs

```bash
python data/compare_confinement.py
```

This prints the digitized run table and the mapped width gradients. The mapping is

```
sigma_eq = 0.15 * (1 - W_v / 2.9)
```

where `W_v` is valley width and 2.9 m is the widest run. It is a monotone rescaling
that places the observed transitional runs inside the theoretical cross-enhancement
window, not a calibration of the theory against the experiments.

### 6. Run the verification audits

```bash
python code/verify_equilibria.py
python code/verify_linear_tracking.py
```

`verify_equilibria.py` solves the amplitude system in closed form and reports the
radial Jacobian at the equilibrium. `verify_linear_tracking.py` follows both bar
branches from `sigma = 0.15` in 0.001 steps, reports overlap quality and eigenpair
residuals, locates the equal-damping crossing, and asserts the growth-rate ordering
against the computed records.

## Expected Numerical Results

### Reference linear eigenvalues and Landau coefficients

Dual-wavenumber assembly at `beta = 15.0`, `sigma = 0.15`, `F = 0.5`, `Cf = 0.01`,
`p = 3`, `k_A = 2.0`, `k_B = 1.0`:

| Quantity | Value |
|---|---|
| `lambda_A` (central bar) | `0.3085 - 5.9616i` |
| `lambda_B` (alternate bar) | `0.1629 - 4.7062i` |
| `gamma_AA` | `-0.0724 + 0.1384i` |
| `gamma_BB` | `0.0548 + 1.0238i` |
| `gamma_AB` | `0.0029 + 0.8665i` |
| `gamma_BA` | `-0.0566 + 0.1593i` |

`Re(gamma_BA) = -0.0566 < 0` is the cross-enhancement result: the central bar promotes
alternate-bar growth. `Re(gamma_AB) = 0.0029 > 0` is weakly competitive in the opposite
direction, so the coupling is strongly asymmetric.

### Single-carrier baseline, main-text Table 2

Both adjoints assembled on `k = k_B = 1.0`, second harmonic at `2 k_B = 2.0`:

| Quantity | Value |
|---|---|
| `lambda_A` | `0.3085 - 5.9616i` |
| `lambda_B` | `0.1629 - 4.7062i` |
| `gamma_AA` | `-0.0030 + 0.0773i` |
| `gamma_BB` | `0.0548 + 1.0238i` |
| `gamma_AB` | `0.0137 + 0.4416i` |
| `gamma_BA` | `-0.0129 + 0.1475i` |

This is the illustrative normal-form assembly reported in main-text Table 2. The refined
dual-wavenumber decomposition above is the one used for the sensitivity analyses.

### Mixed state and amplification

| Quantity | Rounded Table 2 coefficients | Full-precision solver output |
|---|---|---|
| `abs(A_eq)` | 20.42 | 20.74 |
| `abs(B_eq)` | 10.06 | 10.20 |
| `abs(B_iso)` | 1.72 | 1.72 |
| Amplification | 5.83x | 5.91x |

The mixed state is a stable radial node: trace `-9.28`, determinant `9.79`, eigenvalues
`-8.07` and `-1.21`. `verify_equilibria.py` reproduces these values.

### Table S1: wavenumber pairing

All eight rows reproduce `data/sensitivity_wavenumbers.csv` at `beta = 15.0`:

| Configuration | `sigma` | `k_A` | `k_B` | `Re(lambda_A)` | `Re(lambda_B)` | `Re(gamma_BA)` |
|---|---|---|---|---|---|---|
| Subharmonic | 0.15 | 1.00 | 0.50 | 0.3074 | 0.1203 | -0.0005 |
| Subharmonic | 0.15 | 1.50 | 0.75 | 0.3074 | 0.1435 | -0.0272 |
| Subharmonic (ref) | 0.15 | 2.00 | 1.00 | 0.3085 | 0.1629 | -0.0566 |
| Subharmonic | 0.15 | 2.50 | 1.25 | 0.3083 | 0.1785 | -0.1033 |
| Co-evolving | 0.05 | 1.50 | 1.50 | 0.0964 | 0.0552 | -0.0428 |
| Co-evolving | 0.05 | 2.00 | 2.00 | 0.0962 | 0.0607 | -0.0742 |
| Co-evolving | 0.05 | 2.50 | 2.50 | 0.0961 | 0.0642 | -0.1152 |
| Co-evolving | 0.15 | 2.00 | 2.00 | 0.3085 | 0.2083 | -0.1369 |

`Re(gamma_BA) < 0` holds for every pairing, subharmonic and co-evolving alike.

### Table S2: fifth-order regularization

| `g_5` | `abs(A_eq)` | `abs(B_eq)` | `abs(B_iso)` | Amplification |
|---|---|---|---|---|
| `1.0e-8` | 33.66 | 16.42 | 1.72 | 9.53x |
| `1.0e-7` | 29.34 | 14.34 | 1.72 | 8.32x |
| `5.0e-7` | 23.25 | 11.41 | 1.72 | 6.62x |
| `1.0e-6` | 20.42 | 10.06 | 1.72 | 5.83x |
| `5.0e-6` | 14.49 | 7.24 | 1.72 | 4.20x |
| `1.0e-5` | 12.36 | 6.24 | 1.72 | 3.62x |
| `5.0e-5` | 8.42 | 4.44 | 1.72 | 2.57x |
| `1.0e-4` | 7.12 | 3.86 | 1.72 | 2.24x |

The strong-regularization limit is `(c_0 / g_5)^(1/4) = 7.19`; the vanishing
regularization saturation is `sqrt(-c_0 / c_1) = 34.50`. The amplification exceeds
unity across all four orders of magnitude.

### Table S3: transverse bed-slope deflection

| `r_s` | `Re(lambda_A)` | `Re(lambda_B)` | `Re(gamma_AA)` | `Re(gamma_BB)` | `Re(gamma_AB)` | `Re(gamma_BA)` |
|---|---|---|---|---|---|---|
| 0.3 | 0.3192 | 0.1739 | -0.0365 | 0.0002 | -0.0450 | -0.0088 |
| 0.4 | 0.3125 | 0.1670 | -0.0599 | 0.0351 | -0.0140 | -0.0395 |
| 0.5 | 0.3085 | 0.1629 | -0.0724 | 0.0548 | 0.0029 | -0.0566 |
| 0.6 | 0.3058 | 0.1601 | -0.0800 | 0.0646 | 0.0112 | -0.0675 |
| 0.7 | 0.3038 | 0.1582 | -0.0850 | 0.0691 | 0.0147 | -0.0750 |

`Re(gamma_BA) < 0` holds across the whole tested range, so cross-enhancement is not an
artifact of the reference value `r_s = 0.5`.

### Table 3: Landlab flat-bed noise ensemble

Five width modulations, five seeds, 2000 morphological steps, `dx = 1.0 m`. Values are
ensemble means and standard deviations, reproducing `data/landlab_results_noise.csv`:

| `a` | `A_c` (m) | `A_a` (m) | `A_a / A_c` |
|---|---|---|---|
| 0.00 | `0.127 +/- 0.097` | `0.086 +/- 0.028` | `0.95 +/- 0.47` |
| 0.05 | `0.091 +/- 0.055` | `0.071 +/- 0.027` | `0.92 +/- 0.44` |
| 0.10 | `0.092 +/- 0.068` | `0.079 +/- 0.014` | `1.11 +/- 0.49` |
| 0.15 | `0.061 +/- 0.033` | `0.059 +/- 0.020` | `1.06 +/- 0.33` |
| 0.20 | `0.061 +/- 0.032` | `0.058 +/- 0.021` | `1.05 +/- 0.34` |

Both centreline relief and cross-channel asymmetry develop from a flat bed at every
tested modulation. A one-way ANOVA on the individual-run ratio gives `F = 0.179`, so
modulation amplitude does not exert a statistically significant control on the proxy
ratio.

### Discretization sensitivity

Straight channel (`sigma = 0`, co-evolving pair, branches tracked from `sigma = 0.15`):

| `N` | `Re(gamma_BA)` |
|---|---|
| 16 | `+5.07e-4` |
| 20 | `+4.10e-4` |
| 24 | `+3.45e-4` |
| 32 | `+2.61e-4` |
| 40 | `+2.10e-4` |

These values are the discretization noise floor. A computed `Re(gamma_BA)` below about
`5e-4` in magnitude cannot be distinguished from zero at these resolutions.

Long subharmonic pair (`sigma = 0.15`, `k_A = 1.0`, `k_B = 0.5`, `L_B = 12.6`
half-widths): `Re(gamma_BA)` stays strictly negative across `N in [16, 64]`, ranging
from `-8.6e-3` to `-3.1e-3`.

### Harmonic operator convention

| Convention | `Re(gamma_AA)` | `Re(gamma_AB)` | `Re(gamma_BA)` | `Re(gamma_BB)` |
|---|---|---|---|---|
| Unshifted quasi-steady | -0.0724 | 0.0029 | -0.0566 | 0.0548 |
| Frequency-shifted | -0.0724 | 0.0388 | +0.0217 | 0.0548 |

The self-interaction coefficients are unchanged. The cross-coupling is sensitive to the
mixed harmonic operator choice, shifting `Re(gamma_BA)` from `-0.0566` to `+0.0217`.
The unshifted operator is the standard quasi-steady morphodynamic choice: water waves
adjust orders of magnitude faster than the bed, with a velocity ratio of order `1e3`.

## Reproducibility Checklist

- [ ] Dependencies installed from `requirements.txt`
- [ ] `python -c "import sys; sys.path.insert(0, 'code'); import linear_solver"` succeeds
- [ ] All seven figure scripts write PDF and PNG into `figures/`
- [ ] `python code/verify_equilibria.py` reports a stable radial node
- [ ] `python code/verify_linear_tracking.py` exits with all assertions verified
- [ ] `python data/compare_confinement.py` prints the flume run table
- [ ] Regenerated CSVs match the committed `data/` files within solver tolerance

## License

This reproducibility package is provided for academic use. Please cite the
corresponding manuscript when using these scripts.
