"""Two-dimensional morphodynamic benchmark for variable-width channels.

This script provides an independent, physics-based cross-check of the weakly
nonlinear theory.  It solves the two-dimensional shallow-water equations with
Landlab's ``OverlandFlow`` component on a raster grid, then advances the bed with
a finite-difference Exner equation using the Meyer-Peter and Muller bedload
formula.

The channel half-width varies along the streamwise coordinate as

.. math::

    b(x) = B_0 \\left[1 + a \\cos(k_w x)\\right],

where ``a`` is the width-modulation amplitude, the direct counterpart of the
theory's width gradient ``sigma``.  Nodes outside the instantaneous channel are
masked and treated as closed boundaries, so the flow field is confined to a
planform of varying width.

Protocols
---------
``noise_only``
    Gaussian random bed perturbation only, with no organised seed.  This tests
    whether bar structure emerges spontaneously from noise, following the
    flat-bed initialization of Crosato et al. (2012).
``seeded``
    Antisymmetric alternate-bar seed superposed on Gaussian noise, used to check
    that the diagnostic proxies respond to a known transverse mode.

Diagnostics
-----------
Two spatial proxies are extracted from the final detrended bed topography:

* ``A_c``: standard deviation of the centreline bed elevation, a proxy for
  central-bar relief.
* ``A_a``: standard deviation of the cross-channel asymmetry, a proxy for
  alternate-bar relief.

Both are bulk spatial standard deviations in metres.  They are not projections
onto the theoretical eigenmodes, so the benchmark supports qualitative
comparison of trends, not coefficient-by-coefficient agreement.

Outputs
-------
``<out-dir>/landlab_results<tag>.csv``
    One row of scalar diagnostics per run.
``<out-dir>/landlab_topo<tag>_<i>.npz``
    Final bed topography per run (arrays ``x``, ``y``, ``zb``, ``active``).

This single-module layout is deliberate: the numerical protocol, parameter block,
grid helpers, flow helpers, Exner update, and diagnostic extraction stay in one
file so the benchmark remains runnable as a standalone script.

Usage
-----
.. code-block:: console

    python code/landlab_sweep.py --mode noise_only --steps 2000 --out-dir data
    python code/landlab_sweep.py --mode seeded --tag _seeded --seeds 42
"""

import argparse
import csv
import os

import numpy as np
from landlab import RasterModelGrid
from landlab.components import OverlandFlow

# ------------------------------------------------------------------
# Physical and numerical parameters
# ------------------------------------------------------------------
B0 = 7.5                # mean half-width [m]; full width 15 m for beta = 15
D0 = 1.0                # reference flow depth [m]
G = 9.81                # gravitational acceleration [m s^-2]
CF = 0.01               # friction coefficient
S0 = 0.001              # bed slope [m/m]

K_WIDTH = 2.0 * np.pi / 200.0   # width-modulation wavenumber [rad m^-1]
L_DOMAIN = 400.0                # streamwise extent [m]

# dx = 1.0 m places about 15 nodes across the mean full width, which resolves
# transverse modes up to m ~ 4 and separates m = 1 from m = 2.  A coarser
# dx = 2.0 m leaves about 7 nodes across the width, which does not.
DX = 1.0
DT_FLOW = 0.25                  # hydrodynamic time step [s]
DT_MORPH = 50.0                 # morphodynamic time step [s]
N_MORPH_STEPS = 2000            # total Exner steps (100,000 s = 27.8 h)
N_SUBSTEPS = 10                 # flow substeps per morphodynamic step
MANNINGS_N = 0.02               # Manning roughness [s m^-1/3]
H_INLET = 0.5                   # prescribed inlet water depth [m]

MPM_COEFF = 5.0                 # MPM prefactor [m^2 s^-1]
MPM_EXP = 1.5                   # exponent on excess Shields stress
SHIELDS_CRIT = 1e-5             # calibrated critical Shields stress
POROSITY = 0.4                  # bed sediment porosity
ROUGHNESS_EPS = 1e-3            # regularisation [m]
PERTURBATION_AMP = 0.02         # initial random bed perturbation amplitude [m]
BAR_SEED_AMP = 0.05             # organised seed amplitude (seeded protocol) [m]
BAR_SEED_WL = 40.0              # organised seed wavelength [m]

SIGMA_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]
NOISE_SEEDS = [1, 7, 42, 101, 2024]

DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


# ------------------------------------------------------------------
# Grid helpers
# ------------------------------------------------------------------
def make_masked_grid(dx: float, length: float, half_width: float):
    """Create a raster grid covering the widest section.

    The row count is forced odd so the channel centreline falls exactly on a grid
    row.  An even row count places the centreline between rows, which biases the
    left/right asymmetry proxy.
    """
    ny = int(np.ceil(2.0 * half_width / dx)) + 4
    if ny % 2 == 0:
        ny += 1
    nx = int(length / dx) + 1
    return RasterModelGrid((ny, nx), xy_spacing=dx)


def set_mask_and_bed(grid: RasterModelGrid, b0: float, sigma: float, kx: float,
                     rng: np.random.Generator, seed_mode: str = "noise_only"):
    """Define the active channel mask and initialise the bed elevation.

    Args:
        grid: Raster grid.
        b0: Mean half-width [m].
        sigma: Width-modulation amplitude.
        kx: Width-modulation wavenumber [rad m^-1].
        rng: Random generator supplying the Gaussian perturbation.
        seed_mode: ``'seeded'`` or ``'noise_only'``.

    Returns:
        ``(active, half_width)``, the boolean active-node mask and the
        half-width evaluated at every node.
    """
    x = grid.node_x
    y_centre = grid.node_y.mean()
    y_rel = grid.node_y - y_centre
    half_width = b0 * (1.0 + sigma * np.cos(kx * x))
    active = np.abs(y_rel) <= half_width

    zb = grid.add_zeros("topographic__elevation", at="node")
    zb[:] = -S0 * x

    noise = PERTURBATION_AMP * rng.normal(size=x.shape)

    if seed_mode == "seeded":
        # The transverse sign must be built from the row index: absolute y is
        # strictly positive across the channel, so np.sign(y) would be
        # identically +1 and would produce a symmetric seed.  Using the row
        # offset keeps every node strictly +1 or -1 with no centreline zero.
        n_rows = grid.shape[0]
        row_of_node = np.arange(grid.number_of_nodes) // grid.shape[1]
        mid = n_rows // 2
        trans_sign = np.where(row_of_node < mid, -1.0, 1.0)
        seed = BAR_SEED_AMP * np.sin(2.0 * np.pi * x / BAR_SEED_WL) * trans_sign
        zb[active] = (seed + noise)[active]
    elif seed_mode == "noise_only":
        zb[active] = noise[active]
    else:
        raise ValueError(f"Unknown seed_mode: {seed_mode}")

    zb[~active] = 10.0

    grid.status_at_node[~active] = grid.BC_NODE_IS_CLOSED
    grid.status_at_node[active] = grid.BC_NODE_IS_CORE

    left = active & (grid.x_of_node == 0)
    right = active & (grid.x_of_node >= grid.x_of_node.max() - 1e-6)
    grid.status_at_node[left] = grid.BC_NODE_IS_FIXED_VALUE
    grid.status_at_node[right] = grid.BC_NODE_IS_FIXED_VALUE

    return active, half_width


# ------------------------------------------------------------------
# Flow helpers
# ------------------------------------------------------------------
def reset_flow_fields(grid: RasterModelGrid, active: np.ndarray):
    """Initialise water depth and surface unit-flux fields."""
    grid.add_zeros("surface_water__depth", at="node")

    if "surface_water__unit_flux" not in grid.at_node:
        grid.add_field(
            "surface_water__unit_flux",
            np.zeros((grid.number_of_nodes, 2)),
            at="node",
            clobber=True,
        )

    surf = grid.at_node["surface_water__depth"]
    surf[:] = 0.0
    surf[active] = H_INLET

    flux = grid.at_node["surface_water__unit_flux"]
    flux[:] = 0.0


def apply_wall_bc(grid: RasterModelGrid, active: np.ndarray):
    """Reset inactive water depths/fluxes and enforce inlet/outlet heads."""
    surf = grid.at_node["surface_water__depth"]
    surf[~active] = 0.0

    flux = grid.at_node["surface_water__unit_flux"]
    flux[~active, :] = 0.0

    left = active & (grid.x_of_node == 0)
    right = active & (grid.x_of_node >= grid.x_of_node.max() - 1e-6)
    surf[left] = H_INLET
    surf[right] = H_INLET
    grid.status_at_node[left] = grid.BC_NODE_IS_FIXED_VALUE
    grid.status_at_node[right] = grid.BC_NODE_IS_FIXED_VALUE


# ------------------------------------------------------------------
# Sediment transport and Exner update
# ------------------------------------------------------------------
def compute_mpm_flux(grid: RasterModelGrid, active: np.ndarray) -> tuple:
    """Return the Meyer-Peter Muller bedload flux [m^2 s^-1] at active nodes."""
    surf = grid.at_node["surface_water__depth"]

    qx, qy = grid.map_link_vector_components_to_node(
        grid.at_link["surface_water__discharge"]
    )
    u = np.divide(qx, surf + ROUGHNESS_EPS, where=active, out=np.zeros_like(surf))
    v = np.divide(qy, surf + ROUGHNESS_EPS, where=active, out=np.zeros_like(surf))

    speed = np.sqrt(u * u + v * v)
    shields = CF * speed ** 2 / (2.0 * G * D0)
    excess = np.maximum(shields - SHIELDS_CRIT, 0.0)

    q_mag = np.zeros_like(surf)
    q_mag[active] = MPM_COEFF * excess[active] ** MPM_EXP

    qs_x = np.zeros_like(surf)
    qs_y = np.zeros_like(surf)
    mask = active & (speed > ROUGHNESS_EPS)
    qs_x[mask] = q_mag[mask] * u[mask] / speed[mask]
    qs_y[mask] = q_mag[mask] * v[mask] / speed[mask]
    return qs_x, qs_y


def d_dxi(field: np.ndarray, grid: RasterModelGrid, active: np.ndarray) -> np.ndarray:
    """Return the centred longitudinal derivative ``d(field)/dx``."""
    res = np.zeros(grid.number_of_nodes)
    core = grid.core_nodes
    nbrs = grid.adjacent_nodes_at_node[core, :]
    east = nbrs[:, 0]
    west = nbrs[:, 2]
    valid = (
        (east != grid.BAD_INDEX) & active[east] & active[core]
        & (west != grid.BAD_INDEX) & active[west]
    )
    res[core] = np.where(valid, (field[east] - field[west]) / (2.0 * grid.dx), 0.0)
    return res


def d_deta(field: np.ndarray, grid: RasterModelGrid, active: np.ndarray) -> np.ndarray:
    """Return the centred transverse derivative ``d(field)/dy``."""
    res = np.zeros(grid.number_of_nodes)
    core = grid.core_nodes
    nbrs = grid.adjacent_nodes_at_node[core, :]
    north = nbrs[:, 1]
    south = nbrs[:, 3]
    valid = (
        (north != grid.BAD_INDEX) & active[north] & active[core]
        & (south != grid.BAD_INDEX) & active[south]
    )
    res[core] = np.where(valid, (field[north] - field[south]) / (2.0 * grid.dx), 0.0)
    return res


def update_bed(grid: RasterModelGrid, active: np.ndarray, dt_morph: float):
    """Advance the bed with the Exner equation using the MPM flux divergence."""
    qs_x, qs_y = compute_mpm_flux(grid, active)
    div_q = d_dxi(qs_x, grid, active) + d_deta(qs_y, grid, active)

    zb = grid.at_node["topographic__elevation"]
    zb[active] -= dt_morph * div_q[active] / (1.0 - POROSITY)
    zb[~active] = 10.0


# ------------------------------------------------------------------
# Amplitude extraction and transverse modal decomposition
# ------------------------------------------------------------------
def detrend_bed(grid: RasterModelGrid, active: np.ndarray) -> tuple:
    """Return detrended bed elevation and coordinates on the logical grid."""
    ny, nx = grid.shape
    x2 = grid.node_x.reshape((ny, nx))
    y2 = grid.node_y.reshape((ny, nx))
    zb = grid.at_node["topographic__elevation"].reshape((ny, nx))
    mask = active.reshape((ny, nx))

    zb_d = np.full_like(zb, np.nan)
    for r in range(ny):
        m = mask[r, :]
        if m.sum() > 3:
            p = np.polyfit(x2[r, m], zb[r, m], 1)
            zb_d[r, m] = zb[r, m] - np.polyval(p, x2[r, m])
    return zb_d, x2, y2, mask


def extract_amplitudes(grid: RasterModelGrid, active: np.ndarray) -> tuple:
    """Extract the two spatial proxy amplitudes.

    ``A_c`` is the standard deviation of the detrended centreline bed elevation.
    ``A_a`` is the standard deviation of the detrended left-minus-right
    asymmetry.  Both are bulk spatial standard deviations in metres and are not
    projections onto the theoretical eigenmodes.
    """
    ny, nx = grid.shape
    zb_d, x2, y2, mask = detrend_bed(grid, active)
    mid_row = ny // 2

    cm = mask[mid_row, :]
    amp_a = float(np.nanstd(zb_d[mid_row, cm])) if cm.any() else 0.0

    rows = np.arange(ny)[:, None]
    left = rows < mid_row
    right = rows >= mid_row
    valid = mask & (left | right)

    with np.errstate(invalid="ignore", all="ignore"):
        lm = np.where(valid & left, zb_d, np.nan)
        rm = np.where(valid & right, zb_d, np.nan)
        n_l = np.sum(valid & left, axis=0)
        n_r = np.sum(valid & right, axis=0)
        sl = np.nansum(lm, axis=0)
        sr = np.nansum(rm, axis=0)
        usable = (n_l > 0) & (n_r > 0)
        diff = np.full(nx, np.nan)
        diff[usable] = (sl[usable] / n_l[usable]) - (sr[usable] / n_r[usable])
        amp_b = float(np.nanstd(diff)) if np.isfinite(diff).any() else 0.0

    return amp_a, amp_b


def transverse_mode_power(grid: RasterModelGrid, active: np.ndarray,
                          m_max: int = 4) -> dict:
    """Fourier-decompose the detrended bed in the transverse direction.

    Returns RMS amplitude in metres for symmetric (central-like) and
    antisymmetric (alternate-like) components.  Columns with fewer than
    ``2 * m_max + 3`` active nodes are skipped.
    """
    ny, nx = grid.shape
    zb_d, x2, y2, mask = detrend_bed(grid, active)

    acc = {("sym", m): [] for m in range(1, m_max + 1)}
    acc.update({("anti", m): [] for m in range(1, m_max + 1)})

    for c in range(nx):
        col_mask = mask[:, c]
        n = int(col_mask.sum())
        if n < 2 * m_max + 3:
            continue
        v = zb_d[col_mask, c]
        yy = y2[col_mask, c]
        if v.size != n or not np.all(np.isfinite(v)):
            continue
        yc = yy - yy.mean()
        hw = np.abs(yc).max()
        if hw <= 0:
            continue
        for m in range(1, m_max + 1):
            cs = np.cos(m * np.pi * yc / hw)
            sn = np.sin(m * np.pi * yc / hw)
            acc[("sym", m)].append(2.0 * np.sum(v * cs) / n)
            acc[("anti", m)].append(2.0 * np.sum(v * sn) / n)

    out = {}
    for key, arr in acc.items():
        out[key] = float(np.sqrt(np.mean(np.square(arr)))) if arr else float("nan")
    return out


def save_topography(grid, active, out_path):
    """Save the final bed topography and coordinates for later comparison."""
    zb = grid.at_node["topographic__elevation"]
    np.savez_compressed(
        out_path,
        x=grid.node_x,
        y=grid.node_y,
        zb=zb,
        active=active.astype(np.int8),
        shape=np.array(grid.shape),
        dx=grid.dx,
    )


# ------------------------------------------------------------------
# Single simulation
# ------------------------------------------------------------------
def run_one(sigma: float, rng: np.random.Generator,
            seed_mode: str = "noise_only",
            n_steps: int = None,
            dx: float = None,
            dt_flow: float = None) -> dict:
    """Run one morphodynamic simulation and return its diagnostics."""
    n_steps = N_MORPH_STEPS if n_steps is None else n_steps
    dx = DX if dx is None else dx
    dt_flow = DT_FLOW if dt_flow is None else dt_flow

    grid = make_masked_grid(dx, L_DOMAIN, B0 * (1.0 + sigma))
    active, _ = set_mask_and_bed(grid, B0, sigma, K_WIDTH, rng, seed_mode=seed_mode)
    reset_flow_fields(grid, active)
    apply_wall_bc(grid, active)

    flow = OverlandFlow(grid, steep_slopes=True, mannings_n=MANNINGS_N)

    for _ in range(n_steps):
        for _ in range(N_SUBSTEPS):
            flow.run_one_step(dt=dt_flow)
            apply_wall_bc(grid, active)
        update_bed(grid, active, DT_MORPH)

    amp_a, amp_b = extract_amplitudes(grid, active)
    modes = transverse_mode_power(grid, active)

    zb = grid.at_node["topographic__elevation"]
    relief = float(np.std((zb - (-S0 * grid.node_x))[active]))

    return {
        "_grid": grid,
        "_active": active,
        "sigma": sigma,
        "seed_mode": seed_mode,
        "n_steps": n_steps,
        "dx": dx,
        "A_c": amp_a,
        "A_a": amp_b,
        "ratio": amp_b / (amp_a + 1e-12),
        "sym_m1_rms": modes[("sym", 1)],
        "sym_m2_rms": modes[("sym", 2)],
        "anti_m1_rms": modes[("anti", 1)],
        "anti_m2_rms": modes[("anti", 2)],
        "bed_relief_std": relief,
    }


# ------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Landlab morphodynamic sweep across width modulations."
    )
    ap.add_argument("--mode", default="noise_only",
                    choices=["noise_only", "seeded"],
                    help="Initial bed protocol.")
    ap.add_argument("--steps", type=int, default=N_MORPH_STEPS,
                    help="Number of morphodynamic steps.")
    ap.add_argument("--dx", type=float, default=DX, help="Grid spacing [m].")
    ap.add_argument("--dt-flow", type=float, default=DT_FLOW,
                    help="Hydrodynamic time step [s].")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Random seeds; defaults per protocol.")
    ap.add_argument("--sigmas", type=float, nargs="*", default=None,
                    help="Width-modulation amplitudes; defaults to 0 to 0.20.")
    ap.add_argument("--tag", default=None,
                    help="Suffix for output file names (defaults to '_noise' or '_seeded').")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="Directory for CSV and topography output.")
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else (
        NOISE_SEEDS if args.mode == "noise_only" else [42]
    )
    sigmas = args.sigmas if args.sigmas else SIGMA_VALUES
    tag = (
        f"_{'noise' if args.mode == 'noise_only' else 'seeded'}"
        if args.tag is None else args.tag
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"landlab_results{tag}.csv")
    rows = []
    for sigma in sigmas:
        for sd in seeds:
            rng = np.random.default_rng(sd)
            print(
                f"  sigma={sigma:.2f} seed={sd} mode={args.mode} "
                f"steps={args.steps} dx={args.dx}",
                flush=True,
            )
            res = run_one(sigma, rng, seed_mode=args.mode, n_steps=args.steps,
                          dx=args.dx, dt_flow=args.dt_flow)
            res["rng_seed"] = sd
            rows.append(res)
            topo_path = os.path.join(
                args.out_dir, f"landlab_topo{tag}_{len(rows):02d}.npz"
            )
            save_topography(res.pop("_grid"), res.pop("_active"), topo_path)
            print(
                "     -> A_c=%.4f A_a=%.4f ratio=%.3f anti_m1=%.4f "
                "anti_m2=%.4f sym_m2=%.4f"
                % (res["A_c"], res["A_a"], res["ratio"],
                   res["anti_m1_rms"], res["anti_m2_rms"], res["sym_m2_rms"]),
                flush=True,
            )

    fieldnames = [k for k in rows[0].keys() if not k.startswith("_")]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {out_csv}")
    print(f"Runs: {len(rows)}")


if __name__ == "__main__":
    main()
