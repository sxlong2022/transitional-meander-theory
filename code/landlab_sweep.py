"""
Landlab morphodynamic sweep for variable-width channels.

Solves 2D shallow-water hydraulics with Landlab's OverlandFlow component and
updates the bed with a finite-difference Exner equation using the Meyer-Peter
and Muller (MPM) bedload formula. Width varies along x as

    b(x) = B0 * (1 + sigma * cos(k_width * x))

and nodes outside the channel are masked. The central and alternate bar
amplitudes extracted at equilibrium are written to
`theory/benchmarks/landlab_sweep_results.csv`.
"""

import os

import numpy as np
import pandas as pd
from landlab import RasterModelGrid
from landlab.components import OverlandFlow

# ------------------------------------------------------------------
# Physical and numerical parameters
# ------------------------------------------------------------------
B0 = 7.5               # mean half-width [m] (full width = 15 m for beta=15)
D0 = 1.0               # reference flow depth [m]
U0 = 1.0               # reference velocity [m s^-1]
G = 9.81               # gravitational acceleration [m s^-2]
CF = 0.01              # friction coefficient
S0 = 0.001             # bed slope [m/m]

K_WIDTH = 2.0 * np.pi / 200.0   # width-variation wavenumber [rad m^-1]
L_DOMAIN = 400.0                # streamwise extent [m]
DX = 2.0                        # grid spacing [m]
DT_FLOW = 0.5                   # hydrodynamic time step [s]
DT_MORPH = 50.0                 # morphodynamic time step [s]
N_MORPH_STEPS = 200             # total Exner steps
N_SUBSTEPS = 10                 # flow substeps per morphodynamic step
MANNINGS_N = 0.02               # Manning roughness [s m^-1/3]
H_INLET = 0.5                   # prescribed inlet water depth [m]

MPM_COEFF = 5.0                 # MPM prefactor [m^2 s^-1]
MPM_EXP = 1.5                   # exponent on excess Shields stress
SHIELDS_CRIT = 1e-5             # calibrated critical Shields stress
POROSITY = 0.4                  # bed sediment porosity
ROUGHNESS_EPS = 1e-3            # regularisation [m]
PERTURBATION_AMP = 0.02         # initial bed perturbation amplitude [m]
BAR_SEED_AMP = 0.05             # seed amplitude for alternate bars [m]
BAR_SEED_WL = 40.0              # seed wavelength for alternate bars [m]
SEED = 42

SIGMA_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]

# ------------------------------------------------------------------
# Grid helpers
# ------------------------------------------------------------------
def make_masked_grid(dx: float, length: float, half_width: float):
    """Create a RasterModelGrid large enough to cover the widest section."""
    ny = int(2.0 * half_width / dx) + 5  # ensure odd number of rows
    if ny % 2 == 0:
        ny += 1
    nx = int(length / dx) + 1
    return RasterModelGrid((ny, nx), xy_spacing=dx)


def set_mask_and_bed(grid: RasterModelGrid, b0: float, sigma: float, kx: float,
                     rng: np.random.Generator):
    """Define active channel mask and initialise bed elevation."""
    x = grid.node_x
    y_rel = grid.node_y - grid.node_y.mean()  # centreline at y = 0
    half_width = b0 * (1.0 + sigma * np.cos(kx * x))
    active = np.abs(y_rel) <= half_width

    zb = grid.add_zeros("topographic__elevation", at="node")
    zb[:] = -S0 * x

    # Seed a narrow-band alternate-bar perturbation plus random noise
    x_a = x[active]
    y_a = grid.node_y[active]
    seed = BAR_SEED_AMP * np.sin(2.0 * np.pi * x_a / BAR_SEED_WL) * np.sign(y_a)
    zb[active] = seed + PERTURBATION_AMP * rng.normal(size=seed.shape)
    zb[~active] = 10.0  # high inactive terrain

    # Closed banks; inlet/outlet will be overwritten below
    grid.status_at_node[~active] = grid.BC_NODE_IS_CLOSED
    grid.status_at_node[active] = grid.BC_NODE_IS_CORE

    # Boundary conditions: fixed water depth at inlet and outlet
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

    # OverlandFlow uses surface_water__unit_flux as an (N, 2) buffer
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
    """
    Meyer-Peter Muller bedload flux (m2 s^-1) at active nodes.
    Velocity is recovered from link discharges mapped to nodes.
    """
    surf = grid.at_node["surface_water__depth"]
    h_link = grid.at_link["surface_water__depth"]

    # Map link discharge (m2 s-1) to node vector components for velocity.
    qx, qy = grid.map_link_vector_components_to_node(grid.at_link["surface_water__discharge"])
    u = np.divide(qx, surf + ROUGHNESS_EPS, where=(active), out=np.zeros_like(surf))
    v = np.divide(qy, surf + ROUGHNESS_EPS, where=(active), out=np.zeros_like(surf))

    speed = np.sqrt(u * u + v * v)
    shields = CF * speed ** 2 / (2.0 * G * D0)
    excess = np.maximum(shields - SHIELDS_CRIT, 0.0)

    q_mag = np.zeros_like(surf)
    q_mag[active] = MPM_COEFF * excess[active] ** MPM_EXP

    qx = np.zeros_like(surf)
    qy = np.zeros_like(surf)
    mask = active & (speed > ROUGHNESS_EPS)
    qx[mask] = q_mag[mask] * u[mask] / speed[mask]
    qy[mask] = q_mag[mask] * v[mask] / speed[mask]
    return qx, qy


def d_dxi(field: np.ndarray, grid: RasterModelGrid, active: np.ndarray) -> np.ndarray:
    """Centred longitudinal derivative d(field)/dx using east/west neighbours."""
    res = np.zeros(grid.number_of_nodes)
    core = grid.core_nodes
    nbrs = grid.adjacent_nodes_at_node[core, :]  # columns [E, N, W, S]
    east = nbrs[:, 0]
    west = nbrs[:, 2]
    valid = (
        (east != grid.BAD_INDEX) & active[east] & active[core]
        & (west != grid.BAD_INDEX) & active[west]
    )
    res[core] = np.where(valid, (field[east] - field[west]) / (2.0 * grid.dx), 0.0)
    return res


def d_deta(field: np.ndarray, grid: RasterModelGrid, active: np.ndarray) -> np.ndarray:
    """Centred transverse derivative d(field)/dy using north/south neighbours."""
    res = np.zeros(grid.number_of_nodes)
    core = grid.core_nodes
    nbrs = grid.adjacent_nodes_at_node[core, :]  # columns [E, N, W, S]
    north = nbrs[:, 1]
    south = nbrs[:, 3]
    valid = (
        (north != grid.BAD_INDEX) & active[north] & active[core]
        & (south != grid.BAD_INDEX) & active[south]
    )
    res[core] = np.where(valid, (field[north] - field[south]) / (2.0 * grid.dx), 0.0)
    return res


def update_bed(grid: RasterModelGrid, active: np.ndarray, dt_morph: float):
    """Exner bed update with MPM flux divergence."""
    qx, qy = compute_mpm_flux(grid, active)
    div_q = d_dxi(qx, grid, active) + d_deta(qy, grid, active)

    zb = grid.at_node["topographic__elevation"]
    # Exner: (1 - p) dz_b/dt + div(q_b) = 0
    zb[active] -= dt_morph * div_q[active] / (1.0 - POROSITY)
    zb[~active] = 10.0  # keep banks inactive


# ------------------------------------------------------------------
# Amplitude extraction
# ------------------------------------------------------------------
def extract_amplitudes(grid: RasterModelGrid, active: np.ndarray) -> tuple:
    """
    Extract central-bar |A| and alternate-bar |B| amplitudes from bed topo.

    A: standard deviation of the detrended centreline bed elevation.
    B: standard deviation of the detrended left-minus-right asymmetry.
    """
    ny, nx = grid.shape
    x = grid.node_x.reshape((ny, nx))
    zb = grid.at_node["topographic__elevation"].reshape((ny, nx))
    mask = active.reshape((ny, nx))

    # Central bar amplitude along the centreline
    mid_row = ny // 2
    centre_mask = mask[mid_row, :]
    if centre_mask.any():
        xc = x[mid_row, centre_mask]
        zc = zb[mid_row, centre_mask]
        p = np.polyfit(xc, zc, 1)
        a_fit = zc - np.polyval(p, xc)
        amp_a = float(np.std(a_fit))
    else:
        amp_a = 0.0

    # Alternate bar amplitude from left/right asymmetry
    rows = np.arange(ny)[:, None]
    left_idx = rows < mid_row
    right_idx = rows >= mid_row
    valid = mask & (left_idx | right_idx)

    left_mean = zb.copy()
    left_mean[~valid | ~left_idx] = np.nan
    right_mean = zb.copy()
    right_mean[~valid | ~right_idx] = np.nan

    with np.errstate(invalid="ignore", all="ignore"):
        z_left = np.nanmean(left_mean, axis=0)
        z_right = np.nanmean(right_mean, axis=0)
        diff = z_left - z_right
        valid_col = np.isfinite(diff)
        if valid_col.any():
            xd = x[0, valid_col]
            dd = diff[valid_col]
            p = np.polyfit(xd, dd, 1)
            b_fit = dd - np.polyval(p, xd)
            amp_b = float(np.std(b_fit))
        else:
            amp_b = 0.0

    return amp_a, amp_b


# ------------------------------------------------------------------
# Single simulation
# ------------------------------------------------------------------
def run_one(sigma: float, rng: np.random.Generator) -> dict:
    """Run a Landlab morphodynamic loop for one width gradient."""
    print(f"  Running sigma = {sigma:.3f} ...", flush=True)

    grid = make_masked_grid(DX, L_DOMAIN, B0 * (1.0 + sigma))
    active, _ = set_mask_and_bed(grid, B0, sigma, K_WIDTH, rng)
    reset_flow_fields(grid, active)
    apply_wall_bc(grid, active)

    # OverlandFlow creates water__unit_flux at links; we do not use node flux
    flow = OverlandFlow(grid, steep_slopes=True, mannings_n=MANNINGS_N)

    for step in range(N_MORPH_STEPS):
        for _ in range(N_SUBSTEPS):
            flow.run_one_step(dt=DT_FLOW)
            apply_wall_bc(grid, active)

        update_bed(grid, active, DT_MORPH)

        if step == 0 or (step + 1) % 50 == 0:
            amp_a, amp_b = extract_amplitudes(grid, active)
            print(f"    step {step + 1}: |A|={amp_a:.3f}, |B|={amp_b:.3f}",
                  flush=True)

    amp_a, amp_b = extract_amplitudes(grid, active)
    ratio = amp_b / (amp_a + 1e-12)
    return {
        "sigma": sigma,
        "simulated_A_eq": round(amp_a, 4),
        "simulated_B_eq": round(amp_b, 4),
        "simulated_ratio": round(ratio, 4),
    }


# ------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    out_path = os.path.join(os.path.dirname(__file__), "landlab_sweep_results.csv")

    results = []
    for sigma in SIGMA_VALUES:
        results.append(run_one(sigma, rng))

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
