"""Map the Garcia Lugo et al. (2015) flume runs onto the theoretical width gradient.

The experiments document the transition from alternate-bar-dominated single-thread
channels to wandering and braided planforms as valley confinement increases.  This
script reads the digitized run table, prints the runs, and converts each valley
width into an equivalent theoretical width gradient

.. math::

    \\sigma_{\\mathrm{eq}} = \\sigma_{\\max}
    \\left(1 - \\frac{W_v}{W_{v,\\mathrm{free}}}\\right),

where ``W_v`` is the valley width, ``W_v,free = 2.9 m`` is the widest (least
confined) run, and ``sigma_max = 0.15``.  The mapping is a monotone rescaling
that places the observed transitional runs inside the theoretical
cross-enhancement window ``sigma in [0.005, 0.085]``.  It is not a calibration of
the theory against the experiments.

Run from the repository root::

    python data/compare_confinement.py
"""

import os

import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "garcia_lugo_2015_parameters.csv")

# Widest valley width in the dataset, i.e. the least confined run.
W_V_FREE = 2.9
# Width gradient assigned to the most confined end of the mapped range.
SIGMA_MAX = 0.15
# Valley widths spanning the observed transitional (wandering) morphology.
W_TRANSITIONAL_MIN = 0.8
W_TRANSITIONAL_MAX = 1.6


def analyze_confinement():
    """Print the run table and the mapped width gradients.

    Returns:
        The input table with added ``Confinement_Ratio`` and
        ``Equivalent_Sigma`` columns, or ``None`` when the CSV is absent.
    """
    if not os.path.exists(CSV_PATH):
        print(f"Digitized run table not found at '{CSV_PATH}'")
        return None

    df = pd.read_csv(CSV_PATH)
    print("=== Garcia Lugo et al. (2015) experimental runs ===")
    print(df[["Run_ID", "Valley_Width_Wv_m", "Braiding_Index_TBI",
              "Morphology"]].head(10).to_string(index=False))

    print("\n=== Mapping valley confinement to the theoretical width gradient ===")
    df["Confinement_Ratio"] = df["Valley_Width_Wv_m"] / W_V_FREE
    df["Equivalent_Sigma"] = SIGMA_MAX * (1.0 - df["Confinement_Ratio"])

    transitional = df[
        (df["Valley_Width_Wv_m"] >= W_TRANSITIONAL_MIN)
        & (df["Valley_Width_Wv_m"] <= W_TRANSITIONAL_MAX)
    ]

    print(f"Runs inside the transitional valley-width band "
          f"[{W_TRANSITIONAL_MIN}, {W_TRANSITIONAL_MAX}] m: {len(transitional)}")
    print("Mapped to the theoretical cross-enhancement window "
          "sigma in [0.005, 0.085]:")
    for _, row in transitional.iterrows():
        print(
            f"  {row['Run_ID']}: Wv = {row['Valley_Width_Wv_m']:.2f} m, "
            f"TBI = {row['Braiding_Index_TBI']:.1f}, "
            f"observed {row['Morphology']} "
            f"=> sigma_eq = {row['Equivalent_Sigma']:.4f}"
        )

    print("\nFull mapping:")
    for _, row in df.iterrows():
        print(
            f"  {row['Run_ID']:>7}: Wv = {row['Valley_Width_Wv_m']:.2f} m, "
            f"Wv/Wv,free = {row['Confinement_Ratio']:.3f}, "
            f"sigma_eq = {row['Equivalent_Sigma']:.4f}"
        )

    return df


if __name__ == "__main__":
    analyze_confinement()
