#!/usr/bin/env python
"""Standalone constrained multipole solver for Integrated Photonics example.

Implements requested updates from III_Multipole_Expansion.ipynb:
- Enforce electrode voltage bound: |V| <= 9.9 V
- Leave U4/U5 unconstrained (free)
- Keep notebook defaults for geometry, exclusions, and target point
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.multipoles import MultipoleControl


VOLTAGE_MAPPING = {
    "DC1": "DC21",
    "DC2": "DC20",
    "DC3": "DC19",
    "DC4": "DC18",
    "DC5": "DC17",
    "DC6": "DC16",
    "DC7": "DC15",
    "DC8": "DC14",
    "DC9": "DC13",
    "DC10": "DC12",
    "DC11": "DC10",
    "DC12": "DC9",
    "DC13": "DC8",
    "DC14": "DC7",
    "DC15": "DC6",
    "DC16": "DC5",
    "DC17": "DC4",
    "DC18": "DC3",
    "DC19": "DC2",
    "DC20": "DC1",
    "DC21": "DC11",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained multipole voltage solve for Integrated Photonics")
    parser.add_argument("--u2", type=float, default=6.0, help="Target U2 value (default: 6.0)")
    parser.add_argument("--zl-position", type=float, default=-220.001, help="zl_position in um (default: -220.001)")
    parser.add_argument("--max-voltage", type=float, default=9.9, help="Absolute electrode voltage bound in V")
    parser.add_argument("--save-plot", type=Path, default=None, help="Optional output path for bar/scatter plot")
    return parser.parse_args()


def build_controller(trap: dict, xl: float, yl: float, zl: float, roi: list[int], order: int) -> MultipoleControl:
    strs = list(trap["electrodes"].keys())

    excl = {
        "RF": "gnd",
        "DC20": "gnd",
    }

    controlled_electrodes: list[str] = []
    for electrode in strs:
        if electrode in excl and excl[electrode] != "gnd":
            trap["electrodes"][excl[electrode]]["potential"] = (
                trap["electrodes"][excl[electrode]]["potential"]
                + trap["electrodes"][electrode]["potential"]
            )
        elif electrode not in excl:
            controlled_electrodes.append(electrode)

    used_order1multipoles = ["Ex", "Ey", "Ez"]
    used_order2multipoles = ["U1", "U2", "U3", "U4", "U5"]
    used_multipoles = used_order1multipoles + used_order2multipoles

    position = [xl, yl, zl]
    s = MultipoleControl(trap, position, roi, controlled_electrodes, used_multipoles, order)
    s.electrode_positions = OrderedDict(
        [
            ("DC1", [0, 1]),
            ("DC2", [0, 2]),
            ("DC3", [0, 3]),
            ("DC4", [0, 4]),
            ("DC5", [0, 5]),
            ("DC6", [0, 6]),
            ("DC7", [0, 7]),
            ("DC8", [0, 8]),
            ("DC9", [0, 9]),
            ("DC10", [0, 10]),
            ("DC11", [2, 1]),
            ("DC12", [2, 2]),
            ("DC13", [2, 3]),
            ("DC14", [2, 4]),
            ("DC15", [2, 5]),
            ("DC16", [2, 6]),
            ("DC17", [2, 7]),
            ("DC18", [2, 8]),
            ("DC19", [2, 9]),
            ("DC20", [2, 10]),
            ("DC21", [1, 1]),
            ("RF", [1, 2]),
        ]
    )
    return s


def solve_bounded(
    s: MultipoleControl,
    target_coeffs: dict[str, float],
    max_voltage: float,
    free_multipoles: tuple[str, ...] = ("U4", "U5"),
) -> tuple[pd.Series, list[str], object]:
    free = set(free_multipoles)
    constrained = [m for m in s.used_multipoles if m in target_coeffs and m not in free]
    if not constrained:
        raise ValueError("No constrained multipoles found. Check target_coeffs.")

    A = s.expansion_matrix.loc[constrained, s.controlled_elecs].to_numpy(dtype=float)
    b = np.array([target_coeffs[m] for m in constrained], dtype=float)

    res = lsq_linear(A, b, bounds=(-max_voltage, max_voltage), method="trf")
    if not res.success:
        raise RuntimeError(f"Bounded solve failed: status={res.status}, message={res.message}")

    voltages = pd.Series(res.x, index=s.controlled_elecs, dtype=float)
    voltages = voltages.clip(-max_voltage, max_voltage)
    return voltages, constrained, res


def electrode_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("DC") and name[2:].isdigit():
        return (0, f"{int(name[2:]):03d}")
    return (1, name)


def plot_solution(s: MultipoleControl, voltages: pd.Series, height_um: float, out_path: Path | None = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Trap Height: {height_um:0.1f} um")

    ax1.bar(s.controlled_elecs, voltages.loc[s.controlled_elecs].values)
    ax1.set_ylabel("V")
    ax1.set_xticklabels(s.controlled_elecs, rotation=45, fontsize=8)

    xpos = [s.electrode_positions[ele][0] for ele in s.controlled_elecs]
    ypos = [s.electrode_positions[ele][1] for ele in s.controlled_elecs]
    sc = ax2.scatter(xpos, ypos, 500, voltages.loc[s.controlled_elecs].values, cmap="bwr")
    fig.colorbar(sc, ax=ax2)
    ax2.set_ylabel("axial electrode location")
    ax2.set_xlabel("radial electrode location")
    ax2.set_xlim(min(xpos) - 1, max(xpos) + 1)
    ax2.set_ylim(min(ypos) - 1, max(ypos) + 1)

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    radius = 500e-3
    area = 1e-4
    trap_path = SCRIPT_DIR / "inter_results" / "htrap" / f"htrap_{radius}_{area}_simulation.pkl"
    if not trap_path.exists():
        raise FileNotFoundError(f"Simulation file not found: {trap_path}")

    with trap_path.open("rb") as f:
        trap = pd.read_pickle(f)

    xl = 3.75e-3
    yl = 77e-3
    zl0 = (args.zl_position + 0.0) * 1e-3
    roi = [2, 2, 2]
    order = 2

    s = build_controller(trap, xl=xl, yl=yl, zl=zl0, roi=roi, order=order)
    s.update_origin_roi([xl, yl, zl0], roi)

    u2_value = args.u2
    target_coeffs = {
        "Ez": -0.0,
        "Ex": -0.2,
        "Ey": 6.0,
        "U2": u2_value,
        "U5": 0.0,
        "U1": 0.0,
        "U3": u2_value / 3.0,
        "U4": 0.0,
    }

    voltages, constrained, res = solve_bounded(
        s=s,
        target_coeffs=target_coeffs,
        max_voltage=args.max_voltage,
        free_multipoles=("U4", "U5"),
    )

    voltages_dict = {k: float(v) for k, v in voltages.items()}
    voltages_dict["DC20"] = 0.0

    remapped_voltages = {VOLTAGE_MAPPING.get(k, k): v for k, v in voltages_dict.items()}

    achieved = s.setVoltages(pd.Series(voltages_dict))
    target_series = pd.Series({k: target_coeffs[k] for k in constrained})
    achieved_series = achieved.loc[constrained]
    errors = achieved_series - target_series

    sorted_voltages = dict(sorted(voltages_dict.items(), key=lambda kv: electrode_sort_key(kv[0])))
    sorted_remapped = dict(sorted(remapped_voltages.items(), key=lambda kv: electrode_sort_key(kv[0])))

    print("=== Solve Summary ===")
    print(f"Constrained multipoles: {constrained}")
    print("Free multipoles: ['U4', 'U5']")
    print(f"Bounded least-squares status: {res.status} ({res.message})")
    print(f"Max |electrode voltage|: {max(abs(v) for v in voltages_dict.values()):.6f} V")

    print("\n=== Voltages (controller basis) ===")
    print(sorted_voltages)

    print("\n=== Voltages (remapped like notebook) ===")
    print(sorted_remapped)

    print("\n=== Achieved vs Target (constrained multipoles) ===")
    compare = pd.DataFrame({"target": target_series, "achieved": achieved_series, "error": errors})
    print(compare)

    print("\n=== Free Multipoles Result (not constrained) ===")
    print({"U4": float(achieved.loc["U4"]), "U5": float(achieved.loc["U5"])})

    if args.save_plot is not None:
        plot_solution(s, pd.Series(voltages_dict), height_um=yl * 1e3, out_path=args.save_plot)
        print(f"\nSaved plot: {args.save_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
