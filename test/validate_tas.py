"""
validate_tas.py -- Standalone test that wind_utils.calc_tas reproduces nimbus
TASX, plus a demonstration of how perturbing the TAS calculation propagates
into UIC/VIC/WIC. Mirrors the structure of validate_winds.py and
validate_akrd_ssrd.py.

Reproduction path (default, faithful to nimbus on GOTHAAM):

    eop = EWX / PSFDC               (TASFLG/setEOP path; humidity correction)
    mach = mach(QCFRC, PSFC, eop)   (MACHFR; humidity-corrected)
    TAS  = mach * sqrt(Cp/Cv * R * (ATX + Kelvin))  (Cp/Cv/R use eop)

For comparison the script also reports:
  - "dry"          : eop = 0 everywhere
  - "default-eop"  : eop = EWX / PSFC (uses the same p as the mach calc; the
                     default of calc_tas when no p_for_eop is supplied)

Then it recomputes UIC/VIC/WIC using each TAS variant in place of file TASX
and reports the resulting wind residual, so you can see how much a TAS
perturbation is worth in the wind product.

Run:
    cd test
    python validate_tas.py

Optional:
    python validate_tas.py --file OTHER.nc --show
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

# wind_utils.py lives one directory up
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import wind_utils  # noqa: E402


# --- column groups -----------------------------------------------------------

# Inputs we need to (a) reproduce TASX and (b) recompute UIC/VIC/WIC against it.
NEEDED = [
    "Time",
    # TAS reproduction inputs
    "QCFRC", "PSFC", "PSFDC", "ATX", "EWX",
    "MACHFR", "TASX", "TASFLG",
    # Winds inputs
    "TASX", "VEWC", "VNSC", "GGVSPD",
    "PITCH", "ROLL", "THDG",
    "AKRD", "SSLIP",
    "UIC", "VIC", "WIC",
]


# --- helpers ----------------------------------------------------------------

def load_dataframe(nc_path):
    """Read NEEDED variables, decoding _FillValue (-32767) to NaN."""
    with netCDF4.Dataset(nc_path) as nc:
        cols = {}
        for name in dict.fromkeys(NEEDED):  # dedupe while preserving order
            a = nc.variables[name][:]
            cols[name] = (a.filled(np.nan).astype(np.float64)
                          if np.ma.isMaskedArray(a)
                          else a.astype(np.float64))
    return pd.DataFrame(cols)


def report(name, ref, test):
    diff = test - ref
    valid = np.isfinite(ref) & np.isfinite(test)
    nan_mismatch = int((np.isfinite(ref) ^ np.isfinite(test)).sum())
    if not valid.any():
        print(f"  {name}: no overlapping valid samples")
        return
    d = diff[valid]
    print(f"  {name:<22s}: N={valid.sum():>5d}  "
          f"max|diff|={np.max(np.abs(d)):.3e}  "
          f"rms={np.sqrt(np.mean(d * d)):.3e}  "
          f"NaN-mismatch={nan_mismatch}")


def _winds_with_tas(df, tas_array, label):
    """Run calc_winds with ``tas_array`` substituted for TASX."""
    df2 = df.copy()
    col = f"TAS_{label}"
    df2[col] = tas_array
    u, v, w, ux, vy, wf = wind_utils.calc_winds(
        df2, aircraft="C130",
        akrd=df2["AKRD"].to_numpy(), sslip=df2["SSLIP"].to_numpy(),
        tas_var=col, vew_var="VEWC", vns_var="VNSC", vspd_var="GGVSPD",
        dt=1.0, wind_type="corrected",
    )
    return u, v, w


# --- main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=os.path.join(HERE, "GOTHAAMrf18.nc"),
                        help="netCDF file to validate against (default: GOTHAAMrf18.nc)")
    parser.add_argument("--out", default=os.path.join(HERE, "tas_validation.png"),
                        help="Output PNG path (default: tas_validation.png)")
    parser.add_argument("--show", action="store_true",
                        help="Also call plt.show() after saving the figure.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"netCDF file not found: {args.file}")

    print(f"Loading {args.file} ...")
    df = load_dataframe(args.file)
    n = len(df)
    print(f"  {n} samples, "
          f"TAS in [{np.nanmin(df['TASX']):.2f}, {np.nanmax(df['TASX']):.2f}] m/s")
    nflg = int(np.nansum(df["TASFLG"].to_numpy() == 1.0))
    print(f"  TASFLG=1 (50%%-RH fallback) samples: {nflg}/{int(np.isfinite(df['TASFLG']).sum())}")
    print()

    q     = df["QCFRC"].to_numpy()
    p     = df["PSFC"].to_numpy()
    psfdc = df["PSFDC"].to_numpy()
    atx   = df["ATX"].to_numpy()
    ewx   = df["EWX"].to_numpy()

    # -- Validation #1: faithful path (eop uses PSFDC, mach uses PSFC)
    print("Validation #1: calc_tas(QCFRC, PSFC, ATX, EWX, p_for_eop=PSFDC)")
    tas_faith = wind_utils.calc_tas(q, p, atx, ewx=ewx, p_for_eop=psfdc)
    mach_py   = wind_utils.calc_mach(q, p, eop=ewx/psfdc)
    print("  Differences (python_port - nimbus):")
    report("MACHFR",            df["MACHFR"].to_numpy(), mach_py)
    report("TASX (faithful)",   df["TASX"].to_numpy(),   tas_faith)
    print()

    # -- Validation #2: default path (eop = EWX/p, where p is the mach's p)
    print("Validation #2: calc_tas(QCFRC, PSFC, ATX, EWX)   # eop = EWX/PSFC")
    tas_dflt = wind_utils.calc_tas(q, p, atx, ewx=ewx)
    print("  Differences (python_port - nimbus):")
    report("TASX (default eop)", df["TASX"].to_numpy(), tas_dflt)
    print()

    # -- Validation #3: dry-air calculation (no humidity correction)
    print("Validation #3: calc_tas(QCFRC, PSFC, ATX)   # dry air, eop=0")
    tas_dry = wind_utils.calc_tas(q, p, atx)
    print("  Differences (python_port - nimbus):")
    report("TASX (dry)",        df["TASX"].to_numpy(), tas_dry)
    print()

    # -- Winds sensitivity: feed each TAS variant into calc_winds and diff
    # against file UIC/VIC/WIC. Anchors how much a TAS perturbation costs.
    print("Wind sensitivity: substituting each TAS variant for TASX in calc_winds.")
    print("  (reference for the diffs below is the *file* UIC/VIC/WIC)")
    variants = [
        ("faithful", tas_faith),
        ("default-eop", tas_dflt),
        ("dry",      tas_dry),
    ]
    u_ref = df["UIC"].to_numpy()
    v_ref = df["VIC"].to_numpy()
    w_ref = df["WIC"].to_numpy()
    winds_by_variant = {}
    for label, tas_arr in variants:
        u, v, w = _winds_with_tas(df, tas_arr, label)
        winds_by_variant[label] = (u, v, w)
        print(f"\n  TAS variant '{label}':")
        report(f"UIC ({label})", u_ref, u)
        report(f"VIC ({label})", v_ref, v)
        report(f"WIC ({label})", w_ref, w)
    print()

    # ----- Plot -------------------------------------------------------------
    # Row 0: TAS overlay (nimbus + the three python variants)
    # Row 1: TAS residual (py - nimbus) for each variant
    # Rows 2/3: UIC residual; rows 4/5: VIC residual; rows 6/7: WIC residual
    t = df["Time"].to_numpy()
    tasx_ref = df["TASX"].to_numpy()

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(
        5, 1, height_ratios=[3, 1.4, 1.4, 1.4, 1.4], hspace=0.22,
    )
    shared_x = None

    # --- TAS overlay
    ax = fig.add_subplot(gs[0])
    ax.plot(t, tasx_ref, color="C0", linewidth=0.8, label="nimbus TASX")
    colors = {"faithful": "C1", "default-eop": "C2", "dry": "C3"}
    for label, tas_arr in variants:
        ax.plot(t, tas_arr, color=colors[label], linewidth=0.5,
                linestyle="--", label=f"calc_tas '{label}'")
    ax.set_ylabel("TAS [m/s]")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelbottom=False)
    shared_x = ax

    # --- TAS residual (one axis, all three variants)
    ax = fig.add_subplot(gs[1], sharex=shared_x)
    for label, tas_arr in variants:
        ax.plot(t, tas_arr - tasx_ref, color=colors[label], linewidth=0.5,
                label=label)
    ax.axhline(0, color="k", linewidth=0.3)
    ax.set_ylabel("TAS - nimbus\n[m/s]", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelbottom=False)

    # --- Winds residuals (UIC/VIC/WIC), one row per component
    for k, (name, ref) in enumerate(zip(["UIC", "VIC", "WIC"],
                                         [u_ref, v_ref, w_ref])):
        ax = fig.add_subplot(gs[2 + k], sharex=shared_x)
        for label, _ in variants:
            py = winds_by_variant[label][k]
            ax.plot(t, py - ref, color=colors[label], linewidth=0.5,
                    label=label)
        ax.axhline(0, color="k", linewidth=0.3)
        ax.set_ylabel(f"{name} - nimbus\n[m/s]", fontsize=9)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85, ncol=3)
        if k < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (s since 00:00 UTC)")

    fig.suptitle(
        f"Python port wind_utils.calc_tas vs nimbus TASX, and resulting wind "
        f"residuals\n{os.path.basename(args.file)}"
    )
    fig.subplots_adjust(top=0.94)

    fig.savefig(args.out, dpi=120)
    print(f"Saved figure -> {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
