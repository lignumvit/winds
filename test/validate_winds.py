"""
validate_winds.py -- Standalone equivalent of the two validation cells in
wind_testing.ipynb. Loads the netCDF in this directory, runs calc_winds in
both modes (akrd= direct, and aoa_coefs + aoa_coefs_alt + aos_coefs with the
mach-threshold switch), prints per-variable diff stats vs nimbus, and saves
the same residual figure the notebook draws.

Run:
    cd test
    python validate_winds.py

Optional:
    python validate_winds.py --file OTHER.nc --show
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

# Inputs nimbus' WIC depends on, plus the columns the recompute path needs
# (ADIFR/BDIFR/QCF/PSFD) and the variables we'll diff against.
NEEDED = [
    "Time",
    "TASX", "VEWC", "VNSC", "GGVSPD",
    "PITCH", "ROLL", "THDG",
    "AKRD", "SSLIP",
    "UIC", "VIC", "WIC", "UXC", "VYC", "WINDSFLG",
    "ADIFR", "BDIFR", "QCF", "PSFD",
]


# --- helpers ----------------------------------------------------------------

def load_dataframe(nc_path):
    """Read NEEDED variables, decoding _FillValue (-32767) to NaN."""
    with netCDF4.Dataset(nc_path) as nc:
        cols = {}
        for name in NEEDED:
            a = nc.variables[name][:]
            cols[name] = (a.filled(np.nan).astype(np.float64)
                          if np.ma.isMaskedArray(a)
                          else a.astype(np.float64))
        akrd_coeff      = [float(c) for c in nc.variables["AKRD"].getncattr("AKRD_COEFF")]
        akrd_coeff_alt  = [float(c) for c in nc.variables["AKRD"].getncattr("AKRD_COEFF_ALT")]
        sslip_coeff     = [float(c) for c in nc.variables["SSRD"].getncattr("CalibrationCoefficients")]
    return pd.DataFrame(cols), akrd_coeff, akrd_coeff_alt, sslip_coeff


def report(name, ref, test):
    diff = test - ref
    valid = np.isfinite(ref) & np.isfinite(test)
    nan_mismatch = int((np.isfinite(ref) ^ np.isfinite(test)).sum())
    if not valid.any():
        print(f"  {name}: no overlapping valid samples")
        return
    d = diff[valid]
    print(f"  {name:<10s}: N={valid.sum():>5d}  "
          f"max|diff|={np.max(np.abs(d)):.3e}  "
          f"rms={np.sqrt(np.mean(d * d)):.3e}  "
          f"NaN-mismatch={nan_mismatch}")


# --- main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=os.path.join(HERE, "GOTHAAMrf18.nc"),
                        help="netCDF file to validate against (default: GOTHAAMrf18.nc)")
    parser.add_argument("--out", default=os.path.join(HERE, "winds_validation.png"),
                        help="Output PNG path (default: winds_validation.png)")
    parser.add_argument("--show", action="store_true",
                        help="Also call plt.show() after saving the figure.")
    parser.add_argument("--mach-threshold", type=float, default=0.24,
                        help="Mach threshold for AKRD coef switch (default: 0.24)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"netCDF file not found: {args.file}")

    print(f"Loading {args.file} ...")
    df, aoa_coefs, aoa_coefs_alt, aos_coefs = load_dataframe(args.file)
    print(f"  {len(df)} samples, "
          f"TAS in [{np.nanmin(df['TASX']):.2f}, {np.nanmax(df['TASX']):.2f}] m/s")
    print(f"  AKRD_COEFF     (clean) = {aoa_coefs}")
    print(f"  AKRD_COEFF_ALT (dirty) = {aoa_coefs_alt}")
    print(f"  SSRD coefs              = {aos_coefs}")
    print()

    # -- Validation #1: file's AKRD/SSLIP passed in directly
    print("Validation #1: akrd= and sslip= passed directly")
    u1, v1, w1, ux1, vy1, wf1 = wind_utils.calc_winds(
        df, aircraft="C130",
        akrd=df["AKRD"].to_numpy(), sslip=df["SSLIP"].to_numpy(),
        tas_var="TASX", vew_var="VEWC", vns_var="VNSC", vspd_var="GGVSPD",
        dt=1.0, wind_type="corrected",
    )
    print("  Differences (python_port - nimbus):")
    report("UIC",      df["UIC"].to_numpy(),      u1)
    report("VIC",      df["VIC"].to_numpy(),      v1)
    report("WIC",      df["WIC"].to_numpy(),      w1)
    report("UXC",      df["UXC"].to_numpy(),      ux1)
    report("VYC",      df["VYC"].to_numpy(),      vy1)
    report("WINDSFLG", df["WINDSFLG"].to_numpy(), wf1)
    print()

    # -- Validation #2: recompute AKRD/SSLIP from coefs, with mach-threshold switch
    print(f"Validation #2: aoa_coefs + aoa_coefs_alt + aos_coefs "
          f"(mach_threshold={args.mach_threshold})")
    u2, v2, w2, ux2, vy2, wf2 = wind_utils.calc_winds(
        df, aircraft="C130",
        aoa_coefs=aoa_coefs, aoa_coefs_alt=aoa_coefs_alt,
        mach_threshold=args.mach_threshold,
        aos_coefs=aos_coefs,
        tas_var="TASX", vew_var="VEWC", vns_var="VNSC", vspd_var="GGVSPD",
        qcf_var="QCF", psf_var="PSFD",
        dt=1.0, wind_type="corrected",
    )
    akrd_py2  = df["AKRD_test"].to_numpy()
    sslip_py2 = df["SSLIP_test"].to_numpy()
    print("  Differences (python_port - nimbus):")
    report("AKRD",     df["AKRD"].to_numpy(),     akrd_py2)
    report("SSLIP",    df["SSLIP"].to_numpy(),    sslip_py2)
    report("UIC",      df["UIC"].to_numpy(),      u2)
    report("VIC",      df["VIC"].to_numpy(),      v2)
    report("WIC",      df["WIC"].to_numpy(),      w2)
    report("UXC",      df["UXC"].to_numpy(),      ux2)
    report("VYC",      df["VYC"].to_numpy(),      vy2)
    report("WINDSFLG", df["WINDSFLG"].to_numpy(), wf2)

    # how many samples landed in each coefficient set
    mach_dry = wind_utils.calc_mach_dry(df["QCF"].to_numpy(),
                                       df["PSFD"].to_numpy())
    n_alt   = int(np.sum((mach_dry <  args.mach_threshold) & np.isfinite(mach_dry)))
    n_clean = int(np.sum((mach_dry >= args.mach_threshold) & np.isfinite(mach_dry)))
    print(f"\n  Mach-threshold switch: {n_alt} samples used ALT (dirty), "
          f"{n_clean} used clean.")
    print()

    # -- Plot: 5 components x (tall overlay + short residual), plus a coef-set
    # strip at the bottom showing where the dirty coef set was active.
    t = df["Time"].to_numpy()
    py_components  = [("UIC", u2), ("VIC", v2), ("WIC", w2),
                      ("UXC", ux2), ("VYC", vy2)]
    fig = plt.figure(figsize=(12, 17))
    gs = fig.add_gridspec(
        11, 1,
        height_ratios=[3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 0.8],
        hspace=0.18,
    )
    shared_x = None
    for k, (name, py) in enumerate(py_components):
        ref = df[name].to_numpy()

        ax_top = fig.add_subplot(gs[2 * k], sharex=shared_x)
        ax_top.plot(t, ref, color="C0", linewidth=0.8, label=f"nimbus {name}")
        ax_top.plot(t, py,  color="C1", linewidth=0.6, linestyle="--",
                    label="python port")
        ax_top.set_ylabel(f"{name} [m/s]")
        ax_top.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax_top.grid(True, alpha=0.3)
        ax_top.tick_params(labelbottom=False)
        if shared_x is None:
            shared_x = ax_top

        ax_resid = fig.add_subplot(gs[2 * k + 1], sharex=shared_x)
        ax_resid.plot(t, py - ref, color="C3", linewidth=0.5)
        ax_resid.axhline(0, color="k", linewidth=0.3)
        ax_resid.set_ylabel("py - nimbus\n[m/s]", fontsize=8)
        ax_resid.grid(True, alpha=0.3)
        ax_resid.tick_params(labelbottom=False)

        # annotate max abs diff inside the residual axis
        valid = np.isfinite(ref) & np.isfinite(py)
        if valid.any():
            mad = float(np.max(np.abs((py - ref)[valid])))
            ax_resid.text(0.99, 0.92, f"max|diff| = {mad:.2e}",
                          transform=ax_resid.transAxes, ha="right", va="top",
                          fontsize=8,
                          bbox=dict(facecolor="white", edgecolor="none",
                                    alpha=0.7, pad=1.5))

    # coef-set strip
    ax_strip = fig.add_subplot(gs[-1], sharex=shared_x)
    ax_strip.fill_between(t, 0, (mach_dry < args.mach_threshold).astype(float),
                          step="mid", alpha=0.5, color="C3")
    ax_strip.set_ylabel("coef set")
    ax_strip.set_yticks([0, 1])
    ax_strip.set_yticklabels(["clean", "dirty"])
    ax_strip.set_xlabel("Time (s since 00:00 UTC)")
    ax_strip.grid(True, alpha=0.3)

    fig.suptitle(
        f"Python port w/ AOA-coefs + mach_threshold={args.mach_threshold} vs nimbus"
        f"\n{os.path.basename(args.file)}"
    )
    fig.subplots_adjust(top=0.95)

    fig.savefig(args.out, dpi=120)
    print(f"Saved figure -> {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
