"""
validate_akrd_ssrd.py -- Standalone test that the akrd / ssrd functions in
wind_utils (calc_akrd for AKRD, calc_angle for SSRD) reproduce the AKRD and
SSRD variables stored in the netCDF, to numerical precision. Mirrors the
structure of validate_winds.py.

The recomputation reads the calibration coefficients straight off the file
(AKRD.AKRD_COEFF, AKRD.AKRD_COEFF_ALT, SSRD.CalibrationCoefficients) and the
input columns AKRD/SSRD depend on (ADIFR, BDIFR, QCF, PSFD -- see each
variable's Dependencies attribute).

For projects like GOTHAAM where on-ground flap state isn't logged, nimbus'
akrd.c picks between AKRD_COEFF (clean) and AKRD_COEFF_ALT (dirty) by dry
Mach number. The on-disk AKRDFLG variable records that choice per sample
(0 = clean, 1 = dirty), so we use it directly to switch coefficient sets and
also cross-check it against a mach-threshold reconstruction.

Run:
    cd test
    python validate_akrd_ssrd.py

Optional:
    python validate_akrd_ssrd.py --file OTHER.nc --show
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

# wind_utils.py lives one directory up
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import wind_utils  # noqa: E402


# --- helpers ----------------------------------------------------------------

def _read(nc, name):
    """Read a variable, returning float64 with _FillValue decoded to NaN."""
    a = nc.variables[name][:]
    return (a.filled(np.nan).astype(np.float64)
            if np.ma.isMaskedArray(a)
            else a.astype(np.float64))


def report(name, ref, test):
    diff = test - ref
    valid = np.isfinite(ref) & np.isfinite(test)
    nan_mismatch = int((np.isfinite(ref) ^ np.isfinite(test)).sum())
    if not valid.any():
        print(f"  {name}: no overlapping valid samples")
        return
    d = diff[valid]
    print(f"  {name:<18s}: N={valid.sum():>5d}  "
          f"max|diff|={np.max(np.abs(d)):.3e}  "
          f"rms={np.sqrt(np.mean(d * d)):.3e}  "
          f"NaN-mismatch={nan_mismatch}")


# --- main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=os.path.join(HERE, "GOTHAAMrf18.nc"),
                        help="netCDF file to validate against (default: GOTHAAMrf18.nc)")
    parser.add_argument("--out", default=os.path.join(HERE, "akrd_ssrd_validation.png"),
                        help="Output PNG path (default: akrd_ssrd_validation.png)")
    parser.add_argument("--show", action="store_true",
                        help="Also call plt.show() after saving the figure.")
    parser.add_argument("--mach-threshold", type=float, default=0.24,
                        help="Mach threshold for AKRD coef switch (default: 0.24)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"netCDF file not found: {args.file}")

    print(f"Loading {args.file} ...")
    with netCDF4.Dataset(args.file) as nc:
        time    = _read(nc, "Time")
        adifr   = _read(nc, "ADIFR")
        bdifr   = _read(nc, "BDIFR")
        qcf     = _read(nc, "QCF")
        psfd    = _read(nc, "PSFD")
        akrd_ref = _read(nc, "AKRD")
        ssrd_ref = _read(nc, "SSRD")
        akrdflg  = _read(nc, "AKRDFLG")
        aoa_coefs     = [float(c) for c in nc.variables["AKRD"].getncattr("AKRD_COEFF")]
        aoa_coefs_alt = [float(c) for c in nc.variables["AKRD"].getncattr("AKRD_COEFF_ALT")]
        ssrd_coefs    = [float(c) for c in nc.variables["SSRD"].getncattr("CalibrationCoefficients")]

    n = len(time)
    print(f"  {n} samples")
    print(f"  AKRD_COEFF     (clean) = {aoa_coefs}")
    print(f"  AKRD_COEFF_ALT (dirty) = {aoa_coefs_alt}")
    print(f"  SSRD coefs              = {ssrd_coefs}")
    print()

    # -- dry Mach: shared by both coef paths and used for the switch
    mach_dry = wind_utils.calc_mach_dry(qcf, psfd)
    a_ratio = adifr / qcf
    b_ratio = bdifr / qcf

    # -- AKRD via mach-threshold switch (matches calc_winds' internal logic)
    akrd_clean = wind_utils.calc_akrd(np.array([a_ratio, mach_dry]),
                                      *aoa_coefs,     qcf)
    akrd_dirty = wind_utils.calc_akrd(np.array([a_ratio, mach_dry]),
                                      *aoa_coefs_alt, qcf)
    use_alt_mt = mach_dry < args.mach_threshold
    akrd_py_mt = np.where(use_alt_mt, akrd_dirty, akrd_clean)

    # -- AKRD via the file's own AKRDFLG (0=clean, 1=dirty)
    use_alt_flg = (akrdflg == 1)
    akrd_py_flg = np.where(use_alt_flg, akrd_dirty, akrd_clean)

    # -- SSRD: single coef set, no switch
    ssrd_py = wind_utils.calc_angle(b_ratio, *ssrd_coefs, qcf)

    # -- cross-check the mach-threshold switch against AKRDFLG
    flg_valid = np.isfinite(akrdflg)
    n_disagree = int(np.sum(use_alt_mt[flg_valid] != use_alt_flg[flg_valid]))
    n_alt_mt   = int(use_alt_mt[flg_valid].sum())
    n_alt_flg  = int(use_alt_flg[flg_valid].sum())
    print(f"Coefficient-set switch (over {int(flg_valid.sum())} flagged samples):")
    print(f"  ALT (dirty) samples by mach_threshold={args.mach_threshold}: {n_alt_mt}")
    print(f"  ALT (dirty) samples by file AKRDFLG               : {n_alt_flg}")
    print(f"  per-sample disagreement: {n_disagree}")
    print()

    # -- diff stats vs nimbus
    print("Validation #1: AKRD reproduced via mach-threshold switch")
    print("  Differences (python_port - nimbus):")
    report("AKRD (mach-thr)", akrd_ref, akrd_py_mt)
    print()

    print("Validation #2: AKRD reproduced via file AKRDFLG")
    print("  Differences (python_port - nimbus):")
    report("AKRD (AKRDFLG)",  akrd_ref, akrd_py_flg)
    print()

    print("Validation #3: SSRD reproduced from CalibrationCoefficients")
    print("  Differences (python_port - nimbus):")
    report("SSRD",            ssrd_ref, ssrd_py)
    print()

    # -- Plot: AKRD and SSRD overlay + residuals, with a coef-set strip below
    py_components = [
        ("AKRD", akrd_ref, akrd_py_mt),
        ("SSRD", ssrd_ref, ssrd_py),
    ]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        5, 1, height_ratios=[3, 1, 3, 1, 0.8], hspace=0.18,
    )
    shared_x = None
    for k, (name, ref, py) in enumerate(py_components):
        ax_top = fig.add_subplot(gs[2 * k], sharex=shared_x)
        ax_top.plot(time, ref, color="C0", linewidth=0.8, label=f"nimbus {name}")
        ax_top.plot(time, py,  color="C1", linewidth=0.6, linestyle="--",
                    label="python port")
        ax_top.set_ylabel(f"{name} [deg]")
        ax_top.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax_top.grid(True, alpha=0.3)
        ax_top.tick_params(labelbottom=False)
        if shared_x is None:
            shared_x = ax_top

        ax_resid = fig.add_subplot(gs[2 * k + 1], sharex=shared_x)
        ax_resid.plot(time, py - ref, color="C3", linewidth=0.5)
        ax_resid.axhline(0, color="k", linewidth=0.3)
        ax_resid.set_ylabel("py - nimbus\n[deg]", fontsize=8)
        ax_resid.grid(True, alpha=0.3)
        ax_resid.tick_params(labelbottom=False)

        valid = np.isfinite(ref) & np.isfinite(py)
        if valid.any():
            mad = float(np.max(np.abs((py - ref)[valid])))
            ax_resid.text(0.99, 0.92, f"max|diff| = {mad:.2e}",
                          transform=ax_resid.transAxes, ha="right", va="top",
                          fontsize=8,
                          bbox=dict(facecolor="white", edgecolor="none",
                                    alpha=0.7, pad=1.5))

    # coef-set strip: where the dirty (ALT) AKRD coefs were active
    ax_strip = fig.add_subplot(gs[-1], sharex=shared_x)
    ax_strip.fill_between(time, 0, use_alt_mt.astype(float),
                          step="mid", alpha=0.5, color="C3",
                          label=f"mach<{args.mach_threshold}")
    ax_strip.set_ylabel("coef set")
    ax_strip.set_yticks([0, 1])
    ax_strip.set_yticklabels(["clean", "dirty"])
    ax_strip.set_xlabel("Time (s since 00:00 UTC)")
    ax_strip.grid(True, alpha=0.3)
    ax_strip.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        f"Python port wind_utils.calc_akrd / calc_angle vs nimbus AKRD / SSRD"
        f"\n{os.path.basename(args.file)}"
    )
    fig.subplots_adjust(top=0.93)

    fig.savefig(args.out, dpi=120)
    print(f"Saved figure -> {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
