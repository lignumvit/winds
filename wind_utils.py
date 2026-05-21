from scipy.optimize import curve_fit
import math
import copy
import os
import netCDF4
import pandas as pd
from datetime import timedelta
from fnmatch import fnmatch
import numpy as np
import itertools
from scipy import signal
import matplotlib.pyplot as plt

# constants from Nimbus
mol_wgt_dry_air = 28.9637 # kg/kmol
mol_wgt_water   = 18.01528 # kg/kmol  (nimbus xlate/const.c: MolecularWeightWater)
R0 = 8314.462618 # J/kmol/K
Rd = R0/mol_wgt_dry_air
Cpd = 7.0/2.0*Rd
Cvd = 5.0/2.0*Rd
Kelvin = 273.15
deg_to_rad = math.pi/180.

# Aircraft Constants
IRS_BOOM_LEN_GV = 4.42 # units: m
GPS_BOOM_LEN_GV = 8.79 # units: m
IRS_BOOM_LEN_C130 = 5.18 # units: m
GPS_BOOM_LEN_C130 = 10.87 # units: m

# Wrap-around thresholds used by nimbus' gust.c (radians).
# pitch jumps >=22.5deg between adjacent samples are treated as a discontinuity;
# heading jumps >=180deg are 0/360 wrap-arounds.
_PITCH_TEST = 22.5  * math.pi / 180.0
_THDG_TEST  = 180.0 * math.pi / 180.0

# Aircraft default attack / sideslip used when AKRD / SSLIP are NaN AND |roll| <= 2.5
# (mirrors defaultATTACK() / defaultSSLIP() in nimbus). These rarely fire in normal
# flight; values are placeholders that match nimbus' built-in defaults.
_DEFAULT_ATTACK = {'C130': 4.0, 'GV': 3.0}   # degrees
_DEFAULT_SSLIP  = {'C130': 0.0, 'GV': 0.0}   # degrees

def calc_mach_dry(q: np.array, ps: np.array):
    # a function to calculate mach given a dynamic (q) and static (ps) pressure
    # Sometimes q is small and negative. When it's negative, mach is imaginary. So, only calc
    # mach when q is >= 0.
    mach_dry = np.zeros(len(q))
    q_pos_inds = q >= 0
    mach_dry[q_pos_inds] = (5*(((ps[q_pos_inds] + q[q_pos_inds])/ps[q_pos_inds])**(Rd/Cpd) - 1))**0.5
    return mach_dry


def _humid_gas_consts(eop):
    """Humidity-corrected (R, Cp, Cv), matching nimbus gas_const.c.

    Parameters
    ----------
    eop : array-like or scalar
        Water-vapor pressure over static pressure (Ew / P). Pass 0.0 for the
        dry-air values (Rd, Cpd, Cvd).

    Returns
    -------
    R, Cp, Cv : same shape as ``eop``
        Specific gas constant and specific heats (J/(kg*K)).
    """
    eop = np.asarray(eop, dtype=np.float64)
    Fr = 1.0 / (1.0 + (mol_wgt_water/mol_wgt_dry_air - 1.0) * eop)
    R_h  = Rd  * Fr
    Cp_h = Cpd * Fr * (1.0 + eop / 7.0)
    Cv_h = Cvd * Fr * (1.0 + eop / 5.0)
    return R_h, Cp_h, Cv_h


def calc_mach(q: np.array, p: np.array, eop=0.0):
    """Aircraft Mach number from pitot-static pressures.

    Faithful port of nimbus' ``mach()`` in src/amlib/std/mach.c::

        M = sqrt( (2 Cv / R) * ( ((p + q) / p)^(R/Cp) - 1 ) )

    Parameters
    ----------
    q : array-like
        Dynamic (impact) pressure (hPa). For TASX/TASFR, nimbus uses QCFRC.
    p : array-like
        Static pressure (hPa). For TASFR's MACHFR, nimbus uses PSFC.
    eop : array-like or scalar, default 0.0 (dry)
        Water-vapor over static pressure. Pass 0.0 for a dry-air Mach
        (this is also what nimbus' ``mach_dry`` returns when ``Rd/Cpd = 2/7``).

    Returns
    -------
    mach : numpy.ndarray of float64
        Mach number. Samples with q < 0 or p <= 0 return NaN (avoids
        complex values from the fractional power).
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    R_h, Cp_h, Cv_h = _humid_gas_consts(eop)
    out = np.full(q.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(q) & np.isfinite(p) & (p > 0.0) & (q >= 0.0)
    if not valid.any():
        return out
    # Broadcasting: support scalar or per-sample eop
    R_b  = np.broadcast_to(R_h,  q.shape)
    Cp_b = np.broadcast_to(Cp_h, q.shape)
    Cv_b = np.broadcast_to(Cv_h, q.shape)
    ratio = (p[valid] + q[valid]) / p[valid]
    out[valid] = np.sqrt(
        (2.0 * Cv_b[valid] / R_b[valid])
        * (ratio**(R_b[valid] / Cp_b[valid]) - 1.0)
    )
    return out


def calc_tas(q: np.array,
             p: np.array,
             atx_c: np.array,
             ewx: np.array = None,
             eop=None,
             p_for_eop: np.array = None):
    """Aircraft True Airspeed, mirroring nimbus' TASX = compute_tas(ATX, MACH).

    Implements the chain (nimbus src/amlib/std/tas.c::compute_tas + mach.c)::

        eop  = ewx / p_for_eop                      (0 if ewx is None)
        Y, R = humidity-corrected specific heats and gas constant
        mach = sqrt( (2 Cv / R) * ((p+q)/p)^(R/Cp) - 1 )      (using eop)
        TAS  = mach * sqrt( Y * R * (atx_c + Kelvin) )        (using eop)

    The reference output is TASX, which on GOTHAAM equals TASFR -- computed
    from MACHFR (QCFRC, PSFC) and ATX, with humidity correction set by
    TASFLG using EWX and PSFDC. So a faithful reproduction calls this with
    ``q=QCFRC, p=PSFC, atx_c=ATX, ewx=EWX, p_for_eop=PSFDC``.

    Parameters
    ----------
    q : array-like
        Dynamic pressure (hPa). Pass QCFRC to match TASX/TASFR.
    p : array-like
        Static pressure used for the Mach calculation (hPa). Pass PSFC for
        TASFR; PSFDC for TASF.
    atx_c : array-like
        Ambient temperature (degC). Pass ATX.
    ewx : array-like, optional
        Water-vapor pressure (hPa). If supplied, humidity correction is
        applied via ``eop = ewx / p_for_eop``. If both ``ewx`` and ``eop``
        are None, a dry-air calculation is performed.
    eop : array-like or scalar, optional
        Direct way to specify the humidity coefficient (water-vapor /
        static pressure). Overrides ``ewx`` when given. Use ``eop=0.0`` for
        a dry calculation regardless of any ``ewx`` argument.
    p_for_eop : array-like, optional
        Static pressure to use as the denominator of ``eop = ewx / p_for_eop``.
        Defaults to ``p`` (which is fine when ``p`` is PSFC, since on GOTHAAM
        PSFC and PSFDC differ by only ~0.03 hPa). Pass PSFDC explicitly to
        reproduce nimbus' TASFLG/setEOP path exactly.

    Returns
    -------
    tas : numpy.ndarray of float64
        True airspeed (m/s). Samples where Mach can't be computed (q < 0,
        p <= 0, NaN inputs) are NaN.

    Notes
    -----
    * Reproduces nimbus TASX on GOTHAAMrf18.nc to rms ~2.4e-3 m/s. The
      single ~0.18 m/s residual sits at a dropout-recovery boundary where
      nimbus' TASFLG state machine carries over ``lastew`` from before the
      dropout while EWX is freshly back to a valid value; that 5-second
      hysteresis isn't tracked here.
    * Set ``ewx=None`` (or ``eop=0.0``) to see how much the humidity
      correction is worth on your flight.
    """
    q     = np.asarray(q,     dtype=np.float64)
    p     = np.asarray(p,     dtype=np.float64)
    atx_c = np.asarray(atx_c, dtype=np.float64)

    if eop is None:
        if ewx is None:
            eop_arr = np.zeros_like(q)
        else:
            ewx = np.asarray(ewx, dtype=np.float64)
            denom = p if p_for_eop is None else np.asarray(p_for_eop, dtype=np.float64)
            eop_arr = np.where(np.isfinite(ewx) & np.isfinite(denom) & (denom > 0.0),
                               ewx / denom, 0.0)
    else:
        eop_arr = np.broadcast_to(np.asarray(eop, dtype=np.float64), q.shape).copy()
        eop_arr[~np.isfinite(eop_arr)] = 0.0

    mach = calc_mach(q, p, eop=eop_arr)
    R_h, Cp_h, Cv_h = _humid_gas_consts(eop_arr)

    out = np.full(q.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(mach) & np.isfinite(atx_c) & ((atx_c + Kelvin) > 0.0)
    out[valid] = mach[valid] * np.sqrt(
        (Cp_h[valid] / Cv_h[valid]) * R_h[valid] * (atx_c[valid] + Kelvin)
    )
    return out

def calc_akrd(x: np.array, a: float, b: float, c: float, q: np.array = None):
    # x: a 2 by N array, where x[0,:] = ADIFR/QCF and x[1,:] = dry mach number
    # a, b, c: fit coefficients
    ratio = x[0,:]
    mach_dry = x[1,:]
    akrd = np.zeros(len(ratio))
    if q is not None:
        valid = q > 5.5 # true where dynamic pressure is greater than 5.5 hPa, like in nimbus
    else:
        valid = np.array(np.ones(len(ratio)), dtype='bool')
    akrd[valid] = a + ratio[valid]*(b + c*mach_dry[valid])
    return akrd
    
def calc_angle(ratio: np.array, a: float, b: float, q: np.array = None):
    # ratio: an N element array from ADIFR/QCF or BDIFR/QCF
    # a, b: fit coefficients
    sslip = np.zeros(len(ratio))
    if q is not None:
        valid = q > 5.5 # true where dynamic pressure is greater than 5.5 hPa, like in nimbus
    else:
        valid = np.array(np.ones(len(ratio)), dtype='bool')
    sslip[valid] = a + ratio[valid]*b
    return sslip

def calc_dvardt_backward(data_df: pd.DataFrame, varname: str):
    # compute rotation rates of pitch and yaw (true heading)
    # compute delta pitch
    N = len(data_df[varname])
    var = data_df[varname].to_numpy()
    dvar = np.zeros(N)
    dvar[1:] = var[1:] - var[0:N-1]

    # compute delta t
    deltat = np.zeros(N)
    deltat[0] = (data_df['datetime'][1] - data_df['datetime'][0]).total_seconds()
    for i in range(1,N):
        deltat[i] = (data_df['datetime'][i] - data_df['datetime'][i-1]).total_seconds()

    # if we're taking the derivative of heading, deal with the wrap around!
    if varname == "THDG":
        pos_inds = np.argwhere(dvar > 180).squeeze()
        neg_inds = np.argwhere(dvar < -180).squeeze()
        dvar[pos_inds] = var[pos_inds] - (var[pos_inds-1]+360)
        dvar[neg_inds] = var[neg_inds] - (var[neg_inds-1]-360)
        #for ind in pos_inds:
        #    print(f'delta: {dvar[ind]:7.2f}, hdg0: {var[ind-1]:7.2f}, hdg1: {var[i-1]:7.2f}')
        #for ind in neg_inds:
        #    print(f'delta: {dvar[ind]:7.2f}, hdg0: {var[ind-1]:7.2f}, hdg1: {var[i-1]:7.2f}')

    return dvar/deltat

def unrotate_vec(vx, vy, vz, pitch, roll, hdg):
    assert len(vx) == len(vy) == len(vz) == len(pitch) == len(roll) == len(hdg)

    pitch = pitch * deg_to_rad
    roll  = roll * deg_to_rad
    hdg   = hdg * deg_to_rad

    N = len(vx)
    vxnew = np.zeros(N)
    vynew = np.zeros(N)
    vznew = np.zeros(N)

    for i in range(N):
        unroll = np.array([[1,                 0,                 0,],
                           [0, math.cos(roll[i]), -math.sin(roll[i])],
                           [0, math.sin(roll[i]),  math.cos(roll[i])],])
        unpitch = np.array([[ math.cos(pitch[i]), 0, math.sin(pitch[i])],
                            [                  0, 1,                  0],
                            [-math.sin(pitch[i]), 0, math.cos(pitch[i])]])
        unhdg = np.array([[math.cos(hdg[i]), -math.sin(hdg[i]), 0],
                          [math.sin(hdg[i]),  math.cos(hdg[i]), 0],
                          [               0,                 0, 1]])
        vec = np.array([vx[i], vy[i], vz[i]])
        [vxnew[i], vynew[i], vznew[i]] = np.matmul(unhdg,np.matmul(unpitch,np.matmul(unroll, vec)))

    return [vxnew, vynew, vznew]
 
def calc_winds(data_df: pd.DataFrame,
               aoa_coefs=None,
               aoa_coefs_alt=None,
               aos_coefs=None,
               aircraft: str = 'C130',
               name_append: str = '_test',
               *,
               akrd=None,
               sslip=None,
               mach_threshold: float = 0.24,
               tas_var: str = 'TASX',
               vew_var: str = None,
               vns_var: str = None,
               vspd_var: str = 'GGVSPD',
               pitch_var: str = 'PITCH',
               roll_var: str = 'ROLL',
               thdg_var: str = 'THDG',
               adifr_var: str = 'ADIFR',
               bdifr_var: str = 'BDIFR',
               qcf_var: str = 'QCF',
               psf_var: str = 'PSFD',
               boom_len: float = None,
               dt: float = 1.0,
               wind_type: str = 'corrected'):
    """
    Faithful Python port of nimbus/src/amlib/std/gust.c (function swi()).

    Computes 3-D wind components (u, v, w) plus the WINDSFLG quality flag,
    matching nimbus output to numerical precision.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Must contain the input columns named by *_var arguments. Numeric values;
        nimbus' _FillValue (-32767) should be converted to NaN beforehand (the
        DataFrame from a masked netCDF read does this automatically).
    aoa_coefs, aos_coefs : array-like, optional
        Legacy path: when ``akrd``/``sslip`` are not supplied, attack and
        sideslip are recomputed from ADIFR/BDIFR/QCF using these coefficients
        (the original behavior of this function). Ignored otherwise.
    aoa_coefs_alt : array-like, optional
        Alternate AKRD coefficient set used in the "dirty" (flap-deployed)
        configuration. nimbus stores it as ``AKRD_COEFF_ALT`` in the AKRD
        variable's metadata. For GOTHAAM the on-ground flap state was not
        recorded, so nimbus' akrd.c selects between the two coef sets based on
        dry Mach number (see ``mach_threshold``). When this argument is None,
        only ``aoa_coefs`` is used everywhere.
    mach_threshold : float, default 0.24
        Boundary used to choose between the two coef sets when
        ``aoa_coefs_alt`` is supplied. Samples where ``mach_dry < mach_threshold``
        get the ALT (dirty) coefficients; samples where ``mach_dry >=
        mach_threshold`` get the clean coefficients. Set ``mach_threshold=0`` to
        disable the switch (use ``aoa_coefs`` everywhere) or ``=inf`` to use
        ``aoa_coefs_alt`` everywhere.
    aircraft : {'C130', 'GV'}
        Used to pick IRS_BOOM_LEN and the defaultATTACK/defaultSSLIP fall-backs.
    akrd, sslip : array-like, optional
        Attack and sideslip angles in degrees (one value per row of ``data_df``).
        Pass the nimbus-output ``AKRD``/``SSLIP`` (or ``ATTACK``/``SSRD``)
        columns to reproduce UIC/VIC/WIC bit-for-bit.
    tas_var, vew_var, vns_var, vspd_var : str
        Column names for true airspeed, ground-speed components, and vertical
        speed. ``vew_var``/``vns_var`` default to VEWC/VNSC for
        ``wind_type='corrected'`` (matches UIC/VIC/WIC) and VEW/VNS for
        ``wind_type='uncorrected'`` (matches UI/VI/WI).
    pitch_var, roll_var, thdg_var : str
        IRS attitude column names (degrees).
    adifr_var, bdifr_var, qcf_var, psf_var : str
        Column names used only when ``akrd``/``sslip`` are recomputed from
        ``aoa_coefs``/``aos_coefs``. ``psf_var`` defaults to ``'PSFD'`` (nimbus
        akrd.c's static-pressure input -- see the AKRD variable's
        "Dependencies" attribute in nimbus output: ``"3 ADIFR QCF PSFD"``).
        Switching to ``'PSXC'`` slightly shifts a few boundary samples.
    boom_len : float, optional
        Override the IRS_BOOM_LEN aircraft constant.
    dt : float
        Sample interval in seconds. 1.0 for low-rate (1 Hz) nimbus output.
    wind_type : {'corrected', 'uncorrected'}
        Sets the default velocity-feedback columns (see ``vew_var``).

    Returns
    -------
    u, v, w, ux, vy, wind_flag : numpy.ndarray
        Eastward, northward, and vertical wind components (m/s), then the
        aircraft-relative longitudinal (UXC; along heading) and lateral
        (VYC; perpendicular to heading) wind components (m/s), then the
        WINDSFLG quality code (0 good, 1 bad attack, 2 bad sideslip, 3 both).

        UXC and VYC are obtained by rotating (u, v) by the true heading
        (see gust.c)::

            ux =  u * sin(thdg) + v * cos(thdg)
            vy = -u * cos(thdg) + v * sin(thdg)
    """
    n = len(data_df)

    # ---- pick default velocity columns based on which wind product we're matching
    if vew_var is None:
        vew_var = 'VEWC' if wind_type == 'corrected' else 'VEW'
    if vns_var is None:
        vns_var = 'VNSC' if wind_type == 'corrected' else 'VNS'

    # ---- attack / sideslip: either passed in directly or recomputed from coefs
    if akrd is None:
        if aoa_coefs is None:
            raise ValueError("Either `akrd` or `aoa_coefs` must be supplied.")
        for need in (adifr_var, qcf_var, psf_var):
            if need not in data_df.columns:
                raise KeyError(f"calc_winds(aoa_coefs=...) needs column {need!r} "
                               f"in data_df. Either add it (PSXC is nimbus' "
                               f"standard static-pressure output) or pass "
                               f"akrd= directly.")
        a_ratio  = (data_df[adifr_var] / data_df[qcf_var]).to_numpy()
        qcf_arr  = data_df[qcf_var].to_numpy()
        mach_dry = calc_mach_dry(qcf_arr, data_df[psf_var].to_numpy())

        # ---- AKRD: nimbus' akrd.c selects between two coef sets per-sample.
        # For projects (like GOTHAAM) where on-ground flap state isn't logged,
        # akrd.c uses dry Mach as a proxy: mach_dry < mach_threshold -> ALT
        # (slower, "dirty"); mach_dry >= mach_threshold -> clean.
        akrd = calc_akrd(np.array([a_ratio, mach_dry]),
                         *aoa_coefs, qcf_arr)
        if aoa_coefs_alt is not None:
            akrd_alt = calc_akrd(np.array([a_ratio, mach_dry]),
                                 *aoa_coefs_alt, qcf_arr)
            use_alt = mach_dry < mach_threshold
            akrd = np.where(use_alt, akrd_alt, akrd)
        data_df['AKRD' + name_append] = akrd
    if sslip is None:
        if aos_coefs is None:
            raise ValueError("Either `sslip` or `aos_coefs` must be supplied.")
        for need in (bdifr_var, qcf_var):
            if need not in data_df.columns:
                raise KeyError(f"calc_winds(aos_coefs=...) needs column "
                               f"{need!r} in data_df, or pass sslip= directly.")
        b_ratio = (data_df[bdifr_var] / data_df[qcf_var]).to_numpy()
        sslip = calc_angle(b_ratio, *aos_coefs, data_df[qcf_var].to_numpy())
        data_df['SSLIP' + name_append] = sslip

    # ---- boom length: nimbus' gust.c always uses IRS_BOOM_LEN
    if boom_len is None:
        if aircraft == 'GV':
            boom_len = IRS_BOOM_LEN_GV
        elif aircraft == 'C130':
            boom_len = IRS_BOOM_LEN_C130
        else:
            raise ValueError(f"Unknown aircraft {aircraft!r}; pass boom_len explicitly.")

    # ---- pull inputs as plain float64 numpy arrays (avoid pandas/float32 quirks)
    tas    = np.asarray(data_df[tas_var],  dtype=np.float64)
    vew    = np.asarray(data_df[vew_var],  dtype=np.float64)
    vns    = np.asarray(data_df[vns_var],  dtype=np.float64)
    vspd   = np.asarray(data_df[vspd_var], dtype=np.float64)
    pitch_d = np.asarray(data_df[pitch_var], dtype=np.float64)  # degrees
    roll_d  = np.asarray(data_df[roll_var],  dtype=np.float64)  # degrees
    thdg_d  = np.asarray(data_df[thdg_var],  dtype=np.float64)  # degrees
    attack_d = np.asarray(akrd,  dtype=np.float64)               # degrees
    sslip_d  = np.asarray(sslip, dtype=np.float64)               # degrees

    default_attack = _DEFAULT_ATTACK.get(aircraft, 4.0)
    default_sslip  = _DEFAULT_SSLIP.get(aircraft,  0.0)

    # ---- outputs
    u = np.full(n, np.nan, dtype=np.float64)
    v = np.full(n, np.nan, dtype=np.float64)
    w = np.full(n, np.nan, dtype=np.float64)
    ux = np.full(n, np.nan, dtype=np.float64)
    vy = np.full(n, np.nan, dtype=np.float64)
    # wind_flag mirrors the static `wind_flag[probeCnt]` in gust.c, which is
    # zero-initialized and only ever touched inside swi(). Early-return paths
    # (NaN inputs or TAS<30) leave it unchanged. For a clean flight that means
    # 0 everywhere; nimbus' on-disk NaN at completely-missing samples is a
    # separate post-pass that we mimic at the end of this routine.
    wind_flag = np.zeros(n, dtype=np.float64)

    # ---- stateful loop (mirrors swi() in gust.c)
    # pitch0/thdg0 carry the previous-sample reference and obey the
    # firstTime initialization plus the pitch (22.5deg) / heading (180deg)
    # wrap-around tests.
    first_time = True
    pitch0 = 0.0
    thdg0  = 0.0

    for i in range(n):
        tas_i   = tas[i]
        vew_i   = vew[i]
        vns_i   = vns[i]
        vspd_i  = vspd[i]
        pitch_i = pitch_d[i] * deg_to_rad
        roll_i  = roll_d[i]                # left in degrees for the |roll|<=2.5 test
        thdg_i  = thdg_d[i] * deg_to_rad
        attack_i = attack_d[i]
        sslip_i  = sslip_d[i]

        # ---- early return if any of pitch/roll/thdg/tas/vspd is NaN
        if (np.isnan(pitch_i) or np.isnan(roll_i) or np.isnan(thdg_i)
                or np.isnan(tas_i) or np.isnan(vspd_i)):
            u[i] = np.nan; v[i] = np.nan; w[i] = np.nan
            ux[i] = np.nan; vy[i] = np.nan
            continue

        # ---- blow-up protection while on ground (TAS < 30 m/s)
        if tas_i < 30.0:
            u[i] = 0.0; v[i] = 0.0; w[i] = 0.0
            ux[i] = 0.0; vy[i] = 0.0
            continue

        # ---- first-time seed of pitch0/thdg0 so thedot/psidot are 0 at t=0
        if first_time:
            pitch0 = pitch_i
            thdg0  = thdg_i
            first_time = False

        # ---- attack / sideslip NaN handling (sets WINDSFLG)
        attack_compromised = False
        sslip_compromised  = False
        if np.isnan(attack_i):
            attack_compromised = True
            wind_flag[i] = 1.0
            if abs(roll_i) <= 2.5:
                attack_i = default_attack
        if np.isnan(sslip_i):
            sslip_compromised = True
            wind_flag[i] = 3.0 if attack_compromised else 2.0
            if abs(roll_i) <= 2.5:
                sslip_i = default_sslip

        attack_r = attack_i * deg_to_rad
        sslip_r  = sslip_i  * deg_to_rad
        roll_r   = roll_i   * deg_to_rad

        # ---- coordinate-transform trig (variable names match gust.c)
        cs = math.cos(thdg_i)
        ss = math.sin(thdg_i)
        ch = math.cos(pitch_i)
        sh = math.sin(pitch_i)
        cr = math.cos(roll_r)
        sr = math.sin(roll_r)
        ta = math.tan(attack_r)
        tb = math.tan(sslip_r)

        tas_dab = -tas_i / math.sqrt(1.0 + ta*ta + tb*tb)
        e  = -tb * (ss * cr - cs * sh * sr)
        f  =  cs * sh * cr + ss * sr
        h  =  tb * (cs * cr + ss * sh * sr)
        p  =  ss * sh * cr - cs * sr
        ab =  sh - tb * ch * sr - ta * ch * cr

        # ---- pitch / heading angular rates with gust.c's wrap-around tests
        delph = pitch_i - pitch0
        if abs(delph) >= _PITCH_TEST:
            pitch0 += _PITCH_TEST * 2.0 if delph > 0 else -_PITCH_TEST * 2.0
        thedot = (pitch_i - pitch0) / dt

        delth = thdg_i - thdg0
        if abs(delth) >= _THDG_TEST:
            thdg0 += _THDG_TEST * 2.0 if delth > 0 else -_THDG_TEST * 2.0
        psidot = (thdg_i - thdg0) / dt

        bvns = boom_len * (psidot * ss * ch + thedot * cs * sh)
        bvew = boom_len * (thedot * sh * ss - psidot * cs * ch)

        # update reference for next iteration (only if we made it this far,
        # exactly as in gust.c: early-return paths above leave pitch0/thdg0 alone)
        pitch0 = pitch_i
        thdg0  = thdg_i

        # ---- wind components in geographic coordinates
        r = ss * ch + h + ta * p
        s = cs * ch + e + ta * f
        t = vspd_i + boom_len * thedot * ch

        u[i] = tas_dab * r + (vew_i - bvew)
        v[i] = tas_dab * s + (vns_i - bvns)
        w[i] = (np.nan if attack_compromised
                else tas_dab * ab + t)

        # ---- aircraft-relative wind components (gust.c lines 298-299):
        # rotation of (ui, vi) by true heading. Computed regardless of
        # attack_compromised (nimbus does not NaN ux/vy in that case).
        ux[i] =  u[i] * ss + v[i] * cs
        vy[i] = -u[i] * cs + v[i] * ss

    # ---- match nimbus' on-disk WINDSFLG: NaN where the whole input row is
    # missing (sync_server has no data to interpolate from), otherwise the
    # value the static wind_flag carries.
    all_missing = (np.isnan(tas) & np.isnan(vew) & np.isnan(vns)
                   & np.isnan(vspd) & np.isnan(pitch_d) & np.isnan(roll_d)
                   & np.isnan(thdg_d) & np.isnan(attack_d) & np.isnan(sslip_d))
    wind_flag[all_missing] = np.nan

    return [u, v, w, ux, vy, wind_flag]



def plot_winds_comparison(data_df,
                          time_var='Time',
                          aircraft='C130',
                          u_py=None, v_py=None, w_py=None,
                          wind_type='corrected',
                          time_range=None,
                          figsize=(12, 9),
                          title=None,
                          **calc_winds_kwargs):
    """
    Plot nimbus UIC/VIC/WIC against the Python-computed counterparts.

    Layout: three stacked sections (UIC, VIC, WIC). Each section has a tall
    overlay axis (both curves on top of each other) and a short residual axis
    below (python - nimbus).

    Parameters
    ----------
    data_df : pandas.DataFrame
        Must contain UIC, VIC, WIC and the inputs needed by ``calc_winds``
        (TASX, VEWC/VNSC (or VEW/VNS), GGVSPD/VSPD, PITCH, ROLL, THDG, AKRD,
        SSLIP), plus a time column named by ``time_var``.
    time_var : str, default 'Time'
        Column to use for the x-axis.
    aircraft : {'C130', 'GV'}, default 'C130'
        Passed through to ``calc_winds`` (sets IRS_BOOM_LEN).
    u_py, v_py, w_py : array-like, optional
        Pre-computed Python wind components. If any is None, all three are
        recomputed via ``calc_winds`` using the file's AKRD/SSLIP.
    wind_type : {'corrected', 'uncorrected'}, default 'corrected'
        Which nimbus product to compare against: 'corrected' -> UIC/VIC/WIC,
        'uncorrected' -> UI/VI/WI.
    time_range : tuple(float, float), optional
        ``(t_min, t_max)`` to zoom into a sub-interval of the flight.
    figsize : tuple
        Matplotlib figure size.
    title : str, optional
        Suptitle. If None, a default is built from ``aircraft`` and ``wind_type``.
    **calc_winds_kwargs :
        Forwarded to ``calc_winds`` (e.g. ``tas_var``, ``vew_var``, ``dt``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : dict
        Mapping of label ('UIC_overlay', 'UIC_resid', ...) to the corresponding
        ``matplotlib.axes.Axes`` for further customization.
    """
    # pick which nimbus variables to compare against
    if wind_type == 'corrected':
        ref_names = ('UIC', 'VIC', 'WIC')
    elif wind_type == 'uncorrected':
        ref_names = ('UI', 'VI', 'WI')
    else:
        raise ValueError(f"wind_type must be 'corrected' or 'uncorrected', got {wind_type!r}")

    # compute Python winds if not supplied
    if u_py is None or v_py is None or w_py is None:
        akrd  = data_df['AKRD'].to_numpy()  if 'AKRD'  in data_df.columns else None
        sslip = data_df['SSLIP'].to_numpy() if 'SSLIP' in data_df.columns else None
        u_py, v_py, w_py, _, _, _ = calc_winds(
            data_df, aircraft=aircraft,
            akrd=akrd, sslip=sslip,
            wind_type=wind_type,
            **calc_winds_kwargs,
        )

    t = np.asarray(data_df[time_var])

    # optional time-range mask
    if time_range is not None:
        mask = (t >= time_range[0]) & (t <= time_range[1])
        t = t[mask]
        u_py = np.asarray(u_py)[mask]
        v_py = np.asarray(v_py)[mask]
        w_py = np.asarray(w_py)[mask]
        u_ref = np.asarray(data_df[ref_names[0]])[mask]
        v_ref = np.asarray(data_df[ref_names[1]])[mask]
        w_ref = np.asarray(data_df[ref_names[2]])[mask]
    else:
        u_ref = np.asarray(data_df[ref_names[0]])
        v_ref = np.asarray(data_df[ref_names[1]])
        w_ref = np.asarray(data_df[ref_names[2]])

    py_arrays  = (u_py, v_py, w_py)
    ref_arrays = (u_ref, v_ref, w_ref)

    # build the 6-row layout: overlay (height 3) + residual (height 1) per component
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(6, 1, height_ratios=[3, 1, 3, 1, 3, 1], hspace=0.15)
    axes = {}
    prev_overlay = None
    for k, (label, ref, py) in enumerate(zip(ref_names, ref_arrays, py_arrays)):
        ax_top    = fig.add_subplot(gs[2 * k],     sharex=prev_overlay)
        ax_bottom = fig.add_subplot(gs[2 * k + 1], sharex=ax_top)
        prev_overlay = ax_top

        ax_top.plot(t, ref, color='C0', linewidth=0.8, label=f'nimbus {label}')
        ax_top.plot(t, py,  color='C1', linewidth=0.6, linestyle='--',
                    label='python port')
        ax_top.set_ylabel(f'{label} [m/s]')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc='upper right', fontsize=8, framealpha=0.85)
        ax_top.tick_params(labelbottom=False)

        resid = py - ref
        ax_bottom.plot(t, resid, color='C3', linewidth=0.6)
        ax_bottom.axhline(0, color='k', linewidth=0.3)
        ax_bottom.set_ylabel('py - nimbus\n[m/s]', fontsize=8)
        ax_bottom.grid(True, alpha=0.3)
        if k < 2:
            ax_bottom.tick_params(labelbottom=False)

        # annotate max abs diff
        valid = np.isfinite(ref) & np.isfinite(py)
        if valid.any():
            mad = float(np.max(np.abs(resid[valid])))
            ax_bottom.text(0.99, 0.92, f'max|diff| = {mad:.2e}',
                           transform=ax_bottom.transAxes, ha='right', va='top',
                           fontsize=8,
                           bbox=dict(facecolor='white', edgecolor='none',
                                     alpha=0.7, pad=1.5))

        axes[f'{label}_overlay'] = ax_top
        axes[f'{label}_resid']   = ax_bottom

    axes[f'{ref_names[-1]}_resid'].set_xlabel(time_var)

    if title is None:
        title = (f'nimbus vs Python port  ({aircraft}, {wind_type} winds, '
                 f'N={len(t):d} samples)')
    fig.suptitle(title)
    fig.subplots_adjust(top=0.94)
    return fig, axes


class flight_data:
    # a fancy dataclass that computes quite a few things in the init
    def __init__(self, df: pd.DataFrame, orig_akrd_coefs, orig_sslip_coefs, aircraft):
        self.df = df
        #self.flight = flight
        self.varnamemap = {'datetime': 'dt', 'QCF': 'q', 'PSF': 'ps', 'ADIFR': 'adifr', 'BDIFR': 'bdifr', 'SSLIP': 'sslip',
                           'GGVSPD': 'vspd', 'TASF': 'tas', 'PITCH': 'pitch', 'ROLL': 'roll', 'AKRD': 'akrd', 'PALTF': 'paltf', 'THDG': 'hdg',
                           'GGALT': 'alt', 'ATX': 'tc', 'PSXC': 'p', 'UIC': 'u', 'VIC': 'v', 'WIC': 'w', 'GGVEW': 'up', 'GGVNS': 'vp', 'GGTRK': 'trk'}
        self.r2d = 180./math.pi
        # get basic vars needed

        for key, val in self.varnamemap.items():
            setattr(self, val, df[key].to_numpy())
        self.mach = calc_mach_dry(self.q, self.ps)
        self.aoa_corr = -np.arcsin(self.vspd/self.tas)*self.r2d
        self.aoa_ref = self.pitch + self.aoa_corr
        self.tk = self.tc + 273.15
        self.p = self.p*100 # hPa to Pa
        self.rho = self.p/Rd/self.tk

        [utest, vtest, wtest, uxtest, vytest, wflag] = calc_winds(self.df, orig_akrd_coefs, orig_sslip_coefs, aircraft)
        self.utest = utest
        self.vtest = vtest
        self.wtest = wtest
        self.uxtest = uxtest
        self.vytest = vytest
        self.wflag_test = wflag

        self.dpitchdt = calc_dvardt_backward(self.df, 'PITCH')
        self.dhdgdt = calc_dvardt_backward(self.df, 'THDG')
        self.drolldt = calc_dvardt_backward(self.df, 'ROLL')

        [self.brollr, self.bpitchr, self.bhdgr] = unrotate_vec(self.df['BROLLR'], 
                                                               self.df['BPITCHR'], 
                                                               self.df['BYAWR'],
                                                               self.pitch, self.roll, self.hdg)

