"""TokaMaker interface – perturbed Grad-Shafranov equilibrium workflow
=====================================================================

Routines that interact directly with TokaMaker to iterate perturbed
kinetic / current-density profiles toward target :math:`I_p` and
:math:`l_i` values.

Provides:
  - ``fit_inductive_profile`` – spline-based fit of the inductive
    current-density profile, scaled to match a target :math:`l_i` proxy.
  - ``perturb_kinetic_equilibrium`` – perturbs kinetic + current-density
    profiles and iterates to match :math:`I_p` and :math:`l_i` targets
    via TokaMaker.
  - ``generate_perturbed_equilibria`` – batch driver that archives
    perturbed equilibria to HDF5.
  - ``reconstruct_equilibrium`` – reconstruct a single equilibrium from
    a geqdsk reference and kinetic profiles, matching :math:`l_i(1)`
    via secant iteration.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

# Pad used when tracing the lock-mode reference LCFS contour.  Smaller
# than the OFT solver psi_pad (typically 1e-3) so the dpsi/grad_psi
# conversion in psi_boundary_deviation isn't contaminated by the pad
# itself — at psi_pad=1e-3 the phantom offset is ~15 mm on DIII-D.
# 1e-6 is safely traceable while keeping the phantom < 1 μm.
_LOCK_REF_TRACE_PAD = 1.0e-6

from .sampling import (
    generate_perturbed_GPR,
    calc_cylindrical_li_proxy,
    get_li_proxy_geometry,
    calc_cylindrical_li_proxy_fast,
    _draw_monotonic_perturbation,
    EC,
    _MAX_PRESSURE_ITER,
    _MAX_LI_ITER,
)
from .utils import (
    Ip_flux_integral_vs_target,
    store_equilibrium,
    store_baseline_profiles,
)

# ---- Adaptive corrective iteration ----
def _corrective_jphi_iteration(mygs, psi_N, target_jphi, pp_prof,
                                Ip_target, pax_target, psi_pad,
                                min_iters=2, max_iters=8,
                                rtol=0.05, verbose=True):
    r"""Iterate TokaMaker input j_phi until the output matches a target.

    Uses Newton correction: ``input += (target - output)`` each step.
    Starts with *min_iters*, then checks whether the edge spike RMS
    is still improving by more than *rtol* relative per step.  Stops
    when converged or *max_iters* is reached.

    Parameters
    ----------
    mygs : TokaMaker
        GS solver (in a solved state with current profiles set).
    psi_N : ndarray
        Normalised flux grid.
    target_jphi : ndarray
        Target j_phi profile [A/m²] (e.g. j_inductive + spike_profile).
    pp_prof : dict
        Pressure gradient profile dict for ``set_profiles``.
    Ip_target : float
        Plasma current target [A].
    pax_target : float
        On-axis pressure target [Pa].
    psi_pad : float
        LCFS padding.
    min_iters : int
        Minimum iterations before checking convergence (default 2).
    max_iters : int
        Maximum iterations (default 8).
    rtol : float
        Relative improvement threshold — stop if
        ``|rms_new - rms_old| / rms_old < rtol`` (default 0.05 = 5%).
    verbose : bool
        Print per-iteration diagnostics.

    Returns
    -------
    j_phi_output : ndarray
        Converged GS output j_phi [A/m²].
    n_iters : int
        Number of iterations performed.
    edge_rms_history : list of float
        Edge RMS per iteration [A/m²].
    """
    from OpenFUSIONToolkit.TokaMaker.util import get_jphi_from_GS

    npsi = len(psi_N)
    edge_mask = psi_N > 0.9
    j_phi_input = target_jphi.copy()
    # Initialise j_phi_output so the function can return safely if the
    # very first solve fails — without this, hitting ``break`` on the
    # first iteration left j_phi_output unbound and the return raised
    # a NameError that escaped to the bouquet runner.
    j_phi_output = target_jphi.copy()
    edge_rms_history = []

    for it in range(max_iters):
        ffp = {"type": "jphi-linterp", "y": j_phi_input.copy(), "x": psi_N}
        mygs.set_targets(Ip=Ip_target, pax=pax_target)
        mygs.set_profiles(pp_prof=pp_prof, ffp_prof=ffp)
        try:
            mygs.solve()
        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"  [jphi_corr iter {it+1}] solve failed: {e}")
            break

        _, f, fp, _, pp = mygs.get_profiles(npsi=npsi, psi_pad=psi_pad)
        _, _, ravgs, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
        j_phi_output = get_jphi_from_GS(f * fp, pp, ravgs[0], ravgs[1])

        diff = j_phi_output - target_jphi
        rms_edge = float(np.sqrt(np.mean(diff[edge_mask]**2)))
        edge_rms_history.append(rms_edge)

        if verbose:
            print(f"  [jphi_corr iter {it+1}] edge RMS = {rms_edge/1e6:.6f} MA/m²")

        # Check convergence after min_iters
        if it >= min_iters - 1 and len(edge_rms_history) >= 2:
            prev_rms = edge_rms_history[-2]
            if prev_rms > 0:
                rel_change = abs(rms_edge - prev_rms) / prev_rms
                if rel_change < rtol:
                    if verbose:
                        print(f"  [jphi_corr] converged at iter {it+1} "
                              f"(rel_change={rel_change:.4f} < {rtol})")
                    break

        # Newton correction
        j_phi_input = j_phi_input + (target_jphi - j_phi_output)
        j_phi_input = np.maximum(j_phi_input, 0.0)

    return j_phi_output, it + 1, edge_rms_history


# ---- j_phi profile classifier ----
def classify_jphi_profile(psi_N, eqdsk_jphi, spike_profile,
                          edge_psi_min=0.5, prominence_frac=0.15):
    r"""Classify the edge current profile to determine reconstruction strategy.

    Parameters
    ----------
    psi_N : ndarray
        Normalised poloidal flux grid.
    eqdsk_jphi : ndarray
        Toroidal current density from the geqdsk [A/m²].
    spike_profile : ndarray
        Isolated edge bootstrap spike from ``analyze_bootstrap_edge_spike``
        [A/m²].  Flat shelf in the core, rising spike at the edge.
    edge_psi_min : float
        Inner boundary of the edge search window (default 0.5).
    prominence_frac : float
        Minimum peak prominence as a fraction of the edge range
        (default 0.15).

    Returns
    -------
    mode : str
        One of ``'H_mode'``, ``'Lmode_like_jphi'``, ``'L_mode'``.
    metrics : dict
        Edge spike metrics for reconstruction quality tracking.
    """
    from scipy.signal import find_peaks

    metrics = {}
    edge_mask = psi_N >= edge_psi_min

    # --- Check Sauter spike ---
    spike_edge = spike_profile[edge_mask]
    psi_edge = psi_N[edge_mask]
    shelf_val = spike_profile[0]
    spike_range = np.max(spike_edge) - shelf_val

    if spike_range > 0:
        min_prom_spike = prominence_frac * spike_range
        peaks_s, _ = find_peaks(spike_edge, height=shelf_val,
                                prominence=min_prom_spike)
    else:
        peaks_s = np.array([], dtype=int)

    if len(peaks_s) == 0:
        # No Sauter edge spike → L_mode
        metrics['spike_height_sauter'] = 0.0
        metrics['spike_psiN_sauter'] = None
        metrics['spike_height_geqdsk'] = None
        metrics['spike_psiN_geqdsk'] = None
        metrics['spike_height_ratio'] = None
        metrics['spike_psiN_offset'] = None
        print(f"[classify] L_mode — no Sauter edge spike detected")
        return 'L_mode', metrics

    # Sauter spike exists — record its peak
    best_s = peaks_s[np.argmax(spike_edge[peaks_s])]
    metrics['spike_height_sauter'] = float(spike_edge[best_s])
    metrics['spike_psiN_sauter'] = float(psi_edge[best_s])

    # --- Check geqdsk for an edge peak ---
    geqdsk_edge = eqdsk_jphi[edge_mask]
    geqdsk_baseline = geqdsk_edge[0]  # value at edge_psi_min
    geqdsk_range = np.max(geqdsk_edge) - geqdsk_baseline

    # Use prominence only — no height filter. The edge spike may be
    # below the core value (j_phi decreases from core to edge) but
    # still be a significant local peak.
    geqdsk_max = np.max(geqdsk_edge)
    if geqdsk_max > 0:
        min_prom_g = prominence_frac * geqdsk_max
        peaks_g, props_g = find_peaks(geqdsk_edge, prominence=min_prom_g)
        # Filter to psi_N > 0.85 (true edge peaks, not shoulder of core)
        if len(peaks_g) > 0:
            far_edge = psi_edge[peaks_g] > 0.85
            if np.any(far_edge):
                peaks_g = peaks_g[far_edge]
            else:
                peaks_g = np.array([], dtype=int)
    else:
        peaks_g = np.array([], dtype=int)

    if len(peaks_g) > 0:
        # geqdsk has an edge peak → H_mode
        best_g = peaks_g[np.argmax(geqdsk_edge[peaks_g])]
        metrics['spike_height_geqdsk'] = float(geqdsk_edge[best_g])
        metrics['spike_psiN_geqdsk'] = float(psi_edge[best_g])
        metrics['spike_height_ratio'] = (
            metrics['spike_height_sauter'] / metrics['spike_height_geqdsk']
        )
        metrics['spike_psiN_offset'] = (
            metrics['spike_psiN_sauter'] - metrics['spike_psiN_geqdsk']
        )
        print(f"[classify] H_mode — geqdsk peak at psi_N={metrics['spike_psiN_geqdsk']:.4f} "
              f"({metrics['spike_height_geqdsk']/1e6:.4f} MA/m²), "
              f"Sauter peak at psi_N={metrics['spike_psiN_sauter']:.4f} "
              f"({metrics['spike_height_sauter']/1e6:.4f} MA/m²), "
              f"height ratio={metrics['spike_height_ratio']:.2f}, "
              f"psiN offset={metrics['spike_psiN_offset']:.4f}")
        return 'H_mode', metrics
    else:
        # geqdsk has no edge peak → Lmode_like_jphi
        metrics['spike_height_geqdsk'] = None
        metrics['spike_psiN_geqdsk'] = None
        metrics['spike_height_ratio'] = None
        metrics['spike_psiN_offset'] = None
        print(f"[classify] Lmode_like_jphi — no geqdsk edge peak, "
              f"Sauter spike at psi_N={metrics['spike_psiN_sauter']:.4f} "
              f"({metrics['spike_height_sauter']/1e6:.4f} MA/m²)")
        return 'Lmode_like_jphi', metrics


# ---- Shelf-blend decomposition helper ----
def _shelf_blend_decompose(psi_N, j_phi_total, spike_profile,
                           eqdsk_jphi=None):
    r"""Decompose j_phi into j_inductive + spike_profile.

    In the core (where the spike shelf is flat), j_inductive = j_phi - spike.
    At the edge (beyond the shelf), j_inductive is replaced by an
    optimised cubic Hermite that:

    - matches value and derivative of the core j_inductive at the
      exact shelf-end index (C1 join, no blend window),
    - minimises ``||j_ind_bridge + spike - eqdsk_jphi||²`` in the
      edge region by optimising the two free slopes,
    - is constrained to be monotonically decreasing and non-negative,
    - arrives at ``max(eqdsk_jphi[-1] - spike_profile[-1], 0)``
      at psi_N = 1.

    Parameters
    ----------
    psi_N : ndarray
        Normalised flux grid.
    j_phi_total : ndarray
        Total toroidal current density [A/m²].
    spike_profile : ndarray
        Isolated edge bootstrap spike [A/m²] (flat shelf + edge peak).
    eqdsk_jphi : ndarray or None
        Original geqdsk j_phi [A/m²].  Used to set the edge boundary
        value and as the optimisation target.  If ``None``, a simple
        linear taper is used.

    Returns
    -------
    j_inductive : ndarray
        Inductive component (non-negative, smooth edge taper).
    shelf_psi : float
        psi_N location where the shelf ends.
    """
    j_ind_raw = j_phi_total - spike_profile

    # Find where the shelf ends
    shelf_val = spike_profile[0]
    shelf_end = 0
    for i in range(1, len(spike_profile)):
        if abs(spike_profile[i] - shelf_val) / max(abs(shelf_val), 1e-30) < 1e-6:
            shelf_end = i
        else:
            break
    shelf_psi = psi_N[shelf_end]

    # Edge target: spike[-1] + j_ind[-1] = eqdsk[-1], unless spike dominates
    if eqdsk_jphi is not None:
        edge_target = max(eqdsk_jphi[-1] - spike_profile[-1], 0.0)
    else:
        edge_target = 0.0

    # Value at shelf end (from core subtraction)
    val_at_shelf = max(j_ind_raw[shelf_end], 0.0)

    # Estimate the slope that j_inductive needs at the shelf end so that
    # the TOTAL j_phi = j_ind + spike has a smooth derivative there.
    #
    # On the core side: dj_phi/dpsi = dj_ind/dpsi + 0 (spike is flat).
    # On the edge side: dj_phi/dpsi = dj_ind/dpsi + dspike/dpsi.
    # For C1 in total j_phi: the Hermite's dj_ind/dpsi at t=0 should
    # equal the core dj_phi/dpsi minus the spike's derivative just
    # past the shelf end.
    #
    # Core total j_phi slope (5-point stencil):
    n_stencil = min(5, shelf_end)
    if n_stencil >= 2:
        _sl_idx = shelf_end - n_stencil
        _sl_dy = j_ind_raw[shelf_end] - j_ind_raw[_sl_idx]
        _sl_dx = psi_N[shelf_end] - psi_N[_sl_idx]
        core_jphi_slope = _sl_dy / _sl_dx if _sl_dx > 0 else 0.0
    else:
        core_jphi_slope = 0.0

    # Spike derivative just past the shelf end
    if shelf_end < len(psi_N) - 2:
        spike_slope_at_edge = ((spike_profile[shelf_end + 2] - spike_profile[shelf_end])
                               / (psi_N[shelf_end + 2] - psi_N[shelf_end]))
    else:
        spike_slope_at_edge = 0.0

    # j_inductive start slope = core total slope - spike slope
    # so that dj_phi/dpsi = dj_ind/dpsi + dspike/dpsi = core total slope
    core_slope_est = core_jphi_slope - spike_slope_at_edge

    interval = 1.0 - shelf_psi

    if interval <= 0 or eqdsk_jphi is None:
        # Fallback: linear taper
        j_ind = j_ind_raw.copy()
        for i in range(len(psi_N)):
            if psi_N[i] > shelf_psi:
                t = (psi_N[i] - shelf_psi) / max(interval, 1e-30)
                j_ind[i] = val_at_shelf * (1 - t) + edge_target * t
        return np.maximum(j_ind, 0.0), shelf_psi

    # Hermite bridge: 4 DOF, 3 fixed, 1 optimised.
    #   Start value = val_at_shelf (C0 match, fixed)
    #   Start slope = core_slope_est (C1 match, fixed)
    #   End value   = edge_target (fixed)
    #   End slope   = optimised to minimise ||j_ind + spike - eqdsk||²
    edge_mask = psi_N >= shelf_psi
    psi_edge = psi_N[edge_mask]
    spike_edge = spike_profile[edge_mask]
    eqdsk_edge = eqdsk_jphi[edge_mask]

    m0_fixed = core_slope_est * interval  # scaled start slope (exact C1)

    def _build_hermite_arr(m1_scaled):
        """Build Hermite on edge grid with fixed m0."""
        t = (psi_edge - shelf_psi) / interval
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        return h00 * val_at_shelf + h10 * m0_fixed + h01 * edge_target + h11 * m1_scaled

    def _cost(m1_scaled):
        bridge = np.maximum(_build_hermite_arr(m1_scaled), 0.0)
        total = bridge + spike_edge
        residual = total - eqdsk_edge
        cost = np.mean(residual**2)
        # Penalty for non-monotonic bridge
        dbridge = np.diff(bridge)
        violations = np.sum(np.maximum(dbridge, 0.0)**2)
        return cost + 10.0 * violations

    from scipy.optimize import minimize_scalar

    res = minimize_scalar(
        _cost, bounds=(-5.0 * val_at_shelf, 5.0 * val_at_shelf),
        method='bounded',
    )
    m1_opt = res.x
    m0_opt = m0_fixed

    # Build final bridge on the full grid (inclusive of shelf_end index)
    j_ind = j_ind_raw.copy()
    for i in range(len(psi_N)):
        if psi_N[i] >= shelf_psi:
            t = (psi_N[i] - shelf_psi) / interval
            t = min(t, 1.0)
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2
            j_ind[i] = (h00 * val_at_shelf + h10 * m0_opt
                        + h01 * edge_target + h11 * m1_opt)

    return np.maximum(j_ind, 0.0), shelf_psi


# ---- Spline-based inductive profile fitting ----
def fit_inductive_profile(mygs, eqdsk_jtor, j_BS_isolated, psi_N, psi_pad,
                            baseline_li_proxy,
                            k=3, psi_bridge=0.99,
                            rescale_j_BS=False,
                            shelf_psi_N=0.0,
                            n_spline_pts=32):
    r"""Fit a smooth inductive current profile and scale it to match
    a target cylindrical :math:`l_i` proxy.

    Fits a ``scipy.interpolate.UnivariateSpline`` to
    ``eqdsk_jtor - j_BS_isolated``, enforcing a zero edge anchor and
    optionally bridging over the edge region with *psi_bridge*.  Scales
    the resulting inductive basis profile (and optionally the bootstrap
    current) so that the total :math:`j_\phi` reproduces
    *baseline_li_proxy*.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker Grad-Shafranov solver object.
    eqdsk_jtor : ndarray
        1-D target total :math:`j_{\rm tor}` from the geqdsk [A m\ :sup:`-2`].
    j_BS_isolated : ndarray
        1-D isolated bootstrap current profile [A m\ :sup:`-2`].
    psi_N : ndarray
        1-D normalised poloidal flux grid.
    psi_pad : float
        Padding inside the LCFS for the :math:`l_i` proxy calculation.
    baseline_li_proxy : float
        Target cylindrical :math:`l_i` proxy value.
    k : int
        Spline order (default 3).
    psi_bridge : float
        :math:`\hat{\psi}` above which data are replaced by the edge
        anchor point (default 0.99).
    rescale_j_BS : bool
        If ``True``, jointly optimise a bootstrap rescaling factor to
        minimise the RMS residual against *eqdsk_jtor*.
        ``False`` (default) scales the inductive profile only.
    shelf_psi_N : float
        If > 0, apply a flat shelf to *j_BS_isolated* for
        :math:`\hat{\psi} <` *shelf_psi_N*, using the value of
        *j_BS_isolated* at that location.  ``0`` disables the shelf.

    Returns
    -------
    dict
        ``'j_inductive_fit'`` -- fitted inductive profile (scaled)
            [A m\ :sup:`-2`].
        ``'j_phi_fit'`` -- total :math:`j_\phi = j_{\rm ind} + b_{\rm scale}\,j_{\rm BS}`
            [A m\ :sup:`-2`].
        ``'fit_li'`` -- :math:`l_i` proxy of ``j_phi_fit``.
        ``'ind_scale'`` -- inductive scaling factor applied.
        ``'bs_scale'`` -- bootstrap scaling factor (1.0 when
            ``rescale_j_BS=False``).
        ``'j_BS_used'`` -- *j_BS_isolated* after optional shelving.
        ``'spline'`` -- the fitted ``UnivariateSpline`` object.
    """
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import brentq, minimize_scalar

    j_BS_work = j_BS_isolated.copy()

    # ---- Optional shelf on j_BS_isolated ----
    if shelf_psi_N > 0.0:
        shelf_idx = np.searchsorted(psi_N, shelf_psi_N)
        shelf_idx = min(shelf_idx, len(psi_N) - 1)
        shelf_val = j_BS_work[shelf_idx]
        j_BS_work[:shelf_idx] = shelf_val

    # ---- Build the spline basis (at bs_scale = 1) ----
    residual = eqdsk_jtor - j_BS_work
    mask_core = psi_N <= psi_bridge

    edge_target = max(eqdsk_jtor[-1] - j_BS_work[-1], 0.0)
    psi_trusted = np.concatenate([psi_N[mask_core], [1.0]])
    res_trusted = np.concatenate([residual[mask_core], [edge_target]])

    # Use a smoothing spline followed by PCHIP to eliminate ringing.
    # Step 1: smooth the residual with a generous smoothing factor
    # to remove high-frequency noise while preserving the overall shape.
    # Step 2: evaluate on a coarser grid and use PchipInterpolator
    # (shape-preserving, monotonicity-respecting, C1) for the final
    # profile on the full psi_N grid.
    from scipy.interpolate import PchipInterpolator

    # Smoothing factor: tighter when more points are used (captures
    # finer features); looser for the default 32-point subsampling.
    _s_scale = 0.1 * (32 / max(n_spline_pts, 1))
    _s_factor = len(psi_trusted) * np.var(res_trusted) * _s_scale
    _smooth_spline = UnivariateSpline(psi_trusted, res_trusted, k=k,
                                       s=_s_factor)

    # Subsample for PCHIP.  More points = finer feature resolution
    # but more susceptible to oscillation.  Default 32 is conservative;
    # increase to 64-128 to capture shorter length-scale structure.
    _n_sub = min(n_spline_pts, len(psi_N))
    _psi_sub = np.linspace(psi_N[0], psi_N[-1], _n_sub)
    _res_sub = _smooth_spline(_psi_sub)
    _res_sub = np.maximum(_res_sub, 0.0)

    # Force the edge point to match the edge anchor exactly
    # (smoothing spline may not interpolate through it)
    _res_sub[-1] = max(edge_target, 0.0)

    _pchip = PchipInterpolator(_psi_sub, _res_sub)
    j_inductive_basis = _pchip(psi_N)
    j_inductive_basis = np.maximum(j_inductive_basis, 0.0)

    # ---- Helper: solve ind_scale for a given bs_scale via brentq ----
    def _solve_ind_scale(bs_scale):
        def _li_residual(scale):
            j_phi = scale * j_inductive_basis + bs_scale * j_BS_work
            return calc_cylindrical_li_proxy(mygs, j_phi, psi_pad) - baseline_li_proxy

        s_lo, s_hi = 0.5, 2.0
        f_lo, f_hi = _li_residual(s_lo), _li_residual(s_hi)
        for _ in range(10):
            if f_lo * f_hi < 0:
                break
            s_lo *= 0.5
            s_hi *= 2.0
            f_lo, f_hi = _li_residual(s_lo), _li_residual(s_hi)

        if f_lo * f_hi < 0:
            return brentq(_li_residual, s_lo, s_hi, xtol=1e-6)
        else:
            return 1.0  # fallback

    if not rescale_j_BS:
        # ---- v1: scale inductive profile only ----
        ind_scale = _solve_ind_scale(1.0)
        bs_scale_out = 1.0
    else:
        # ---- v2: jointly optimise bs_scale and ind_scale ----
        def _rms_for_bs_scale(bs_scale):
            isc = _solve_ind_scale(bs_scale)
            j_phi = isc * j_inductive_basis + bs_scale * j_BS_work
            return np.sqrt(np.mean((j_phi - eqdsk_jtor)**2))

        result = minimize_scalar(_rms_for_bs_scale, bounds=(0.0, 4.0),
                                    method='bounded', options={'xatol': 1e-4})
        bs_scale_out = result.x
        ind_scale = _solve_ind_scale(bs_scale_out)

    j_inductive_fit = ind_scale * j_inductive_basis
    j_phi_fit = j_inductive_fit + bs_scale_out * j_BS_work
    fit_li = calc_cylindrical_li_proxy(mygs, j_phi_fit, psi_pad)

    return {
        'j_inductive_fit': j_inductive_fit,
        'j_phi_fit': j_phi_fit,
        'fit_li': fit_li,
        'ind_scale': ind_scale,
        'bs_scale': bs_scale_out,
        'j_BS_used': j_BS_work,
        'spline': _pchip,
    }

# ====================================================================
#  Locked-coil helpers
# ====================================================================
#  These support the "lock_coils" mode of generate_bouquet, where each
#  perturbed equilibrium is solved with coil currents pinned to the
#  reconstructed-baseline values.  Boundary deviations and X-point
#  deviations from the baseline are recorded as diagnostics and (in
#  PR2) used as accept/reject filters.
# ====================================================================
def lock_coils_to_baseline(mygs, ref_coil_currents, weight=1.0e2,
                            vsc_weight=1.0, per_coil_weight=None):
    r"""Soft-lock coils via strong regularisation toward reference values.

    Replaces the existing coil regularisation with terms that pin each
    coil close to ``ref_coil_currents`` through TokaMaker's reg
    framework, then re-asserts the saddle constraint as cleared.
    **Isoflux is intentionally left in place** so the inverse solver
    has constraint freedom to converge under perturbed profiles —
    pure forward GS solves at fixed currents (``set_coil_currents`` +
    ``set_isoflux(None)``) fail with "Matrix solve failed for
    targets" on physical DIII-D meshes.

    The default ``weight=1e2`` produces ~1% coil drift on typical
    profile perturbations on DIII-D (max |drift| ~ 1.7 kA on
    180 kA F-coils).  Raise the weight for tighter coil pinning at
    the cost of larger boundary deviation; lower it for the inverse
    trade-off.  Drift saturates above ``weight ≈ 1e4`` because
    isoflux + Ip targets become the binding constraints.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker GS solver, already set up and converged on the
        baseline equilibrium (e.g. via ``reconstruct_equilibrium``).
    ref_coil_currents : dict
        Mapping ``{coil_name: current_A}`` from
        ``mygs.get_coil_currents()`` on the baseline solve.
    weight : float
        Default regularisation weight per coil reg term.  Default
        ``1e2``.  Used for any coil not listed in
        ``per_coil_weight``.
    vsc_weight : float
        Regularisation weight on the virtual stability circuit
        (``#VSC``) term.  Default ``1.0``.  Raise for tighter
        vertical-stability coil control (e.g. on DIII-D, where
        ``F9A``/``F9B`` form the VSC pair and would otherwise
        absorb residual perturbation drift).
    per_coil_weight : dict or None
        Per-coil weight overrides as ``{coil_name: weight}``.  Coils
        not in the dict get the default ``weight``.  Useful when
        certain coils need tighter (or looser) control than others —
        e.g. pinning DIII-D's vertical-stability coils via
        ``per_coil_weight={'F9A': 1e4, 'F9B': 1e4}``.

    Notes
    -----
    Call once after the baseline reconstruction; no per-iteration
    re-pinning is needed because the reg is part of the inverse
    solve at every step.  After installing the lock, run one more
    ``mygs.solve()`` to obtain the lock-mode baseline state — this
    is the reference state for downstream perturbation deviation
    measurements.
    """
    if per_coil_weight is None:
        per_coil_weight = {}
    mygs.set_saddles(None)
    rt = []
    for name, target in ref_coil_currents.items():
        w = per_coil_weight.get(name, weight)
        rt.append(mygs.coil_reg_term({name: 1.0}, target=target,
                                       weight=w))
    rt.append(mygs.coil_reg_term({'#VSC': 1.0}, target=0.0,
                                   weight=vsc_weight))
    mygs.set_coil_reg(reg_terms=rt)


def psi_boundary_deviation(mygs, ref_R, ref_Z):
    r"""Spatial deviation of the current LCFS from reference points.

    For each reference boundary point :math:`(R, Z)`, evaluates
    :math:`\psi` at that location and converts the offset from
    :math:`\psi_{\rm LCFS}` into a spatial distance via the local
    poloidal field magnitude :math:`|\nabla\psi| = R\,|B_p|`:

    .. math::

       \delta r \;\approx\; \frac{|\psi(R,Z) - \psi_{\rm LCFS}|}
                                 {R\,|B_p(R,Z)|}.

    This avoids brittle contour extraction of the perturbed LCFS and
    reuses TokaMaker's field evaluators directly.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker solver in its post-``solve`` state.
    ref_R, ref_Z : ndarray
        1-D reference boundary point coordinates [m].  Typically the
        TokaMaker baseline LCFS extracted right after
        ``reconstruct_equilibrium``.

    Returns
    -------
    rms_m : float
        RMS deviation across all reference points [m].
    max_m : float
        Maximum deviation across all reference points [m].
    devs_m : ndarray
        Per-point deviation [m], aligned with ``ref_R`` / ``ref_Z``.
    """
    psi_eval = mygs.get_field_eval('psi')
    B_eval = mygs.get_field_eval('B')
    psi_lcfs = float(mygs.psi_bounds[0])
    psi_axis = float(mygs.psi_bounds[1])
    dpsi_total = abs(psi_lcfs - psi_axis)

    devs = np.empty(len(ref_R), dtype=np.float64)
    for k, (R, Z) in enumerate(zip(ref_R, ref_Z)):
        pt = np.array([R, Z], dtype=np.float64)
        psi_val = psi_eval.eval(pt)[0]
        B_vec = B_eval.eval(pt)        # [BR, Bt, BZ]
        BR, BZ = B_vec[0], B_vec[2]
        Bpol = np.sqrt(BR * BR + BZ * BZ)
        grad_psi = R * Bpol            # |grad psi|
        dpsi = abs(psi_val - psi_lcfs)
        if grad_psi > 1e-6:
            devs[k] = dpsi / grad_psi
        else:
            # Fallback: dimensionless dpsi/dpsi_total (rare, near
            # nulls).  Caller can detect via huge values relative to
            # machine size.
            devs[k] = dpsi / dpsi_total if dpsi_total > 0 else 0.0

    rms_m = float(np.sqrt(np.mean(devs * devs)))
    max_m = float(np.max(devs))
    return rms_m, max_m, devs


def xpoint_deviation(mygs, ref_xpts, max_pair_dist_m=0.1):
    r"""Per-X-point deviation between current and reference equilibria.

    Pairs each baseline X-point to the nearest perturbed X-point via
    Euclidean distance and returns per-pair displacement.  Pairs
    farther than ``max_pair_dist_m`` are treated as "lost" (NaN
    deviation) — this catches transient solves where an X-point
    disappears or splits.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker solver in its post-``solve`` state.
    ref_xpts : ndarray, shape (n_ref, 2)
        Baseline X-point coordinates ``[(R, Z), ...]`` from
        ``mygs.get_xpoints()[0].copy()`` taken on the baseline solve.
    max_pair_dist_m : float
        Maximum allowed pairing distance [m].  Above this, the
        baseline X-point is reported as unmatched (NaN deviation).

    Returns
    -------
    rms_m : float
        RMS deviation across paired X-points [m].  NaN if every
        baseline X-point is unmatched.
    max_m : float
        Max deviation across paired X-points [m].  NaN if all
        unmatched.
    devs_m : ndarray, shape (n_ref,)
        Per-baseline-X-point deviation [m].  NaN entries indicate
        unmatched baseline X-points.
    paired_RZ : ndarray, shape (n_ref, 2)
        Coordinates of the paired perturbed X-point for each baseline
        X-point.  NaN rows for unmatched baselines.
    """
    ref_xpts = np.asarray(ref_xpts, dtype=np.float64)
    n_ref = ref_xpts.shape[0]
    devs_m = np.full(n_ref, np.nan, dtype=np.float64)
    paired_RZ = np.full((n_ref, 2), np.nan, dtype=np.float64)

    if n_ref == 0:
        return float('nan'), float('nan'), devs_m, paired_RZ

    pert_xpts, _ = mygs.get_xpoints()
    if pert_xpts is None or len(pert_xpts) == 0:
        return float('nan'), float('nan'), devs_m, paired_RZ

    pert_xpts = np.asarray(pert_xpts, dtype=np.float64)

    # Pair each baseline X-point to its nearest perturbed counterpart.
    # n_ref is small (typically 1–2 for diverted DIII-D), so an O(n*m)
    # loop is fine and avoids a SciPy KDTree dependency just for this.
    for i in range(n_ref):
        d2 = np.sum((pert_xpts - ref_xpts[i]) ** 2, axis=1)
        j = int(np.argmin(d2))
        d = float(np.sqrt(d2[j]))
        if d <= max_pair_dist_m:
            devs_m[i] = d
            paired_RZ[i] = pert_xpts[j]

    finite = devs_m[np.isfinite(devs_m)]
    if finite.size == 0:
        rms_m = float('nan')
        max_m = float('nan')
    else:
        rms_m = float(np.sqrt(np.mean(finite * finite)))
        max_m = float(np.max(finite))
    return rms_m, max_m, devs_m, paired_RZ


# ====================================================================
#  Core perturbation routine
# ====================================================================
def perturb_kinetic_equilibrium(
    mygs,
    psi_N,
    pressure,
    ne,
    te,
    ni,
    ti,
    input_j_phi,
    sigma_ne,
    sigma_te,
    sigma_ni,
    sigma_ti,
    sigma_jphi,
    n_ls,
    t_ls,
    j_ls,
    Ip_target,
    l_i_target,
    Zeff,
    npsi,
    p_thresh=0.5,
    input_jinductive=None,
    l_i_tolerance=0.05,
    l_i_proxy_threshold=5.0,
    psi_pad=1e-3,
    constrain_sawteeth=True,
    recalculate_j_BS=True,
    isolate_edge_jBS=True,
    scale_jBS=1.0,
    diagnostic_plots=False,
    max_pressure_iter=_MAX_PRESSURE_ITER,
    max_li_iter=_MAX_LI_ITER,
    psi_N_kinetic=None,
    max_proxy_draws=500,
    ref_lcfs_R=None,
    ref_lcfs_Z=None,
    ref_x_points=None,
    ref_coil_currents=None,
):
    r"""Perturb kinetic and current-density profiles and iterate to
    match :math:`I_p` and :math:`l_i` targets.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker Grad-Shafranov solver object.
    psi_N : ndarray
        1-D normalised poloidal flux grid :math:`\hat{\psi}`.
    pressure : ndarray
        1-D baseline total pressure [Pa].
    ne : ndarray
        1-D electron density [m\ :sup:`-3`].
    te : ndarray
        1-D electron temperature [eV].
    ni : ndarray
        1-D ion density [m\ :sup:`-3`].
    ti : ndarray
        1-D ion temperature [eV].
    input_j_phi : ndarray
        1-D toroidal current density [A/m\ :sup:`2`]; must be the
        *inductive* component when ``recalculate_j_BS=True``.
    sigma_ne : ndarray
        1-D experimental :math:`1\sigma` for :math:`n_e` [m\ :sup:`-3`].
    sigma_te : ndarray
        1-D experimental :math:`1\sigma` for :math:`T_e` [eV].
    sigma_ni : ndarray
        1-D experimental :math:`1\sigma` for :math:`n_i` [m\ :sup:`-3`].
    sigma_ti : ndarray
        1-D experimental :math:`1\sigma` for :math:`T_i` [eV].
    sigma_jphi : ndarray
        1-D experimental :math:`1\sigma` for :math:`j_\phi` [A/m\ :sup:`2`].
    n_ls : float
        GPR length-scale for density profiles.
    t_ls : float
        GPR length-scale for temperature profiles.
    j_ls : float or ndarray
        GPR length-scale for :math:`j_\phi`.  A 1-D array gives a
        non-stationary Gibbs kernel (see ``sigmoid_length_scale``).
    Ip_target : float
        Target plasma current [A].
    l_i_target : float
        Target internal inductance.
    Zeff : float
        Effective ion charge (scalar).
    npsi : int
        Normalised poloidal flux grid size.
    p_thresh : float
        Acceptable :math:`\langle P \rangle` mismatch [%].
    input_jinductive : ndarray or None
        Dimensionless inductive :math:`j_\phi` shape (required when
        ``recalculate_j_BS=True``).
    l_i_tolerance : float
        :math:`l_i` matching tolerance [%].
    l_i_proxy_threshold : float
        Proxy :math:`l_i` relative error threshold [%].
    psi_pad : float
        Padding inside the LCFS for profile queries.
    constrain_sawteeth : bool
        Reject equilibria with :math:`q_0 < 1`.
    recalculate_j_BS : bool
        Recompute bootstrap current for perturbed profiles.
    isolate_edge_jBS : bool
        Separate the edge bootstrap-current spike from the core
        contribution inside ``solve_with_bootstrap``.
    scale_jBS : float
        Multiplicative scale factor applied to :math:`j_{\rm BS}` in
        ``solve_with_bootstrap``.  A value of 1.0 applies no scaling.
    diagnostic_plots : bool
        Show diagnostic matplotlib figures (including inside
        ``solve_with_bootstrap`` and ``find_optimal_scale``).
    max_pressure_iter : int
        Safety cap on pressure-matching loop.
    max_li_iter : int
        Safety cap on :math:`l_i`-matching loop.
    psi_N_kinetic : ndarray or None
        Kinetic-profile grid (may extend past psi_N = 1 into the SOL).
        If provided, ``ne``, ``te``, ``ni``, ``ti`` and their sigmas
        must be on this grid.  GPR sampling is done on this grid;
        perturbed profiles are then interpolated onto ``psi_N`` for
        pressure matching and equilibrium solving.  Returned
        perturbed profiles are on ``psi_N_kinetic``.  If ``None``,
        ``psi_N`` is used for everything (original behaviour).
    ref_lcfs_R, ref_lcfs_Z : ndarray or None
        Reference LCFS coordinates [m] from the reconstructed-baseline
        TokaMaker solve.  When supplied (i.e. ``generate_bouquet`` was
        called with ``lock_coils=True``), boundary deviation is
        evaluated at these points after the perturbed solve and added
        to ``diagnostics`` (``boundary_rms_mm``, ``boundary_max_mm``,
        ``boundary_devs_mm``).
    ref_x_points : ndarray or None, shape (n_ref, 2)
        Baseline X-point coordinates ``[(R, Z), ...]`` from
        ``mygs.get_xpoints()[0]`` on the baseline solve.  When
        supplied, X-point deviation is computed and added to
        ``diagnostics`` (``x_point_rms_mm``, ``x_point_max_mm``,
        ``x_point_devs_mm``, ``x_points_RZ``).
    ref_coil_currents : dict or None
        ``{name: current_A}`` from the baseline solve.  When supplied,
        post-solve coil drift is recorded in
        ``diagnostics['coil_drift_A']`` so the lock can be verified.

    Returns
    -------
    tuple
        ``(ne_perturb, te_perturb, ni_perturb, ti_perturb,
        w_ExB, output_jphi, diagnostics)``

        When ``psi_N_kinetic`` is provided, the kinetic profiles
        (``ne_perturb`` etc.) are on the ``psi_N_kinetic`` grid.
    """

    # ----------------------------------------------------------------
    #  1.  Lazy OFT imports (deferred so GPR-only use works without OFT)
    # ----------------------------------------------------------------
    from scipy.optimize import root_scalar
    from scipy.interpolate import interp1d
    from OpenFUSIONToolkit.TokaMaker.util import get_jphi_from_GS
    from OpenFUSIONToolkit.TokaMaker.bootstrap import (
        solve_with_bootstrap,
        find_optimal_scale,
    )

    # ----------------------------------------------------------------
    #  2.  Validate inputs and set up dual grids
    # ----------------------------------------------------------------
    if recalculate_j_BS and input_jinductive is None:
        raise ValueError(
            "input_jinductive must be provided when recalculate_j_BS=True"
        )

    # Kinetic grid: either the user-supplied extended grid or psi_N
    psi_kin = psi_N_kinetic if psi_N_kinetic is not None else psi_N
    _dual_grid = psi_N_kinetic is not None

    def _kin_to_eq(arr_kin):
        """Interpolate a profile from kinetic grid onto equilibrium grid."""
        if not _dual_grid:
            return arr_kin
        return interp1d(psi_kin, arr_kin, kind='linear',
                        bounds_error=False,
                        fill_value=(arr_kin[0], arr_kin[-1]))(psi_N)

    # ----------------------------------------------------------------
    #  3.  Perturb kinetic profiles to match <P>
    # ----------------------------------------------------------------
    inp_avg = mygs.flux_integral(psi_N, pressure)

    p_err = np.inf
    p_iter = 0
    print("Searching for pressure profile match...")

    while p_err > p_thresh:
        p_iter += 1
        if p_iter > max_pressure_iter:
            raise RuntimeError(
                f"Pressure match not found within {max_pressure_iter} iterations "
                f"(last error {p_err:.2f}% vs threshold {p_thresh}%)"
            )

        # GPR sampling on psi_kin (kinetic grid, may include SOL)
        ne_perturb = _draw_monotonic_perturbation(
            psi_kin, ne / ne[0], sigma_ne / ne[0], n_ls
        ) * ne[0]

        te_perturb = _draw_monotonic_perturbation(
            psi_kin, te / te[0], sigma_te / te[0], t_ls
        ) * te[0]

        ni_perturb = _draw_monotonic_perturbation(
            psi_kin, ni / ni[0], sigma_ni / ni[0], n_ls
        ) * ni[0]

        ti_perturb = _draw_monotonic_perturbation(
            psi_kin, ti / ti[0], sigma_ti / ti[0], t_ls
        ) * ti[0]

        # Pressure matching on equilibrium grid (psi_N, confined only)
        ne_eq = _kin_to_eq(ne_perturb)
        te_eq = _kin_to_eq(te_perturb)
        ni_eq = _kin_to_eq(ni_perturb)
        ti_eq = _kin_to_eq(ti_perturb)

        pres_tmp = EC * (ne_eq * te_eq + ni_eq * ti_eq)
        tmp_avg = mygs.flux_integral(psi_N, pres_tmp)
        p_err = np.mean(np.abs(inp_avg - tmp_avg) / inp_avg) * 100.0

    mygs.set_targets(Ip=Ip_target, pax=pres_tmp[0])

    # ----------------------------------------------------------------
    #  3b. Optional diagnostic plots for kinetic profiles
    # ----------------------------------------------------------------
    if diagnostic_plots:
        fig, ax = plt.subplots(2, 2, figsize=(9, 5), sharex=True)
        _pairs = [
            #  axis       orig  pert        scale  σ_phys     color        label       ylabel
            (ax[0, 0], ne, ne_perturb, 1.0,  sigma_ne, "tab:red",    r"$n_e$", r"n [m$^{-3}$]"),
            (ax[0, 1], ni, ni_perturb, 1.0,  sigma_ni, "tab:orange", r"$n_i$", None),
            (ax[1, 0], te, te_perturb, 1e-3, sigma_te, "tab:blue",   r"$T_e$", r"T [keV]"),
            (ax[1, 1], ti, ti_perturb, 1e-3, sigma_ti, "tab:cyan",   r"$T_i$", None),
        ]
        for a, orig, pert, scale, sig, clr, lbl, ylabel in _pairs:
            a.plot(psi_N, pert * scale, c=clr, ls="--", alpha=0.5)
            a.plot(psi_N, orig * scale, c=clr, lw=2, label=f"input {lbl}")
            a.fill_between(
                psi_N,
                (orig - sig) * scale,
                (orig + sig) * scale,
                alpha=0.3, color=clr,
                label=r"$\pm\,1\sigma_{\rm exp}$",
            )
            a.legend(loc="best")
            a.grid(ls=":")
            if ylabel:
                a.set_ylabel(ylabel)
        ax[1, 0].set_xlabel(r"$\hat{\psi}$")
        ax[1, 1].set_xlabel(r"$\hat{\psi}$")
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------
    #  4.  Bootstrap-current recalculation (optional)
    # ----------------------------------------------------------------
    j0_scales = []
    Ip_scales = []
    iteration_l_is = []
    iteration_Ips = []

    # ----------------------------------------------------------------
    #  4b.  Bootstrap recompute on perturbed kinetics
    # ----------------------------------------------------------------
    # Architectural choice: ``solve_with_bootstrap`` is used **only** as
    # a bootstrap calculator.  Its ``j_inductive`` and ``total_j_phi``
    # outputs are *discarded* because SWB only amplitude-scales whatever
    # ``inductive_jphi`` seed it receives — it does not reshape it, so
    # those outputs inherit the seed's shape rather than producing a
    # physically reconstructed inductive.
    #
    # The inductive shape we use downstream is ``input_jinductive`` (=
    # ``result['j_inductive_fit']`` from ``reconstruct_equilibrium``).
    # That profile was *spline-fit* by ``fit_inductive_profile`` to
    # ``eqdsk_jtor − j_BS`` and is the right shape for the converged
    # equilibrium — we just GPR-perturb it for σ > 0.
    #
    # This makes σ → 0 collapse cleanly to reconstruct's converged
    # state rather than to whatever shape SWB happens to amplitude-
    # scale a seed into.
    if recalculate_j_BS:
        # ---- SWB call hygiene ---------------------------------------
        # solve_with_bootstrap is sensitive to two things beyond
        # kinetics:
        #
        #   1. mygs state (q profile, FSA quantities) — drifts
        #      between calls if perturb is invoked from a non-recon
        #      state.
        #   2. inductive_jphi seed shape — SWB amplitude-only-scales
        #      its seed and its bootstrap iteration's convergence path
        #      depends on the seed.  Feeding recon's *converged* peaky
        #      ``j_inductive_fit`` produces a 22% bootstrap drift vs
        #      feeding the same broad ``create_power_flux_fun`` seed
        #      that recon used.
        #
        # Both fixes here:
        #   (a) State anchor: one solve at recon's exact j_phi +
        #       targets to put mygs in recon's converged state.
        #   (b) Synthesise the SAME broad SWB seed that
        #       ``reconstruct_equilibrium`` uses internally.  This
        #       decouples the SWB seed (purely a bootstrap-iteration
        #       starting point) from ``input_jinductive`` (the GPR
        #       baseline shape used downstream).
        _pre_pp = {"type": "linterp",
                   "y": np.gradient(pressure) /
                        (np.gradient(psi_N) *
                         (mygs.psi_bounds[1] - mygs.psi_bounds[0])),
                   "x": psi_N}
        _pre_pp["y"][-1] = 0.0
        _pre_ffp = {"type": "jphi-linterp",
                    "y": input_j_phi.copy(),
                    "x": psi_N}
        mygs.set_targets(Ip=Ip_target, pax=pressure[0])
        mygs.set_profiles(pp_prof=_pre_pp, ffp_prof=_pre_ffp)
        try:
            mygs.solve()
        except (ValueError, RuntimeError):
            # Anchor failed (rare); fall through and let SWB cope.
            pass

        # SWB seed: broad ``(1−ψ̂^1.5)^1.5`` shape — matches the seed
        # passed to SWB inside ``reconstruct_equilibrium`` so the
        # bootstrap iteration follows the same convergence path.
        from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
        _swb_seed = create_power_flux_fun(npsi, 1.5, 1.5)['y']

        results = solve_with_bootstrap(
            mygs,
            ne_eq, te_eq, ni_eq, ti_eq,
            Zeff, Ip_target, _swb_seed,
            scale_jBS=scale_jBS,
            isolate_edge_jBS=isolate_edge_jBS,
            diagnostic_plots=False,
            verbose=False,
        )
        eq_stats = mygs.get_stats(lcfs_pad=psi_pad)

        # Bootstrap pieces — kept.
        full_j_BS = results["j_BS"]
        spike_profile = results["isolated_j_BS"]

        # Reconstructed total at recon's inductive shape + this run's
        # bootstrap.  This is the "baseline j_phi" from which we'll
        # GPR-perturb in the l_i loop.
        baseline_jphi = input_jinductive + spike_profile
        baseline_li_proxy = calc_cylindrical_li_proxy(
            mygs, baseline_jphi, psi_pad)

        j0_scales.append(results["scale_j0"])
        Ip_scales.append(results["scale_Ip"])
        iteration_l_is.append(eq_stats["l_i"])
        iteration_Ips.append(eq_stats["Ip"])
    else:
        # When bootstrap is not recalculated, freeze it at the recon-
        # supplied value: ``spike_profile = input_j_phi - input_jinductive``.
        # Since the user passes ``input_j_phi = result['j_phi_fit']``
        # and ``input_jinductive = result['j_inductive_fit']``, this
        # difference equals ``result['j_BS_used']`` exactly — recon's
        # converged bootstrap, used verbatim.  No SWB-state drift.
        spike_profile = input_j_phi - input_jinductive
        full_j_BS = spike_profile.copy()
        baseline_jphi = input_j_phi.copy()
        baseline_li_proxy = calc_cylindrical_li_proxy(
            mygs, input_j_phi, psi_pad)

    # ----------------------------------------------------------------
    #  5.  l_i matching loop
    # ----------------------------------------------------------------
    l_i = np.inf
    final_scale_j0 = 1.0
    final_scale_Ip = 1.0
    # Inductive seed = recon's spline-fit (not SWB's amplitude-scaled
    # output).  GPR-perturbed below.
    matched_j_inductive = (input_jinductive.copy()
                           if recalculate_j_BS else input_j_phi.copy())

    # The proxy target starts at the baseline proxy value but is
    # adaptively corrected after each TokaMaker solve to account for
    # the systematic offset between the cylindrical proxy and the
    # actual equilibrium l_i.  This makes the proxy filter select
    # profiles that land near l_i_target in equilibrium space rather
    # than in proxy space.
    proxy_target = baseline_li_proxy

    for li_iter in range(1, max_li_iter + 1):
        if 100.0 * abs(l_i - l_i_target) / l_i_target <= l_i_tolerance:
            break

        t_phase = time.perf_counter()

        # ---- 5a. Draw j_phi perturbation matching l_i proxy --------
        # GPR seed is recon's spline-fit inductive (NOT SWB's amplitude-
        # scaled output, NOT input_j_phi which already includes the
        # bootstrap).  The bootstrap (perturbed via SWB or frozen from
        # recon) is added back as ``spike_profile`` after the GPR draw.
        # Always use input_jinductive when provided so we don't double-
        # add the bootstrap.
        if input_jinductive is not None:
            step_j_phi = input_jinductive
        else:
            # Legacy fallback: user didn't supply an inductive
            # decomposition, so treat input_j_phi as a single profile
            # and zero the spike (matches old behaviour).
            step_j_phi = input_j_phi
        j_phi_0 = step_j_phi[0]

        # Pre-compute geometry once for the inner proxy loop (the
        # equilibrium state doesn't change between proxy evaluations).
        _geo = get_li_proxy_geometry(mygs, npsi, psi_pad)

        l_i_rel_err = np.inf
        proxy_draws = 0
        while l_i_rel_err > l_i_proxy_threshold:
            proxy_draws += 1
            if proxy_draws > max_proxy_draws:
                raise RuntimeError(
                    f"Proxy match not found after {max_proxy_draws} draws "
                    f"(last err={l_i_rel_err:.3f}%, threshold={l_i_proxy_threshold}%). "
                    f"Try increasing l_i_proxy_threshold or max_proxy_draws."
                )
            jphi_perturb = generate_perturbed_GPR(
                psi_N,
                step_j_phi / j_phi_0,
                sigma_profile=sigma_jphi / j_phi_0,   # normalised σ
                length_scale=j_ls,
                n_samples=1,
                diag_plot=False,
            )
            jphi_perturb *= j_phi_0

            if np.any(jphi_perturb < 0.0):
                continue

            # Architectural change: no a_optimal scaling.  Previously
            # ``a_optimal`` was solved via ``Ip_flux_integral_vs_target``
            # to make ``∫(a × jphi_perturb + spike) = Ip_target`` exactly.
            # That used the *current* mygs state's volume metric, which
            # drifts during the perturb pipeline and produces a_optimal
            # values that cause large profile-magnitude shifts (down to
            # 0.6× of recon's profile in σ → 0 testing) — the dominant
            # source of the perturbed-vs-recon equilibrium offset.
            #
            # The minimal-resolve test (set profile + Ip target + solve)
            # reproduces recon's equilibrium to <0.5% l_i without this
            # scaling.  We therefore use the GPR-drawn inductive plus
            # the spike directly; the inverse solver naturally settles
            # near the right Ip when given recon's profile shape.
            a_optimal = 1.0
            matched_jphi_perturb = jphi_perturb + spike_profile

            # Fast proxy: uses cached geometry, no TokaMaker calls
            tmp_li_proxy = calc_cylindrical_li_proxy_fast(
                matched_jphi_perturb, _geo
            )
            l_i_rel_err = (
                100.0 * abs(tmp_li_proxy - proxy_target) / proxy_target
            )

        dt_proxy = time.perf_counter() - t_phase
        print(f"  [li_iter={li_iter}] Proxy matched in {proxy_draws} draws "
              f"({dt_proxy:.1f}s, err={l_i_rel_err:.3f}%)")

        # ---- 5b. Set up GS profiles --------------------------------
        psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
        pprime_tmp = np.gradient(pres_tmp) / (np.gradient(psi_N) * psi_range)
        pprime_tmp[-1] = 0.0

        pp_prof = {"type": "linterp", "y": pprime_tmp, "x": psi_N}
        ffp_prof = {
            "type": "jphi-linterp",
            "y": matched_jphi_perturb,
            "x": psi_N,
        }

        matched_j_inductive = jphi_perturb        # no a_optimal scaling

        # ---- 5c. Skip ``find_optimal_scale`` entirely --------------
        # Both ``scale_j0`` (profile shape scaling) and ``scale_Ip``
        # (Ip target scaling) introduced shifts away from the
        # reconstruction baseline.  The minimal-resolve sanity check
        # (set recon's exact profile + Ip target + solve) reproduces
        # recon to <0.5% l_i, so neither compensation is needed.
        # The inverse solver's natural Ip undershoot (~0.7%) is small
        # and physically acceptable.
        final_scale_j0 = 1.0
        final_scale_Ip = 1.0

        # Quick q_0 check before the corrective iter
        if constrain_sawteeth:
            mygs.set_targets(Ip=Ip_target, pax=pres_tmp[0])
            mygs.set_profiles(pp_prof=pp_prof, ffp_prof=ffp_prof)
            mygs.solve()
            _, q_pre, _, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
            if q_pre[0] < 1.0:
                print("Skipping this equilibrium, q_0 < 1.0 (pre-check)")
                l_i = np.inf
                continue

        # ---- 5d. Definitive sawtooth constraint (after Ip scaling) --
        if constrain_sawteeth:
            _, q, _, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
            if q[0] < 1.0:
                print("Skipping this equilibrium, q_0 < 1.0")
                l_i = np.inf
                continue

        j0_scales.append(final_scale_j0)
        Ip_scales.append(final_scale_Ip)

        # ---- 5e. Adaptive corrective iteration ----------------------
        # Iterate TokaMaker until its output j_phi matches the intended
        # input (j_inductive*scale + spike).  This compensates for
        # geometry coupling that distorts the edge profile.
        pprime_tmp = np.gradient(pres_tmp) / (
            np.gradient(psi_N) * psi_range
        )
        pprime_tmp[-1] = 0.0
        pp_prof = {"type": "linterp", "y": pprime_tmp, "x": psi_N}

        target_jphi_perturb = matched_j_inductive * final_scale_j0 + spike_profile

        output_jphi, _n_corr, _corr_hist = _corrective_jphi_iteration(
            mygs, psi_N, target_jphi_perturb, pp_prof,
            Ip_target * final_scale_Ip, pres_tmp[0], psi_pad,
            min_iters=2, max_iters=8, rtol=0.05, verbose=False,
        )
        if _n_corr > 2:
            print(f"  [jphi correction] {_n_corr} iterations, "
                  f"edge RMS: {_corr_hist[0]/1e6:.4f} → {_corr_hist[-1]/1e6:.4f} MA/m²")

        if diagnostic_plots:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(psi_N, matched_jphi_perturb, label=r"Input $j_\phi$")
            ax.plot(psi_N, output_jphi, label=r"Converged $j_\phi$")
            ax.fill_between(
                psi_N,
                input_j_phi - sigma_jphi,
                input_j_phi + sigma_jphi,
                alpha=0.3, label=r"$\pm\,1\sigma_{\rm exp}$ envelope",
            )
            ax.set_ylim(
                0.0,
                max(input_j_phi[0], (input_j_phi + sigma_jphi)[0]),
            )

            ax.legend(loc="best")
            ax.set_xlabel(r"$\hat{\psi}$")
            ax.set_ylabel(r"$j_\phi$ [A/m$^2$]")
            ax.set_title(f"jphi-linterp  |  l_i iter {li_iter}")
            plt.tight_layout()
            plt.show()

        eq_stats = mygs.get_stats(lcfs_pad=psi_pad)
        Ip = eq_stats["Ip"]
        l_i = eq_stats["l_i"]

        # Compute cylindrical proxy on the FINAL converged j_phi to
        # measure the proxy-vs-TokaMaker offset.  This diagnostic
        # helps calibrate the l_i_proxy_threshold parameter.
        final_li_proxy = calc_cylindrical_li_proxy_fast(output_jphi, _geo)
        proxy_vs_real = 100.0 * (final_li_proxy - l_i) / l_i if l_i != 0 else 0.0

        Ip_err = 100.0 * abs(Ip - Ip_target) / Ip_target

        # Adaptive proxy target correction: use the observed
        # proxy-to-equilibrium mapping to predict what proxy value
        # would produce l_i_target in the actual equilibrium.
        if l_i > 0 and np.isfinite(l_i):
            corrected_target = final_li_proxy * (l_i_target / l_i)
            # Blend: 70% new correction, 30% old target (smooths noise)
            proxy_target = 0.7 * corrected_target + 0.3 * proxy_target

        print(f"  l_i target (equil):   {l_i_target:.4f}")
        print(f"  proxy target:         {proxy_target:.4f}  (corrected)")
        print(f"  matched l_i (equil):  {l_i:.4f}")
        print(f"  matched l_i (proxy):  {final_li_proxy:.4f}")
        print(f"  Ip error vs target:   {Ip_err:.3f}%")
        print(f"  proxy vs real l_i:    {proxy_vs_real:+.2f}%")
        _li_pct_err = 100.0 * abs(l_i - l_i_target) / l_i_target if l_i_target != 0 else float('inf')
        print(f"  l_i error:            {_li_pct_err:.2f}% (tolerance: {l_i_tolerance:.2f}%)")

        iteration_l_is.append(l_i)
        iteration_Ips.append(Ip)
    else:
        # Fired only if the for-loop exhausted without break
        raise RuntimeError(
            f"l_i match not found after {max_li_iter} iterations "
            f"(last l_i error = {100.0 * abs(l_i - l_i_target) / l_i_target:.2f}%, "
            f"tolerance = {l_i_tolerance:.2f}%, "
            f"target={l_i_target:.4f}, best={l_i:.4f}).\n"
            f"Try reducing your kinetic profile uncertainties or "
            f"increasing l_i_tolerance."
        )

    # ----------------------------------------------------------------
    #  6.  Package outputs
    # ----------------------------------------------------------------
    # NOTE: w_ExB (E×B rotation) is not yet computed from the
    # perturbed equilibrium.  A zero placeholder is stored so the
    # output tuple and HDF5 schema remain forward-compatible.
    w_ExB = np.zeros_like(psi_N)

    # Shelf-blend decomposition: j_inductive tapers to zero at the
    # edge where the Sauter spike dominates.
    j_inductive_consistent, _ = _shelf_blend_decompose(
        psi_N, output_jphi, spike_profile, eqdsk_jphi=input_j_phi
    )

    diagnostics = {
        "j0_scales": j0_scales,
        "Ip_scales": Ip_scales,
        "iteration_l_is": iteration_l_is,
        "iteration_Ips": iteration_Ips,
        "j_inductive": j_inductive_consistent,
        "j_BS": full_j_BS,
        "j_BS_edge": spike_profile,
    }

    # ----------------------------------------------------------------
    #  6b.  Lock-coil diagnostics (boundary + X-point deviation)
    # ----------------------------------------------------------------
    # Populated only when lock-coil reference data is supplied (i.e.
    # generate_bouquet was called with lock_coils=True).  In that mode
    # mygs has just completed a forward solve at fixed coil currents,
    # so the live psi/B fields and X-point list reflect the perturbed
    # equilibrium and we can measure displacement from the baseline.
    if ref_lcfs_R is not None and ref_lcfs_Z is not None:
        bnd_rms_m, bnd_max_m, bnd_devs_m = psi_boundary_deviation(
            mygs, ref_lcfs_R, ref_lcfs_Z
        )
        diagnostics["boundary_rms_mm"] = bnd_rms_m * 1e3
        diagnostics["boundary_max_mm"] = bnd_max_m * 1e3
        diagnostics["boundary_devs_mm"] = bnd_devs_m * 1e3

    if ref_x_points is not None and len(ref_x_points) > 0:
        xpt_rms_m, xpt_max_m, xpt_devs_m, paired_RZ = xpoint_deviation(
            mygs, ref_x_points
        )
        diagnostics["x_point_rms_mm"] = xpt_rms_m * 1e3 \
            if np.isfinite(xpt_rms_m) else float('nan')
        diagnostics["x_point_max_mm"] = xpt_max_m * 1e3 \
            if np.isfinite(xpt_max_m) else float('nan')
        diagnostics["x_point_devs_mm"] = xpt_devs_m * 1e3
        diagnostics["x_points_RZ"] = paired_RZ

    # Surface coil drift so testing can verify the pin held.
    if ref_coil_currents is not None:
        cur, _ = mygs.get_coil_currents()
        diagnostics["coil_drift_A"] = {
            n: float(cur[n] - ref_coil_currents[n])
            for n in ref_coil_currents
            if n in cur
        }

    return (
        ne_perturb,
        te_perturb,
        ni_perturb,
        ti_perturb,
        w_ExB,
        output_jphi,
        diagnostics,
    )


# ====================================================================
#  Top-level scan driver
# ====================================================================
def generate_bouquet(
    mygs,
    psi_N,
    n_equils,
    header,
    input_j_phi,
    ne,
    te,
    ni,
    ti,
    sigma_ne,
    sigma_te,
    sigma_ni,
    sigma_ti,
    sigma_jphi,
    n_ls,
    t_ls,
    j_ls,
    initial_Ip_target,
    l_i_target,
    Zeff,
    input_jinductive=None,
    l_i_tolerance=0.03,
    l_i_proxy_threshold=5.0,
    psi_pad=1e-3,
    constrain_sawteeth=True,
    recalculate_j_BS=True,
    isolate_edge_jBS=True,
    jBS_scale_range=None,
    diagnostic_plots=True,
    scan_val=None,
    pfile_bytes=None,
    Zeff_profile=None,
    baseline_eqdsk_bytes=None,
    baseline_pfile_bytes=None,
    psi_N_kinetic=None,
    max_proxy_draws=500,
    lock_coils=False,
    lock_coils_weight=1.0e2,
    lock_coils_vsc_weight=1.0,
    lock_coils_per_coil_weight=None,
    coil_drift_threshold_A=None,
    boundary_max_threshold_mm=None,
    xpoint_max_threshold_mm=None,
):
    r"""Generate a batch of perturbed equilibria and archive to HDF5.

    Parameters
    ----------
    mygs : TokaMaker
        TokaMaker GS solver object.
    psi_N : ndarray
        1-D normalised flux grid :math:`\hat{\psi}`.
    n_equils : int
        Number of perturbed equilibria to generate.
    header : str
        Base name for the HDF5 database.
    input_j_phi : ndarray
        1-D baseline toroidal current density [A/m\ :sup:`2`].
    ne : ndarray
        1-D electron density [m\ :sup:`-3`].
    te : ndarray
        1-D electron temperature [eV].
    ni : ndarray
        1-D ion density [m\ :sup:`-3`].
    ti : ndarray
        1-D ion temperature [eV].
    sigma_ne : ndarray
        1-D experimental :math:`1\sigma` for :math:`n_e` [m\ :sup:`-3`].
    sigma_te : ndarray
        1-D experimental :math:`1\sigma` for :math:`T_e` [eV].
    sigma_ni : ndarray
        1-D experimental :math:`1\sigma` for :math:`n_i` [m\ :sup:`-3`].
    sigma_ti : ndarray
        1-D experimental :math:`1\sigma` for :math:`T_i` [eV].
    sigma_jphi : ndarray
        1-D experimental :math:`1\sigma` for :math:`j_\phi` [A/m\ :sup:`2`].
    n_ls : float
        GPR length-scale for density profiles.
    t_ls : float
        GPR length-scale for temperature profiles.
    j_ls : float or ndarray
        GPR length-scale for :math:`j_\phi`.  A 1-D array gives a
        non-stationary Gibbs kernel (see ``sigmoid_length_scale``).
    initial_Ip_target : float
        Target plasma current [A].
    l_i_target : float
        Target internal inductance.
    Zeff : float
        Effective ion charge.
    input_jinductive : ndarray or None
        Dimensionless inductive :math:`j_\phi` shape.
    l_i_tolerance : float
        :math:`l_i` tolerance [%].
    l_i_proxy_threshold : float
        Proxy :math:`l_i` relative-error threshold [%].
    psi_pad : float
        LCFS padding for profile queries.
    constrain_sawteeth : bool
        Reject equilibria with :math:`q_0 < 1`.
    recalculate_j_BS : bool
        Recompute bootstrap current each iteration.
    isolate_edge_jBS : bool
        Separate the edge bootstrap-current spike from the core
        contribution inside ``solve_with_bootstrap``.
    jBS_scale_range : list of two floats, or None
        Bounds ``[lo, hi]`` for a uniformly distributed multiplicative
        scale factor applied to :math:`j_{\rm BS}`.  For example,
        ``[0.8, 1.2]`` draws from :math:`\mathcal{U}(0.8, 1.2)`.
        When ``None``, no additional scaling is applied
        (``scale_jBS = 1.0`` for every sample).
    diagnostic_plots : bool
        Show diagnostic matplotlib figures.
    scan_val : str, float, int, or None
        Optional scan-point label for nested HDF5 storage.
        ``None`` gives the flat layout.
    pfile_bytes : bytes or None
        Raw p-file content to store alongside each equilibrium.
    Zeff_profile : array-like or None
        1-D effective charge profile to store in HDF5.
    lock_coils : bool
        When ``True``, install a strong coil regularisation that pins
        each perturbed-equilibrium coil current near the
        reconstruction-baseline value.  Isoflux is retained so the
        inverse solver has constraint freedom; the saddle constraint
        is cleared.  Boundary and X-point deviations from the *post-
        lock* baseline are recorded as diagnostics and HDF5 fields —
        useful for filtering perturbations that move the boundary far
        from the reconstruction.  Default ``False`` preserves the
        legacy workflow with no coil pinning.
    lock_coils_weight : float
        Default regularisation weight per coil reg term when
        ``lock_coils=True``.  Default ``1e2`` (~1% drift on typical
        perturbations on DIII-D).  Drift saturates above ``1e4``
        because isoflux + Ip become the binding constraints.  Used
        for any coil not listed in ``lock_coils_per_coil_weight``.
    lock_coils_vsc_weight : float
        Regularisation weight on the virtual stability circuit
        (``#VSC``) term.  Default ``1.0``.  On DIII-D the VSC pair
        (F9A/F9B) tends to absorb residual drift because the
        antisymmetric mode is the plasma's only vertical-position
        DOF; raising this weight (e.g. ``1e2``) tightens vertical
        control alongside the individual coil reg.
    lock_coils_per_coil_weight : dict or None
        Per-coil weight overrides as ``{coil_name: weight}``.  Coils
        not in the dict get ``lock_coils_weight``.  Useful for
        tightening specific coils (e.g.
        ``{'F9A': 1e4, 'F9B': 1e4}`` to pin DIII-D's VSC pair while
        keeping the rest at default).
    coil_drift_threshold_A : float or None
        If set (and ``lock_coils=True``), perturbed equilibria whose
        max ``|coil_drift|`` exceeds this threshold (in Amps) are
        discarded — not stored to HDF5.  This filters out
        perturbations where the inverse solver had to drift coils
        far enough that the lock is functionally broken.  Default
        ``None`` (no filter).  Recommended starting value on DIII-D:
        ~5000 A.
    boundary_max_threshold_mm : float or None
        If set (and ``lock_coils=True``), perturbed equilibria whose
        max boundary deviation from the lock-mode baseline LCFS
        exceeds this threshold (in millimetres) are discarded.  This
        is the most physically-defensible realism filter: anything
        moving the LCFS by more than the magnetics-diagnostic noise
        floor would have been detected experimentally and is therefore
        not a plausible alternative equilibrium.  Default ``None`` (no
        filter).  Recommended starting value on DIII-D: ~10 mm
        (roughly the magnetics-resolution boundary uncertainty for a
        well-fit reconstruction).
    xpoint_max_threshold_mm : float or None
        If set (and ``lock_coils=True``), perturbed equilibria whose
        max X-point displacement from any lock-mode-baseline X-point
        exceeds this threshold (in millimetres) are discarded.  Useful
        because X-point position is a localised feature easy to miss
        with a boundary-only filter — sparse isoflux can let the
        X-point cusp drift even when the upper LCFS stays close to
        baseline.  Default ``None`` (no filter).

    Returns
    -------
    list[dict]
        Diagnostics from each equilibrium.
    """
    all_diagnostics = []

    # self-consistent pressure for baseline <P>
    # When kinetic profiles are on a different grid, interpolate
    # onto the equilibrium grid for pressure/GS calculations.
    if psi_N_kinetic is not None:
        from scipy.interpolate import interp1d as _interp1d_bg
        _kin2eq = lambda arr: _interp1d_bg(
            psi_N_kinetic, arr, kind='linear', bounds_error=False,
            fill_value=(arr[0], arr[-1]))(psi_N)
        pressure = EC * (_kin2eq(ne) * _kin2eq(te) + _kin2eq(ni) * _kin2eq(ti))
    else:
        pressure = EC * (ne * te + ni * ti)
    npsi = len(psi_N)

    # --- Baseline solve (sawtooth detection + lock-coil ref capture) ---
    # Always run a fresh baseline solve when sawtooth detection is on
    # OR when lock_coils is on:
    #   - sawtooth: q_0 must reflect the baseline state (not a
    #     corrective-iteration state that may have altered q_0).
    #   - lock_coils: we capture coil currents, the LCFS contour, and
    #     X-points from this solve to use as the reference for
    #     perturbed-equilibrium deviation measurements.
    ref_coil_currents = None
    ref_lcfs_R = None
    ref_lcfs_Z = None
    ref_x_points = None
    if constrain_sawteeth or lock_coils:
        _pp_check = {"type": "linterp",
                      "y": np.gradient(pressure) / (np.gradient(psi_N)
                           * (mygs.psi_bounds[1] - mygs.psi_bounds[0])),
                      "x": psi_N}
        _pp_check["y"][-1] = 0.0
        _ffp_check = {"type": "jphi-linterp",
                       "y": input_j_phi.copy(), "x": psi_N}
        mygs.set_targets(Ip=initial_Ip_target, pax=pressure[0])
        mygs.set_profiles(pp_prof=_pp_check, ffp_prof=_ffp_check)
        mygs.solve()

        if constrain_sawteeth:
            _, q_baseline_check, _, _, _, _ = mygs.get_q(
                npsi=len(psi_N), psi_pad=psi_pad
            )
            if q_baseline_check[0] < 1.0:
                print(
                    f"NOTE: Baseline equilibrium has q(0) = {q_baseline_check[0]:.4f} < 1.0 "
                    f"(sawtoothing plasma).\n"
                    f"      Overriding constrain_sawteeth = False so perturbed "
                    f"equilibria are not rejected."
                )
                constrain_sawteeth = False

        if lock_coils:
            # 1. Capture coil currents from the reconstruction
            #    baseline (pre-lock state) — these are the targets
            #    the lock will pin to.
            _ref_dict, _ = mygs.get_coil_currents()
            ref_coil_currents_initial = dict(_ref_dict)

            # 2. Install the soft lock (strong reg targeting the
            #    reconstruction-baseline coil values; isoflux retained
            #    so the inverse solver has constraint freedom).
            lock_coils_to_baseline(
                mygs, ref_coil_currents_initial,
                weight=lock_coils_weight,
                vsc_weight=lock_coils_vsc_weight,
                per_coil_weight=lock_coils_per_coil_weight,
            )

            # 3. Re-solve under the lock to obtain the *lock-mode*
            #    baseline.  The inverse solver finds a slightly
            #    different equilibrium because the reg targets shifted
            #    from zero (reconstruction) to ref_coils (lock).  This
            #    new state is the true reference for perturbed-
            #    equilibrium deviations.
            mygs.set_targets(Ip=initial_Ip_target, pax=pressure[0])
            mygs.set_profiles(pp_prof=_pp_check, ffp_prof=_ffp_check)
            mygs.solve()

            # 4. Capture lock-mode reference state.
            _ref_dict_post, _ = mygs.get_coil_currents()
            ref_coil_currents = dict(_ref_dict_post)
            _lcfs_RZ = mygs.trace_surf(1.0 - _LOCK_REF_TRACE_PAD)
            if _lcfs_RZ is None:
                # Fall back to a slightly larger pad if the tight one
                # fails (rare; happens when the surface is too close
                # to the X-point cusp for the tracer).
                _lcfs_RZ = mygs.trace_surf(1.0 - 1e-4)
            if _lcfs_RZ is None or len(_lcfs_RZ) == 0:
                raise RuntimeError(
                    "lock_coils=True: failed to extract lock-mode "
                    "baseline LCFS via mygs.trace_surf().")
            ref_lcfs_R = _lcfs_RZ[:, 0].copy()
            ref_lcfs_Z = _lcfs_RZ[:, 1].copy()

            _xpts, _ = mygs.get_xpoints()
            ref_x_points = (np.asarray(_xpts, dtype=np.float64).copy()
                            if _xpts is not None else
                            np.zeros((0, 2), dtype=np.float64))

            # Coil drift induced by the lock install (interesting to
            # see how far the reg targets pulled the inverse solution
            # away from the original reconstruct-only converged state).
            _max_install_drift = max(
                abs(ref_coil_currents[n] - ref_coil_currents_initial[n])
                for n in ref_coil_currents
                if n in ref_coil_currents_initial)
            print(f"[lock_coils] installed (weight={lock_coils_weight:.1e}); "
                  f"post-lock baseline captured: "
                  f"{len(ref_coil_currents)} coils, "
                  f"{len(ref_lcfs_R)} LCFS pts, "
                  f"{len(ref_x_points)} X-pts; "
                  f"install drift max = {_max_install_drift:.1f} A")

    # Pre-compute jBS scale factors for the whole batch (if requested).
    # Uses a uniform distribution within the specified range so that
    # extreme values (which can make l_i matching very difficult) are
    # strictly bounded.
    if jBS_scale_range is not None:
        lo, hi = jBS_scale_range
        jBS_scales = np.random.uniform(lo, hi, size=n_equils)
    else:
        jBS_scales = np.ones(n_equils)

    # Store baseline profiles and uncertainties so the .h5 file is
    # self-contained (the plotting GUI only needs the file path).
    #
    # Recompute the baseline p-file's rotation profiles using the same
    # midplane method we use for perturbed p-files.  This ensures that
    # baseline and perturbed omghb / Er are computed consistently and
    # can be compared directly in plots.
    stored_pfile_bytes = baseline_pfile_bytes
    if baseline_pfile_bytes is not None and baseline_eqdsk_bytes is not None:
        try:
            from .io.pfile import PFile as _PFile
            from .io import GEQDSKEquilibrium as _GEQDSK
            from scipy.interpolate import interp1d

            pf_bl = _PFile.from_bytes(baseline_pfile_bytes)
            eq_bl = _GEQDSK.from_bytes(baseline_eqdsk_bytes)
            psi_pf = pf_bl.psinorm_for("ne")
            dpsi_bl = eq_bl.psi_boundary - eq_bl.psi_axis
            psi_Wb_bl = psi_pf * dpsi_bl + eq_bl.psi_axis

            pf_bl.compute_diamagnetic_rotations(psi_Wb_bl)

            mid_bl = eq_bl.midplane
            psi_eq = eq_bl.psi_N
            R_bl = interp1d(
                psi_eq, mid_bl["R"],
                fill_value="extrapolate")(psi_pf)
            Bp_bl = interp1d(
                psi_eq, mid_bl["Bp"],
                fill_value="extrapolate")(psi_pf)
            Bt_bl = interp1d(
                psi_eq, mid_bl["Bt"],
                fill_value="extrapolate")(psi_pf)

            pf_bl.compute_rotation_decomposition(
                R=R_bl, Bp=Bp_bl, Bt=Bt_bl, psi=psi_Wb_bl)
            stored_pfile_bytes = pf_bl.to_bytes()
        except Exception as exc:
            import traceback
            print(f"  WARNING: could not recompute baseline rotations: {exc}")
            traceback.print_exc()

    store_baseline_profiles(
        header, psi_N,
        ne, te, ni, ti,
        pressure, input_j_phi,
        sigma_ne, sigma_te, sigma_ni, sigma_ti, sigma_jphi,
        initial_Ip_target, l_i_target,
        scan_val=scan_val,
        eqdsk_bytes=baseline_eqdsk_bytes,
        pfile_bytes=stored_pfile_bytes,
        psi_N_kinetic=psi_N_kinetic,
        coils_locked=lock_coils,
        ref_coil_currents=ref_coil_currents,
        ref_lcfs_R=ref_lcfs_R,
        ref_lcfs_Z=ref_lcfs_Z,
        ref_x_points=ref_x_points,
        lock_coils_weight=(lock_coils_weight if lock_coils else None),
        coil_drift_threshold_A=coil_drift_threshold_A,
        boundary_max_threshold_mm=boundary_max_threshold_mm,
        xpoint_max_threshold_mm=xpoint_max_threshold_mm,
    )

    t_batch_start = time.perf_counter()
    elapsed_times = []

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    pbar = (
        _tqdm(range(n_equils), desc="Bouquet", unit="eq")
        if _tqdm is not None
        else None
    )
    eq_iter = pbar if pbar is not None else range(n_equils)

    for count in eq_iter:
        scale_jBS = float(jBS_scales[count])
        eta_str = ""
        if elapsed_times:
            avg_s = np.mean(elapsed_times)
            remaining = avg_s * (n_equils - count)
            eta_min = remaining / 60.0
            eta_str = f"  ETA: {eta_min:.1f} min"
        print(f"\n{'='*60}")
        print(f"  Equilibrium {count+1}/{n_equils}  "
              f"(scale_jBS={scale_jBS:.4f}){eta_str}")
        print(f"{'='*60}")
        t_start = time.perf_counter()

        # No per-iteration re-pin under strong-reg lock — the
        # regularisation is part of every inverse solve and the
        # reconstruct/perturb workflow only set_targets/set_profiles
        # between solves, which doesn't disturb the reg.

        try:
            (
                ne_perturb,
                te_perturb,
                ni_perturb,
                ti_perturb,
                w_ExB,
                jphi_perturb,
                diagnostics,
            ) = perturb_kinetic_equilibrium(
                mygs,
                psi_N,
                pressure,
                ne, te, ni, ti,
                input_j_phi,
                sigma_ne,
                sigma_te,
                sigma_ni,
                sigma_ti,
                sigma_jphi,
                n_ls, t_ls, j_ls,
                initial_Ip_target,
                l_i_target,
                Zeff,
                npsi,
                input_jinductive=input_jinductive,
                l_i_tolerance=l_i_tolerance,
                l_i_proxy_threshold=l_i_proxy_threshold,
                psi_pad=psi_pad,
                constrain_sawteeth=constrain_sawteeth,
                recalculate_j_BS=recalculate_j_BS,
                isolate_edge_jBS=isolate_edge_jBS,
                scale_jBS=scale_jBS,
                diagnostic_plots=diagnostic_plots,
                psi_N_kinetic=psi_N_kinetic,
                max_proxy_draws=max_proxy_draws,
                ref_lcfs_R=ref_lcfs_R,
                ref_lcfs_Z=ref_lcfs_Z,
                ref_x_points=ref_x_points,
                ref_coil_currents=ref_coil_currents,
            )
        except (RuntimeError, ValueError) as e:
            print(f"\n  STOPPED: {e}")
            print(f"  Skipping equilibrium {count+1}/{n_equils}.\n")
            if pbar is not None:
                pbar.update(1)
            continue

        elapsed = time.perf_counter() - t_start
        elapsed_times.append(elapsed)
        total_elapsed = time.perf_counter() - t_batch_start
        print(f"  Wall-clock time: {elapsed:.1f}s  "
              f"(total: {total_elapsed/60:.1f} min, "
              f"avg: {np.mean(elapsed_times):.1f}s/eq)")

        if pbar is not None:
            pbar.set_postfix_str(
                f"avg={np.mean(elapsed_times):.0f}s/eq, "
                f"total={total_elapsed/60:.1f}min"
            )

        diagnostics['time'] = elapsed

        # ---- save geqdsk to a temporary file, archive, delete -------
        eqdsk_filename = f"{header}_count={count}.geqdsk"
        full_path = os.path.abspath(eqdsk_filename)
        print(f"  Saving to: {full_path}")

        mygs.save_eqdsk(
            eqdsk_filename,
            nr=257, nz=257,
            truncate_eq=False,
            lcfs_pad=psi_pad,
        )

        eq_stats_std = mygs.get_stats(li_normalization="std", lcfs_pad=psi_pad)
        li1 = eq_stats_std["l_i"]
        eq_stats_iter = mygs.get_stats(li_normalization="iter", lcfs_pad=psi_pad)
        li3 = eq_stats_iter["l_i"]

        # Pressure on the equilibrium grid (for storage and plotting).
        # Interpolate kinetic profiles onto psi_N if on a different grid.
        if psi_N_kinetic is not None:
            from scipy.interpolate import interp1d as _interp1d_pp
            _to_eq = lambda arr: _interp1d_pp(
                psi_N_kinetic, arr, kind='linear', bounds_error=False,
                fill_value=(arr[0], arr[-1]))(psi_N)
            pressure_perturb = EC * (_to_eq(ne_perturb) * _to_eq(te_perturb)
                                      + _to_eq(ni_perturb) * _to_eq(ti_perturb))
        else:
            pressure_perturb = EC * (ne_perturb * te_perturb
                                      + ni_perturb * ti_perturb)

        # Extract coil currents from TokaMaker
        coil_current_dict, _ = mygs.get_coil_currents()

        # ---- Coil-drift filter ----------------------------------------
        # If the user set a ``coil_drift_threshold_A`` and any coil's
        # |drift| from the lock-mode baseline exceeds it, drop this
        # equilibrium.  Filters perturbations where the inverse solver
        # had to drift coils far enough that the lock is broken.
        if coil_drift_threshold_A is not None and lock_coils \
                and ref_coil_currents is not None:
            _drifts = {n: float(coil_current_dict[n] - ref_coil_currents[n])
                       for n in ref_coil_currents
                       if n in coil_current_dict}
            if _drifts:
                _max_drift = max(abs(v) for v in _drifts.values())
                if _max_drift > coil_drift_threshold_A:
                    _worst = max(_drifts, key=lambda k: abs(_drifts[k]))
                    print(f"  [coil_drift_filter] discarded: "
                          f"max |drift| = {_max_drift:.0f} A on {_worst} "
                          f"> threshold {coil_drift_threshold_A:.0f} A")
                    if pbar is not None:
                        pbar.update(1)
                    continue

        # ---- Boundary-deviation filter --------------------------------
        # Discard perturbations whose LCFS moved farther from the lock-
        # mode baseline than the magnetics-diagnostic noise floor — these
        # would have been distinguishable in the experimental
        # reconstruction and are therefore implausible alternatives.
        if boundary_max_threshold_mm is not None and lock_coils:
            _bnd_max = diagnostics.get("boundary_max_mm", float('nan'))
            if np.isfinite(_bnd_max) and _bnd_max > boundary_max_threshold_mm:
                print(f"  [boundary_filter] discarded: "
                      f"max boundary dev = {_bnd_max:.1f} mm "
                      f"> threshold {boundary_max_threshold_mm:.1f} mm")
                if pbar is not None:
                    pbar.update(1)
                continue

        # ---- X-point-deviation filter ---------------------------------
        # Independent of the boundary filter because the X-point cusp is
        # a localised feature; a perturbation can shift the X-point even
        # when the rest of the LCFS stays well within the boundary
        # tolerance (or vice versa).
        if xpoint_max_threshold_mm is not None and lock_coils:
            _xpt_max = diagnostics.get("x_point_max_mm", float('nan'))
            if np.isfinite(_xpt_max) and _xpt_max > xpoint_max_threshold_mm:
                print(f"  [xpoint_filter] discarded: "
                      f"max X-point dev = {_xpt_max:.1f} mm "
                      f"> threshold {xpoint_max_threshold_mm:.1f} mm")
                if pbar is not None:
                    pbar.update(1)
                continue

        # ---- Build a perturbed p-file from the baseline p-file --------
        # Start from the baseline so that profiles we don't perturb
        # (beam density, rotation, kpol, etc.) are preserved as-is.
        # Replace ne, te, ni, ti with the perturbed values, then
        # recompute derived quantities (nz1, ptot, diamagnetic
        # rotations, ExB decomposition) self-consistently.
        perturbed_pfile_bytes = pfile_bytes  # fallback: original bytes
        if pfile_bytes is not None:
            try:
                from .io.pfile import PFile as _PFile
                from .io import GEQDSKEquilibrium as _GEQDSK
                from scipy.interpolate import interp1d

                pf = _PFile.from_bytes(pfile_bytes)

                # Interpolate perturbed profiles → p-file grid.
                # When psi_N_kinetic is provided, the perturbed
                # profiles already cover the SOL — no extrapolation
                # needed.  Otherwise, fall back to the equilibrium
                # grid with edge-value fill (no cubic extrapolation).
                _psi_src = (psi_N_kinetic if psi_N_kinetic is not None
                            else psi_N)
                for pf_key, arr_si, scale in [
                    ("ne", ne_perturb, 1e-20),   # m^-3 → 10^20/m^3
                    ("te", te_perturb, 1e-3),     # eV   → keV
                    ("ni", ni_perturb, 1e-20),
                    ("ti", ti_perturb, 1e-3),
                ]:
                    if pf_key in pf:
                        psi_grid = pf.psinorm_for(pf_key)
                        vals = interp1d(
                            _psi_src, arr_si * scale,
                            kind="cubic",
                            bounds_error=False,
                            fill_value=(arr_si[0] * scale,
                                        arr_si[-1] * scale),
                        )(psi_grid)
                        pf.set_profile(pf_key, psi_grid, vals)

                # Keep baseline nz1 rather than recomputing from
                # quasi-neutrality.  Bouquet perturbs ne and ni
                # independently, which can push nz1 = (ne-ni-nb)/Z
                # negative — an unphysical result that produces
                # sign-flipped diamagnetic terms and spikes in Er/omghb.
                # The baseline nz1 is a physically consistent impurity
                # density and a reasonable approximation for the
                # perturbed case since we are not perturbing the
                # impurity content itself.

                # Recompute total pressure
                pf.compute_pressure()

                # Recompute diamagnetic rotations + decomposition
                # using the perturbed equilibrium just written to disk
                eq = _GEQDSK(full_path)
                psi_pf = pf.psinorm_for("ne")
                dpsi = eq.psi_boundary - eq.psi_axis
                psi_Wb = psi_pf * dpsi + eq.psi_axis

                pf.compute_diamagnetic_rotations(psi_Wb)

                # Outboard midplane R, Bp, Bt for ExB / Er / Hahm-Burrell
                mid = eq.midplane
                psi_eq = eq.psi_N
                R_mid = interp1d(
                    psi_eq, mid["R"],
                    fill_value="extrapolate")(psi_pf)
                Bp_mid = interp1d(
                    psi_eq, mid["Bp"],
                    fill_value="extrapolate")(psi_pf)
                Bt_mid = interp1d(
                    psi_eq, mid["Bt"],
                    fill_value="extrapolate")(psi_pf)

                pf.compute_rotation_decomposition(
                    R=R_mid, Bp=Bp_mid, Bt=Bt_mid, psi=psi_Wb)

                perturbed_pfile_bytes = pf.to_bytes()
            except Exception as exc:
                import traceback
                print(f"  WARNING: could not build perturbed p-file: {exc}")
                traceback.print_exc()

        store_equilibrium(
            header, count, full_path,
            psi_N,
            jphi_perturb,
            diagnostics["j_BS"],
            diagnostics["j_inductive"],
            ne_perturb, te_perturb,
            ni_perturb, ti_perturb,
            w_ExB,
            li1, li3,
            scan_val=scan_val,
            pressure=pressure_perturb,
            j_BS_edge=diagnostics["j_BS_edge"],
            pfile_bytes=perturbed_pfile_bytes,
            Zeff=Zeff_profile,
            coil_currents=coil_current_dict,
            psi_N_kinetic=psi_N_kinetic,
            boundary_devs_mm=diagnostics.get("boundary_devs_mm"),
            boundary_rms_mm=diagnostics.get("boundary_rms_mm"),
            boundary_max_mm=diagnostics.get("boundary_max_mm"),
            x_point_devs_mm=diagnostics.get("x_point_devs_mm"),
            x_point_rms_mm=diagnostics.get("x_point_rms_mm"),
            x_point_max_mm=diagnostics.get("x_point_max_mm"),
            x_points_RZ=diagnostics.get("x_points_RZ"),
            coil_drift_A=diagnostics.get("coil_drift_A"),
        )

        # Clean up on-disk eqdsk after archiving
        try:
            os.remove(full_path)
            print(f"  Deleted temporary file: {full_path}")
        except OSError as exc:
            print(f"  WARNING: could not delete {full_path}: {exc}")

        all_diagnostics.append(diagnostics)

    if pbar is not None:
        pbar.close()

    return all_diagnostics


# ====================================================================
#  Single-equilibrium reconstruction from geqdsk + kinetic profiles
# ====================================================================
def reconstruct_equilibrium(mygs, eqdsk, ne, te, ni, ti, Zeff,
                            isoflux_pts, weights, psi_pad,
                            guess_jinductive,n_k,psi_bridge,rescale_j_BS,
                            shelf_psi_N,initialize_psi=True,
                            isolate_edge_jBS=True,
                            n_spline_pts=32):
    r"""Reconstruct a single Grad-Shafranov equilibrium from a geqdsk
    reference and kinetic profiles, matching the EFIT :math:`l_i(1)`.

    The workflow is:

    1. Set isoflux boundary targets from the geqdsk LCFS.
    2. Compute bootstrap current via ``solve_with_bootstrap``.
    3. Fit a smooth inductive current profile with
       :func:`fit_inductive_profile`.
    4. Iterate on the inductive scale factor (hybrid secant–bisection
       with step clamping and psi save/restore) until the TokaMaker
       :math:`l_i(1)` matches the geqdsk value.
    5. Correct residual Ip drift by iterating on the Ip target passed
       to ``mygs.set_targets(Ip=...)`` via a secant search.  The
       :math:`j_\phi` profile shape is preserved; TokaMaker internally
       enforces the Ip constraint.

    Parameters
    ----------
    mygs : TokaMaker
        Initialised TokaMaker GS solver (mesh, regions, coils already
        set up).
    eqdsk : GEQDSKEquilibrium
        Parsed geqdsk equilibrium object.
    ne : ndarray
        Electron density on ``eqdsk.psi_N`` [m\ :sup:`-3`].
    te : ndarray
        Electron temperature on ``eqdsk.psi_N`` [eV].
    ni : ndarray
        Ion density on ``eqdsk.psi_N`` [m\ :sup:`-3`].
    ti : ndarray
        Ion temperature on ``eqdsk.psi_N`` [eV].
    Zeff : ndarray
        Effective charge on ``eqdsk.psi_N``.
    isoflux_pts : ndarray, shape (N, 2)
        :math:`(R, Z)` coordinates of isoflux constraint points
        [m].  Passed to ``mygs.set_isoflux``.
    weights : ndarray, shape (N,)
        Weights for each isoflux constraint point.
    psi_pad : float
        Padding inside the LCFS for :math:`l_i` evaluation.
    guess_jinductive : ndarray
        Initial guess for the inductive current-density profile,
        passed to ``solve_with_bootstrap`` as the starting
        :math:`j_{\rm inductive}` shape.
    n_k : int
        Spline order for :func:`fit_inductive_profile` (``k``
        parameter).
    psi_bridge : float
        :math:`\hat{\psi}` above which edge data are replaced by a
        zero anchor in :func:`fit_inductive_profile`.
    rescale_j_BS : bool
        If ``True``, jointly optimise a bootstrap rescaling factor
        in :func:`fit_inductive_profile`.
    shelf_psi_N : float
        If > 0, apply a flat shelf to :math:`j_{\rm BS}` for
        :math:`\hat{\psi} <` *shelf_psi_N* in
        :func:`fit_inductive_profile`.  ``0`` disables the shelf.
    initialize_psi : bool
        If ``True`` (default), call ``mygs.init_psi`` using LCFS
        geometry estimated from the geqdsk boundary.  Set to ``False``
        to skip initialisation (e.g. when reusing a prior solution).
    isolate_edge_jBS : bool
        If ``True`` (default), isolate the edge bootstrap spike from
        the core bootstrap current via ``analyze_bootstrap_edge_spike``.
        If ``False``, the full bootstrap current is used as the spike
        profile (no core/edge separation).

    Returns
    -------
    dict
        Result dictionary containing reconstructed profiles, fields,
        and comparison data keyed as documented inline.
    """
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
    from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap

    if initialize_psi:
        # Estimate shape parameters from geqdsk LCFS geometry
        geo = eqdsk.geometry
        R0 = geo['R'][-1]
        Z0 = geo['Z'][-1]
        a = geo['a'][-1]
        kappa = geo['kappa'][-1]
        delta = geo['delta'][-1]
        mygs.init_psi(R0, Z0, a, kappa, delta)

    eqdsk_jtor = abs(eqdsk.j_tor_averaged_direct)

    # ---- 2. Bootstrap current ----
    results = solve_with_bootstrap(
        mygs, ne, te, ni, ti, Zeff,
        abs(eqdsk.Ip), guess_jinductive,
        scale_jBS=1.0,
        isolate_edge_jBS=isolate_edge_jBS,
        diagnostic_plots=False,
    )

    j_BS_isolated = results['isolated_j_BS']

    # ---- 2b. Classify the j_phi profile ----
    jphi_mode, spike_metrics = classify_jphi_profile(
        eqdsk.psi_N, eqdsk_jtor, j_BS_isolated
    )

    # Pre-compute shelf location (needed for mode-dependent iteration)
    _, _shelf_psi_recon = _shelf_blend_decompose(
        eqdsk.psi_N, eqdsk_jtor, j_BS_isolated, eqdsk_jphi=eqdsk_jtor
    )  # just to get shelf_psi; j_ind result discarded

    # ---- 3. Fit inductive profile ----
    baseline_li_proxy = calc_cylindrical_li_proxy(mygs, eqdsk_jtor, psi_pad)

    fit_result = fit_inductive_profile(
        mygs, eqdsk_jtor, j_BS_isolated, eqdsk.psi_N, psi_pad,
        baseline_li_proxy,
        k=n_k, psi_bridge=psi_bridge,
        rescale_j_BS=rescale_j_BS,
        shelf_psi_N=shelf_psi_N,
        n_spline_pts=n_spline_pts,
    )

    j_inductive_fit_raw = fit_result['j_inductive_fit']
    scale_opt = fit_result['ind_scale']
    bs_scale_opt = fit_result['bs_scale']
    j_BS_isolated = fit_result['j_BS_used']

    print(f"[fit] ind_scale={scale_opt:.6f}  bs_scale={bs_scale_opt:.6f}  "
          f"li_proxy={fit_result['fit_li']:.6f}  (target={baseline_li_proxy:.6f})")

    # Smooth the shelf→spike transition in j_BS_isolated to eliminate
    # second-derivative discontinuities that TokaMaker's geometry
    # coupling amplifies into visible divots in the output j_phi.
    # Apply a localised Gaussian filter only around the transition zone.
    from scipy.ndimage import gaussian_filter1d

    _shelf_val_sm = j_BS_isolated[0]
    _shelf_end_sm = 0
    for _i in range(1, len(j_BS_isolated)):
        if abs(j_BS_isolated[_i] - _shelf_val_sm) / max(abs(_shelf_val_sm), 1e-30) < 1e-6:
            _shelf_end_sm = _i
        else:
            break

    # Smooth a window around the shelf end (±10 indices)
    _sm_half = 10
    _sm_lo = max(0, _shelf_end_sm - _sm_half)
    _sm_hi = min(len(j_BS_isolated), _shelf_end_sm + _sm_half + 1)
    _sm_sigma = 3.0  # Gaussian width in grid indices

    _smoothed_section = gaussian_filter1d(j_BS_isolated[_sm_lo:_sm_hi], sigma=_sm_sigma)

    # Blend smoothed section back — only modify the transition zone,
    # preserve the exact shelf value in the core and exact spike beyond
    j_BS_isolated_smooth = j_BS_isolated.copy()
    for _i in range(_sm_lo, _sm_hi):
        _local = _i - _sm_lo
        # Blend weight: 1 at shelf_end, 0 at edges of window
        _dist = abs(_i - _shelf_end_sm) / _sm_half
        _w = max(0.0, 1.0 - _dist)  # triangular window
        j_BS_isolated_smooth[_i] = (_w * _smoothed_section[_local]
                                     + (1 - _w) * j_BS_isolated[_i])

    j_BS_isolated = j_BS_isolated_smooth

    # Use the spline-fit j_inductive directly. The corrective iteration
    # (section 7) will drive TokaMaker's output to match the target
    # j_phi = j_inductive_fit + j_BS_isolated, compensating for any
    # geometry-coupling distortion at the edge.
    j_inductive_fit = j_inductive_fit_raw

    # DEBUG: check spline fit for divot before corrective iteration
    _d2_max = 0
    _d2_idx = 0
    for _di in range(1, len(eqdsk.psi_N) - 1):
        _s1 = (j_inductive_fit[_di] - j_inductive_fit[_di-1]) / (eqdsk.psi_N[_di] - eqdsk.psi_N[_di-1])
        _s2 = (j_inductive_fit[_di+1] - j_inductive_fit[_di]) / (eqdsk.psi_N[_di+1] - eqdsk.psi_N[_di])
        _d2 = abs(_s2 - _s1) / 1e6
        if _d2 > _d2_max:
            _d2_max = _d2
            _d2_idx = _di
    print(f"[spline_check] max |d²j_ind/dpsi²| = {_d2_max:.4f} MA/m²/psiN² "
          f"at index {_d2_idx} (psi_N={eqdsk.psi_N[_d2_idx]:.5f})")

    # ---- 4. Pressure and GS profiles ----
    pres_tmp = 1.6022e-19 * (ne * te + ni * ti)
    psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
    pprime_tmp = np.gradient(pres_tmp) / (np.gradient(eqdsk.psi_N) * psi_range)
    pprime_tmp[-1] = 0.0

    pp_prof = {"type": "linterp", "y": pprime_tmp, "x": eqdsk.psi_N}
    ffp_prof = {
        "type": "jphi-linterp",
        "y": j_inductive_fit + j_BS_isolated,
        "x": eqdsk.psi_N,
    }

    mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)
    mygs.solve()

    # ---- 5. Hybrid secant–bisection iteration to match eqdsk li(1) ----
    #
    # Guard-rails that prevent TokaMaker from being given profiles too
    # far from the last converged state:
    #   a) The secant step is clamped to ±max_step_frac of the current
    #      ind_factor so the GS solver always starts close to its
    #      previous solution.
    #   b) Once we have a bracket (one point above, one below target)
    #      bisection is used whenever the (clamped) secant would escape
    #      the bracket.
    #   c) The last converged psi is saved with get_psi / set_psi so
    #      that a non-converged solve does not poison subsequent
    #      iterations.
    li_target = eqdsk.li["li(1)_EFIT"]
    li_tol = 0.001
    max_li_iters = 20
    max_step_frac = 0.10  # cap secant steps at ±10 % of current value

    # -- save / restore helpers for the last known-good psi state -----
    _last_good_psi = mygs.get_psi(False).copy()

    def _save_psi():
        nonlocal _last_good_psi
        _last_good_psi = mygs.get_psi(False).copy()

    def _restore_psi():
        mygs.set_psi(_last_good_psi, update_bounds=True)

    def _solve_and_get_li(ind_factor):
        """Set profiles with scaled j_inductive, solve, return li(1).

        Saves psi on success; restores the previous good psi on
        TokaMaker solve failure so the next attempt starts clean.
        """
        ffp_tmp = {
            "type": "jphi-linterp",
            "y": ind_factor * j_inductive_fit + j_BS_isolated,
            "x": eqdsk.psi_N,
        }
        mygs.set_profiles(ffp_prof=ffp_tmp, pp_prof=pp_prof)
        try:
            mygs.solve()
        except ValueError:
            print(f"[li match]   solve failed for ind_factor={ind_factor:.6f}, "
                  "restoring last good psi")
            _restore_psi()
            return None  # signal failed solve
        _save_psi()
        eq_stats = mygs.get_stats(li_normalization='std', lcfs_pad=psi_pad)
        return eq_stats['l_i']

    # -- bracket bookkeeping ------------------------------------------
    # bracket_lo: (ind, li) with li < li_target  (err < 0)
    # bracket_hi: (ind, li) with li > li_target  (err > 0)
    bracket_lo = bracket_hi = None

    def _update_bracket(ind, li):
        nonlocal bracket_lo, bracket_hi
        if li < li_target:
            if bracket_lo is None or abs(li - li_target) < abs(bracket_lo[1] - li_target):
                bracket_lo = (ind, li)
        else:
            if bracket_hi is None or abs(li - li_target) < abs(bracket_hi[1] - li_target):
                bracket_hi = (ind, li)

    # -- initial two evaluations --------------------------------------
    eq_stats_0 = mygs.get_stats(li_normalization='std', lcfs_pad=psi_pad)
    ind_0, li_0 = 1.0, eq_stats_0['l_i']
    _save_psi()
    _update_bracket(ind_0, li_0)

    ind_1 = 1.05
    li_1_sec = _solve_and_get_li(ind_1)
    if li_1_sec is not None:
        _update_bracket(ind_1, li_1_sec)

    print(f"[li match] target={li_target:.6f}")
    print(f"[li match] iter 0: ind_factor={ind_0:.6f}  li={li_0:.6f}  err={li_0 - li_target:.6f}")
    print(f"[li match] iter 1: ind_factor={ind_1:.6f}  li={li_1_sec:.6f}  err={li_1_sec - li_target:.6f}")

    for li_iter in range(2, max_li_iters):
        err_0 = li_0 - li_target
        err_1 = li_1_sec - li_target

        if li_1_sec is not None and abs(err_1) < li_tol:
            print(f"[li match] converged at iter {li_iter}: "
                  f"ind_factor={ind_1:.6f}  li={li_1_sec:.6f}")
            break

        # -- propose next ind_factor ----------------------------------
        use_bisection = False

        if li_1_sec is None:
            # Previous solve failed — fall back to bisection if we have
            # a bracket, otherwise halve the step toward last good point
            use_bisection = True
        else:
            denom = err_1 - err_0
            if abs(denom) < 1e-14:
                use_bisection = True
            else:
                ind_secant = ind_1 - err_1 * (ind_1 - ind_0) / denom
                ind_secant = max(ind_secant, 0.0)

        if use_bisection and bracket_lo is not None and bracket_hi is not None:
            ind_new = 0.5 * (bracket_lo[0] + bracket_hi[0])
            print(f"[li match]   bisection -> {ind_new:.6f}")
        elif use_bisection:
            # No bracket yet — retreat halfway toward ind_0
            ind_new = 0.5 * (ind_0 + ind_1)
            print(f"[li match]   midpoint fallback -> {ind_new:.6f}")
        else:
            # Clamp secant step to ±max_step_frac of current value
            max_delta = max_step_frac * abs(ind_1)
            ind_clamped = np.clip(ind_secant,
                                  ind_1 - max_delta,
                                  ind_1 + max_delta)
            if ind_clamped != ind_secant:
                print(f"[li match]   clamped secant {ind_secant:.6f} "
                      f"-> {ind_clamped:.6f}")

            # If we have a bracket, ensure we stay inside it
            if bracket_lo is not None and bracket_hi is not None:
                blo, bhi = sorted([bracket_lo[0], bracket_hi[0]])
                if not (blo <= ind_clamped <= bhi):
                    ind_new = 0.5 * (bracket_lo[0] + bracket_hi[0])
                    print(f"[li match]   secant escaped bracket, "
                          f"bisection -> {ind_new:.6f}")
                else:
                    ind_new = ind_clamped
            else:
                ind_new = ind_clamped

        # -- evaluate ---------------------------------------------------
        ind_0, li_0 = ind_1, li_1_sec if li_1_sec is not None else li_0
        ind_1 = ind_new
        li_1_sec = _solve_and_get_li(ind_1)
        if li_1_sec is not None:
            _update_bracket(ind_1, li_1_sec)

        li_disp = f"{li_1_sec:.6f}" if li_1_sec is not None else "FAILED"
        err_disp = (f"{li_1_sec - li_target:.6f}"
                    if li_1_sec is not None else "N/A")
        print(f"[li match] iter {li_iter}: ind_factor={ind_1:.6f}  "
              f"li={li_disp}  err={err_disp}")
    else:
        print(f"[li match] WARNING: did not converge within "
              f"{max_li_iters} iterations")

    # Ensure the final state is from a converged solve
    if li_1_sec is None:
        _restore_psi()

    _eq_stats_final = mygs.get_stats(li_normalization='std', lcfs_pad=psi_pad)
    final_li = _eq_stats_final['l_i']
    Ip_tokamaker = _eq_stats_final['Ip']
    print(f"[li match] final li(1)={final_li:.6f}  target={li_target:.6f}  |err|={abs(final_li - li_target):.6f}")

    # ---- 6. Ip-correction secant (set_targets Ip rescaling) ----------
    #
    # The jphi-linterp profile type has inherent Ip drift because the
    # actual integrated current depends on the equilibrium geometry.
    # We correct this by iterating on the Ip target passed to
    # mygs.set_targets() — TokaMaker internally rescales the current
    # density to enforce the constraint.  The j_phi profile *shape*
    # is unchanged, so li is minimally perturbed.
    Ip_desired = abs(eqdsk.Ip)
    Ip_tol = 0.0005  # relative tolerance on Ip
    max_Ip_iters = 8

    # Current j_phi components after li convergence (fixed throughout)
    j_ind_li = ind_1 * j_inductive_fit  # li-matched inductive profile
    j_phi_li = j_ind_li + j_BS_isolated

    def _solve_and_get_Ip(Ip_trial):
        """Set Ip target, solve, return (actual Ip, li)."""
        mygs.set_targets(Ip=Ip_trial,pax=pres_tmp[0])
        try:
            mygs.solve()
        except ValueError:
            print(f"[Ip match]   solve failed for Ip_target={Ip_trial:.1f}, "
                  "restoring last good psi")
            _restore_psi()
            return None, None
        _save_psi()
        stats = mygs.get_stats(li_normalization='std', lcfs_pad=psi_pad)
        return stats['Ip'], stats['l_i']

    Ip_err_rel = abs(Ip_tokamaker - Ip_desired) / Ip_desired
    if Ip_err_rel > Ip_tol:
        print(f"\n[Ip match] desired={Ip_desired:.1f}  current={Ip_tokamaker:.1f}  "
              f"err={100 * (Ip_tokamaker - Ip_desired) / Ip_desired:+.4f}%")

        # Two initial evaluations for secant:
        # pt 0: Ip_target = Ip_desired (the true target), result already known
        # pt 1: Ip_target adjusted by a first-order correction
        t0, Ip_0 = Ip_desired, Ip_tokamaker
        t1 = Ip_desired * (Ip_desired / Ip_tokamaker)  # first-order correction
        Ip_1, li_1_Ip = _solve_and_get_Ip(t1)

        if Ip_1 is not None:
            print(f"[Ip match] iter 0: Ip_target={t0:.1f}  Ip={Ip_0:.1f}")
            print(f"[Ip match] iter 1: Ip_target={t1:.1f}  Ip={Ip_1:.1f}  "
                  f"li={li_1_Ip:.6f}")

            for Ip_iter in range(2, max_Ip_iters):
                e0 = Ip_0 - Ip_desired
                e1 = Ip_1 - Ip_desired

                if abs(e1 / Ip_desired) < Ip_tol:
                    print(f"[Ip match] converged at iter {Ip_iter}: "
                          f"Ip_target={t1:.1f}  Ip={Ip_1:.1f}  li={li_1_Ip:.6f}")
                    break

                denom = e1 - e0
                if abs(denom) < 1.0:
                    print(f"[Ip match] secant denominator ~0, stopping")
                    break

                t_new = t1 - e1 * (t1 - t0) / denom
                t_new = max(t_new, 0.5 * Ip_desired)  # safety floor

                t0, Ip_0 = t1, Ip_1
                t1 = t_new
                Ip_1, li_1_Ip = _solve_and_get_Ip(t1)
                if Ip_1 is None:
                    print(f"[Ip match] solve failed, keeping previous result")
                    break

                print(f"[Ip match] iter {Ip_iter}: Ip_target={t1:.1f}  "
                      f"Ip={Ip_1:.1f}  li={li_1_Ip:.6f}")
            else:
                print(f"[Ip match] WARNING: did not converge within "
                      f"{max_Ip_iters} iterations")
    else:
        print(f"\n[Ip match] already within tolerance: "
              f"err={100 * (Ip_tokamaker - Ip_desired) / Ip_desired:+.4f}%")

    # -- Final stats after Ip correction --------------------------------
    _eq_stats_final = mygs.get_stats(li_normalization='std', lcfs_pad=psi_pad)
    final_li = _eq_stats_final['l_i']
    Ip_tokamaker = _eq_stats_final['Ip']
    print(f"[final] li(1)={final_li:.6f}  Ip={Ip_tokamaker:.1f}  "
          f"Ip_err={100 * (Ip_tokamaker - Ip_desired) / Ip_desired:+.4f}%  "
          f"li_err={abs(final_li - li_target):.6f}")

    # ---- 7. Mode-dependent corrective iteration ----
    #
    # TokaMaker's jphi-linterp converts j_phi to FF' using flux-surface
    # geometry (<R>, <1/R>).  The output j_phi generally differs from
    # the input because the geometry changes after solving.  Iterate
    # the input to drive the output toward the geqdsk target.
    #
    # For Lmode_like_jphi: only correct in the core (psi_N < shelf),
    # preserving the Sauter edge spike that the geqdsk lacks.
    Ip_final_target = abs(eqdsk.Ip)

    if jphi_mode == 'L_mode':
        # TODO: validate with L-mode test data
        j_BS_isolated_corr = np.zeros_like(eqdsk.psi_N)
        corr_target = eqdsk_jtor.copy()
        print("[reconstruct] L_mode: zeroing j_BS_isolated, using geqdsk j_phi as j_inductive")
    else:
        # H_mode and Lmode_like_jphi: target = j_inductive_fit + j_BS_isolated.
        # Always trust the Sauter edge spike rather than the geqdsk edge.
        # The spline fit already matches the geqdsk in the core
        # (since it fits eqdsk - j_BS), so no blend with eqdsk_jtor
        # is needed — avoiding blend-induced kinks and preserving
        # the Sauter edge structure.
        corr_target = (j_ind_li + j_BS_isolated).copy()
        j_BS_isolated_corr = j_BS_isolated.copy()

    # Adaptive corrective iteration
    j_phi_output_corr, _n_corr, _corr_hist = _corrective_jphi_iteration(
        mygs, eqdsk.psi_N, corr_target, pp_prof,
        Ip_final_target, pres_tmp[0], psi_pad,
        min_iters=2, max_iters=8, rtol=0.05, verbose=True,
    )

    # ---- 8. Final profiles ----
    # The corrective iteration drove TokaMaker's output toward
    # corr_target (= Hermite-bridged j_inductive + j_BS in the edge,
    # geqdsk in the core).  The output j_phi_output_corr is smooth
    # and self-consistent.  Decompose by simple subtraction — no
    # re-running the Hermite bridge, which would create a different
    # optimisation and introduce kinks.
    j_phi_final = j_phi_output_corr.copy()
    j_BS_final = j_BS_isolated_corr.copy()

    if jphi_mode == 'L_mode':
        # TODO: validate with L-mode test data
        j_ind_final = j_phi_final.copy()
        j_BS_final = np.zeros_like(j_phi_final)
    else:
        j_ind_final = j_phi_final - j_BS_final
        j_ind_final = np.maximum(j_ind_final, 0.0)

    # ---- 9. Reconstruction quality metrics ----
    _edge_mask = eqdsk.psi_N > 0.9
    _core_mask = eqdsk.psi_N < 0.8

    # Boundary deviation: nearest-neighbor distance from geqdsk boundary
    # points to the TokaMaker LCFS contour (same method as plotting.py)
    from scipy.spatial import cKDTree as _cKDTree_q
    _psi_arr = mygs.get_psi(False)
    _psi_lcfs = float(mygs.psi_bounds[0])
    _fig_tmp, _ax_tmp = plt.subplots(1, 1)
    try:
        _cs = _ax_tmp.tricontour(
            mygs.r[:, 0], mygs.r[:, 1], mygs.lc, _psi_arr,
            levels=[_psi_lcfs])
        _segs = [v for seg in _cs.allsegs for v in seg if len(v) > 4]
    finally:
        plt.close(_fig_tmp)

    if _segs:
        _lcfs_pts = max(_segs, key=len)
        _tree = _cKDTree_q(_lcfs_pts)
        _devs, _ = _tree.query(isoflux_pts)
        _bnd_rms_mm = float(np.sqrt(np.mean(_devs**2)) * 1e3)
        _bnd_max_mm = float(np.max(_devs) * 1e3)
    else:
        _bnd_rms_mm = float('nan')
        _bnd_max_mm = float('nan')

    quality = {
        'jphi_mode': jphi_mode,
        **spike_metrics,
        'jphi_core_rms': float(np.sqrt(np.mean(
            (j_phi_final[_core_mask] - eqdsk_jtor[_core_mask])**2))),
        'jphi_edge_rms': float(np.sqrt(np.mean(
            (j_phi_final[_edge_mask] - eqdsk_jtor[_edge_mask])**2))),
        'li_error': float(abs(final_li - li_target)),
        'Ip_error_pct': float(100 * abs(Ip_tokamaker - Ip_desired) / Ip_desired),
        'boundary_rms_mm': _bnd_rms_mm,
        'boundary_max_dev_mm': _bnd_max_mm,
    }
    print(f"[quality] mode={jphi_mode}, core_rms={quality['jphi_core_rms']/1e6:.4f} MA/m², "
          f"edge_rms={quality['jphi_edge_rms']/1e6:.4f} MA/m², "
          f"li_err={quality['li_error']:.6f}, Ip_err={quality['Ip_error_pct']:.4f}%, "
          f"bnd_rms={_bnd_rms_mm:.2f} mm, bnd_max={_bnd_max_mm:.2f} mm")

    # FF' from the converged TokaMaker equilibrium
    _, F_prof, Fp_prof, _, _ = mygs.get_profiles(psi=eqdsk.psi_N)
    ffprime_tokamaker = F_prof * Fp_prof

    return {
        'ne': ne.copy(),
        'te': te.copy(),
        'ni': ni.copy(),
        'ti': ti.copy(),
        'Zeff': Zeff.copy(),
        'isoflux_pts': isoflux_pts.copy(),
        'weights': weights.copy(),
        'psi_lcfs_val': float(mygs.psi_bounds[0]),
        'j_inductive_fit': j_ind_final.copy(),
        'j_phi_fit': j_phi_final.copy(),
        'j_BS_used': j_BS_final.copy(),
        'psi': mygs.get_psi(False),
        'pprime': pprime_tmp.copy(),
        'ffprime': ffprime_tokamaker.copy(),
        'ind_factor_final': ind_1,
        'bs_factor_final': 1.0,
        'Ip_tokamaker': Ip_tokamaker,
        'eqdsk_jtor': eqdsk_jtor.copy(),
        'eqdsk_psi_N': eqdsk.psi_N.copy(),
        'eqdsk_pres': eqdsk.pres.copy(),
        'eqdsk_boundary_R': eqdsk.boundary_R.copy(),
        'eqdsk_boundary_Z': eqdsk.boundary_Z.copy(),
        'eqdsk_ffprim': eqdsk.ffprim.copy(),
        'eqdsk_li': dict(eqdsk.li),
        'eqdsk_Ip': eqdsk.Ip,
        'pres_tokamaker': pres_tmp.copy(),
        'psi_N_grid': eqdsk.psi_N.copy(),
        'li_final': final_li,
        'quality': quality,
    }
