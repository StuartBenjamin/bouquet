
import numpy as np

# ====================================================================
#  Uncertainty envelope builder.
#  Multiply against your profile of choice to give σ(ψ_N).
# ====================================================================
def new_uncertainty_profiles(
    psi_N,
    uncertainty,
    falloff_exp=None,
    shelf=0.0,
    edge_val=0.0,
    falloff_loc=0.8,
    tail_alpha=2.5,
):
    r"""Build a 1-D uncertainty envelope over normalised flux.

    Two modes are supported:

    * **Power-law mode** (``falloff_exp`` is not ``None``):
      :math:`u(\hat{\psi}) = U\,(1 - \hat{\psi})^{\mathrm{falloff\_exp}} + S`

      where *S* is the ``shelf`` parameter — a minimum fractional
      uncertainty that prevents the envelope from decaying to zero at
      the edge.

    * **Flat + tail mode** (default, ``falloff_exp`` is ``None``):
      constant value :math:`U` for
      :math:`\hat{\psi} \le \hat{\psi}_{\rm loc}`, then a cosine
      (or cosh) decay to ``edge_val`` at :math:`\hat{\psi}=1`
      controlled by ``tail_alpha``.  The shelf is added after the
      tail decay.

    Parameters
    ----------
    psi_N : ndarray
        1-D array of normalised poloidal flux :math:`\hat{\psi}`.
    uncertainty : float
        Scalar amplitude :math:`U` of the envelope.
    falloff_exp : float or None
        Exponent for the power-law branch (``None`` selects flat + tail).
    shelf : float
        Minimum fractional uncertainty floor added to the envelope.
        Prevents the profile from decaying all the way to zero.
    edge_val : float
        Envelope value at :math:`\hat{\psi}=1` (flat + tail mode).
    falloff_loc : float
        :math:`\hat{\psi}_{\rm loc}` where the tail begins.
    tail_alpha : float
        Sharpness exponent :math:`\alpha` of the cosine/cosh tail.

    Returns
    -------
    ndarray
        1-D array of the same length as ``psi_N``.
    """
    # ---- power-law branch -------------------------------------------
    if falloff_exp is not None:
        return uncertainty * (1.0 - psi_N) ** falloff_exp + shelf

    # ---- flat + cosine/cosh tail branch -----------------------------
    profile_left = uncertainty * np.ones_like(psi_N)
    stitch_height = uncertainty                        # value at falloff_loc

    dist_to_edge = max(1.0 - falloff_loc, 1e-5)       # avoid division by zero

    if edge_val < stitch_height:
        # --- cosine decay ---
        target_cos = np.clip(
            (edge_val / stitch_height) ** (1.0 / tail_alpha), -1.0, 1.0
        )
        omega = np.arccos(target_cos) / dist_to_edge

        base_cos = np.maximum(np.cos(omega * (psi_N - falloff_loc)), 0.0)
        profile_right = stitch_height * (base_cos ** tail_alpha)
    else:
        # --- cosh rise (rare) ---
        target_cosh = (edge_val / stitch_height) ** (1.0 / tail_alpha)
        omega = np.arccosh(target_cosh) / dist_to_edge
        profile_right = stitch_height * (
            np.cosh(omega * (psi_N - falloff_loc)) ** tail_alpha
        )

    profile = np.where(psi_N <= falloff_loc, profile_left, profile_right)

    if edge_val >= 0.0:
        profile = np.maximum(profile, 0.0)

    return profile + shelf


# ====================================================================
#  Synthetic "IDA-like" fractional-sigma profiles (sine basis).
#
#  A shareable, non-proprietary stand-in for the DIII-D 204441@4400 IDA
#  uncertainty envelopes, fit to the real fractional-sigma SHAPE with an
#  all-sine basis so it can ship in the public example + golden tests:
#
#     s(psi)   = sin(pi/2 * psi_N)              (0 at axis -> 1 at edge)
#     frac(psi)= const
#                + sum_k  c_k  * sin(k*pi*psi_N)   (core/mid oscillation; Te,Ti)
#                + sum_p  e_p  * s**p              (edge rise/spike; e_p >= 0)
#
#  Coefficients were obtained by bounded least squares against the real
#  204441@4400 IDA fractional sigma (edge powers constrained >= 0 so the
#  edge can only add, never dig a pre-edge trough).  ne uses no harmonics
#  (flat core + smooth edge); Te/Ti carry the mid-radius shoulder/bump.
#  These constants ARE the golden definition -- do not refit casually.
# ====================================================================
_SYNTHETIC_IDA_SIGMA = {
    # channel: (const, {harmonic_k: coeff}, {edge_power_p: coeff})  [fractional]
    "ne": (0.0238, {},                                  {64: 0.0361}),
    "te": (0.0317, {1: 0.0292, 2: -0.0035, 3: 0.0059},  {64: 0.0846}),
    "ti": (0.0569, {1: 0.0139, 2: 0.0111, 3: 0.0002},   {16: 0.0045, 64: 0.0348}),
}
# aliases -> canonical channel key
_SIGMA_ALIASES = {
    "ne": "ne", "n_e": "ne", "density": "ne", "ni": "ne", "n_i": "ne",
    "te": "te", "t_e": "te",
    "ti": "ti", "t_i": "ti", "t_12c6": "ti",
}


def synthetic_ida_sigma(psi_N, channel):
    r"""Synthetic IDA-like **fractional** 1-sigma uncertainty envelope.

    Returns the fractional uncertainty :math:`\sigma_X / X` on a sine basis
    (see module notes), calibrated to the DIII-D 204441@4400 IDA shape.
    Multiply by the profile to get the absolute sigma, e.g.
    ``sigma_ne = synthetic_ida_sigma(psi_N, 'ne') * ne_SI``.

    Parameters
    ----------
    psi_N : array_like
        Normalised poloidal flux (0 = axis, 1 = LCFS).  Values > 1 (SOL)
        are clamped to 1 so the envelope holds its edge value.
    channel : str
        One of ``ne``/``ni``/``te``/``ti`` (aliases ``n_e``, ``t_e``,
        ``t_i``, ``t_12c6``, ``density`` accepted).  ``ni`` reuses ``ne``.

    Returns
    -------
    ndarray
        Fractional sigma, same shape as ``psi_N``, clipped to >= 0.

    Notes
    -----
    Used by the D3D-like example notebook and the backend golden tests as a
    deterministic, non-proprietary uncertainty model.  The j_phi envelope is
    handled separately (flat 10 % fractional), as in the 204441 workflow.
    """
    key = _SIGMA_ALIASES.get(str(channel).strip().lower())
    if key is None:
        raise ValueError(
            f"Unknown channel {channel!r}; expected one of {sorted(set(_SIGMA_ALIASES))}"
        )
    const, harm, edge = _SYNTHETIC_IDA_SIGMA[key]
    psi = np.clip(np.asarray(psi_N, dtype=float), 0.0, 1.0)
    s = np.sin(np.pi / 2.0 * psi)
    frac = np.full_like(psi, float(const))
    for k, c in harm.items():
        frac = frac + c * np.sin(k * np.pi * psi)
    for p, e in edge.items():
        frac = frac + e * s ** p
    return np.clip(frac, 0.0, None)
