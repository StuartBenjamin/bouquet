# Reconstruction leaves a small constant boundary offset from jphi-linterp edge / separatrix handling

**Labels:** `reconstruction`, `accuracy`, `investigation`

## Summary

When `reconstruct_equilibrium` (and the per-draw bouquet solves) run in
**`jphi-linterp`** mode — we specify the flux-surface-averaged toroidal current
`⟨j_φ⟩` and let TokaMaker back-solve `FF'` — the realized current near the
separatrix systematically **overshoots** the specified profile. This produces a
small but **constant** offset between the jphi-linterp equilibrium and recon's
native inverse-mode LCFS, and forces two workarounds we currently rely on
(`jphi_baseline=True` referencing + `save_eqdsk(truncate_eq=True)`).

The offset is benign today (it's constant, not a bias, and the
`jphi_baseline` reference absorbs most of it), but it sets a ~0.5 mm floor on
every downstream boundary-deviation measurement and means the g-file we write
doesn't faithfully represent the pedestal-foot / separatrix current. This
issue tracks investigating a cleaner edge/separatrix treatment.

## Evidence

The opt-in backend systematics test (`tests/test_systematics.py`, Mode 1:
`pin_jphi=True`, `σ=0`) drives every draw to reproduce the baseline exactly.
It lands at:

- **boundary RMS ≈ 0.525 mm** (identical across draws → deterministic, *not* a
  random bias),
- **max coil drift ≈ 0.054 %**.

With `pin_jphi` the bootstrap/SWB call is bypassed, so this ~0.5 mm is **purely**
the jphi-linterp edge-representation residual — it is not bootstrap-related.

Spot value (D3D-like baseline, 257-pt grid): near `ψ_N ≈ 0.996` the input
`⟨j_φ⟩ ≈ 0.20 MA/m²` is realized as `≈ 0.29 MA/m²` (~5 % of peak) after the
solve. That edge overshoot is what displaces the LCFS by ~0.5 mm.

## Root cause (current understanding)

1. **jphi-linterp → FF' conversion uses flux-surface geometry** (`⟨R⟩`, `⟨1/R⟩`)
   that changes after the solve, so the output `⟨j_φ⟩` differs from the input,
   most visibly in the steep-gradient edge/pedestal region.
2. **`save_eqdsk(truncate_eq=False)`** holds the profile past `lcfs_pad` and
   forces an EFIT free-boundary BC at the last node (`P'=0`, `FF'` snaps),
   producing a derivative kink in the edge `⟨j_φ⟩`. We work around this with
   `truncate_eq=True`, which genuinely truncates at `lcfs_pad` and gives a
   monotone edge — but that's a save-time patch, not a solve-time fix.
3. **No explicit separatrix-current model.** The reconstruction doesn't assign a
   defined `j_φ` at the LCFS / pedestal foot; the value falls out of the
   linterp + BC interaction rather than being modeled.

## Why it matters

- Sets a ~0.5 mm structural floor on per-draw boundary-deviation diagnostics
  (`plot_traces`, `filter_boundaries`, the systematics test threshold).
- The written g-file's edge `⟨j_φ⟩` is a representation artifact, not the
  intended profile — relevant for anyone consuming the separatrix current.
- We currently mask it with `jphi_baseline=True` (reference per-draw diagnostics
  to the unperturbed jphi-linterp baseline so σ=0 lands ~0). That's the right
  pragmatic default, but it hides rather than removes the offset.

## Proposed investigation

- [ ] Quantify the edge `⟨j_φ⟩` input-vs-output mismatch vs `ψ_N` resolution
      (we already bump `nlevels` to 257 → 15 pedestal points; does finer help,
      or is it a quadrature/cut-cell limitation near the separatrix?).
- [ ] Decide whether `jphi-linterp` should explicitly model the pedestal-foot /
      separatrix current (a defined `j_sep`) rather than letting the BC snap it.
- [ ] Evaluate whether the corrective-iteration (`_corrective_jphi_iteration`)
      should also target the edge node, or whether an inverse-mode edge anchor
      is preferable.
- [ ] Determine whether `truncate_eq=True` should be the long-term default for
      saved g-files, or replaced by a solve-time edge treatment.
- [ ] Re-run `tests/test_systematics.py` after any change — the ~0.5 mm floor is
      the regression metric.

## References

- Backend tracking task #41 ("Revisit reconstruction's handling of
  `j_separatrix` (edge/LCFS current) values").
- `tests/test_systematics.py` (Mode 1 = the no-systematic floor measurement).
- `bouquet/TokaMaker_interface.py`: `reconstruct_equilibrium`,
  `_corrective_jphi_iteration`, the `jphi_baseline` baseline-solve block, and
  `safe_save_eqdsk(truncate_eq=...)`.
