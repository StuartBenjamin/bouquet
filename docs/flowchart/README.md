# bouquet workflow & logic map

> **Stale as of 1.4.0 (the full logic map only).** `graph.json` and the
> rendered `l1_full.svg` / `l0_overview.svg` / `index.html` were last
> extracted before the unified engine (index.html rendered 2026-09-18,
> graph.json hand-patched to 2026-10-10). None of the 588 nodes points into
> `bouquet/engine.py`, `engine_draws.py`, `solver_geometry.py`,
> `boundary_balance.py` or `kinetic_sampler.py`, and most `file:line` anchors
> no longer match the code, so a patch of a few nodes would not make the map
> honest; it needs a re-extraction. The physics workflow
> (`physics_workflow.svg` / `.pdf`) is unaffected. Nodes that 1.4.0 changed:
> `fn.run.isoflux_weights or np.ones` and `fn.run.isoflux_pts or
> eqdsk.isoflux_pts` (now the boundary set: 256 points per unit length at a
> constant total of 2e5, `solver_geometry.case_geometry`, with the report-only
> partition probe after the reconstruction); `gate.iso_update` (engine draws
> hold the reconstruction's isoflux); the homotopy nodes
> (`fn.TokaMaker_interface.for F_tol, VSC_tol in ...homotopy_passes`,
> `gate.homotopy_feasible`, `gate.homotopy_early_stop`: the VSC bound is
> floored at 10 kA, and a capped stage may carry the `vertical_runaway`
> label); `gate.draw_accepted` (the per-draw wall budget, the kinetic clips
> on every route); `gate.in_spec_verdict`, `gate.coil_spec`,
> `gate.boundary_rms` (in spec = the χ² coil test AND the boundary cut, the
> magnetics); `cfg.generationconfig.l_i_tolerance`, `cfg.l_i_tolerance`,
> `gate.li_band`, `gate.li_band_loop`, `gate.li_acceptance` (the l_i band is
> opt-in, default off; it still constructs the legacy draws); and the
> missing nodes for the unified reconstruction and draw loop, the
> bistability check and the axis stamps. Until it is re-extracted, read
> [engine.md](../engine.md) and [workflows.md](../workflows.md) for the
> 1.4.0 logic.

Two layers, one page (`index.html`, fully self-contained):

1. **The physics workflow** (`physics_workflow.svg` / `.pdf`) — the
   user-facing diagram: only physics quantities and equilibrium/profile
   variables, complete across the perturbation, matching, and solve steps.
   Embedded in the top-level README; the PDF is paper-figure quality.
2. **The full logic map** (`l1_full.svg` + interactive pan/zoom/search) —
   every config knob, decision gate, and stored artifact (550+ nodes), each
   with a `file:line` anchor.

- **View**: <https://d-burg.github.io/bouquet/flowchart/> (GitHub Pages), or
  open `index.html` locally.
- **Regenerate everything**:

  ```bash
  python docs/flowchart/build.py     # needs graphviz `dot` on PATH
  ```

## Editing the physics workflow (new diagnostics / data sources)

The physics diagram is hand-authored DATA in `build.py`: `PHYS_NODES` (one
dict per box: `id`, `cluster`, `kind`, `lines`) and `PHYS_EDGES` (one tuple
per arrow). To add e.g. MSE q-profile constraints: add one node dict to the
`inputs` cluster, one edge tuple, rerun `build.py` — there is a commented
MSE example in place. Labels are graphviz HTML-like markup; use the
`V("n","e")` helper for italic variables with subscripts.

## Editing the full logic map

Source of truth is `graph.json` (extracted from the code by a fleet of
agents, adversarially spot-verified, then hand-patched). Edit it (keep
`file:line` anchors honest) and rerun `build.py`. It is a descriptive
snapshot of the commit stamped in the page footer — regenerate (or
re-extract) after significant refactors.

The map is a descriptive snapshot of the commit stamped in the page footer —
it can drift from the code across refactors; regenerate (or re-extract) when
it does.
