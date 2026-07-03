# bouquet workflow & logic map

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
