#!/usr/bin/env python3
"""Render the bouquet logic-flow map from graph.json.

graph.json is the source of truth (extracted from the code with file:line
anchors on every node). This script renders it with graphviz `dot`:

  * l0_overview.svg — one node per pipeline stage (README-embeddable)
  * l1_full.svg     — the full quantity/decision graph, clustered by stage
  * index.html      — self-contained zoomable page (inline SVG + vendored
                      svg-pan-zoom + a node search box)

Regenerate after editing graph.json:  python docs/flowchart/build.py
Requires the `dot` binary (graphviz) on PATH.
"""
import html
import json
import os
import subprocess
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")

# stage ordering (roughly upstream -> downstream) and cluster colors
STAGE_ORDER = [
    "Configuration",
    "Input readers (g-file / p-file / IDA)",
    "IMAS/OMAS reader",
    "Baseline & uncertainty resolution",
    "GS reconstruction",
    "Bouquet orchestrator / IMAS forward solve",
    "Draw loop (GPR sampling)",
    "Per-draw solve (SWB / l_i / homotopy)",
    "Archive, filtering & plotting",
    "Process-parallel path",
    "Shared quantities",
]
STAGE_FILL = {
    "Configuration": "#eef4fb",
    "Input readers (g-file / p-file / IDA)": "#f0f7ee",
    "IMAS/OMAS reader": "#f0f7ee",
    "Baseline & uncertainty resolution": "#fdf6ec",
    "GS reconstruction": "#fdf0f0",
    "Bouquet orchestrator / IMAS forward solve": "#f3eefb",
    "Draw loop (GPR sampling)": "#fff8e7",
    "Per-draw solve (SWB / l_i / homotopy)": "#fdf0f0",
    "Archive, filtering & plotting": "#eefbf4",
    "Process-parallel path": "#f4f4f4",
}
TYPE_STYLE = {  # type -> (shape, fillcolor)
    "input":     ("parallelogram", "#e9edc9"),
    "knob":      ("note",          "#8ecae6"),
    "quantity":  ("ellipse",       "#ffd166"),
    "transform": ("box",           "#a7c957"),
    "decision":  ("diamond",       "#f28482"),
    "artifact":  ("cylinder",      "#cdb4db"),
}


def esc(s):
    return str(s or "").replace("\\", "\\\\").replace('"', '\\"')


def load():
    with open(GRAPH) as fh:
        return json.load(fh)


def node_dot(n):
    shape, fill = TYPE_STYLE[n["type"]]
    anchor = f'{n.get("file", "")}:{n.get("line", "")}' if n.get("file") else ""
    tip = " — ".join(x for x in (anchor, n.get("desc", "")) if x) or n["id"]
    return (f'"{esc(n["id"])}" [label="{esc(n.get("label") or n["id"])}", '
            f'shape={shape}, style="filled", fillcolor="{fill}", '
            f'tooltip="{esc(tip)}", href="#", fontsize=10];')


def render_l1(g):
    by_stage = defaultdict(list)
    for n in g["nodes"]:
        by_stage[n.get("subsystem", "Shared quantities")].append(n)
    lines = [
        "digraph bouquet {",
        '  graph [rankdir=LR, fontname="Helvetica", fontsize=18, ',
        '         ranksep=0.7, nodesep=0.25, concentrate=true, ',
        '         tooltip="bouquet logic flow"];',
        '  node  [fontname="Helvetica"];',
        '  edge  [fontname="Helvetica", fontsize=8, color="#666666", '
        '         arrowsize=0.6];',
    ]
    for i, stage in enumerate(STAGE_ORDER):
        nodes = by_stage.get(stage, [])
        if not nodes:
            continue
        if stage == "Shared quantities":       # unclustered: dot places them
            for n in nodes:                     # near their consumers
                lines.append("  " + node_dot(n))
            continue
        lines.append(f'  subgraph cluster_{i} {{')
        lines.append(f'    label="{esc(stage)}"; labeljust=l; fontsize=16;')
        lines.append(f'    style="filled,rounded"; fillcolor="{STAGE_FILL[stage]}";'
                     f' color="#999999";')
        for n in nodes:
            lines.append("    " + node_dot(n))
        lines.append("  }")
    for e in g["edges"]:
        style = ', style=dashed, color="#b56576"' if e.get("kind") == "control" else ""
        tip = esc(e.get("desc") or f'{e["src"]} -> {e["dst"]}')
        lines.append(f'  "{esc(e["src"])}" -> "{esc(e["dst"])}" '
                     f'[tooltip="{tip}", edgetooltip="{tip}"{style}];')
    lines.append("}")
    return "\n".join(lines)


def render_l0(g):
    """Stage overview: collapse hops through shared quantities into direct
    stage->stage edges (stage A writes qty q, stage B reads q  =>  A -> B)."""
    SHARED = "Shared quantities"
    sub_of = {n["id"]: n.get("subsystem", SHARED) for n in g["nodes"]}
    direct = Counter()
    derived = Counter()
    writers = defaultdict(set)   # shared qty id -> stages writing it
    readers = defaultdict(set)   # shared qty id -> stages reading it
    for e in g["edges"]:
        a, b = sub_of.get(e["src"]), sub_of.get(e["dst"])
        if not a or not b or a == b:
            continue
        if a != SHARED and b != SHARED:
            direct[(a, b)] += 1
        elif b == SHARED and a != SHARED:
            writers[e["dst"]].add(a)
        elif a == SHARED and b != SHARED:
            readers[e["src"]].add(b)
    for q in set(writers) & set(readers):
        for wa in writers[q]:
            for rb in readers[q]:
                if wa != rb:
                    derived[(wa, rb)] += 1
    # keep every direct stage->stage edge; keep derived (via shared
    # quantities) only when substantial, so the overview stays readable
    counts = Counter(direct)
    for k, c in derived.items():
        if c >= 4 and k not in counts:
            counts[k] = c
    n_in = Counter(n.get("subsystem") for n in g["nodes"])
    lines = [
        "digraph bouquet_overview {",
        '  graph [rankdir=LR, fontname="Helvetica", ranksep=0.55, '
        '         nodesep=0.35, tooltip="bouquet pipeline overview"];',
        '  node  [fontname="Helvetica", shape=box, style="filled,rounded", '
        '         fontsize=13, margin="0.22,0.12"];',
        '  edge  [fontname="Helvetica", fontsize=10, color="#666666"];',
    ]
    for stage in STAGE_ORDER:
        if stage == "Shared quantities" or not n_in.get(stage):
            continue
        fill = STAGE_FILL[stage]
        lines.append(f'  "{esc(stage)}" [fillcolor="{fill}", '
                     f'label="{esc(stage)}\\n({n_in[stage]} nodes)", href="#"];')
    # invisible spine pins the canonical pipeline order left-to-right; real
    # edges are drawn constraint=false so they overlay without re-ranking
    spine = [s for s in STAGE_ORDER if s != "Shared quantities" and n_in.get(s)]
    for a, b in zip(spine, spine[1:]):
        lines.append(f'  "{esc(a)}" -> "{esc(b)}" [style=invis, weight=100];')
    order = {s: i for i, s in enumerate(spine)}
    for (a, b), c in sorted(counts.items(), key=lambda kv: -kv[1]):
        back = order.get(a, 0) >= order.get(b, 0)   # against the pipeline
        extra = ', constraint=false, color="#b56576"' if back else ""
        lines.append(f'  "{esc(a)}" -> "{esc(b)}" [label="{c}"{extra}, '
                     f'penwidth={min(1 + c / 8, 5):.1f}];')
    lines.append("}")
    return "\n".join(lines)


# ===========================================================================
#  The physics workflow (the new-user view)
#
#  Hand-authored (NOT extracted from code): only physics quantities and
#  equilibrium/profile variables, but complete across the perturbation,
#  matching, and solve steps. Style follows the bouquet paper's workflow
#  figure (required/optional inputs, iterate loops, repeat x N_ens bracket).
#
#  TO EXTEND (new diagnostic, data source, constraint): add one dict to
#  PHYS_NODES and its arrows to PHYS_EDGES — the renderer does the rest.
#  Labels are graphviz HTML-like markup; use V("n","e") for italic
#  variables with subscripts and SUP() for superscripts.
# ===========================================================================

def V(base, sub=None):
    """Italic math variable with optional subscript: V('n','e') -> n_e."""
    s = f"<i>{base}</i>"
    return f"{s}<sub>{sub}</sub>" if sub else s


def SUP(base, sup):
    return f"{base}<sup>{sup}</sup>"


# node kinds -> graphviz style
_PHYS_KIND = {
    "required": 'style="filled,rounded", fillcolor="#dbeefc", color="#4da3dd", penwidth=1.6',
    "optional": 'style="filled,rounded,dashed", fillcolor="#f2f2f2", color="#888888"',
    "baseline": 'style="filled,rounded", fillcolor="#e8f4fd", color="#2779b0", penwidth=1.6',
    "derived":  'style="filled,rounded", fillcolor="#eef7ea", color="#4c8c3f", penwidth=1.6',
    "sample":   'style="filled,rounded", fillcolor="#cfe6f5", color="#1f6699", penwidth=1.6',
    "compute":  'style="filled,rounded", fillcolor="#fdf3d8", color="#c99b2f", penwidth=1.6',
    "solve":    'style="filled,rounded", fillcolor="#fbe8d9", color="#c06722", penwidth=1.6',
    "tag":      'style="filled,rounded", fillcolor="#efe9f7", color="#7a5ba8", penwidth=1.6',
    "output":   'style="filled,rounded", fillcolor="#fbe4f0", color="#b3487f", penwidth=1.6',
    "gate":     'shape=diamond, style="filled", fillcolor="#fbe3e0", color="#c9524a"',
}

# clusters: id -> (title, graphviz cluster style)
_PHYS_CLUSTERS = {
    "inputs":   ("Inputs        (solid blue = required for its path, dashed = optional)",
                 'style="rounded"; color="#bbbbbb";'),
    "baseline": ("Baseline (once per slice)",
                 'style="filled,rounded"; fillcolor="#f6fbf4"; color="#999999";'),
    "loop":     (f'repeat × {V("N", "ens")}   (each draw independent)',
                 'style="rounded,dashed"; color="#777777";'),
}

PHYS_NODES = [
    # ---- inputs ----------------------------------------------------------
    dict(id="gfile", cluster="inputs", kind="required", lines=[
        "geqdsk",
        f'{V("ψ")}(R,Z), {V("p")}, {V("q")}, {V("j","φ")}, separatrix']),
    dict(id="kin", cluster="inputs", kind="required", lines=[
        f'kinetic profiles + σ({V("ψ","N")}) envelopes',
        f'{V("n","e")}, {V("T","e")}, {V("T","i")} (, {V("n","i")}, {V("ω","tor")})',
        "p-file or kinetic-fit netCDF"]),
    dict(id="ids", cluster="inputs", kind="required", lines=[
        "IMAS/OMAS IDS",
        f'{V("n","e")}, {V("T","e")}, {V("T","i")}, {V("Z","eff")};  '
        f'{V("j","ohmic")}, {V("j","BS")}, {V("j","NBI")};',
        f'{V("p","fast")};  equilibrium anchors']),
    dict(id="lcfs", cluster="inputs", kind="optional", lines=[
        "magnetics-only LCFS",
        "(separatrix target override)"]),
    dict(id="fixed", cluster="inputs", kind="optional", lines=[
        "fixed arrays",
        f'{V("j","NBI")}, {V("j","RF")}, {V("p","fast")}({V("ψ","N")})']),
    dict(id="sigmas", cluster="inputs", kind="optional", lines=[
        "explicit σ profiles",
        "or diagnostic σ source"]),
    # future example (uncomment + add edges when MSE constraints land):
    # dict(id="mse", cluster="inputs", kind="optional", lines=[
    #     "MSE", f'{V("q")}-profile constraints']),

    # ---- baseline --------------------------------------------------------
    dict(id="recon", cluster="baseline", kind="baseline", lines=[
        "GS reconstruction (geqdsk path)",
        f'fit inductive spline, full-Sauter {V("j","BS")}',
        f'{V("j","φ")} = {V("j","ind")} + {V("j","BS")} ;  '
        f'match {V("ℓ","i")}(1) by secant',
        f'targets: {V("I","p")}, {V("ℓ","i")}']),
    dict(id="fsolve", cluster="baseline", kind="baseline", lines=[
        "forward solve (IMAS/OMAS path)",
        f'keep IDS {V("j","ohmic")};  Sauter {V("j","BS")} reconciliation',
        f'(diff: {V("j","BS,diff")} = {SUP(V("j","BS"), "IDS")} − '
        f'{SUP(V("j","BS"), "Sauter")}  |  rescale)',
        f'anchors: {V("p","diff")}, {V("j","φ,diff")} ;  '
        f'{V("ℓ","i")} target = solved {V("ℓ","i")}(1)']),
    dict(id="quasi", cluster="baseline", kind="derived", lines=[
        "quasineutral densities",
        f'{V("n","i")} = {V("n","e")} ({V("Z","imp")} − {V("Z","eff")}) / '
        f'({V("Z","imp")} − 1)']),
    dict(id="envel", cluster="baseline", kind="derived", lines=[
        "uncertainty envelope",
        f'{V("σ","n")}, {V("σ","T")}, {V("σ","Zeff")}, {V("σ","jφ")}  +  '
        f'GP length scales {V("ℓ","n")}, {V("ℓ","T")}, {V("ℓ","j")}']),

    # ---- per-draw loop ---------------------------------------------------
    dict(id="gp", cluster="loop", kind="sample", lines=[
        "GP profile sampling (Gibbs kernel)",
        f'draw {V("n","e")}, {V("T","e")}, {V("T","i")}, {V("Z","eff")}, '
        f'{V("j","ind")} ~ GP(baseline, σ, ℓ)',
        f'within envelopes; SOL included via {V("ψ","N,kin")} grid']),
    dict(id="derive", cluster="loop", kind="compute", lines=[
        "recompute derived",
        f'{V("n","i")} from drawn {V("Z","eff")} (quasineutrality)',
        f'{V("p","tot")} = Σ<sub>a</sub> {V("e")} {V("n","a")} {V("T","a")} + '
        f'{V("p","imp")} + {V("p","fast")} (+ {V("p","diff")})']),
    dict(id="pgate", cluster="loop", kind="gate", lines=[
        f'⟨{V("P")}⟩ within 5%', "of baseline?"]),
    dict(id="boot", cluster="loop", kind="compute", lines=[
        "bootstrap recompute (Sauter)",
        f'{V("j","BS")} = f({V("n")}, {V("T")}, {V("Z","eff")})   '
        "(3 self-consistent iterations)",
        f'parallel→toroidal: {V("c")} = 1 / (⟨{V("R")}⟩⟨1/{V("R")}⟩)',
        f'scale {V("s")} ~ U(0.99, 1.01) · {V("s","0")}']),
    dict(id="rebuild", cluster="loop", kind="compute", lines=[
        "rebuild total current",
        f'{V("j","φ")} = {V("j","ind")} + {V("s")}·{V("j","BS")} '
        f'(+ {V("j","BS,diff")}) + {V("j","NBI")} + {V("j","RF")} '
        f'(+ {V("j","φ,diff")})']),
    dict(id="limatch", cluster="loop", kind="solve", lines=[
        f'{V("ℓ","i")} matching',
        f'(optionally sample target ~ N({V("ℓ","i")}, {V("σ","ℓi")}))',
        f'scale {V("j","ind")} + corrective iteration',
        f'until |{V("ℓ","i")} − target| ≤ 5%']),
    dict(id="gssolve", cluster="loop", kind="solve", lines=[
        "TokaMaker free-boundary GS solve",
        f'isoflux separatrix + {V("I","p")} target',
        "coil homotopy: bounds ±5% → ±2% → ±1% (warm-started)",
        "VSC pair handles vertical control"]),
    dict(id="agate", cluster="loop", kind="gate", lines=[
        "accepted?",
        f'converged, {V("ℓ","i")} in band,',
        f'{V("q","0")} ≥ 1 (optional)']),
    dict(id="inspec", cluster="loop", kind="tag", lines=[
        "in-spec tag",
        "coil drift ≤ 2%;  VSC via quadrature-propagated σ"]),

    # ---- outputs ---------------------------------------------------------
    dict(id="archive", cluster=None, kind="output", lines=[
        "HDF5 ensemble archive (schema v2)",
        f'profiles, {V("j")} components, g-/p-file bytes,',
        "coil currents, targets, provenance (config)"]),
    dict(id="filter", cluster=None, kind="output", lines=[
        "post-filtering (non-destructive)",
        "boundary RMS ≤ 5 mm;  coil spec",
        "⇒ selected (machine-realizable) sub-ensemble"]),
]

# (src, dst, kind, label) — kinds: main | optional | yes | loop | reject
PHYS_EDGES = [
    ("gfile", "recon", "main", None), ("kin", "recon", "main", None),
    ("ids", "fsolve", "main", None), ("lcfs", "fsolve", "optional", None),
    ("kin", "envel", "main", None), ("sigmas", "envel", "optional", None),
    ("fixed", "rebuild", "optional", None),
    ("recon", "quasi", "main", None), ("fsolve", "quasi", "main", None),
    ("quasi", "envel", "main", None), ("envel", "gp", "main", None),
    ("gp", "derive", "main", None), ("derive", "pgate", "main", None),
    ("pgate", "boot", "yes", "yes"),
    ("pgate", "gp", "reject", "resample"),
    ("boot", "rebuild", "main", None), ("rebuild", "limatch", "main", None),
    ("limatch", "gssolve", "main", None),
    ("limatch", "limatch", "loop", " iterate"),
    ("gssolve", "gssolve", "loop", " homotopy\n passes"),
    ("gssolve", "agate", "main", None),
    ("agate", "inspec", "yes", "yes"),
    ("agate", "gp", "reject", "reject draw"),
    ("inspec", "archive", "main", None), ("archive", "filter", "main", None),
]

_PHYS_EDGE_STYLE = {
    "main":     "",
    "yes":      "",
    "optional": ', style=dashed, color="#888888"',
    "loop":     ', color="#c06722", fontcolor="#c06722"',
    "reject":   (', style=dashed, color="#c9524a", fontcolor="#c9524a", '
                 "constraint=false"),
}


def render_physics():
    """Render PHYS_NODES/PHYS_EDGES to DOT (HTML-like labels: italics + subs)."""
    lines = [
        "digraph bouquet_physics {",
        '  graph [rankdir=TB, fontname="Helvetica", fontsize=16, '
        '         ranksep=0.45, nodesep=0.35, '
        '         tooltip="bouquet physics workflow"];',
        '  node  [fontname="Helvetica", fontsize=12, shape=box, '
        '         margin="0.18,0.10"];',
        '  edge  [fontname="Helvetica", fontsize=10, color="#333333", '
        '         arrowsize=0.8];',
    ]
    by_cluster = defaultdict(list)
    for n in PHYS_NODES:
        by_cluster[n.get("cluster")].append(n)

    def node_line(n):
        label = "<" + "<br/>".join(n["lines"]) + ">"
        return f'    {n["id"]} [label={label}, {_PHYS_KIND[n["kind"]]}];'

    for i, (cid, (title, style)) in enumerate(_PHYS_CLUSTERS.items()):
        lines.append(f"  subgraph cluster_p{i} {{")
        lines.append(f"    label=<{title}>; labeljust=l; fontsize=15; {style}")
        for n in by_cluster.get(cid, []):
            lines.append(node_line(n))
        lines.append("  }")
    for n in by_cluster.get(None, []):
        lines.append(node_line(n)[2:])

    for src, dst, kind, lbl in PHYS_EDGES:
        attrs = _PHYS_EDGE_STYLE[kind]
        if lbl:
            attrs = f'label="{esc(lbl)}"' + (", " + attrs.lstrip(", ") if attrs else "")
        lines.append(f"  {src} -> {dst} [{attrs.strip(', ')}];")
    lines.append("}")
    return "\n".join(lines)


def _fix_baseline_shift(svg):
    """Make graphviz sub/superscripts portable across browsers.

    dot emits `baseline-shift="sub|super"` on absolutely-positioned <text>
    elements at FULL font size; browser support for baseline-shift on <text>
    is inconsistent, so subscripts collide with the next line. Bake the
    shift into the y coordinate and use a proper 70% glyph size instead
    (cairo-rendered PNG/PDF outputs are unaffected — they handle it fine).
    """
    import re

    def fix(m):
        tag = m.group(0)
        kind = m.group("kind")
        fs = float(re.search(r'font-size="([\d.]+)"', tag).group(1))
        y = float(re.search(r'y="(-?[\d.]+)"', tag).group(1))
        dy = 0.25 * fs if kind == "sub" else -0.35 * fs
        tag = re.sub(r'\s*baseline-shift="(sub|super)"', "", tag)
        tag = re.sub(r'font-size="[\d.]+"', f'font-size="{fs * 0.7:.2f}"', tag)
        tag = re.sub(r'y="(-?[\d.]+)"', f'y="{y + dy:.2f}"', tag)
        return tag

    return re.sub(r'<text[^>]*baseline-shift="(?P<kind>sub|super)"[^>]*>',
                  fix, svg)


def dot_to_svg(dot_src, out_name):
    out = os.path.join(HERE, out_name)
    res = subprocess.run(["dot", "-Tsvg", "-o", out], input=dot_src.encode(),
                         capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"dot failed for {out_name}:\n"
                           + res.stderr.decode()[:2000])
    with open(out) as fh:
        svg = fh.read()
    fixed = _fix_baseline_shift(svg)
    if fixed != svg:
        with open(out, "w") as fh:
            fh.write(fixed)
    return out


def build_html(g, phys_svg, l0_svg, l1_svg, stamp):
    with open(os.path.join(HERE, "svg-pan-zoom.min.js")) as fh:
        panzoom_js = fh.read()

    def inline(path, svg_id):
        import re
        with open(path) as fh:
            svg = fh.read()
        svg = svg[svg.index("<svg"):]                 # drop xml/doctype prologue
        # responsive SVG: keep the viewBox, drop the fixed pt width/height
        # (svg-pan-zoom's fit/zoom math degenerates with fixed pt dimensions
        # at extreme aspect ratios -> zeroed viewport matrix)
        head, rest = svg.split(">", 1)
        head = re.sub(r'\s(width|height)="[^"]*"', "", head)
        return f'{head} id="{svg_id}">{rest}'

    counts = Counter(n["type"] for n in g["nodes"])
    legend = " ".join(
        f'<span class="chip" style="background:{fill}">{t} ({counts.get(t, 0)})</span>'
        for t, (_s, fill) in TYPE_STYLE.items())
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bouquet — logic flow map</title>
<style>
 body {{ font-family: Helvetica, Arial, sans-serif; margin: 0; color: #222; }}
 header {{ padding: 14px 22px 8px; border-bottom: 1px solid #ddd; }}
 h1 {{ margin: 0 0 4px; font-size: 22px; }}
 .sub {{ color: #666; font-size: 13px; }}
 .chip {{ padding: 2px 8px; border-radius: 10px; font-size: 12px;
          margin-right: 4px; white-space: nowrap; }}
 .bar {{ display: flex; gap: 12px; align-items: center; padding: 8px 22px;
         flex-wrap: wrap; border-bottom: 1px solid #eee; }}
 #search {{ padding: 5px 9px; font-size: 13px; width: 300px;
            border: 1px solid #bbb; border-radius: 6px; }}
 #hits {{ font-size: 12px; color: #666; }}
 #overview {{ padding: 10px 22px; border-bottom: 1px solid #eee; }}
 #overview svg {{ max-width: 100%; height: auto; }}
 #map {{ height: calc(100vh - 60px); min-height: 480px; min-width: 640px; }}
 #map svg {{ width: 100%; height: 100%; }}
 .dim {{ opacity: 0.12; }}
 .hit ellipse, .hit polygon, .hit path, .hit rect {{
     stroke: #d00000 !important; stroke-width: 3px !important; }}
 details summary {{ cursor: pointer; font-size: 14px; padding: 6px 0; }}
 footer {{ padding: 8px 22px; color: #888; font-size: 12px; }}
</style>
</head>
<body>
<header>
 <h1>bouquet — workflow &amp; logic map</h1>
 <div class="sub">Start with the <b>physics workflow</b> (what happens to the
 profiles and equilibrium quantities); open the <b>full logic map</b> below to
 see every config knob, decision gate, and artifact, each anchored to
 <code>file:line</code> (hover). Scroll to zoom, drag to pan the full map.
 {stamp}</div>
</header>
<details id="physics" open>
 <summary><b>The physics workflow</b> — GP perturbation → derived quantities →
 bootstrap → ℓ_i matching → GS solve → archive</summary>
 <div style="padding: 8px 22px; max-width: 1100px; margin: 0 auto;">
 {inline(phys_svg, "svgp")}
 </div>
</details>
<details id="overview">
 <summary>Stage overview of the code ({len(g["nodes"])} nodes, {len(g["edges"])}
 edges in the full map below)</summary>
 {inline(l0_svg, "svg0")}
</details>
<div class="bar">
 <input id="search" type="search"
        placeholder="search nodes… (e.g. j_BS, zeff, homotopy, l_i)">
 <span id="hits"></span>
 {legend}
 <span class="chip" style="background:#eee">dashed edge = control flow</span>
</div>
<div id="map">
 {inline(l1_svg, "svg1")}
</div>
<footer>Generated by <code>docs/flowchart/build.py</code> from
<code>docs/flowchart/graph.json</code>. Edges/nodes are a descriptive snapshot
and may drift from the code; regenerate after refactors.</footer>
<script>{panzoom_js}</script>
<script>
 // Init AFTER full load + a paint: initializing while the (large) inline SVG
 // is still being laid out gives a degenerate screen CTM -> a zeroed
 // viewport matrix and a blank map.
 var pz = null;
 function initPanZoom() {{
   // never init on a zero-size container (backgrounded/embedded tabs can
   // report innerHeight=0): svg-pan-zoom would throw mid-init and corrupt
   // the SVG (consumed viewBox, non-invertible matrix). Wait instead.
   var el = document.getElementById('map');
   if (!el.clientWidth || !el.clientHeight) {{ setTimeout(initPanZoom, 300); return; }}
   pz = svgPanZoom('#svg1', {{zoomEnabled: true, controlIconsEnabled: true,
                             fit: true, center: true, minZoom: 0.05,
                             maxZoom: 60, zoomScaleSensitivity: 0.3}});
 }}
 // setTimeout, not requestAnimationFrame: rAF never fires in backgrounded /
 // embedded tabs, which would stall init forever.
 if (document.readyState === 'complete') {{ setTimeout(initPanZoom, 0); }}
 else {{ window.addEventListener('load', function () {{ setTimeout(initPanZoom, 0); }}); }}
 // keep the viewport sane when the overview panel opens/closes (the map
 // container height changes); the timeout waits out the reflow.
 document.getElementById('overview').addEventListener('toggle', function () {{
   setTimeout(function () {{
     if (!pz) return;
     try {{ pz.resize(); pz.fit(); pz.center(); }} catch (e) {{}}
   }}, 50);
 }});
 var nodes = document.querySelectorAll('#svg1 g.node');
 var hits = document.getElementById('hits');
 document.getElementById('search').addEventListener('input', function (ev) {{
   var q = ev.target.value.trim().toLowerCase();
   var n = 0;
   nodes.forEach(function (g) {{
     var t = (g.textContent || '').toLowerCase();
     if (!q) {{ g.classList.remove('dim', 'hit'); return; }}
     if (t.indexOf(q) >= 0) {{ g.classList.add('hit'); g.classList.remove('dim'); n++; }}
     else {{ g.classList.add('dim'); g.classList.remove('hit'); }}
   }});
   hits.textContent = q ? (n + ' match' + (n === 1 ? '' : 'es')) : '';
 }});
</script>
</body>
</html>
"""
    with open(os.path.join(HERE, "index.html"), "w") as fh:
        fh.write(page)


def main():
    g = load()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                cwd=HERE).stdout.strip()
    except Exception:
        commit = "unknown"
    stamp = f"Snapshot of commit <code>{commit or 'unknown'}</code>."
    phys = dot_to_svg(render_physics(), "physics_workflow.svg")
    # PDF export of the physics diagram (paper-figure candidate)
    subprocess.run(["dot", "-Tpdf", "-o",
                    os.path.join(HERE, "physics_workflow.pdf")],
                   input=render_physics().encode(), check=True)
    l0 = dot_to_svg(render_l0(g), "l0_overview.svg")
    l1 = dot_to_svg(render_l1(g), "l1_full.svg")
    build_html(g, phys, l0, l1, stamp)
    print(f"rendered physics workflow + {len(g['nodes'])} nodes /"
          f" {len(g['edges'])} edges -> physics_workflow.svg/.pdf,"
          f" l0_overview.svg, l1_full.svg, index.html")


if __name__ == "__main__":
    main()
