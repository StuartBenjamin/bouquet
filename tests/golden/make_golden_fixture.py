"""Build the git-tracked golden test fixture + manifest from a full bouquet run.

The backend regression tests (``tests/test_golden_bouquet.py``) run against a
**slimmed** copy of a real bouquet ``.h5`` plus a JSON manifest of expected
values.  This script produces both, so updating the golden set on purpose is a
single, reviewable command.

What it does
------------
1. Reads a full bouquet ``.h5`` (the 30 MB shareable example artifact).
2. Writes ``D3Dlike_Hmode_golden_slim.h5`` next to this script:
   - **keeps the ``*.eqdsk`` geqdsks, gzip-compressed** (~3x; ~0.37 MB each)
     so the geqdsk read/parse/handling path -- a deliberately coarse-at-the-
     separatrix format -- is exercised by real files.  The geqdsks are stored
     as gzipped ``uint8`` under their original ``.eqdsk`` dataset names, so
     every reader (``bytes(grp[k][()])``) is unaffected.
   - drops the ``*.pfile`` byte blobs (not needed for equilibrium-handling
     tests; full p-files stay in the example artifact),
   - extracts the real ``Ip`` from each eqdsk into an ``Ip`` attr,
   - keeps every attr, ``coil_currents``, ``x_points``, both LCFS references,
     and the profile arrays the assertions need.
3. Writes ``golden_manifest.json`` of expected per-draw + baseline values with
   tolerances.

The ``--eqdsk`` flag controls geqdsk retention:
``all`` (default, ~11 MB), ``subset`` (baseline + a few representative draws,
~5 MB), or ``none`` (smallest, ~4 MB, no geqdsk-handling coverage).

Updating the golden set
-----------------------
Re-run the bouquet notebook to produce a fresh full ``.h5``, then::

    python tests/golden/make_golden_fixture.py \
        --source /path/to/D3Dlike_Hmode_golden.h5

Review the git diff of ``golden_manifest.json`` (and the slim ``.h5``) before
committing -- the manifest diff shows exactly which physics values moved.

Note: eventually the default equilibrium interchange should migrate to
IMAS/OMAS; when it does, this fixture can store those instead of geqdsks.
"""
import argparse
import json
import os

import numpy as np
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SOURCE = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "bouquet", "examples", "D3D-like",
    "D3Dlike_Hmode_golden.h5"))
SLIM_NAME = "D3Dlike_Hmode_golden_slim.h5"
MANIFEST_NAME = "golden_manifest.json"

# p-file byte blobs are always dropped (not needed for equilibrium-handling
# tests).  geqdsk retention is controlled by the --eqdsk flag.
_EQDSK_GZIP_LEVEL = 9

# Tolerances the regression test uses when comparing recomputed values to
# the manifest.  Reads/derivations are deterministic, so these are tight;
# small absolute floors guard pure float round-off.
TOLERANCES = {
    "l_i_atol": 1e-4,        # l_i(1)/l_i(3)
    "Ip_rtol": 1e-6,         # plasma current (relative)
    "drift_atol": 1e-4,      # coil drift percentages
    "bnd_atol_mm": 1e-3,     # boundary RMS/max deviation [mm]
    "coil_atol_A": 1e-3,     # per-coil current [A]
    "xpoint_atol_m": 1e-6,   # X-point R,Z [m]
}


def _bp_to_path(bkey):
    return f"scan/{bkey}" if bkey is not None else None


def _scan_keys(hf):
    """Return list of (bkey, group_path_prefix) for each scan value."""
    if "scan" in hf:
        return [(k, f"scan/{k}") for k in sorted(hf["scan"].keys())]
    return [(None, "")]


def _ip_from_eqdsk_bytes(raw):
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
    from bouquet.utils import read_eqdsk_from_bytes
    from bouquet.io import read_geqdsk
    eq = read_eqdsk_from_bytes(raw, read_geqdsk)
    return float(eq.Ip)


def _boundary_devs(bl_boundary, grp):
    from scipy.spatial import cKDTree
    if bl_boundary is None or "perturbed_lcfs_ref" not in grp:
        return (np.nan, np.nan)
    pert = np.asarray(grp["perturbed_lcfs_ref"][()], dtype=float)
    tree = cKDTree(pert)
    devs, _ = tree.query(bl_boundary)
    return (float(np.sqrt(np.mean(devs ** 2)) * 1e3),
            float(np.max(devs) * 1e3))


def _select_subset(draws):
    """Pick a small, diverse set of draw indices for ``--eqdsk subset``.

    Covers the parser-relevant variation: first in-spec, first out-of-spec,
    and the min/max boundary-deviation draws.
    """
    idxs = sorted(int(k) for k in draws)
    chosen = set()
    inspec = [i for i in idxs if draws[str(i)].get("in_spec")]
    outspec = [i for i in idxs if not draws[str(i)].get("in_spec")]
    if inspec:
        chosen.add(inspec[0])
    if outspec:
        chosen.add(outspec[0])
    rms = {i: draws[str(i)].get("bnd_rms_mm", float("nan")) for i in idxs}
    fin = [i for i in idxs if rms[i] == rms[i]]  # drop NaN
    if fin:
        chosen.add(min(fin, key=lambda i: rms[i]))
        chosen.add(max(fin, key=lambda i: rms[i]))
    return chosen


def build(source, out_dir=_HERE, eqdsk="all"):
    if eqdsk not in ("all", "subset", "none"):
        raise ValueError("eqdsk must be 'all', 'subset', or 'none'")
    slim_path = os.path.join(out_dir, SLIM_NAME)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)

    manifest = {
        "source_basename": os.path.basename(source),
        "eqdsk_retention": eqdsk,
        "tolerances": TOLERANCES,
        "scans": {},
    }
    ip_map = {}            # full group path -> Ip (injected as attr in pass 2)
    keep_eqdsk = set()     # full .eqdsk dataset names to KEEP (gzipped)

    # ---- pass 1: read source -> manifest, Ip map, which geqdsks to keep --
    with h5py.File(source, "r") as hf:
        for bkey, prefix in _scan_keys(hf):
            parent = hf[prefix] if prefix else hf
            base = (prefix + "/") if prefix else ""
            scan_entry = {"baseline": {}, "draws": {}}

            bl = parent["_baseline"] if "_baseline" in parent else None
            bl_boundary = None
            if bl is not None and "recon_lcfs_ref" in bl:
                bl_boundary = np.asarray(bl["recon_lcfs_ref"][()], dtype=float)

            if bl is not None:
                eqk = [k for k in bl.keys() if k.endswith(".eqdsk")]
                if eqk:
                    ip = _ip_from_eqdsk_bytes(bytes(bl[eqk[0]][()]))
                    ip_map[f"{base}_baseline"] = ip
                    scan_entry["baseline"]["Ip"] = ip
                    if eqdsk in ("all", "subset"):  # baseline always kept
                        keep_eqdsk.add(f"{base}_baseline/{eqk[0]}")
                        scan_entry["baseline"]["has_eqdsk"] = True
                for k in ("Ip_target", "l_i_target"):
                    if k in bl.attrs:
                        scan_entry["baseline"][k] = float(bl.attrs[k])
                if "x_points" in bl:
                    scan_entry["baseline"]["x_points"] = \
                        np.asarray(bl["x_points"][()], dtype=float).tolist()
                if "diverted" in bl.attrs:
                    scan_entry["baseline"]["diverted"] = bool(bl.attrs["diverted"])

            draw_keys = sorted(
                int(k) for k in parent.keys()
                if k not in ("_baseline", "scan") and str(k).lstrip("-").isdigit())
            eqk_name = {}  # draw idx -> eqdsk dataset name
            for i in draw_keys:
                grp = parent[str(i)]
                rec = {}
                eqk = [k for k in grp.keys() if k.endswith(".eqdsk")]
                if eqk:
                    eqk_name[i] = eqk[0]
                    ip = _ip_from_eqdsk_bytes(bytes(grp[eqk[0]][()]))
                    ip_map[f"{base}{i}"] = ip
                    rec["Ip"] = ip
                for k in ("l_i(1)", "l_i(3)", "max_F_drift_pct",
                          "max_VSC_drift_pct", "inspec_F_max", "inspec_VSC_max",
                          "homotopy_pass", "l_i_target_used"):
                    if k in grp.attrs:
                        rec[k] = float(grp.attrs[k])
                if "in_spec" in grp.attrs:
                    rec["in_spec"] = bool(grp.attrs["in_spec"])
                rms, mx = _boundary_devs(bl_boundary, grp)
                rec["bnd_rms_mm"] = rms
                rec["bnd_max_mm"] = mx
                if "coil_currents [A]" in grp and "coil_names" in grp.attrs:
                    names = json.loads(grp.attrs["coil_names"])
                    vals = np.asarray(grp["coil_currents [A]"][()], dtype=float)
                    rec["coil_currents"] = {n: float(v)
                                            for n, v in zip(names, vals)}
                if "x_points" in grp:
                    rec["x_points"] = \
                        np.asarray(grp["x_points"][()], dtype=float).tolist()
                if "diverted" in grp.attrs:
                    rec["diverted"] = bool(grp.attrs["diverted"])
                scan_entry["draws"][str(i)] = rec

            # decide which draw geqdsks to keep
            if eqdsk == "all":
                keep_draws = set(draw_keys)
            elif eqdsk == "subset":
                keep_draws = _select_subset(scan_entry["draws"])
            else:
                keep_draws = set()
            for i in keep_draws:
                if i in eqk_name:
                    keep_eqdsk.add(f"{base}{i}/{eqk_name[i]}")
                    scan_entry["draws"][str(i)]["has_eqdsk"] = True

            scan_entry["n_draws"] = len(draw_keys)
            scan_entry["n_in_spec"] = sum(
                1 for d in scan_entry["draws"].values() if d.get("in_spec"))
            scan_entry["draw_indices"] = draw_keys
            scan_entry["eqdsk_indices"] = sorted(keep_draws)
            manifest["scans"][str(bkey)] = scan_entry

    # ---- pass 2: clean rebuild (keep gzipped geqdsks, drop pfiles) -------
    if os.path.exists(slim_path):
        os.remove(slim_path)
    with h5py.File(source, "r") as src, h5py.File(slim_path, "w") as dst:
        for k in src.attrs:
            dst.attrs[k] = src.attrs[k]

        # filter flags are run-state, not generation output -- strip them so
        # the fixture is always the canonical "unfiltered" run regardless of
        # whether the source h5 had filters applied (e.g. by the notebook).
        _filter_attrs = ("passes_coil_filter", "passes_boundary_filter",
                         "selected")

        def _copy(name, obj):
            if isinstance(obj, h5py.Group):
                g = dst.require_group(name)
                for ak in obj.attrs:
                    if ak in _filter_attrs:
                        continue
                    g.attrs[ak] = obj.attrs[ak]
                return
            # dataset
            if name.endswith(".pfile"):
                return                      # always dropped
            if name.endswith(".eqdsk"):
                if name not in keep_eqdsk:
                    return                  # dropped per --eqdsk
                raw = bytes(obj[()])
                arr = np.frombuffer(raw, dtype=np.uint8)
                d = dst.create_dataset(name, data=arr, compression="gzip",
                                       compression_opts=_EQDSK_GZIP_LEVEL)
                for ak in obj.attrs:
                    d.attrs[ak] = obj.attrs[ak]
                return
            d = dst.create_dataset(name, data=obj[()])
            for ak in obj.attrs:
                d.attrs[ak] = obj.attrs[ak]
        src.visititems(_copy)

        # inject the extracted Ip as a group attr (so it is asserted even
        # for groups whose geqdsk blob was dropped)
        for gpath, ip in ip_map.items():
            if gpath in dst:
                dst[gpath].attrs["Ip"] = ip

    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    size_mb = os.path.getsize(slim_path) / 1e6
    print(f"[golden] wrote {slim_path}  ({size_mb:.2f} MB, eqdsk={eqdsk})")
    print(f"[golden] wrote {manifest_path}")
    for sk, se in manifest["scans"].items():
        print(f"[golden]   scan {sk}: {se['n_draws']} draws, "
              f"{se['n_in_spec']} in-spec, "
              f"{len(se['eqdsk_indices'])} geqdsks kept, "
              f"indices {se['draw_indices']}")
    return slim_path, manifest_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=_DEFAULT_SOURCE,
                    help="full bouquet .h5 to slim (default: the D3D-like "
                         "example artifact)")
    ap.add_argument("--eqdsk", default="all",
                    choices=("all", "subset", "none"),
                    help="geqdsk retention: all (default, ~11 MB), subset "
                         "(baseline + representative draws, ~5 MB), or none")
    args = ap.parse_args()
    if not os.path.isfile(args.source):
        raise SystemExit(f"source not found: {args.source}")
    build(args.source, eqdsk=args.eqdsk)
