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
import hashlib
import json
import os
import platform

import numpy as np
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SOURCE = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "bouquet", "examples", "D3D-like",
    "D3Dlike_Hmode_golden.h5"))
SLIM_NAME = "D3Dlike_Hmode_golden_slim.h5"
MANIFEST_NAME = "golden_manifest.json"
RNG_MANIFEST_NAME = "rng_stream_manifest.json"

# Seed for the pinned draw stream.  Changing it re-pins every value in
# rng_stream_manifest.json, so don't, unless that is the intent.
RNG_STREAM_SEED = 20260804

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


def _is_eqdsk_name(k):
    """Both archive generations: legacy stores '<header>_count=N.eqdsk';
    the class-API schema stores a dataset simply named 'eqdsk'."""
    return k == "eqdsk" or k.endswith(".eqdsk")


def _is_pfile_name(k):
    return k == "pfile" or k.endswith(".pfile")


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


def _digest(arr):
    """SHA-256 over the raw float64 bytes -- a bitwise fingerprint."""
    return hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def blas_provenance():
    """What the drawn values depend on besides bouquet itself.

    The draw factorises its kernel through LAPACK, so the build is part of the
    bitwise identity: reproducible on one machine, close-but-not-equal across
    machines (see ``draw_stream``).
    """
    blas = "unknown"
    try:                                        # numpy >= 1.25
        cfg = np.__config__.show(mode="dicts")
        dep = cfg["Build Dependencies"]["blas"]
        blas = f"{dep.get('name', '?')}-{dep.get('version', '?')}"
    except Exception:
        pass
    return {"platform": platform.system(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "blas": blas}


def draw_stream(psi_kin, psi_N, profiles, sigmas, seed=RNG_STREAM_SEED):
    """The seeded GPR draw stream, in the draw path's own order.

    Replays exactly what ``perturb_kinetic_equilibrium`` does for one draw --
    ne, Te, ni, Ti through ``_draw_monotonic_perturbation`` (each normalised by
    its on-axis value), then the :math:`j_\\phi` GPR candidate -- off ONE
    ``make_rng(seed)`` Generator.  No solver and no mesh, so this is cheap
    enough to pin as a golden.

    How reproducible is it?  The GP draw is a fixed-order Cholesky factor
    applied to the seeded normals, so the values are

      * **bitwise** stable for a given seed on a given machine -- the contract
        ``generate()`` relies on, and what the manifest's SHA-256 pins;
      * stable to ~1e-9 across LAPACK/libm builds: Cholesky has no discrete
        choices (no eigenvector signs, no null-space basis, no pivots), so a
        different build can only differ by rounding.  The eigh factorisation
        this replaced handed LAPACK both a sign convention and a null-space
        basis, and the same seed differed by 1.3% between the macOS and Linux
        CI builds.

    Bitwise equality across machines is still not a claim we can make --
    rounding inside LAPACK and libm is build-dependent -- so the manifest
    carries :func:`blas_provenance` and the test compares numerically
    off-machine, bitwise on-machine.

    Returns ``{channel: ndarray}`` on the grid each channel is drawn on.
    """
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
    from bouquet.sampling import make_rng, _draw_monotonic_perturbation
    from bouquet.sampling import generate_perturbed_GPR

    rng = make_rng(seed)
    out = {}
    for ch, ls in (("ne", 0.5), ("te", 0.4), ("ni", 0.5), ("ti", 0.4)):
        p = np.asarray(profiles[ch], dtype=float)
        s = np.asarray(sigmas[ch], dtype=float)
        out[ch] = _draw_monotonic_perturbation(
            psi_kin, p / p[0], s / p[0], ls, rng=rng) * p[0]
    jp = np.asarray(profiles["jphi"], dtype=float)
    sj = np.asarray(sigmas["jphi"], dtype=float)
    out["jphi"] = generate_perturbed_GPR(
        psi_N, jp / jp[0], sigma_profile=sj / jp[0], length_scale=0.25,
        n_samples=1, rng=rng, diag_plot=False) * jp[0]
    return out


def build_rng_stream(source_slim=None, out_dir=_HERE, seed=RNG_STREAM_SEED):
    """Pin the seeded draw stream against the golden baseline profiles.

    Reads the baseline profiles + sigma envelopes out of the slim fixture and
    writes ``rng_stream_manifest.json``: a SHA-256 of each drawn channel plus a
    few sampled values, so the git diff shows both that something moved and
    roughly where.  This is the golden that became possible only once the seed
    actually reached the GPR -- before that the stream was OS entropy and no
    draw-level value could be pinned at all.
    """
    slim = source_slim or os.path.join(out_dir, SLIM_NAME)
    with h5py.File(slim, "r") as hf:
        bkeys = sorted(hf["scan"].keys()) if "scan" in hf else [None]
        bl = hf[f"scan/{bkeys[0]}/_baseline"] if bkeys[0] is not None \
            else hf["_baseline"]
        psi_kin = np.asarray(bl["psi_N_kinetic"][()], dtype=float)
        psi_N = np.asarray(bl["psi_N"][()], dtype=float)
        profiles = {"ne": bl["n_e"][()], "te": bl["T_e"][()],
                    "ni": bl["n_i"][()], "ti": bl["T_i"][()],
                    "jphi": bl["j_phi"][()]}
        sigmas = {"ne": bl["sigma_ne"][()], "te": bl["sigma_te"][()],
                  "ni": bl["sigma_ni"][()], "ti": bl["sigma_ti"][()],
                  "jphi": bl["sigma_jphi"][()]}

    drawn = draw_stream(psi_kin, psi_N, profiles, sigmas, seed=seed)
    man = {
        "seed": int(seed),
        "source_fixture": os.path.basename(slim),
        "grids": {"psi_N_kinetic": int(psi_kin.size),
                  "psi_N": int(psi_N.size)},
        # which machine pinned it -- the SHA-256s are only meaningful there
        "pinned_on": blas_provenance(),
        "channels": {},
    }
    for ch, arr in drawn.items():
        a = np.asarray(arr, dtype=float)
        idx = [0, a.size // 4, a.size // 2, (3 * a.size) // 4, a.size - 1]
        man["channels"][ch] = {
            "n": int(a.size),
            "sha256": _digest(a),
            "sample_indices": idx,
            "sample_values": [float(a[i]) for i in idx],
            "min": float(a.min()), "max": float(a.max()),
        }
    path = os.path.join(out_dir, RNG_MANIFEST_NAME)
    with open(path, "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
    print(f"[golden] wrote {path}  (seed={seed}, "
          f"{len(man['channels'])} channels)")
    return path


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
                eqk = [k for k in bl.keys() if _is_eqdsk_name(k)]
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
                eqk = [k for k in grp.keys() if _is_eqdsk_name(k)]
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
                    # legacy layout: bracketed dataset + JSON names attr
                    names = json.loads(grp.attrs["coil_names"])
                    vals = np.asarray(grp["coil_currents [A]"][()], dtype=float)
                    rec["coil_currents"] = {n: float(v)
                                            for n, v in zip(names, vals)}
                elif "coil_currents" in grp and "coil_names" in grp:
                    # schema-v2 layout: clean dataset names
                    names = [n.decode() if isinstance(n, bytes) else str(n)
                             for n in grp["coil_names"][()]]
                    vals = np.asarray(grp["coil_currents"][()], dtype=float)
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
            if _is_pfile_name(name.rsplit("/", 1)[-1]):
                return                      # always dropped
            if _is_eqdsk_name(name.rsplit("/", 1)[-1]):
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
    ap.add_argument("--rng-stream-only", action="store_true",
                    help="re-pin rng_stream_manifest.json from the EXISTING "
                         "slim fixture and stop (no --source needed)")
    args = ap.parse_args()
    if args.rng_stream_only:
        build_rng_stream()
        raise SystemExit(0)
    if not os.path.isfile(args.source):
        raise SystemExit(f"source not found: {args.source}")
    build(args.source, eqdsk=args.eqdsk)
    build_rng_stream()
