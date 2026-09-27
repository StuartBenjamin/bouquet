"""BouquetArchive reader (Phase 2: F13, F12-read).

Uses the committed golden fixture (real scan/-layout archive) for the round-trip,
plus a synthetic h5 for the suffix-scan (legacy vs schema-v2 eqdsk names)."""

import os
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import bouquet as bq

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")
_has_golden = os.path.isfile(_GOLDEN)


@pytest.mark.skipif(not _has_golden, reason="golden fixture absent")
class TestArchiveGolden:
    def test_scan_and_indices(self):
        ar = bq.BouquetArchive(_GOLDEN)
        assert ar.scan_keys == ["0"]
        sc = ar["0"]
        assert len(sc.indices) >= 1
        assert sc.indices == sorted(sc.indices)          # sorted, gap-tolerant
        # single scan -> scan(None) resolves it
        assert ar.scan().scan_key == "0"

    def test_selected_excluded_partition(self):
        sc = bq.BouquetArchive(_GOLDEN)["0"]
        alln = {d.count for d in sc.all}
        seln = {d.count for d in sc.selected}
        excn = {d.count for d in sc.excluded}
        assert seln | excn == alln and not (seln & excn)   # a partition

    def test_draw_view_scalars_profiles_bytes(self):
        sc = bq.BouquetArchive(_GOLDEN)["0"]
        d = sc[sc.indices[0]]
        assert isinstance(d.li1, float) and isinstance(d.li3, float)
        assert "psi_N" in d.profiles and d.profiles["psi_N"].ndim == 1
        assert isinstance(d.attrs, dict) and "l_i(1)" in d.attrs
        assert d.eqdsk_bytes is not None                  # suffix scan finds it

    def test_equilibrium_parse(self):
        sc = bq.BouquetArchive(_GOLDEN)["0"]
        eq = sc[sc.indices[0]].equilibrium()
        assert abs(eq.Ip) > 0 and len(eq.psi_N) > 0

    def test_crosscheck_direct_h5(self):
        # Cross-check against a direct h5py read of the same group. (We avoid
        # load_equilibrium here precisely because it reconstructs the exact
        # {header}_{scan}_{count}.eqdsk name and fails on a renamed file -- F11 --
        # which the archive's suffix scan handles.)
        ar = bq.BouquetArchive(_GOLDEN)
        sc = ar["0"]
        c = sc.indices[0]
        d = sc[c]
        with h5py.File(_GOLDEN, "r") as hf:
            grp = hf[f"scan/0/{c}"]
            assert np.allclose(d.profiles["psi_N"], np.array(grp["psi_N"]))
            assert abs(d.li1 - float(grp.attrs["l_i(1)"])) < 1e-9
            eq_ds = "eqdsk"
            assert d.eqdsk_bytes == bytes(grp[eq_ds][()])

    def test_iteration(self):
        sc = bq.BouquetArchive(_GOLDEN)["0"]
        assert len(list(sc)) == len(sc.indices)

    def test_spread_reports_global_scalars(self):
        # <P> + beta_N reported alongside the l_i variance, in one call.
        sc = bq.BouquetArchive(_GOLDEN).scan()
        out = sc.spread("all", print_table=False)
        for q in ("l_i(1)", "l_i(3)", "<P> [kPa]", "beta_N"):
            st = out[q]
            assert st is not None and st["n"] >= 1 and st["std"] >= 0.0
            assert st["min"] <= st["mean"] <= st["max"]
            assert 0.0 <= st["rel_std"] < 1.0            # a sane fractional spread
        # selected is a subset of all
        sel = sc.spread("selected", print_table=False)
        assert sel["<P> [kPa]"]["n"] <= out["<P> [kPa]"]["n"]


class TestSuffixScan:
    def _make(self, path, eqdsk_name):
        with h5py.File(path, "w") as hf:
            g = hf.require_group("scan/4400/0")
            g["psi_N"] = np.linspace(0, 1, 5)
            g["j_phi"] = np.ones(5)
            g.attrs["l_i(1)"] = 1.05
            g.attrs["l_i(3)"] = 0.85
            g.attrs["selected"] = True
            g.create_dataset(eqdsk_name, data=np.void(b"GEQDSK-BYTES"))

    def test_legacy_name(self, tmp_path):
        p = str(tmp_path / "leg.h5")
        self._make(p, "leg_4400_0.eqdsk")          # legacy {header}_{scan}_{count}
        d = bq.BouquetArchive(p)["4400"][0]
        assert d.eqdsk_bytes == b"GEQDSK-BYTES"
        assert "psi_N" in d.profiles

    def test_v2_fixed_name(self, tmp_path):
        p = str(tmp_path / "v2.h5")
        self._make(p, "eqdsk")                     # schema-v2 fixed name
        d = bq.BouquetArchive(p)["4400"][0]
        assert d.eqdsk_bytes == b"GEQDSK-BYTES"


class TestBinarySourceDedup:
    """Binary IDA .cdf sources must NOT be duplicated into every draw group
    (the new IDA-database files are ~190 MB -> a 20-draw archive hit ~4 GB).
    Text p-files stay per-draw (they carry the draw's perturbed kinetics)."""

    def _store(self, path, pfile_bytes):
        from bouquet.utils import store_equilibrium, store_baseline_profiles
        stem = os.path.splitext(path)[0]
        psi = np.linspace(0, 1, 9)
        one = np.ones(9)
        eq_path = stem + "_in.eqdsk"
        with open(eq_path, "wb") as fh:
            fh.write(b"GEQDSK-BYTES")
        store_baseline_profiles(
            stem, psi, one, one, one, one, one, one,          # ne te ni ti p jphi
            one, one, one, one, one,                          # 5 sigmas
            1e6, 1.0, scan_key="900",
            eqdsk_bytes=b"GEQDSK-BYTES", pfile_bytes=pfile_bytes)
        for c in range(2):
            store_equilibrium(
                stem, c, eq_path, psi,
                one, one, one,                                 # j_phi j_BS j_ind
                one, one, one, one, one,                       # ne Te ni Ti wExB
                1.0, 0.8, scan_key="900", pfile_bytes=pfile_bytes)

    def test_binary_cdf_stored_once_with_reader_fallback(self, tmp_path):
        fake_cdf = b"\x89HDF\r\n\x1a\n" + b"x" * 4096   # netCDF4/HDF5 magic
        p = str(tmp_path / "dedup.h5")
        self._store(p, fake_cdf)
        with h5py.File(p, "r") as hf:
            assert "pfile" in hf["scan/900/_baseline"]          # stored once
            for c in ("0", "1"):
                assert "pfile" not in hf[f"scan/900/{c}"]        # NOT per draw
        # readers still see the bytes via the baseline fallback
        d = bq.BouquetArchive(p)["900"][0]
        assert d.pfile_bytes == fake_cdf
        rec = bq.load_equilibrium(os.path.splitext(p)[0], count=0, scan_key="900")
        assert rec["pfile_bytes"] == fake_cdf

    def test_text_pfile_still_stored_per_draw(self, tmp_path):
        text_pf = b"5 psinorm ne(10^20/m^3) dne/dpsiN\n 0.0 1.0 0.0\n"
        p = str(tmp_path / "textpf.h5")
        self._store(p, text_pf)
        with h5py.File(p, "r") as hf:
            for c in ("0", "1"):
                assert "pfile" in hf[f"scan/900/{c}"]            # per-draw kept


class TestDethread:
    """Phase 3: readers accept a run/archive; bad scan_key raises (F1)."""

    @pytest.mark.skipif(not _has_golden, reason="golden fixture absent")
    def test_resolver_accepts_objects(self):
        from bouquet.utils import _resolve_h5, _default_scan_key
        stem = os.path.splitext(_GOLDEN)[0]
        ar = bq.BouquetArchive(_GOLDEN)
        for ref in (ar, stem, _GOLDEN):
            assert _resolve_h5(ref).endswith("golden_slim.h5")

        class _Gen:  # minimal Bouquet-like duck type
            scan_key = "0"

        class _Cfg:
            output_header = stem
            generation = _Gen()

        class _Run:
            config = _Cfg()

        assert _resolve_h5(_Run()).endswith("golden_slim.h5")
        assert _default_scan_key(_Run(), None) == "0"

    @pytest.mark.skipif(not _has_golden, reason="golden fixture absent")
    def test_select_indices_raises_on_bad_scan_key(self):
        stem = os.path.splitext(_GOLDEN)[0]
        with pytest.raises(KeyError, match="available"):
            bq.select_indices(stem, scan_key="9999")
        assert bq.select_indices(stem, scan_key="0", selection="all")   # good key works

    @pytest.mark.skipif(not _has_golden, reason="golden fixture absent")
    def test_reader_accepts_archive_object(self):
        ar = bq.BouquetArchive(_GOLDEN)
        assert bq.select_indices(ar, scan_key="0", selection="all")


class TestCoilSchemaV2:
    """Regression for the F15 fix: per-draw coil_names is a string DATASET, and
    plot_coil_currents reads it as such (not a JSON attr -> all-NaN heatmap)."""

    @pytest.mark.skipif(not _has_golden, reason="golden fixture absent")
    def test_per_draw_coil_names_is_dataset(self):
        with h5py.File(_GOLDEN, "r") as hf:
            draws = [k for k in hf["scan/0"] if k.isdigit()]
            checked = 0
            for c in draws:
                g = hf[f"scan/0/{c}"]
                if "coil_currents" in g:
                    assert "coil_names" in g and isinstance(g["coil_names"], h5py.Dataset)
                    assert len(g["coil_names"]) > 0
                    assert "coil_names" not in g.attrs      # not the old JSON attr
                    checked += 1
            assert checked > 0

    @pytest.mark.skipif(not _has_golden, reason="golden fixture absent")
    def test_plot_coil_currents_finite_drift(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        res = bq.plot_coil_currents(os.path.splitext(_GOLDEN)[0], scan_key="0")
        fig = res[0] if isinstance(res, tuple) else res
        # the drift heatmap must carry at least some finite values (the bug made
        # every per-draw cell NaN because names were read from an absent attr)
        vals = []
        for ax in fig.axes:
            for im in ax.get_images():
                vals.append(np.asarray(im.get_array(), dtype=float))
            for qm in ax.collections:
                a = getattr(qm, "get_array", lambda: None)()
                if a is not None:
                    vals.append(np.asarray(a, dtype=float).ravel())
        plt.close("all")
        assert vals, "no heatmap array found"
        assert any(np.isfinite(v).any() for v in vals), "coil-drift heatmap all-NaN"


class TestJbsLoopBlockSchemaV3:
    """Schema v3: the self-consistent bootstrap record (``jbs_loop`` block)
    on draw groups and on ``_baseline``; absent = a frozen bootstrap."""

    _REC = {"converged": True, "n_passes": 4, "n_passes_total": 5,
            "r_j": [3e-2, 2e-3, 4e-4, 1e-4], "omega": [0.7, 0.7, 0.7, 0.7],
            "tolerances": {"rtol_j": 1e-3}, "arr": np.arange(3.0)}

    def _store(self, path, rec_draw0, rec_draw1, rec_baseline):
        from bouquet.utils import (store_equilibrium, store_baseline_profiles,
                                   store_baseline_jbs_loop,
                                   initialize_equilibrium_database)
        stem = os.path.splitext(path)[0]
        initialize_equilibrium_database(stem)
        psi = np.linspace(0, 1, 9)
        one = np.ones(9)
        eq_path = stem + "_in.eqdsk"
        with open(eq_path, "wb") as fh:
            fh.write(b"GEQDSK-BYTES")
        store_baseline_profiles(
            stem, psi, one, one, one, one, one, one,
            one, one, one, one, one, 1e6, 1.0, scan_key="7",
            eqdsk_bytes=b"GEQDSK-BYTES")
        for c, rec in ((0, rec_draw0), (1, rec_draw1)):
            store_equilibrium(
                stem, c, eq_path, psi, one, one, one,
                one, one, one, one, one, 1.0, 0.8, scan_key="7",
                jbs_loop=rec)
        store_baseline_jbs_loop(stem, rec_baseline, scan_key="7")
        return stem

    def test_version_is_3_and_the_block_is_named_in_the_schema(self):
        from bouquet import schema
        assert schema.SCHEMA_VERSION == 3
        assert schema.JBS_LOOP_ATTRS == ("jbs_converged", "jbs_n_passes",
                                         "jbs_loop_json")
        assert schema.JBS_LOOP_SINCE_SCHEMA == 3

    def test_block_round_trips_on_draws_and_baseline(self, tmp_path):
        from bouquet.utils import load_jbs_loop
        stem = self._store(str(tmp_path / "v3.h5"), self._REC, None,
                           dict(self._REC, n_passes_total=None))
        with h5py.File(stem + ".h5", "r") as hf:
            assert hf.attrs["schema_version"] == 3
            a0 = hf["scan/7/0"].attrs
            assert bool(a0["jbs_converged"]) is True
            assert int(a0["jbs_n_passes"]) == 5      # all loops of the draw
            ab = hf["scan/7/_baseline"].attrs
            assert int(ab["jbs_n_passes"]) == 4      # a baseline has n_passes
            # a frozen draw carries none of the block
            assert not any(k in hf["scan/7/1"].attrs
                           for k in ("jbs_converged", "jbs_n_passes",
                                     "jbs_loop_json"))
        r0 = load_jbs_loop(stem, 0, scan_key="7")
        assert r0["r_j"] == self._REC["r_j"] and r0["arr"] == [0.0, 1.0, 2.0]
        assert load_jbs_loop(stem, 1, scan_key="7") is None
        assert load_jbs_loop(stem, "_baseline", scan_key="7")["n_passes"] == 4

    def test_archive_views_read_the_block(self, tmp_path):
        stem = self._store(str(tmp_path / "views.h5"), self._REC, None,
                           self._REC)
        sc = bq.BouquetArchive(stem + ".h5")["7"]
        assert sc[0].jbs_converged is True and sc[0].jbs_loop["n_passes"] == 4
        assert sc[1].jbs_converged is None and sc[1].jbs_loop is None
        assert sc.baseline_jbs_loop["converged"] is True
        assert sc.bootstrap_model == "self-consistent Redl bootstrap"

    def test_a_frozen_archive_reads_as_legacy(self, tmp_path):
        """v2 archives and jbs_self_consistent=False runs carry no block:
        nothing to migrate, and the readers say 'frozen'."""
        stem = self._store(str(tmp_path / "frozen.h5"), None, None, None)
        with h5py.File(stem + ".h5", "r") as hf:
            assert "jbs_loop_json" not in hf["scan/7/_baseline"].attrs
        sc = bq.BouquetArchive(stem + ".h5")["7"]
        assert sc.baseline_jbs_loop is None
        assert all(d.jbs_loop is None for d in sc.all)
        assert sc.bootstrap_model == "frozen SWB bootstrap (legacy)"

    def test_the_block_json_is_plain_json(self, tmp_path):
        import json
        stem = self._store(str(tmp_path / "j.h5"), self._REC, None, None)
        with h5py.File(stem + ".h5", "r") as hf:
            raw = hf["scan/7/0"].attrs["jbs_loop_json"]
        json.loads(raw)                               # must not raise


class TestBootstrapModelLabel:
    """plotting.plot_jphi names the bootstrap model the archive records."""

    def _archive(self, tmp_path, rec):
        stem = TestJbsLoopBlockSchemaV3()._store(
            str(tmp_path / "lbl.h5"), rec, rec, rec)
        return stem + ".h5"

    @pytest.mark.parametrize("rec, want", [
        ({"converged": True, "n_passes": 3}, "self-consistent Redl bootstrap"),
        (None, "frozen SWB bootstrap (legacy)"),
    ])
    def test_plot_jphi_labels_the_bootstrap_model(self, tmp_path, rec, want):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from bouquet.plotting import plot_jphi
        fig, ax = plot_jphi(self._archive(tmp_path, rec), scan_key="7")
        try:
            assert want in ax[2].get_title()
            labels = [ln.get_label() for ln in ax[2].get_lines()]
            assert any(want in lb for lb in labels), labels
            assert not any("SWB baseline" in lb for lb in labels), labels
        finally:
            plt.close(fig)
