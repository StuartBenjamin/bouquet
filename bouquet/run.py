"""The :class:`Bouquet` orchestrator.

Owns the TokaMaker solver (``mygs``), the HDF5 output path, the resolved
baseline and the uncertainty envelope, so the per-draw generation step is
auto-wired instead of threaded by hand.

Composable methods for interactive work (inspect the baseline before spending
GS-solve compute), plus a thin :meth:`run` convenience for scripts/CI::

    import bouquet as bq

    bouquet = bq.Bouquet(config)
    bouquet.setup_solver()
    bouquet.prepare_baseline()      # reconstruction OR imas, transparently
    bouquet.plot_baseline()         # gate: is the baseline good?
    bouquet.generate()
    bouquet.filter()
    bouquet.export()

    # or, once the baseline is trusted:
    bouquet = bq.Bouquet(config).run()
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import BouquetConfig
    from .baseline import Baseline


def _shape_from_boundary(boundary_RZ):
    """LCFS shape params (R0, Z0, a, kappa, delta) from boundary (R,Z) points."""
    import numpy as np

    rz = np.asarray(boundary_RZ, dtype=float)
    R, Z = rz[:, 0], rz[:, 1]
    Rmax, Rmin, Zmax, Zmin = R.max(), R.min(), Z.max(), Z.min()
    R0 = 0.5 * (Rmax + Rmin)
    a = 0.5 * (Rmax - Rmin)
    Z0 = 0.5 * (Zmax + Zmin)
    kappa = (Zmax - Zmin) / (Rmax - Rmin)
    R_upper = R[int(np.argmax(Z))]
    R_lower = R[int(np.argmin(Z))]
    delta = 0.5 * ((R0 - R_upper) + (R0 - R_lower)) / a
    return R0, Z0, a, kappa, delta


class Bouquet:
    """Stateful driver: solver -> baseline -> generate -> filter -> export."""

    def __init__(self, config: "BouquetConfig"):
        self.config = config
        self.mygs = None                          # set by setup_solver()
        self.baseline: Optional["Baseline"] = None
        self._resolved_uncertainty = None         # resolved sigma profiles + length scales
        self.diagnostics: Optional[list] = None   # generate() per-draw output
        self.generation_log: Optional[str] = None # captured generate() solver chatter
        self._selection = None                    # filter() result

    # ── constructors ----------------------------------------------------
    @classmethod
    def from_geqdsk(cls, geqdsk_path, *, profiles, mesh,
                    n_draws=20, header="bouquet", cocos=1, time=None,
                    **solver_kwargs) -> "Bouquet":
        """Minimal constructor for the reconstruction path (g-file + profiles).

        ``profiles`` is an IDA ``.cdf`` or a p-file (auto-detected). Extra
        keyword args go to :class:`SolverConfig` (e.g. ``order``, ``nthreads``).
        Reach into ``bq.uncertainty`` / ``bq.generation`` afterwards for the
        advanced knobs.
        """
        from .config import (BouquetConfig, SolverConfig, ReconstructionSource,
                             GenerationConfig)
        cfg = BouquetConfig(
            source=ReconstructionSource(geqdsk_path=geqdsk_path,
                                        profiles_path=profiles,
                                        cocos=cocos, time=time),
            solver=SolverConfig(mesh_path=mesh, **solver_kwargs),
            generation=GenerationConfig(n_equils=n_draws),
            output_header=header,
        )
        return cls(cfg)

    @classmethod
    def from_imas(cls, ids_path, *, mesh, time=None,
                  n_draws=20, header="bouquet", **solver_kwargs) -> "Bouquet":
        """Minimal constructor for the IMAS/OMAS path (no reconstruction).

        Extra keyword args go to :class:`SolverConfig`. Reach into
        ``bq.uncertainty`` / ``bq.generation`` afterwards for advanced knobs.
        """
        from .config import (BouquetConfig, SolverConfig, ImasSource,
                             GenerationConfig)
        cfg = BouquetConfig(
            source=ImasSource(ids_path=ids_path, time=time),
            solver=SolverConfig(mesh_path=mesh, **solver_kwargs),
            generation=GenerationConfig(n_equils=n_draws),
            output_header=header,
        )
        return cls(cfg)

    # ── ergonomic config accessors (so `bq.uncertainty.ne_scalar_sigma = ...`,
    # `bq.generation.n_equils = ...` read like a control panel) ──
    @property
    def uncertainty(self):
        """The :class:`UncertaintyConfig` (sigma profiles, GPR lengths, switchboard)."""
        return self.config.uncertainty

    @property
    def generation(self):
        """The :class:`GenerationConfig` (n_equils, tolerances, homotopy)."""
        return self.config.generation

    @property
    def solver(self):
        """The :class:`SolverConfig` (mesh, order, threads, F0)."""
        return self.config.solver

    def set_slice(self, *, time=None, header=None) -> "Bouquet":
        """Re-point to a new time slice, reusing the existing solver.

        For multi-slice runs in one kernel (the ``OFT_env`` singleton forbids a
        second solver): keep one :meth:`setup_solver`, then for each slice call
        ``set_slice(time=t, header=...)`` and :meth:`run` (or generate). Clears
        the cached baseline/uncertainty so the next call re-solves; the new
        baseline reconstruction/forward-solve overwrites any prior equilibrium
        state on ``mygs``, so no explicit reset is needed.
        """
        if time is not None:
            if not hasattr(self.config.source, "time"):
                raise TypeError(
                    f"{type(self.config.source).__name__} has no time slice")
            self.config.source.time = time
        if header is not None:
            self.config.output_header = header
        self.baseline = None
        self._resolved_uncertainty = None
        self.diagnostics = None
        self._selection = None
        return self

    # ── stage 1: solver -------------------------------------------------
    def setup_solver(self) -> "Bouquet":
        """Read mesh, build regions, stand up ``mygs``, set isoflux + VSC + reg.

        Common to every baseline source -- perturbed draws are always solved
        with TokaMaker. Returns self for chaining. Idempotent: a no-op if
        ``mygs`` already exists, so it is safe to call once and then reuse the
        solver across multiple baselines/time-slices (OFT_env is a per-process
        singleton, so re-creating it would raise).
        """
        import numpy as np

        if self.mygs is not None:
            return self
        from OpenFUSIONToolkit import OFT_env
        from OpenFUSIONToolkit.TokaMaker import TokaMaker
        from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh

        from .config import ReconstructionSource, ImasSource
        from .io.geqdsk import read_geqdsk

        sc = self.config.solver
        src = self.config.source

        myOFT = OFT_env(nthreads=sc.nthreads)
        mygs = TokaMaker(myOFT)

        mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(sc.mesh_path)
        mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)

        # F0 and reference LCFS boundary come from the g-file (reconstruction)
        # or the IDS vacuum_toroidal_field + boundary outline (IMAS).
        F0 = sc.F0
        eqdsk_ref = None
        boundary_RZ = None
        if isinstance(src, ReconstructionSource):
            eqdsk_ref = read_geqdsk(src.geqdsk_path, cocos=src.cocos)
            if F0 is None:
                F0 = abs(eqdsk_ref.R_center * eqdsk_ref.B_center)
            boundary_RZ = np.column_stack(
                [eqdsk_ref.boundary_R, eqdsk_ref.boundary_Z]
            )
        elif isinstance(src, ImasSource):
            from .io.imas import read_imas_geometry
            _imas_F0, boundary_RZ = read_imas_geometry(src)
            if F0 is None:
                F0 = _imas_F0
        if F0 is None:
            raise ValueError("F0 could not be determined; set SolverConfig.F0")

        mygs.setup(order=sc.order, F0=F0)
        mygs.settings.maxits = 800
        mygs.settings.pm = False
        mygs.update_settings()
        mygs.set_coil_vsc(sc.coil_vsc)

        # Isoflux: explicit config wins; otherwise the source's LCFS boundary
        iso_pts, iso_w = sc.isoflux_pts, sc.isoflux_weights
        if iso_pts is None and boundary_RZ is not None:
            iso_pts = boundary_RZ
            iso_w = np.ones(len(iso_pts)) * 500.0
        if iso_pts is not None:
            mygs.set_isoflux(iso_pts, weights=iso_w)

        # Weak coil regularisation toward zero + small VSC freedom
        reg_terms = [mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0)
                     for name in mygs.coil_sets]
        reg_terms.append(mygs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=1e-2))
        mygs.set_coil_reg(reg_terms=reg_terms)

        self.mygs = mygs
        self._myOFT = myOFT          # keep the env alive
        self._eqdsk_ref = eqdsk_ref
        self._boundary_RZ = boundary_RZ   # LCFS shape for IMAS forward-solve init
        return self

    # ── stage 2: baseline (reconstruction OR imas) ----------------------
    def prepare_baseline(self) -> "Baseline":
        """Resolve the baseline from ``config.source`` and cache it.

        Delegates to :func:`bouquet.baseline.resolve_baseline`, which dispatches
        on source type. Generation depends only on the returned
        :class:`~bouquet.baseline.Baseline`, never on reconstruction directly.
        """
        from .baseline import resolve_baseline
        from .config import ImasSource

        self.baseline = resolve_baseline(self.config, self.mygs)

        # IMAS path: read_imas_baseline does no GS solve, so establish a converged
        # baseline equilibrium on mygs here (the reconstruction path gets this for
        # free from reconstruct_equilibrium). This also sets l_i_target to the
        # TokaMaker-solved li_1 and records IDS-vs-TokaMaker li for sanity.
        if isinstance(self.config.source, ImasSource) and self.mygs is not None:
            self._forward_solve_imas_baseline()

        # Reconstruction path: surface a glanceable quality summary (the verbose
        # solver chatter was captured to baseline.reconstruction_log).
        if self.baseline.reconstruction_metrics is not None:
            self._print_reconstruction_summary()
        return self.baseline

    def reconstruct(self) -> "Baseline":
        """Reconstruct the baseline equilibrium and print a quality summary.

        Intent-revealing entry point for the reconstruction path: ensures the
        solver is up, runs the GS reconstruction, and prints the curated metrics
        so you can confirm at a glance that it succeeded before spending compute
        on the draws. Equivalent to ``setup_solver(); prepare_baseline()``.
        """
        from .config import ReconstructionSource

        if not isinstance(self.config.source, ReconstructionSource):
            raise TypeError(
                "reconstruct() is for a ReconstructionSource; the IMAS path has "
                "no reconstruction step -- call prepare_baseline() (or run())."
            )
        self.setup_solver()
        return self.prepare_baseline()

    def _print_reconstruction_summary(self):
        """Print the reconstruction-fidelity block (TokaMaker vs input, % error).

        Global scalars are shown as ``value (input ref, +/-% err)`` against the
        input g-file's own values; geometric residuals that should be ~0
        (boundary, axis offset, j_phi RMS) are shown absolute. See
        :func:`bouquet.baseline._reconstruction_metrics`.
        """
        m = self.baseline.reconstruction_metrics
        tag = self.config.source.geqdsk_path.split("/")[-1]
        mark = "PASS ✅" if m.get("verdict") == "PASS" else "CHECK ⚠"
        # blank lines so the summary stands out after any solver output above
        print(f"\n\n=== Reconstruction — {tag} {'=' * max(3, 40 - len(tag))} {mark}")

        def line(label, val, ref, err, unit="", fmt=".3f"):
            u = f" {unit}" if unit else ""
            lhs = f"{format(val, fmt)}{u}"
            print(f"  {label:<12} {lhs:<13} (input {format(ref, fmt)}{u}, {err:+.2f}%)")

        print(f"  {'converged':<12} yes")
        line("Ip", m['Ip_MA'], m['Ip_efit_MA'], m['Ip_err_pct'], "MA")
        line("l_i", m['li'], m['li_efit'], m['li_err_pct'])
        line("q0", m['q0'], m['q0_efit'], m['q0_err_pct'], fmt=".2f")
        line("q95", m['q95'], m['q95_efit'], m['q95_err_pct'], fmt=".2f")
        line("beta_N", m['beta_n'], m['beta_n_efit'], m['beta_n_err_pct'], fmt=".2f")
        line("beta_p", m['beta_p'], m['beta_p_efit'], m['beta_p_err_pct'], fmt=".2f")
        line("kappa", m['kappa'], m['kappa_efit'], m['kappa_err_pct'])
        line("delta", m['delta'], m['delta_efit'], m['delta_err_pct'])
        line("j_sep(.99)", m['j_sep_MA'], m['j_sep_efit_MA'], m['j_sep_err_pct'], "MA/m²")
        line("W_MHD", m['W_MHD_MJ'], m['W_MHD_efit_MJ'], m['W_MHD_err_pct'], "MJ")
        print(f"  {'boundary':<12} RMS {m['boundary_rms_mm']:.2f} mm   "
              f"max {m['boundary_max_mm']:.2f} mm   axis off {m['axis_offset_mm']:.2f} mm")
        print(f"  {'jphi resid':<12} core RMS {m['jphi_core_rms_MA']:.3f}   "
              f"edge RMS {m['jphi_edge_rms_MA']:.3f} MA/m²")

    def _forward_solve_imas_baseline(self):
        """Forward GS solve of the IMAS baseline (j_phi + pressure) on mygs.

        Initialises psi from the IDS LCFS shape, then solves with the IMAS total
        toroidal current (jphi-linterp) and the thermal+fast pressure. Sets
        ``l_i_target`` to the TokaMaker-solved li_1 and stores the IDS/TokaMaker
        li_1/li_3 comparison in ``baseline.li_metrics``.
        """
        import numpy as np

        bl = self.baseline
        mygs = self.mygs
        psi_N = np.asarray(bl.psi_N, dtype=float)
        psi_pad = 1e-3
        EC = 1.602176634e-19

        # init psi from the LCFS shape parameters
        R0, Z0, a, kappa, delta = _shape_from_boundary(self._boundary_RZ)
        mygs.init_psi(R0, Z0, a, kappa, delta)

        # kinetic profiles + total pressure on the equilibrium grid (IMAS shares
        # psi_N between the kinetic and current grids).
        def k2e(arr):
            return np.interp(psi_N, np.asarray(bl.psi_N_kinetic, dtype=float),
                             np.asarray(arr, dtype=float))

        ne, te, ni, ti = k2e(bl.ne), k2e(bl.te), k2e(bl.ni), k2e(bl.ti)
        Zeff = np.clip(k2e(bl.Zeff), 1.0, None)
        p_total = EC * (ne * te + ni * ti)
        if bl.p_fast is not None:
            p_total = p_total + k2e(bl.p_fast)

        def solve_jphi(j_phi):
            ffp = {"type": "jphi-linterp", "y": np.asarray(j_phi, dtype=float), "x": psi_N}
            for _ in range(2):   # 2nd pass refines the jphi-linterp flux scaling
                psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
                pp_y = np.gradient(p_total) / (np.gradient(psi_N) * psi_range)
                pp_y[-1] = 0.0
                mygs.set_targets(Ip=bl.Ip_target, pax=float(p_total[0]))
                mygs.set_profiles(
                    pp_prof={"type": "linterp", "y": pp_y, "x": psi_N}, ffp_prof=ffp,
                )
                mygs.solve()

        # Solve with the IMAS total current. The OMAS currents are
        # reconstruction-self-consistent (bootstrap = SWB of the same kinetics),
        # so this baseline l_i matches what the per-draw machinery produces -- the
        # draws anchor to bl.j_inductive and recompute a ~equal SWB spike. An
        # earlier SWB rebuild here shifted the baseline l_i ~6% off the draws and
        # made the l_i target unreachable (draws floored at the low tail); removed.
        solve_jphi(np.asarray(bl.j_phi, dtype=float))

        tok_li1 = float(mygs.get_stats(lcfs_pad=psi_pad, li_normalization="std")["l_i"])
        tok_li3 = float(mygs.get_stats(lcfs_pad=psi_pad, li_normalization="iter")["l_i"])

        metrics = dict(bl.li_metrics or {})
        metrics.update(tokamaker_li_1=tok_li1, tokamaker_li_3=tok_li3)
        bl.li_metrics = metrics
        bl.l_i_target = tok_li1   # per project decision: target TokaMaker li_1
        print(
            f"[imas forward-solve] recalc_jBS={self.config.generation.recalculate_j_BS} "
            f"TokaMaker li_1={tok_li1:.4f} li_3={tok_li3:.4f} | "
            f"IDS li_1={metrics.get('ids_li_1')} li_3={metrics.get('ids_li_3')}"
        )

    def plot_baseline(self):
        """Diagnostic plots: kinetic profiles, separated currents, and (for the
        reconstruction source) the j_phi fit residuals and l_i convergence."""
        raise NotImplementedError

    # ── stage 3: perturbed bouquet --------------------------------------
    def generate(self, n: Optional[int] = None) -> list:
        """Generate the perturbed bouquet and archive to ``{header}.h5``.

        Auto-feeds the baseline (j_phi, j_inductive, l_i_target, Ip_target) and
        the uncertainty envelope (kinetic sigmas, j_phi sigma, GPR lengths).
        ``n`` overrides ``config.generation.n_equils`` for a quick smaller run.

        Requires :meth:`prepare_baseline` first; raises if ``self.baseline`` is
        None.
        """
        import numpy as np
        from .baseline import resolve_uncertainty
        from .TokaMaker_interface import generate_bouquet
        from .utils import initialize_equilibrium_database

        if self.baseline is None:
            raise ValueError("call prepare_baseline() before generate()")
        if self.mygs is None:
            raise ValueError("call setup_solver() before generate()")

        bl = self.baseline
        gc = self.config.generation
        fc = self.config.filtering
        n_equils = int(n if n is not None else gc.n_equils)

        env = resolve_uncertainty(self.config, bl)
        self._resolved_uncertainty = env

        header = self.config.output_header
        initialize_equilibrium_database(header)

        # Restore the reconstruction isoflux targets before sampling (the recon
        # solve leaves them in place; an explicit restore is harmless otherwise).
        if bl.recon is not None and "isoflux_pts" in bl.recon:
            self.mygs.set_isoflux(bl.recon["isoflux_pts"], weights=bl.recon["weights"])

        psi_pad = float(getattr(self.config.source, "psi_pad", 1e-3))

        # Zeff is consumed on the EQUILIBRIUM grid (psi_N) by solve_with_bootstrap,
        # whereas the perturbed kinetic profiles (ne/te/ni/ti, sigmas) live on the
        # kinetic grid (psi_N_kinetic). Interpolate Zeff down to psi_N.
        Zeff_eq = np.clip(
            np.interp(np.asarray(bl.psi_N, dtype=float),
                      np.asarray(bl.psi_N_kinetic, dtype=float),
                      np.asarray(bl.Zeff, dtype=float)),
            1.0, None,
        )

        # The per-draw solver emits a large volume of diagnostic text (homotopy
        # passes, bootstrap/boundary diagnostics, DLSODE chatter) -- enough to
        # bloat a notebook by tens of MB. Capture it unless verbose, so output
        # stays readable; the full text is kept on generation_log for debugging.
        # Set BouquetConfig.verbose=True to stream it (and the tqdm progress bar).
        from .utils import capture_native_output
        verbose = bool(getattr(self.config, "verbose", False))
        with capture_native_output(enabled=not verbose) as _cap:
            self.diagnostics = generate_bouquet(
                self.mygs, np.asarray(bl.psi_N, dtype=float), n_equils, header,
                np.asarray(bl.j_phi, dtype=float),
                bl.ne, bl.te, bl.ni, bl.ti,
                env["sigma_ne"], env["sigma_te"], env["sigma_ni"], env["sigma_ti"],
                env["sigma_jphi"],
                env["n_ls"], env["t_ls"], env["j_ls"],
                bl.Ip_target, bl.l_i_target, Zeff_eq,
                input_jinductive=np.asarray(bl.j_inductive, dtype=float),
                l_i_tolerance=gc.l_i_tolerance,
                psi_pad=psi_pad,
                constrain_sawteeth=gc.constrain_sawteeth,
                recalculate_j_BS=gc.recalculate_j_BS,
                jBS_scale_range=gc.jBS_scale_range,
                diagnostic_plots=gc.diagnostic_plots,
                scan_val=0,
                pfile_bytes=bl.pfile_bytes,
                baseline_eqdsk_bytes=bl.eqdsk_bytes,
                baseline_pfile_bytes=bl.pfile_bytes,
                psi_N_kinetic=np.asarray(bl.psi_N_kinetic, dtype=float),
                coil_drift=gc.coil_drift,
                homotopy_passes=gc.homotopy_passes,
                inspec_F_max=fc.inspec_F_max,
                inspec_VSC_max=fc.inspec_VSC_max,
                seed=gc.seed,
                # Fixed additive components, summed into every draw, never perturbed.
                p_fast=bl.p_fast,
                j_NBI=bl.j_NBI,
                j_RF=bl.j_RF,
                # Switchboard: extra perturbed profiles (passive + active Zeff).
                extra_sigma=env.get("extra_sigma"),
                extra_baseline=env.get("extra_baseline"),
                extra_length_scale=env.get("extra_length_scale"),
            )
        self.generation_log = _cap["text"] or None
        return self.diagnostics

    def plot_bouquet(self, mode: str = "all", selection: str = "all",
                     layout: str = "stack", pub_style: bool = False):
        """Overlay plots of the generated bouquet (kinetic / pressure / j_phi /
        boundary). Thin wrapper over :func:`bouquet.plotting.plot_bouquet` on
        this run's HDF5 (``{output_header}.h5``). ``pub_style=False`` (default)
        is one combined dashboard figure; ``pub_style=True`` (with
        ``layout="row"``) gives the separate side-by-side publication figures."""
        from .plotting import plot_bouquet as _plot_bouquet
        return _plot_bouquet(f"{self.config.output_header}.h5",
                             scan_value=0, mode=mode, selection=selection,
                             layout=layout, pub_style=pub_style)

    # ── stage 4: filter + export ----------------------------------------
    def filter(self, rms_max_mm: Optional[float] = None) -> dict:
        """Mark the machine-realizable subset (coil + boundary filters).

        Non-destructive: writes pass flags into the HDF5. Returns a summary.
        """
        from .filtering import filter_coil_currents, filter_boundaries

        header = self.config.output_header
        fc = self.config.filtering
        rms = fc.rms_max_mm if rms_max_mm is None else rms_max_mm

        coil_summary, _ = filter_coil_currents(
            header,
            F_max_pct=fc.inspec_F_max * 100.0,
            VSC_max_pct=fc.inspec_VSC_max * 100.0,
            apply=True, plot=False,
        )
        bnd_summary, _ = filter_boundaries(
            header, rms_max_mm=rms, apply=True, plot=False,
        )
        self._selection = {"coil": coil_summary, "boundary": bnd_summary}
        self._print_generation_summary(coil_summary, bnd_summary)
        return self._selection

    def _print_generation_summary(self, coil_summary, bnd_summary):
        """Concise post-generation summary (draws / coil spec / boundary / in-spec),
        in the style of the reconstruction summary."""
        from .filtering import select_indices
        header = self.config.output_header
        n_all = len(select_indices(header, selection="all"))
        n_sel = len(select_indices(header, selection="selected"))

        def _first(summary):
            if not summary:
                return {}
            if isinstance(summary, dict):
                return next(iter(summary.values()), {}) or {}
            return summary[0] if len(summary) else {}

        cs = _first(coil_summary)
        rs = (_first(bnd_summary).get("rms_stats") or {})
        fc = self.config.filtering
        tag = header.split("/")[-1]
        frac = 100.0 * n_sel / max(n_all, 1)
        print(f"\n=== Bouquet — {tag} {'=' * max(3, 34 - len(tag))}  "
              f"{n_sel}/{n_all} in-spec ({frac:.0f}%)")
        print(f"  draws         {n_all} generated")
        print(f"  coil spec     {cs.get('n_pass', '?')}/{cs.get('n_total', '?')} "
              f"within ±{fc.inspec_F_max * 100:.0f}%   "
              f"({cs.get('n_fail', '?')} out-of-spec)")
        if rs:
            print(f"  boundary      RMS median {rs.get('median', float('nan')):.2f} mm"
                  f"   max {rs.get('max', float('nan')):.2f} mm")

    def export(self, out_path: Optional[str] = None, selection: str = "selected"):
        """Write a pruned HDF5 with only the selected draws.

        Defaults to ``{header}_selected.h5``.
        """
        from .filtering import export_filtered

        header = self.config.output_header
        out = out_path if out_path is not None else f"{header}_selected.h5"
        export_filtered(header, out, selection=selection, overwrite=True)
        return out

    # ── convenience -----------------------------------------------------
    def run(self) -> "Bouquet":
        """setup_solver -> prepare_baseline -> generate -> filter -> export.

        For scripts / CI where the baseline is already trusted. Returns self,
        fully populated (baseline, diagnostics, HDF5 path all reachable).

        Idempotent on the early stages: if the solver is already up (e.g. you
        called :meth:`reconstruct` first, or are reusing it across slices) it is
        not rebuilt, and an already-prepared baseline is not re-solved.
        """
        self.setup_solver()                       # idempotent
        if self.baseline is None:
            self.prepare_baseline()
        self.generate()
        self.filter()
        self.export()
        return self
