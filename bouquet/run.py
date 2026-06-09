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
        self.uncertainty = None                   # resolved sigma profiles + length scales
        self.diagnostics: Optional[list] = None   # generate() per-draw output
        self._selection = None                    # filter() result

    # ── stage 1: solver -------------------------------------------------
    def setup_solver(self) -> "Bouquet":
        """Read mesh, build regions, stand up ``mygs``, set isoflux + VSC + reg.

        Common to every baseline source -- perturbed draws are always solved
        with TokaMaker. Returns self for chaining.

        F0 (= R0*B0) and the reference boundary come from the reconstruction
        g-file when the source is a :class:`ReconstructionSource`; for an
        :class:`ImasSource` set ``SolverConfig.F0`` explicitly (generation-side
        solver setup for the IMAS path is wired in a later step).
        """
        import numpy as np
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
        return self.baseline

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

        # Initial solve with the IMAS total current (also seeds SWB geometry).
        solve_jphi(np.asarray(bl.j_phi, dtype=float))

        if self.config.generation.recalculate_j_BS:
            # The draws recompute bootstrap via SWB each iteration, which differs
            # from the IDS/generic bootstrap. Rebuild the baseline on that SAME
            # SWB basis (inductive*scale + SWB spike + fixed, Ip-matched) so the
            # forward-solve l_i is reachable by the draws.
            from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap
            from scipy.optimize import root_scalar
            from .utils import Ip_flux_integral_vs_target

            j_fixed = np.zeros_like(psi_N)
            if bl.j_NBI is not None:
                j_fixed = j_fixed + np.asarray(bl.j_NBI, dtype=float)
            if bl.j_RF is not None:
                j_fixed = j_fixed + np.asarray(bl.j_RF, dtype=float)

            jind = np.asarray(bl.j_inductive, dtype=float)
            res = solve_with_bootstrap(
                mygs, ne, te, ni, ti, Zeff, bl.Ip_target, jind,
                scale_jBS=1.0, isolate_edge_jBS=True, diagnostic_plots=False,
            )
            spike = np.asarray(res["isolated_j_BS"], dtype=float)
            root = root_scalar(
                Ip_flux_integral_vs_target,
                args=(mygs, jind, spike + j_fixed, psi_N, bl.Ip_target),
                bracket=[1e-10 * bl.Ip_target, 1e1 * bl.Ip_target],
                method="brentq", rtol=1e-6,
            )
            scale = root.root
            j_phi_swb = scale * jind + spike + j_fixed
            solve_jphi(j_phi_swb)
            bl.j_BS = spike                 # update decomposition to the SWB basis
            bl.j_inductive = scale * jind
            bl.j_phi = j_phi_swb

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
        self.uncertainty = env

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
        )
        return self.diagnostics

    def plot_bouquet(self, mode: str = "all"):
        """Overlay plots of the generated bouquet (boundaries, profiles, coils)."""
        raise NotImplementedError

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
        return self._selection

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
        """
        self.setup_solver()
        self.prepare_baseline()
        self.generate()
        self.filter()
        self.export()
        return self
