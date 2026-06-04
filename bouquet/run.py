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
        """Read mesh, build regions, stand up ``mygs``, set isoflux + VSC.

        Common to every baseline source -- perturbed draws are always solved
        with TokaMaker. Idempotent; returns self for chaining.
        """
        raise NotImplementedError

    # ── stage 2: baseline (reconstruction OR imas) ----------------------
    def prepare_baseline(self) -> "Baseline":
        """Resolve the baseline from ``config.source`` and cache it.

        Delegates to :func:`bouquet.baseline.resolve_baseline`, which dispatches
        on source type. Generation depends only on the returned
        :class:`~bouquet.baseline.Baseline`, never on reconstruction directly.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def plot_bouquet(self, mode: str = "all"):
        """Overlay plots of the generated bouquet (boundaries, profiles, coils)."""
        raise NotImplementedError

    # ── stage 4: filter + export ----------------------------------------
    def filter(self, rms_max_mm: Optional[float] = None) -> dict:
        """Mark the machine-realizable subset (coil + boundary filters).

        Non-destructive: writes pass flags into the HDF5. Returns a summary.
        """
        raise NotImplementedError

    def export(self, out_path: Optional[str] = None, selection: str = "selected"):
        """Write a pruned HDF5 with only the selected draws.

        Defaults to ``{header}_selected.h5``.
        """
        raise NotImplementedError

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
