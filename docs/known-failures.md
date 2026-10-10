# Known failures

Tests marked as expected failures (`xfail`) on purpose, with the reason and what removes the mark.

## Legacy g-file reconstruction: the j_BS loop cycles (finite edge current)

**Test:** `tests/test_gfile_written_contents_solver.py`, arm `("legacy", "legacy")` (xfail).

**Solver:** the legacy reconstruction only (`reconstruction_engine="legacy"`,
`TokaMaker_interface.reconstruct_equilibrium`, the j_BS loop after the corrective j_phi iteration).
The engine arm of the same test passes. swb takes IMAS sources only and never runs this loop.

**Case:** the synthetic D3D-like g-file + p-file example with `separatrix_pressure="legacy"`, on
OpenFUSIONToolkit PSFC_dev (from `solve_bootstrap_improvements` on: 1a89b59, PR4, 335ce5e). The
default `"offset"` arm converges.

**What happens:**
- The loop cycles with period 3 and never has 2 consecutive passing passes. r_j goes 2.9e-4, 1.2e-3,
  9.4e-4 (tol 1e-3) and r_I goes 5e-5, 2.2e-4, 1.7e-4 (tol 1e-4).
- On every third pass, the corrective iteration's second GS solve does not converge
  (`Exceeded "maxits"`, at 800 and still at 4000). The Picard residual stalls at 3.0e-6
  (nl_tol 1e-6) in a 2-cycle: the LCFS flux alternates by 1.0e-5, the lower X-point moves ~3e-5 m,
  and the VSC current alternates between -11647 A and -11639 A.
- The corrective iteration then keeps its first, uncorrected iterate, and that pass jumps.
- Neither lower under-relaxation (urf 0.1, 0.05) nor OFT's tracer backend changes this. It is not a
  limited/diverted switch.

**Cause:** the jphi-linterp request is finite at the LCFS: 0.04–0.08 MA/m² at ψ_N = 1 and
0.07–0.13 MA/m² at ψ_N = 0.996, about 5 % of the 1.5 MA/m² peak. The legacy reconstruction has no
edge taper (`swb_edge_taper_psi0` acts on swb only). OFT's new ε (about 0.3 % on I_BS) moved this
case onto the 2-cycle; with the old ε it converges.

**Evidence that the edge current is the cause:** taper the request to zero over ψ_N 0.995 → 1, with
nothing else changed (same build 335ce5e). The check after the corrective step is then inside
tolerance at once (r_j 8.0e-4, r_I 5.8e-5), the reconstruction converges, and no solve fails.
Realized l_i(3) after the corrective step moves from 0.6560 to 0.6555. This is one case.

**Removes the mark:** an edge taper for the legacy reconstruction (as swb has), or an OFT-side fix of
the 2-cycle.

## test_systematics mode 1: coil drift 1.45 %

See `tests/test_systematics.py` (`_KNOWN_MODE1_COIL_DRIFT_PCT`): every OFT build from 6bd832b on
(exact per-node ⟨R⟩, ⟨1/R⟩ for jphi profiles) replays the legacy golden's pinned j_phi at a 1.45 % max
coil drift against the 0.3 % bar. The boundary holds (0.35 mm). Removed when the golden is
regenerated.
