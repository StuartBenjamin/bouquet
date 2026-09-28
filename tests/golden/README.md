# Golden bouquet test fixtures

Git-tracked regression fixtures for `tests/test_golden_bouquet.py`.

| file | what it is |
|------|------------|
| `D3Dlike_Hmode_golden_slim.h5` | a slimmed real bouquet run (~11.8 MB): `*.pfile` byte blobs dropped, the `*.eqdsk` geqdsks **kept but gzip-compressed** (~3x), `Ip` also extracted into an attr, everything the assertions need kept (attrs, `coil_currents`, `x_points`, both LCFS refs, profiles). |
| `golden_manifest.json` | expected per-draw + baseline values (l_i, Ip, coil drifts, boundary RMS/max, coil currents, X-points) with tolerances. |
| `rng_stream_manifest.json` | the **seeded GPR draw stream**, pinned bitwise (SHA-256 per channel + sampled values), drawn from the slim fixture's baseline profiles + sigma envelopes. |
| `make_golden_fixture.py` | regenerates the three files above from a full run. |

## The draw-stream golden

`rng_stream_manifest.json` is the only draw-*level* golden here, and it became
possible only when `GenerationConfig.seed` started reaching the GPR: before
that, every draw site re-seeded from OS entropy and no drawn value was
reproducible. It replays what `perturb_kinetic_equilibrium` does for one draw
— ne, Te, ni, Ti through `_draw_monotonic_perturbation`, then the `j_phi` GPR
candidate — off one `make_rng(seed)` Generator. Pure NumPy: no solver, no
mesh, so it is bitwise identical on any machine.

Re-pin it on its own (no full run needed) with

```bash
python tests/golden/make_golden_fixture.py --rng-stream-only
```

The manifest carries sampled values and per-channel min/max alongside each
hash, so the git diff shows roughly *where* a stream moved, not just that it
did. A changed hash with unchanged samples means the change is elsewhere in
the profile.

The geqdsks are deliberately retained: geqdsk is a coarse-at-the-separatrix
format and exercising its read/parse path on real files (see the
`test_geqdsk_*` tests) is worthwhile. They are stored as gzipped `uint8`
under their original `.eqdsk` dataset names, so every reader
(`bytes(grp[k][()])`) is unaffected. `make_golden_fixture.py --eqdsk` chooses
retention: `all` (default, ~11.8 MB), `subset` (baseline + representative
draws, ~5 MB), or `none` (~3.7 MB, no geqdsk-handling coverage). Eventually,
when the default interchange migrates to IMAS/OMAS, the fixture can store
those instead.

The full-fidelity 30 MB run (with p-file bytes and uncompressed eqdsks) stays
as the shareable example artifact under
`bouquet/examples/D3D-like/D3Dlike_Hmode_golden.h5` (not tracked here).

## Updating the golden set (on purpose)

1. Re-run the example notebook to produce a fresh full `.h5`.
2. Regenerate the fixture + manifest:
   ```bash
   python tests/golden/make_golden_fixture.py \
       --source /path/to/D3Dlike_Hmode_golden.h5
   ```
   (defaults to the D3D-like example artifact path if `--source` is omitted;
   `rng_stream_manifest.json` is re-pinned from the new slim fixture in the
   same command).
3. Review the `golden_manifest.json` git diff — it shows exactly which physics
   values moved — then commit the new fixture + manifests together.

The builder refuses to write a fixture that names an absolute filesystem path
anywhere (string attrs, `config_json`, geqdsk headers) and reduces the paths
inside the self-consistent bootstrap records (`jbs_loop_json`, which carry the
OFT package path) to basenames: this repository is public, and the OFT build
is identified by the content digests in the provenance stamp instead.

## Pending: the self-consistent-bootstrap refresh

`GenerationConfig.jbs_self_consistent` now defaults to `True`, but the fixture
here is still the frozen-bootstrap run it always was (its stored config
predates the field, so `load_config` reads it with the loop off and
`test_systematics` replays it on the frozen path -- consistent, not stale in
that sense). A loop-on regeneration from this fixture's own config has not
yet produced a usable archive: on this case the standard (l_i-loop) draw's
Gauss-Seidel bootstrap coupling contracts at ~0.38/pass from r_j ~ 2e-2..1e-1
and needs ~7-9 passes against the per-draw ceiling of 6, and the
post-homotopy check (ceiling 2 with the two-consecutive-pass rule) rejects any
draw whose first pass misses. Refresh once the draw ceilings are settled; the
ceilings are an approved convergence setting and are not changed here.

## Known build-dependent failure: `test_systematics::test_mode3_production_reproduces_golden`

On OFT builds that carry the 2026-08 bootstrap stencil change (the current
line, including the macOS development build and the Linux production build)
this replay misses the recorded l_i(1) of draw 3 by 3.74 % against its 3 %
bar (`l_i(1) replay 0.8240 vs golden 0.8560`), with l_i(3) (2.06 %) and the
boundary RMS inside their bars. The miss is the fixture's, not bouquet's: the
commit that produced the fixture reproduces it to four decimals on the newer
OFT, and passes on the older one; the signature (an l_i(1)-only, edge-localised
change) is the end-stencil difference of the bootstrap-gradient formula. The
bar is not widened; the failure is expected until the fixture is regenerated
on the current OFT line (see the pending refresh above).

The `*.h5` glob in `.gitignore` is negated for `tests/golden/*.h5` so the slim
fixture is tracked while ad-hoc run outputs elsewhere stay ignored.
