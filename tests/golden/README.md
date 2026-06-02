# Golden bouquet test fixtures

Git-tracked regression fixtures for `tests/test_golden_bouquet.py`.

| file | what it is |
|------|------------|
| `D3Dlike_Hmode_golden_slim.h5` | a slimmed real bouquet run (~11.8 MB): `*.pfile` byte blobs dropped, the `*.eqdsk` geqdsks **kept but gzip-compressed** (~3x), `Ip` also extracted into an attr, everything the assertions need kept (attrs, `coil_currents`, `x_points`, both LCFS refs, profiles). |
| `golden_manifest.json` | expected per-draw + baseline values (l_i, Ip, coil drifts, boundary RMS/max, coil currents, X-points) with tolerances. |
| `make_golden_fixture.py` | regenerates the two files above from a full run. |

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
   (defaults to the D3D-like example artifact path if `--source` is omitted).
3. Review the `golden_manifest.json` git diff — it shows exactly which physics
   values moved — then commit the new fixture + manifest together.

The `*.h5` glob in `.gitignore` is negated for `tests/golden/*.h5` so the slim
fixture is tracked while ad-hoc run outputs elsewhere stay ignored.
