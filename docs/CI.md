# Continuous integration / test policy

Two tiers:

| tier | what | when | where | time |
|------|------|------|-------|------|
| **fast** | unit suite (golden-fixture reads, filtering, io, etc.) — no OFT | every push + PR | GitHub-hosted `ubuntu-latest` | ~seconds |
| **solver** | live OpenFUSIONToolkit GS solve: reconstruction-free replay + systematics (`-m solver`) | **pre-merge only** (merge queue) + manual | self-hosted **feynman** | ~13 min |

`pytest` is **fast-by-default** (`addopts = -m "not solver"` in `pyproject.toml`).
Run the comprehensive suite explicitly with `pytest -m solver`.

## Workflows
- `.github/workflows/tests.yml` — fast suite on `push`/`pull_request`.
- `.github/workflows/solver.yml` — solver suite on `merge_group` (the merge
  queue) + `workflow_dispatch`, on `runs-on: [self-hosted, feynman]`.

## One-time GitHub setup
1. **Register the feynman runner.** On feynman: Settings → Actions → Runners →
   *New self-hosted runner*; add labels `self-hosted, feynman`.  The runner's
   shell must have Python 3 + `numpy/scipy/h5py/matplotlib` and the **OFT fork
   built** (the jphi-linterp Ip fixes the solver tests depend on).  If the env
   needs activation (conda/modules), do it in the runner service or add a step
   in `solver.yml`.
2. **Set the OFT path.** Repo → Settings → Secrets and variables → Actions →
   *Variables* → add `OFT_PYTHONPATH` = `/path/on/feynman/OpenFUSIONToolkit/build_release/python`.
   (The test also falls back to a sibling `OpenFUSIONToolkit/` checkout.)
3. **Enable the merge queue** for `main` (Settings → General → Pull Requests →
   *Allow merge queue*), so `merge_group` fires the solver gate at merge time.
4. **Branch protection on `main`** (Settings → Branches): require status checks
   `fast` (always) and `solver` (the merge-queue gate), plus a review.

## Result
- Every push → fast feedback (~s).
- Merging to `main` → the comprehensive solver suite must pass once, in the
  merge queue, on feynman — not on every commit.

## Future simplification
Once the OFT Ip fixes are merged upstream **and** OpenFUSIONToolkit publishes to
PyPI, the solver job can move to a GitHub-hosted runner with `pip install
OpenFUSIONToolkit` and drop the feynman dependency.  (Until then, the solver
suite needs the fork build, which is why it runs on feynman.)
