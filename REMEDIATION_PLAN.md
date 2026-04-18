# AAE 568 Remediation Plan

**Status:** active, drafted 2026-04-17.
**Purpose:** Produce the saved artifacts that NARRATIVE.md requires. No narrative changes — pure backfill.
**Governing document:** `NARRATIVE.md`. If a remediation task conflicts with the narrative, the narrative wins.

## Guiding principles

1. **Schema-first.** Define `results_summary.json` format before any instrumentation. Every solver call writes into the same schema.
2. **Harness, not framework.** A `common/` module with result IO + timing wrappers. No class hierarchy, no config files.
3. **Re-run, don't re-parse.** No pickled `OptimizeResult` objects exist from past runs; sweeps must be re-executed with better logging.
4. **Course rigor, not paper rigor.** If Artemis needs hand-tuning to converge, report honestly — don't polish it away.

## Tranche 1 — Foundation (blocking for T2, T3)

- **T1.1 Schema design** — `common/results_schema.py`. Record shape: `{phase, case, method, parameters, cost, converged, residual, wall_time_s, iterations, nfev, njev, n_vars, n_constraints, git_sha, timestamp, python_version, convergence_history?}`.
- **T1.2 IO utilities** — `common/results_io.py`: `save_result()`, `append_to_summary()`, `load_results(filter=...)`. One authoritative `results_summary.json` at project root.
- **T1.3 Timing wrapper** — `common/timing.py`: `timed_solve()` context manager using `time.perf_counter()` around solver call only.
- **T1.4 Reference implementation on Phase 0** — port `Earth-Mars/shooting.py` + `Earth-Mars/ipopt_collocation_2body.py` to emit records. Validates the schema on the simplest case and produces the first populated `results_summary.json`.

**Deliverable:** `common/` committed; `results_summary.json` contains Earth-Mars shooting + IPOPT entries; tabular-agreement printout substantiating narrative P0 claim "agree to machine tolerance."

## Tranche 2 — Phase 1 instrumentation (parallel, after T1)

- **T2.1 Shooting baseline** — re-run `Planer/cr3bp_transfer.py:shooting_cr3bp_min_energy`. Validate J* = 0.04306.
- **T2.2 IPOPT global Bézier** — re-run IPOPT path, capture CasADi `solver.stats()`.
- **T2.3 SLSQP segmented-Bézier N-sweep** — re-run `Planer/cr3bp_transfer_segmented.py` for N ∈ {1,2,4,8,16} with proper `OptimizeResult` capture. One record per N.
- **T2.4 Pareto plot** — `Planer/make_pareto_plot.py` loads `results_summary.json`, plots |J−J*| vs wall time overlaying SLSQP (line over N) and IPOPT (single point), PMP anchor as vertical line.

**Deliverable:** narrative P1 claims "N=16 matches PMP at 0.4% error, 87× slower" and "IPOPT recovers cost" are now backed by the data file.

## Tranche 3 — Phase 2 Artemis instrumentation (parallel, after T1)

- **T3.1 Solver wrappers** — wrap `solve_shooting` and `solve_ipopt_collocation` in `artemis2_ephemeris.py` + `artemis2_full_mission.py` with timing + result capture.
- **T3.2 Multi-guess sweep persistence** — the existing `n_guesses=15` shooting loop must write one record per seed (converged/not, cost, time). Backbone of the "shooting fragility" claim.
- **T3.3 Retuning audit** — `Artemis2/RETUNING_AUDIT.md` enumerates, for each IPOPT option, whether inherited from Phase 1 or locally modified. Substantiates narrative claim "converges without re-tuning."
- **T3.4 Convergence-history plot** — capture IPOPT per-iter {obj, constr_viol} via iteration callback; plot.

**Deliverable:** narrative P2 claims "transfers without retuning," "initial-guess fragility via 15-seed sweep," "convergence history vs OEM" all substantiated by saved artifacts.

## Tranche 4 — Derived artifacts (after T2 + T3)

- **T4.1 Consolidated results table** — LaTeX/markdown table for report, rows = {Earth-Mars, Planar global, Planar segmented N=16, Artemis TLI, Artemis full}, columns = {method, wall-clock, iterations, ΔV/cost, NLP vars}. Falls out of `results_summary.json`.
- **T4.2 Jacobi-constant check** (Phase 1) — one-line-per-method plot confirming dynamics integrity beyond endpoint match.

## Tranche 5 — Optional cleanup

- T5.1 Prune `__pycache__` with mixed Python 3.10/3.12 compiled files.
- T5.2 Remove superseded `AAE568_Project_Notes_041226.docx`.
- T5.3 Decide Phase 3 NRHO code disposition (keep as "future work exhibit" with README, or prune).

## Explicitly descoped

- **STORYLINE.md rewrite** — superseded by NARRATIVE.md; leave as historical draft.
- **Basin-of-convergence sweep for Phase 1 shooting** — Artemis2 multi-guess sweep provides sufficient evidence.
- **Free-t_f transversality** — acknowledged as limitation in NARRATIVE §4, not implemented.
- **Primer vector / Lawden's conditions** — out of scope for course project.

## Dispatch order

Subagent round 1 (parallel, this turn):

| Subagent | Scope | Dependencies |
|---|---|---|
| A | T1.1–T1.4 (foundation + Phase 0 reference impl) | none |
| B | T3.3 only (Artemis retuning audit — read-only) | none |

Round 2 (next turn, after A returns with committed schema): T2 and remainder of T3 launched in parallel.

Round 3: T4 (derived artifacts).
