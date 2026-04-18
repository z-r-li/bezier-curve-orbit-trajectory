# Bézier-Curve Orbit Trajectory Optimization — AAE 568, Spring 2026

Low-thrust trajectory optimization in cislunar regimes, solved by **indirect shooting (PMP)** and **direct transcription (Bézier collocation + NLP)**. The project compares the two method families across a three-phase ladder — Earth→Mars two-body, planar CR3BP Lyapunov transfer, and NASA Artemis II post-flight validation — and isolates the regime where each solver class succeeds or fails.

This repo is the deliverable code, data, and writing pack for the course project. Teammates writing the report or presentation should start at `NARRATIVE.md` and `REPORT_PACK.md`.

---

## Read this first

| File | What it is | When you need it |
| --- | --- | --- |
| **`NARRATIVE.md`** | Locked storyline. Source of truth for every claim in the report. | Writing the report / slides. |
| **`REPORT_PACK.md`** | Pull-quote cheatsheet of numerical claims with their JSON keys and source files. | Dropping numbers into the LaTeX template. |
| **`FIGURES.md`** | Map of every `.png` / `.gif` → the narrative section it supports and its suggested slot. | Choosing figures for report / slides. |
| **`results_table.md`** / **`results_table.tex`** | Consolidated eight-row Phase 0/1/2 results table, auto-generated from JSON. | Main-body table in the report. |
| `results_summary.json` | Authoritative machine-readable record of every solver run. | Sanity-checking numbers; regenerating the table. |
| `REMEDIATION_PLAN.md` | Tranche-based plan (T1–T5) that produced the JSON + table + audit. | Historical / meta; not needed for the writeup. |
| `STORYLINE.md` | Earlier draft of the narrative, kept as historical reference. Superseded by `NARRATIVE.md`. | Rarely. |

---

## Repository layout

```
.
├── AAE568_Proposal.{pdf,tex}          Original project proposal.
├── AAE568_Technical_Documentation.docx   Technical doc draft.
├── AAE568_Controls_Aspects.docx       Controls-methodology writeup draft (paired with References/AAE568_Controls_Aspects.{tex,pdf}).
├── Artemis_II_OEM_2026_04_10_Post-ICPS-Sep-to-EI.asc   NASA post-flight OEM (truth for Phase 2).
│
├── NARRATIVE.md                       Locked narrative — read this before changing anything.
├── REPORT_PACK.md                     Writing-time cheatsheet of numerical claims.
├── FIGURES.md                         Figure → claim map.
├── REMEDIATION_PLAN.md                T1–T5 plan that produced the artifacts.
├── STORYLINE.md                       Superseded draft narrative.
├── results_summary.json               Authoritative solver-result records (27 rows).
├── results_table.{md,tex}             Consolidated Phase 0/1/2 table (auto-generated).
├── make_results_table.py              Regenerates results_table.{md,tex} from JSON.
│
├── common/                            Shared harness — schema + IO + timing.
│   ├── results_schema.py              ResultRecord dataclass (see §“Result record schema”).
│   ├── results_io.py                  Atomic save / append / load.
│   └── timing.py                      timed_solve() context manager + git_sha_or_none().
│
├── Earth-Mars/                        Phase 0 — two-body pipeline check.
│   ├── dynamics.py, bezier.py, shooting.py
│   ├── ipopt_collocation_2body.py     Bézier + IPOPT, exact AD.
│   ├── validate_two_body.py
│   ├── compare_methods.py             Side-by-side (shooting vs Bézier collocation).
│   ├── run_phase0.py                  Instrumented runner that writes to results_summary.json.
│   ├── validation_min_energy.png
│   ├── comparison_shooting_vs_bezier.png
│   └── comparison_animation.gif
│
├── Planer/                            Phase 1 — planar CR3BP L1↔L2 Lyapunov transfer.
│   ├── cr3bp_planar.py                Rotating-frame dynamics, Jacobi integral.
│   ├── cr3bp_transfer.py              Shooting + (legacy) Bézier collocation driver.
│   ├── ipopt_collocation.py           Bézier collocation + IPOPT with exact AD.
│   ├── bezier_segmented.py            Segmented-Bézier NLP harness (SLSQP).
│   ├── cr3bp_transfer_segmented.py    N-sweep driver.
│   ├── run_phase1.py                  Instrumented Phase 1 runner (PMP / IPOPT / N-sweep).
│   ├── make_pareto_plot.py            |J-J*| vs wall-time Pareto chart.
│   ├── make_jacobi_plot.py            ΔC(t) cross-check.
│   ├── nsweep_findings.md             N-sweep mesh-refinement study write-up.
│   ├── cr3bp_transfer_cislunar.png, cr3bp_transfer_comparison.png
│   ├── cr3bp_transfer_segmented_convergence.png, cr3bp_transfer_segmented_trajectories.png
│   ├── pareto_J_error_vs_wall_time.png, pareto_data.csv
│   ├── jacobi_constant.png
│   └── cr3bp_transfer_animation.gif
│
├── Artemis2/                          Phase 2 — Artemis II ephemeris validation.
│   ├── Ephem_Post_TLI/                Post-TLI ballistic arc solve + figures + animation.
│   │   ├── ephem_{2d_projections,3d_trajectory,error_comparison,summary_stats}.png
│   │   └── ephem_animation.gif
│   ├── Ephem_Full/                    Full-mission solve with burn-aware meshing.
│   │   ├── artemis2_ephemeris.py      Post-TLI IPOPT multi-shooting (instrumented).
│   │   ├── artemis2_full_mission.py   Full-mission IPOPT multi-shooting (instrumented).
│   │   ├── full_{2d_projections,3d_trajectory,control_profile,error_comparison}.png
│   │   └── artemis2_full_mission.gif
│   ├── run_phase2.py                  Instrumented Phase 2 runner (15-seed shooting + IPOPT).
│   ├── RETUNING_AUDIT.md              IPOPT option-by-option comparison P1 vs P2 (Option C).
│   ├── convergence_history.png        IPOPT per-iteration {obj, constr_viol, dual_inf}.
│   ├── AAE568_Project_Notes.{docx,pdf}  Phase 2 working notes.
│
├── ThreeD/                            FUTURE-WORK EXHIBIT ONLY (see ThreeD/README.md).
│   └── LEO→NRHO 3D prototypes — descoped from the final report narrative.
│
└── References/                        Supporting material (proposal, lecture-aligned notes).
    ├── AAE568_Controls_Aspects.{tex,pdf}   Controls-derivation writeup.
    ├── AAE568_S26_Project_Template.pdf
    ├── NRHO_transfer_sources.md, phase3_implementation_plan.md
    └── *.mlx                           Early MATLAB prototypes (not in the report path).
```

---

## Three-phase arc at a glance

| Phase | Folder | Problem | What it proves |
| --- | --- | --- | --- |
| **P0** | `Earth-Mars/` | Two-body Earth→Mars, low-thrust, min-energy | Pipeline check. Indirect and direct agree to nine significant figures. |
| **P1** | `Planer/` | Planar CR3BP, L1↔L2 Lyapunov, min-energy low-thrust | Accuracy-vs-cost tradeoff. Segmented Bézier matches PMP; SLSQP loses on wall-clock; IPOPT+AD recovers it. **Motivates the solver pivot.** |
| **P2** | `Artemis2/` | 3D Earth–Moon–Sun N-body (astropy builtin), Artemis II post-flight | IPOPT+AD tool *class* transfers to mission-relevant 3D dynamics. Tolerances and mesh are retuned; see `RETUNING_AUDIT.md`. |

The one-sentence thesis lives in `NARRATIVE.md §1`. The report conclusion should echo §3 Takeaway lines and §4.

---

## How to reproduce `results_summary.json`

Every locked number in `NARRATIVE.md §5` is emitted by a `run_phase*.py` script into `results_summary.json` via the `common/` harness.

```bash
# from repo root
python Earth-Mars/run_phase0.py     # Phase 0: two-body Earth→Mars (PMP + IPOPT)
python Planer/run_phase1.py         # Phase 1: shooting, IPOPT, N-sweep (long — SLSQP N=16 alone ~9 min)
python Artemis2/run_phase2.py       # Phase 2: 15-seed shooting + IPOPT post-TLI + IPOPT full-mission
python make_results_table.py        # regenerates results_table.{md,tex} from the JSON
```

Each solver call is wrapped by `common.timed_solve(...)` and appended to `results_summary.json` via `common.append_to_summary(...)`. Re-runs dedupe by `(phase, case, method, parameters_hash)` — "latest wins." See `common/results_io.py` for the policy.

### Result record schema (`common/results_schema.py`)

```python
ResultRecord(
    phase:              str,          # "0" | "1" | "2"
    case:               str,          # e.g. "planar_cr3bp_L1_L2_lyapunov"
    method:             str,          # "indirect_shooting" | "global_bezier_ipopt" | "segmented_bezier_slsqp" | "multi_shooting_ipopt"
    parameters:         dict,
    cost:               float,        # Bolza J = ∫|u|² dt (no ½ prefactor — see NARRATIVE §6.1)
    converged:          bool,
    residual:           float | None,
    wall_time_s:        float,
    iterations:         int | None,
    nfev, njev:         int | None,
    n_vars, n_constraints: int | None,
    git_sha:            str | None,
    timestamp:          str,          # UTC ISO-8601
    python_version:     str,
    convergence_history: list | None, # IPOPT per-iter trace when available
    notes:              str | None,
)
```

---

## Dependencies

From `requirements.txt`:

```
numpy
scipy
matplotlib
astropy
casadi
```

`casadi` ships IPOPT with the MUMPS linear solver; no separate IPOPT install is required. `astropy` is only used in Phase 2 for solar-system ephemeris (`get_body_barycentric_posvel`, `solar_system_ephemeris`); the default builtin analytical ephemeris is fine for this project — see `NARRATIVE.md §6.5` for the precision caveat.

Tested on Python 3.10 and 3.12.

---

## Contributing workflow (teammates)

- **Do not edit `NARRATIVE.md`** without syncing with Zhuorui first — it is frozen. The report writing follows the narrative, not vice versa.
- **Do not hand-edit `results_summary.json` or `results_table.{md,tex}`.** These regenerate from re-runs; edit the runner or regenerate the table.
- **If a new claim needs a new number:** add the run to the appropriate `run_phase*.py`, re-run, commit the JSON delta. The table rebuilds from there.
- Report / slide drafts that cite numbers should cite them with the JSON key, not a hand-transcribed copy. See `REPORT_PACK.md` for the pull-quote format.
- `ThreeD/` is a future-work exhibit; do not cite its figures in the report — see `ThreeD/README.md`.

---

## License

MIT — see `LICENSE`.
