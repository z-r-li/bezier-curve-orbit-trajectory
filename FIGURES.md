# FIGURES.md — Figure & Animation Catalogue

Every `.png` / `.gif` in the repo, mapped to what it shows, the `NARRATIVE.md` section it supports, and where it belongs in the written report or the presentation.

**Priority tier** follows `NARRATIVE.md §8.4` ("Figures that pull their weight"):

- **A — must include** (the chart makes a locked claim visible)
- **B — strong support** (reinforces a claim, include if space permits)
- **C — nice to have** (illustrative; use in slides, drop from the written report if tight on space)
- **exhibit** — do **not** cite in the report; in the repo for historical / future-work reasons only

## Phase 0 — Two-body Earth→Mars (`Earth-Mars/`)

| File | Shows | Narrative section | Priority | Report slot | Slide slot |
| --- | --- | --- | --- | --- | --- |
| `validation_min_energy.png` | Indirect shooting trajectory for the 2-body min-energy Earth→Mars transfer. | §3 P0, §5.1 | B | Phase 0 figure | P0 single slide |
| `comparison_shooting_vs_bezier.png` | Shooting vs Bézier collocation overlaid — visual confirmation that the two agree trajectory-for-trajectory. | §3 P0 ("agree to nine significant figures"), §5.1 | A | Phase 0 figure | P0 single slide |
| `comparison_animation.gif` | Motion version of the side-by-side comparison. | §3 P0 | C | — | P0 closing slide (optional) |

## Phase 1 — Planar CR3BP L1↔L2 Lyapunov (`Planer/`)

| File | Shows | Narrative section | Priority | Report slot | Slide slot |
| --- | --- | --- | --- | --- | --- |
| `pareto_J_error_vs_wall_time.png` | `\|J − J*\|` vs wall time: SLSQP-segmented line over `N ∈ {1, 2, 4, 8, 16}` plus IPOPT single point plus PMP anchor. **The single chart that makes the method-comparison argument quantitative.** | §3 P1, §5.2 | **A** | Main results figure | Opening slide of Results |
| `cr3bp_transfer_comparison.png` | Three methods (shooting, IPOPT, segmented SLSQP) overlaid on the same L1↔L2 trajectory in the rotating frame. | §3 P1 ("agreement"), §5.2 | **A** | Phase 1 trajectory panel | Right after the Pareto |
| `cr3bp_transfer_segmented_convergence.png` | Monotone-decreasing cost vs `N` for the SLSQP N-sweep — the mesh-refinement story visually. | §3 P1, §5.2 "cost scaling", `Planer/nsweep_findings.md` §2 | **A** | Phase 1 mesh-refinement panel | N-sweep slide |
| `cr3bp_transfer_segmented_trajectories.png` | Segmented trajectories at each `N`, showing visual convergence to the shooting reference. | `Planer/nsweep_findings.md` §2 | B | Appendix (mesh sweep) | — |
| `jacobi_constant.png` | `ΔC(t)` trace for all three methods — dynamics-integrity cue (caveat: `C` is not conserved under thrust). | §3 P1 "Jacobi-constant cross-check", §5.2 | B | Methods / dynamics-integrity figure | Dynamics-integrity slide (optional) |
| `cr3bp_transfer_cislunar.png` | Overview plot of the Earth–Moon rotating frame and both Lyapunov orbits with the transfer. | §3 P1 (context) | B | Phase 1 setup figure | Setup slide |
| `cr3bp_transfer_animation.gif` | Animated L1↔L2 transfer in the rotating frame. | §3 P1 | C (slides only) | **not in report** | Closing-of-talk slide |

## Phase 2 — Artemis II ephemeris validation (`Artemis2/`)

### Post-TLI coast arc (`Artemis2/Ephem_Post_TLI/`)

| File | Shows | Narrative section | Priority | Report slot | Slide slot |
| --- | --- | --- | --- | --- | --- |
| `ephem_3d_trajectory.png` | 3D inertial-frame trajectory for the post-TLI ballistic coast arc, IPOPT result vs NASA OEM truth. | §3 P2, §5.3 "post-TLI" | A | Phase 2 post-TLI figure | Phase 2 intro slide |
| `ephem_2d_projections.png` | XY / XZ / YZ projections of the same arc — useful for showing out-of-plane structure. | §5.3 | B | Phase 2 appendix figure | — |
| `ephem_error_comparison.png` | Position / velocity error between IPOPT multi-shooting result and OEM truth vs time. | §5.3 ("residual `3.5e-10`"), §8.3 | B | Phase 2 validation figure | Validation slide |
| `ephem_summary_stats.png` | Rolled-up Post-TLI summary panel (iters, residual, cost, wall-time). | §5.3 | C | — | Recap slide |
| `ephem_animation.gif` | Post-TLI arc, animated. | §3 P2 | C (slides only) | — | Post-TLI closing slide |

### Full mission (`Artemis2/Ephem_Full/`)

| File | Shows | Narrative section | Priority | Report slot | Slide slot |
| --- | --- | --- | --- | --- | --- |
| `full_3d_trajectory.png` | Full-mission 3D trajectory (day 0 → 8.9) with detected "burn" windows highlighted. | §3 P2, §5.3 "full mission" | A | Phase 2 full-mission figure | Full-mission slide |
| `full_2d_projections.png` | Three orthogonal projections of the full mission. | §5.3 | B | Appendix | — |
| `full_control_profile.png` | Reconstructed `u(t)` on the full mission — shows where the NLP places thrust. | §5.3, §8.3 (**verify ΔV numbers before including!**) | B | Methods figure (captioned carefully) | Discussion slide |
| `full_error_comparison.png` | IPOPT result vs OEM truth, position/velocity residuals vs time across the full mission. | §5.3, §8.3 | B | Validation figure | Validation slide |
| `artemis2_full_mission.gif` | Full mission animated. | §3 P2 | C (slides only) | — | Talk closer |

### Convergence audit (`Artemis2/`)

| File | Shows | Narrative section | Priority | Report slot | Slide slot |
| --- | --- | --- | --- | --- | --- |
| `convergence_history.png` | IPOPT per-iteration trace of objective, primal constraint violation, and dual infeasibility for the full mission — demonstrates the 4-iter interior-point convergence. | §3 P2 ("4 iter / 1.68 s"), §5.3, §8.4 | A | Phase 2 convergence figure | IPOPT convergence slide |

## Exhibit / descoped — `ThreeD/`

Do **not** cite any of these figures in the report or presentation. They are kept in the repo per `ThreeD/README.md` as a future-work exhibit for a LEO→NRHO insertion that was cut when Phase 2 was scoped to the Artemis II OEM validation.

| File | Shows |
| --- | --- |
| `2d_projections.png`, `3d_trajectory.png`, `control_profile.png`, `summary_stats.png` | CR3BP 3D LEO→NRHO prototype outputs. |
| `ephem_nrho_3d.png`, `ephem_nrho_dv_comparison.png`, `ephem_nrho_primer.png` | Ephemeris-based LEO→NRHO prototype outputs (primer-vector diagnostic is partial; no locked claims). |

## Quick-pick priority list (top 6 for a 10-minute talk)

Per `NARRATIVE.md §8.4`, these six images carry the story; cut anything else to stay under slide time:

1. `Planer/pareto_J_error_vs_wall_time.png` — the single most important chart.
2. `Planer/cr3bp_transfer_comparison.png` — visual confirmation of agreement.
3. `Planer/cr3bp_transfer_segmented_convergence.png` — mesh-refinement story.
4. `Planer/jacobi_constant.png` — dynamics integrity (with caveat).
5. `Artemis2/convergence_history.png` — IPOPT's 4-iter convergence on the full mission.
6. `Planer/cr3bp_transfer_animation.gif` — closer of the talk; cut from the written report.

---

All figures above are produced by the corresponding `run_phase*.py` and `make_*_plot.py` scripts; see the `common/` harness and each folder's scripts for the wiring.
