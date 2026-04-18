# Project Storyline — AAE 568, Spring 2026

**Title:** Bézier Curve Collocation for Optimal Trajectory Design in Cislunar Space: A Comparison with the Shooting Method

**Scope (finalized 2026-04-16):** CR3BP method comparison + Artemis II ephemeris validation. The LEO→NRHO ephemeris extension (Phase 3) was cut after Bézier NLP failed to converge under a straight-line warm-start through the Moon keep-out zone; it is retained in the repository but relegated to *Scope and Limitations* in the report.

**Reviewer context:** Course project for AAE 568, not a peer-reviewed submission. The proposal scoped the group and the topic but the details are flexible per Dr. Hwang. The standard for "good enough" is: Bolza formulation correct, Hamiltonian derived, indirect shooting converges, direct transcription converges, comparison is articulated. Conference/journal standards (exhaustive basin-of-convergence sweeps, paper-grade tables, full notation consistency) are *not* the bar.

---

## 1. Narrative arc

The report moves through four increasingly realistic settings, each of which isolates a different lesson from the course:

**Act I — Validation on a textbook problem (`Earth-Mars/`).**
A 2D two-body Earth→Mars minimum-energy transfer. Both methods (indirect shooting via Pontryagin; direct Bézier collocation via IPOPT) converge, produce the same trajectory and control history, and give identical cost to within numerical noise. Pedagogical purpose: *establish that Bézier collocation is a valid alternative to shooting on a problem where we can check the answer by inspection*. This is the "proof the pipeline works" act.

**Act II — Method comparison on nonlinear dynamics (`Planer/`).**
A planar CR3BP L1↔L2 Lyapunov orbit transfer. Same two methods, but now the dynamics are nonlinear and the shooting method's sensitivity to the initial costate guess becomes visible. Bézier collocation's convex hull property and local defect constraints produce a wider basin of convergence than shooting. Pedagogical purpose: *show the phenomenon that motivates direct collocation — indirect methods are mathematically elegant but numerically fragile on realistic dynamics*.

**Act III — Scaling to 3D (`ThreeD/leo_to_nrho_cr3bp.py`).**
3D CR3BP LEO→NRHO transfer. Here the NLP-based approach switches from a Bézier parameterization to **IPOPT direct multiple-shooting** (piecewise-constant controls + RK4 per segment). The reason for the switch is scale: a Bézier NLP on a 6D state with Moon keep-out inequalities at tight collocation spacing becomes large and ill-conditioned; multiple-shooting retains the direct-transcription philosophy while trading the global Bézier basis for local RK4 blocks. Indirect shooting is still run and still compared. Pedagogical purpose: *illustrate that "direct transcription" is a family of methods, and the choice between Bézier and multiple-shooting is a modeling decision driven by problem size and conditioning*.

**Act IV — Ephemeris validation against real flight data (`Artemis2/`).**
The same 3D infrastructure is promoted to Earth–Moon–Sun N-body dynamics via JPL ephemeris (astropy's builtin tables) and applied to the Artemis II TLI→Moon→return trajectory, which actually flew 1–11 April 2026. Indirect shooting and IPOPT multi-shooting both run, both reproduce the NASA OEM post-flight trajectory, and both recover the TLI burn at the correct epoch. Pedagogical purpose: *close the loop — the control techniques developed in idealized CR3BP scale to realistic ephemeris dynamics and match ground truth on a freshly flown mission*.

**Epilogue — Scope and Limitations (`ThreeD/leo_to_nrho_ephem.py`).**
An attempt to extend the Act III methodology to ephemeris LEO→NRHO. Outer phasing sweep (Lambert-based Stage 1 grid + Nelder-Mead Stage 2) recovers a realistic 3.98 km/s two-body baseline at (RAAN=182.18°, ν=2.87°, NRHO phase=0.9805). Newton shooting converges to 4.26 km/s under full ephemeris. IPOPT Bézier fails to converge from a straight-line warm start because interior control points violate the Moon keep-out inequality. Future work: seed the Bézier control points from a shooting solution via least-squares Bernstein fit.

---

## 2. Mapping phases to AAE 568 course material

The companion document `References/AAE568_Controls_Aspects.pdf` already threads the narrative through the course material (Bolza formulation → Hamiltonian/Euler–Lagrange → PMP → TPBVP → shooting → direct collocation → mesh refinement → homotopy). The project output supplies numerical instances of each step:

| Course concept | Where it appears in the code |
|---|---|
| Bolza problem with min-energy Lagrangian | All four acts |
| Hamiltonian + costate dynamics | `Earth-Mars/shooting.py`, `Planer/cr3bp_transfer.py` (`shooting_cr3bp_min_energy`), `ThreeD/leo_to_nrho_cr3bp.py` (`shooting_residual`), `Artemis2/artemis2_ephemeris.py` (`solve_shooting`) |
| PMP optimal control law u*=−½λᵥ | Every indirect-shooting routine |
| TPBVP + `scipy.fsolve` | Every indirect-shooting routine |
| Initial guess problem | Planar CR3BP failure modes; Artemis2 multi-guess sweep (15 seeds) |
| Direct collocation via NLP | `Earth-Mars/ipopt_collocation_2body.py`, `Planer/ipopt_collocation.py`, `ThreeD/leo_to_nrho_cr3bp.py` (`solve_ipopt_cascade`) |
| Mesh refinement | `ThreeD/leo_to_nrho_cr3bp.py:solve_ipopt_cascade` |
| Transversality for free-t_f | (Not instantiated — all problems use fixed t_f. Worth flagging in report.) |

---

## 3. Existing outputs (what's already rendered)

Earth–Mars:
`comparison_shooting_vs_bezier.png`, `validation_min_energy.png`

Planar CR3BP:
`cr3bp_transfer_cislunar.png`, `cr3bp_transfer_comparison.png`

3D CR3BP LEO→NRHO:
`3d_trajectory.png`, `2d_projections.png`, `control_profile.png`, `summary_stats.png`

Artemis II full mission:
`full_3d_trajectory.png`, `full_2d_projections.png`, `full_control_profile.png`, `full_error_comparison.png`

Artemis II post-TLI leg:
`ephem_3d_trajectory.png`, `ephem_2d_projections.png`, `ephem_error_comparison.png`, `ephem_summary_stats.png`, `ephem_animation.gif`

Ephemeris LEO→NRHO (epilogue only):
`ephem_nrho_3d.png`, `ephem_nrho_dv_comparison.png`, `ephem_nrho_primer.png`

---

## 4. Gaps and implementation concerns

These are things I notice that could weaken the report or make reviewers pause. Each is a candidate next-step, not a blocker.

### 4.1 Bézier → multi-shooting is a deliberate methodological progression, not drift

**The observation.** Bézier parameterization in the strict sense (global Bernstein basis, control-point variables, derived boundary derivatives) appears only in `Earth-Mars/` and `Planer/`. The 3D CR3BP transfer and the Artemis II phases use **IPOPT direct multiple-shooting** — state-at-nodes decision variables and piecewise-constant controls, integrated by RK4 per segment. Both sit in the "direct transcription" family, but they parameterize different things: Bézier parameterizes the *path* (position control points); multi-shooting parameterizes *node states* and integrates between them.

**Why this is the right framing for the report.** Global polynomial methods on long intervals are known to struggle with conditioning and Runge-type effects (Betts, *Practical Methods for Optimal Control*, Ch. 4). The standard remedy in both CFD and trajectory optimization is *segmentation*: split the time interval, use a local polynomial on each segment, enforce continuity and dynamics at interior nodes. Direct multi-shooting is a particular choice inside that family, where each segment's polynomial is replaced by an ODE integrator. Pseudospectral methods (Radau, Gauss-Lobatto) are another.

The report can tell this story cleanly as a **progression**:

  - *Act I (2D 2-body).* Problem is small and smooth; global Bézier collocation converges cleanly. Bézier's convex-hull property gives a wider basin of convergence than shooting. Win for direct.
  - *Act II (planar CR3BP).* Still small enough for global Bézier; now the shooting method is visibly fragile on CR3BP nonlinearity. Same win for direct, amplified.
  - *Acts III–IV (3D, 6D state, longer arcs, ephemeris).* Attempting global Bézier on the full transfer becomes ill-conditioned — the LEO→NRHO ephemeris run in the epilogue demonstrates this empirically. Segmenting the time interval and using multiple-shooting is the natural remedy (analog: mesh refinement in CFD). Methodology scales; parameterization changes.

Framed this way, the ephemeris non-convergence is **data supporting the methodological choice**, not a limitation to hide. "We started with global Bézier because it has the cleanest theoretical story; we moved to segmented / multi-shooting form for the larger problems because the global method didn't converge — and here's the result that demonstrates why."

**Optional enhancement (if time permits).** A literal *segmented Bézier* implementation as a bridge between Acts II and III would close the methodological arc with maximum clarity. Partition [0, T] into N intervals, use a degree-d Bernstein polynomial on each, enforce C¹ continuity of r, v at interior boundaries, and enforce dynamics defects at per-segment collocation nodes. Run it on the planar CR3BP L1↔L2 problem for side-by-side comparison with the global Bézier result. One day of work wrapping the existing `BezierCollocation` class in a segment loop. This is pseudospectral h-p collocation and is a named method (GPOPS-II style) — not a toy exercise.

### 4.2 Basin-of-convergence plot is missing

The central pedagogical claim — *"shooting is sensitive to initial costate; direct transcription tolerates bad initial guesses"* — is narrated in `AAE568_Controls_Aspects.tex §5.1` and supported anecdotally by the multi-guess logic in `Artemis2/artemis2_ephemeris.py:solve_shooting` (15 random seeds), but it is not quantified in any plot.

**What's needed:** a sweep over initial-costate magnitude (or initial trajectory perturbation for direct methods), recording convergence / residual. Even a small sweep — 50 random seeds, scatter plot of (seed norm, converged?) for both methods on the planar CR3BP problem — would make the argument visible rather than rhetorical. Estimated effort: half a day, mostly wrapping existing `solve_shooting` in a loop.

### 4.3 Wall-clock / iteration-count comparison table

Several scripts print timing ("residual=..., time=0.17s, nfev=23") but there's no consolidated table in the report.

**What's needed:** a single table with rows = {Earth-Mars, Planar CR3BP, 3D CR3BP, Artemis II TLI, Artemis II full mission} and columns = {method, wall-clock, iterations, converged ΔV, NLP variables}. This is purely a reporting step — data is already in stdout logs — but makes the report quantitative instead of narrative.

### 4.4 Jacobi constant / energy conservation plot (CR3BP cases)

For the planar CR3BP and 3D CR3BP phases, a short plot of Jacobi constant along the transfer under both methods would confirm solution quality beyond endpoint matching. Usually one line per method, very cheap, included in most published CR3BP papers. Not present in `cr3bp_transfer_*.png`.

### 4.5 Primer vector / Lawden's conditions

Mentioned in the course-aspects doc at §5 but not shown in plots. For the impulsive Artemis II case the primer vector check would add a rigorous "is the two-burn solution actually optimal?" diagnostic. The existing `primer_vector_analysis` in `leo_to_nrho_ephem.py` uses a linear interpolation that's vacuous; would need upgrading to the Lion–Handelsman 3×3 STM-block construction for a real plot. Lower priority than 4.1–4.3.

### 4.6 Transversality for free t_f is theorized but not instantiated

`AAE568_Controls_Aspects.tex §2.4` develops the transversality condition for free final time, but all four acts use **fixed** transfer duration. The report claims the theory applies; the code doesn't exercise it. Either:
  - Acknowledge in writing that free-t_f is out of scope for this project, or
  - Add a short homotopy study (e.g., Artemis II TLI arc with t_f as a free variable and an added Hamiltonian = 0 transversality constraint).

Recommendation: one paragraph acknowledgement in §7 Conclusion. Adding free t_f is not worth the effort for a course deadline.

### 4.7 Mesh refinement cascade is not visualized

`leo_to_nrho_cr3bp.py:solve_ipopt_cascade` actually runs a coarse→medium→fine cascade, but the output PNGs don't show the cascade — they show the final-stage trajectory. A 2×2 subplot with "coarse solution, medium, fine, final" would make the mesh refinement argument visible. Probably half a day, using the intermediate solutions already computed inside the cascade.

### 4.8 The ephemeris LEO→NRHO Bezier implementation

As flagged earlier, this file is kept in the repo but the Bezier case doesn't converge. If it stays in the report, the "Future work" paragraph should explicitly describe the shooting-seeded Bernstein fit as the remediation — otherwise a skeptical reader will read the non-convergence as a methodological failure of Bezier rather than a warm-start engineering issue.

---

## 5. Priority ordering for remaining time

Given the course-project context (not a paper submission), most items in §4 are nice-to-haves rather than must-haves. A defensible minimum path to a strong report is:

  1. **Write the §4.1 progression narrative into the Introduction and the Methods section** (half-hour). This is the single most important fix — not because the current code is wrong, but because the report's value rises substantially when the Bézier → multi-shooting transition is explicitly framed as methodological reasoning ("we segmented for the same reason CFD segments: global methods on large intervals are ill-conditioned") rather than left implicit.
  2. **Add a one-paragraph *Scope and Limitations* section** acknowledging free-t_f transversality (theorized, not instantiated) and the ephemeris LEO→NRHO Bézier non-convergence (reframed per §4.1 as supporting the segmentation argument). One hour.
  3. **Consolidated comparison table** (§4.3). Numbers are already in the stdout logs; it's a transcription task. One hour.
  4. **Jacobi constant conservation plot for the CR3BP phases** (§4.4). Trivial — adds a solution-quality checkmark that costs almost nothing.

Stretch goals (do only if time allows):

  5. Basin-of-convergence sweep on the planar CR3BP case (§4.2). Strong visual for the Bézier-vs-shooting claim, half a day.
  6. Implement segmented Bézier on the planar CR3BP problem as a methodological bridge (§4.1 optional enhancement). One day, highest narrative reward.
  7. Mesh refinement cascade visualization (§4.7). The cascade already runs; this just harvests intermediate solutions.

Items 1–4 alone produce a complete, internally consistent report. Items 5–7 turn it into a polished one. Dropped-from-must-have (relative to the earlier version of this doc): the title-rename option, the Bézier-in-Artemis-II add, and most of the quantitative-methodology hardening — all of those were grading against a paper standard, not a course standard.
