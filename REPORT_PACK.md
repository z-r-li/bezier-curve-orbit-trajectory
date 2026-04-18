# REPORT_PACK.md — Writing-Time Cheatsheet

Pull-quote-ready numerical claims for the AAE 568 final report and presentation. Every number below is loaded from `results_summary.json`; every statement of fact is anchored in `NARRATIVE.md` (the locked storyline). When you copy a number into the LaTeX template, cite it from the JSON record (`phase`, `case`, `method`) so the next person can re-derive it.

> Conventions used throughout this file:
> - **`J`** is the Bolza min-energy objective `∫ |u|² dt` (no ½ prefactor) — see `NARRATIVE.md §6.1`.
> - **Wall time** is single-core laptop, captured by `common.timed_solve(...)`.
> - **Iterations** are NLP outer iterations except where labelled `nfev` (function evaluations from SciPy `fsolve`).
> - **Source: JSON row** points at the unique `(phase, case, method[, parameters])` key in `results_summary.json`.
> - **Narrative §** points at the section in `NARRATIVE.md` that locks the framing.

---

## 0. The thesis sentence (use verbatim)

> "A Bézier-segment direct-transcription pipeline is validated against PMP on tractable problems, exposes a scalability limit that motivates IPOPT multi-shooting, and transfers to a mission-relevant Artemis II ephemeris problem with the same tool *class* (IPOPT + automatic differentiation + warm-start-from-reference) — tolerances, mesh schedule, and transcription form are retuned for dimensional 3D dynamics and enumerated in `Artemis2/RETUNING_AUDIT.md`."

— `NARRATIVE.md §1`. The conclusion (§4) and the talk's opening should restate this.

---

## 1. Phase 0 — Earth → Mars (pipeline check)

**Locked claim:** Indirect shooting and direct Bézier collocation agree to nine significant figures.

| Metric | Indirect shooting | Bézier collocation + IPOPT |
| --- | --- | --- |
| Cost J | **1.49004478 × 10⁻²** | **1.49004452 × 10⁻²** |
| Residual | 1.32 × 10⁻¹² | 2.22 × 10⁻¹⁵ |
| Wall time | 0.85 s | 1.22 s |
| Iterations | 24 (`nfev`) | 2 |
| NLP variables | 4 | 896 |
| Method label (JSON) | `indirect_shooting` | `global_bezier_ipopt` (N=16) |

**The headline:** `|ΔJ| ≈ 2.6 × 10⁻⁹` — agreement to the ninth decimal. This is the "machine-tolerance agreement" claim in `NARRATIVE.md §3`.

- Sources (JSON rows): `phase=0`, `case=earth_mars_2body`, `method ∈ {indirect_shooting, global_bezier_ipopt}`.
- Narrative: §3 P0, §5.1.
- Figures: `Earth-Mars/comparison_shooting_vs_bezier.png` (priority A), `Earth-Mars/validation_min_energy.png` (priority B).

---

## 2. Phase 1 — Planar CR3BP L1↔L2 Lyapunov transfer

### 2.1 PMP anchor

**`J* = 4.306 × 10⁻²`, wall time = 6.02 s, 95 `nfev`.** This is the accuracy axis baseline.

- Source: `phase=1`, `case=planar_cr3bp_L1_L2_lyapunov`, `method=indirect_shooting`.
- Narrative: §3 P1, §5.2 first bullet.

### 2.2 Bézier collocation + IPOPT (warm-started from shooting)

| Metric | Value |
| --- | --- |
| Cost J | **4.323 × 10⁻²** (+0.4% vs J\*) |
| Residual | 4.97 × 10⁻¹⁴ |
| Wall time | **2.0 s** |
| Iterations | 5 |
| NLP variables / constraints | 896 / 836 |

- **Cold-start caveat (lock this in the report):** Without warm-start from shooting, IPOPT lands in a non-PMP basin at `J = 0.0878`. The warm-start is **methodology**, not a tuning trick — see `NARRATIVE.md §2.2` and §6.4.
- Source: `phase=1`, `case=planar_cr3bp_L1_L2_lyapunov`, `method=global_bezier_ipopt`. (`global_bezier_ipopt` is a legacy label — actually segmented collocation; see §3 P1.)
- Narrative: §3 P1, §5.2.

### 2.3 Segmented Bézier + SLSQP — N-sweep (the mesh-refinement story)

| N | Converged | Iterations | Wall time (s) | Cost J | `nfev` |
| --- | --- | --- | --- | --- | --- |
| 1 | ✓ | 21 | 0.10 | 7.181 × 10⁻¹ | 863 |
| 2 | ✓ | 169 | 3.17 | 1.281 × 10⁻¹ | 14 552 |
| 4 | ✓ | 198 | 15.17 | 9.013 × 10⁻² | 34 488 |
| 8 | ✓ | 140 | 49.50 | 5.001 × 10⁻² | 48 977 |
| 16 | ✓ | 294 | **528.86** | **4.324 × 10⁻²** | 206 417 |
| 32 | timeout > 900 s | — | — | — | — |

**The 88× headline number:** SLSQP at `N=16` is **528.9 s vs 6.02 s** for shooting, i.e. **~88× slower**. Older drafts (`Planer/nsweep_findings.md` §2) reported "87×" using `467.6 s / 5.33 s` — both are valid as long as numerator and denominator come from the same run; pick one and stick with it. **Recommended:** use 88× from `results_summary.json`.

- Source: `phase=1`, `case=planar_cr3bp_L1_L2_lyapunov`, `method=segmented_bezier_slsqp`, `parameters.N_segments ∈ {1,2,4,8,16}`. The `N=32` row is **not** in the JSON because it timed out (`NARRATIVE.md §8.3`).
- Narrative: §3 P1, §5.2 "Segmented Bézier + SLSQP N-sweep".
- Figures: `Planer/cr3bp_transfer_segmented_convergence.png` (A), `Planer/pareto_J_error_vs_wall_time.png` (A — the master chart), `Planer/cr3bp_transfer_comparison.png` (A).
- Cost-scaling sentence: "Wall clock grows ~10× per doubling of N; cost gap to J\* contracts geometrically." (§5.2)

### 2.4 Jacobi-constant cross-check (dynamics integrity)

All three methods compute the same `ΔC(t)` trajectory to within `O(10⁻²)` — confirming they converge to a common controlled path. **Caveat (lock this caption):** under thrust, `C` is not conserved; the `O(10⁻²)` trace is dominated by work done by `u*(t)`, **not** method error. Do not upgrade the figure caption beyond that. (`NARRATIVE.md §5.2`, §8.6.)

- Figure: `Planer/jacobi_constant.png`.

### 2.5 Phase 1 Takeaway (use verbatim)

> "Segmented direct transcription is correct as a transcription family — the solver *pairing* is what breaks: SLSQP driven by dense finite-difference gradients scales as ≈ 10× per doubling of N on this problem and is impractical past N=16 on a laptop budget. Interior-point NLP with exact (AD) derivatives is the appropriate replacement; no claim is made about SLSQP with analytic gradients or sparse Jacobians."

— `NARRATIVE.md §3 P1 Takeaway`. Used as the bridge into Phase 2.

---

## 3. Phase 2 — Artemis II ephemeris validation

> Truth data: NASA OEM for Artemis II, flown **2026-04-02 through 2026-04-10 UTC**. File: `Artemis_II_OEM_2026_04_10_Post-ICPS-Sep-to-EI.asc`. The OEM starts post-ICPS-separation, i.e. *after* TLI — neither detected burn in the analysis window is the TLI itself. (`NARRATIVE.md §5.3`, §6.5.)

### 3.1 Shooting baseline — 15-seed multi-start sweep

- **Sweep total:** 2033 s of compute over 15 seeds.
- **Convergences below the `1e-4` early-stop threshold:** **1 / 15** (seed = random-normal, scale `1e-6`). Result: `J = 6.21 × 10⁻⁷`, residual `6.2 × 10⁻⁶`, 130 s for that seed.
- **Honest framing (use verbatim):** "The 14 failures ran against a per-seed `maxfev=80` ceiling — budget-constrained, not a basin proof, which is a subtle but honest point for the report." (`NARRATIVE.md §5.3`.)
- Sources: `phase=2`, `case=artemis2_post_tli`, `method=indirect_shooting` — 15 individual rows + 1 best-of-15 rollup row.

### 3.2 IPOPT multi-shooting — Post-TLI coast arc

| Metric | Value |
| --- | --- |
| Cost J | **5.30 × 10⁻¹²** (numerically zero — ballistic coast, `u ≡ 0` is optimal) |
| Residual | 3.49 × 10⁻¹⁰ |
| Wall time | **0.81 s** |
| Iterations | **2** |
| NLP variables / constraints | 906 / 612 |

- **Why it's so fast (lock this caveat):** "this count reflects a near-feasible warm start, not a from-scratch solve: the NASA OEM states are close enough to the NLP manifold that IPOPT polishes rather than searches. The speed is a property of the warm-start quality, not a generic IPOPT claim." (`NARRATIVE.md §5.3`.) Do not state "0.81 s on a 7.5-day arc!" without this sentence.
- **Endpoint residual = 0.000 km is not a validation quantity** — it's a hard equality constraint. The validation quantity is interior-state defect + objective driving to numerical zero. (`NARRATIVE.md §5.3`.)
- Source: `phase=2`, `case=artemis2_post_tli`, `method=multi_shooting_ipopt`.

### 3.3 IPOPT multi-shooting — Full mission (day 0 → 8.9)

| Metric | Value |
| --- | --- |
| Cost J | **9.44 × 10⁻⁴** |
| Residual | 1.92 × 10⁻¹⁰ |
| Wall time | **1.68 s** |
| Iterations | **4** |
| NLP variables / constraints | 2 031 / 1 362 |
| Mesh | 225 segments, 2-min spacing across burn windows (non-uniform) |

- Same "warm-start quality, not generic IPOPT" caveat as §3.2 applies to the 4-iter count.
- **Burn relabel (use verbatim — the heuristic mislabels TLI):**
  - **Burn 1** — days 0.874–0.945, likely the first Orion outbound trajectory correction (OTC-1 or equivalent).
  - **Burn 2** — days 8.88–8.91, a return-phase correction maneuver (not entry interface).
- **Do not report** the heuristic's `total_dv_km_s = 22.59 / 9.72 km/s` numbers as physical ΔV. They are `Σ|v_{i+1} − v_i|` summed over each detection window — an integrated `|a_grav + a_thrust|` magnitude, dominated by gravity. Apollo's TLI was ~3.05 km/s. **Verify before any slide quotes these** (`NARRATIVE.md §5.3`, §8.3).
- Source: `phase=2`, `case=artemis2_full_mission`, `method=multi_shooting_ipopt`.

### 3.4 Shooting-vs-IPOPT headline

The raw 2 033 s vs 0.81 s ratio is **~2 500×**. Use this only with the budget caveat:

> "It is a budget-constrained comparison, not a fundamental speedup: shooting's per-seed `maxfev` was capped at 80 (vs 2 000 in P1) to keep the 15-seed sweep tractable on 7.5-day 3D arcs, and 14/15 seeds exhausted that budget. A budget-matched comparison would give a smaller ratio. The qualitative claim — shooting is basin-fragile on long 3D arcs, IPOPT with AD is not — is robust to that framing; the specific 2 500× number is not."

— `NARRATIVE.md §5.3`. Do not cite `2500×` without the caveat sentence next to it.

### 3.5 "Carries over" framing (Option C — final wording)

> "IPOPT with AD is the right *class* of tool for both P1 and P2; the solver architecture (MUMPS, exact derivatives, warm-start from reference) carries over unchanged. Problem-specific tolerances, mesh schedule, transcription form, and control bounds are retuned for dimensional 3D dynamics — enumerated in `Artemis2/RETUNING_AUDIT.md`. Pedagogical circle closed on tool *class*, not verbatim settings."

— `NARRATIVE.md §3 P2 Takeaway`. **Do not say "converges without re-tuning"** — that wording was retired by `RETUNING_AUDIT.md` in favour of Option C above.

---

## 4. Consolidated table (matches `results_table.md`)

| Phase / Case | Method | NLP vars | Iterations | Wall time (s) | Cost J |
| --- | --- | --- | --- | --- | --- |
| Earth→Mars (2-body) | Indirect shooting | 4 | 24 (nfev) | 0.85 | 0.0149 |
| Earth→Mars (2-body) | Global Bézier + IPOPT | 896 | 2 | 1.2 | 0.0149 |
| Planar CR3BP (L1↔L2) | Indirect shooting | 4 | 95 (nfev) | 6.0 | 0.0431 |
| Planar CR3BP (L1↔L2) | Global Bézier + IPOPT | 896 | 5 | 2.0 | 0.0432 |
| Planar CR3BP (L1↔L2) | Segmented Bézier + SLSQP (N=16) | 700 | 294 | 529 | 0.0432 |
| Artemis II Post-TLI | Indirect shooting (best of 15) | 6 | 1 236 (nfev) | 2.0 × 10³ | 6.21 × 10⁻⁷ |
| Artemis II Post-TLI | IPOPT multi-shooting | 906 | 2 | 0.81 | 5.30 × 10⁻¹² |
| Artemis II Full Mission | IPOPT multi-shooting | 2 031 | 4 | 1.7 | 9.44 × 10⁻⁴ |

For the `.tex` version drop-in, use `results_table.tex` directly.

---

## 5. AAE 568 lecture-content traceability (one-liner per topic)

Every methodological claim traces to a specific `AAE568-Lecture_Notes.pdf` slide range (Prof. Hwang, Spring 2026). Full mapping in `NARRATIVE.md §7`. Quick crib:

- Calculus of variations → pp. 144 → motivates augmented cost.
- Hamiltonian + variation of `J_a` → pp. 145–146, 150 → `H = |u|² + λᵀ(f + Bu)`.
- Euler–Lagrange → p. 147 → state/costate ODEs.
- TPBVP formulation → pp. 148–149 → P0 + P1 shooting code.
- PMP → pp. 154–157 → `u* = -½ λ_v` (unconstrained, no bang–bang).
- Numerical TPBVP / shooting → pp. 190–191 → `fsolve` driver, then direct transcription pivot.
- Constrained optimization (Lagrangian, KKT) → pp. 123–138 → SLSQP / IPOPT background.
- Inequality constraints on control → pp. 161–166 → Phase 2 full-mission `|u| ≤ u_max`.

**Deliberately not used (cite as "future work" if asked):** DP / HJB, LQR / ARE, MPC, Kalman filtering. (`NARRATIVE.md §7`.)

---

## 6. Pre-submission checklist (drawn from `NARRATIVE.md §8`)

- ☐ The thesis sentence (§0 above) appears verbatim in the introduction.
- ☐ The "88× slower" number uses the `(528.9 s, 6.02 s)` pair, not mixed with `nsweep_findings.md`'s `(467.6 s, 5.33 s)` pair.
- ☐ The `2 500×` Shooting-vs-IPOPT ratio is never stated without the budget caveat (§3.4 above).
- ☐ "IPOPT *class* carries over, settings don't" replaces any draft phrasing about "no re-tuning needed" (§3.5 above).
- ☐ The full-mission ΔV numbers `22.59 km/s` and `9.72 km/s` are **not** in the report unless they have been independently recomputed as physical-thrust `|∫a_thrust dt|` (`NARRATIVE.md §5.3`, §8.3).
- ☐ The Post-TLI `J = 5.3 × 10⁻¹²` is annotated as "ballistic coast → optimum is `u ≡ 0`" so it doesn't read as suspiciously tight convergence.
- ☐ Burn labels follow the §3.3 relabel (OTC-1, return-phase correction); the heuristic's "TLI" label is not used.
- ☐ Jacobi-constant figure caption stays bounded — "method agreement on a common controlled arc," not a precision diagnostic.
- ☐ `RETUNING_AUDIT.md` table goes in an appendix, not the methods body.
- ☐ Cold-start basin caveat (`J = 0.0878` non-PMP) is mentioned where the IPOPT P1 result is introduced (`NARRATIVE.md §2.2`).
- ☐ `STORYLINE.md` is **not** referenced anywhere — it is superseded by `NARRATIVE.md`.

---

## 7. Files to cite in the report's "Code & data" footnote

> Code, data, and reproducibility scripts: <https://github.com/z-r-li/bezier-curve-orbit-trajectory>. Authoritative results in `results_summary.json` (regenerated by `python make_results_table.py`). Phase runners: `Earth-Mars/run_phase0.py`, `Planer/run_phase1.py`, `Artemis2/run_phase2.py`. Locked storyline: `NARRATIVE.md`. IPOPT option-by-option audit: `Artemis2/RETUNING_AUDIT.md`.
