# Artemis Retuning Audit (T3.3)

**Purpose:** verify NARRATIVE.md Phase 2 claim that "IPOPT multi-shooting converges **without re-tuning** P1 solver options."

**Method:** line-by-line comparison of IPOPT and shooting options between Phase 1 (`Planer/`) and Phase 2 (`Artemis2/Ephem_Full/`).

**Files inspected:**
- P1 IPOPT: `Planer/ipopt_collocation.py` (class `CR3BPBezierIPOPT.solve`, `mesh_refine_solve`)
- P1 runner: `Planer/cr3bp_transfer.py` (`solve_both` → mesh loop; `solve_shooting`)
- P1 SLSQP (context only): `Planer/bezier_segmented.py`, `Planer/cr3bp_transfer_segmented.py`
- P2 IPOPT + shooting: `Artemis2/Ephem_Full/artemis2_ephemeris.py`
- P2 full-mission IPOPT: `Artemis2/Ephem_Full/artemis2_full_mission.py`

---

## Summary verdict

**ORIGINAL CLAIM TOO STRONG — NARROWED TO OPTION C (see below).**

The first draft of NARRATIVE §Phase 2 claimed IPOPT "converges without re-tuning" P1 solver options. This audit found that to be overstated: Phase 2 uses a different transcription (explicit RK4 multiple-shooting with piecewise-constant control) from Phase 1 (Bézier collocation with Gauss-Legendre nodes); several IPOPT options were added or changed (`acceptable_tol`, `warm_start_init_point`, control-magnitude bounds); the mesh-refinement cascade was dropped; `artemis2_full_mission.py` loosens `tol` from `1e-8` to `1e-6`; and the shooting baseline was retuned (`maxfev` cut from 2000 to 80, guess count grown from 11 to 15).

NARRATIVE.md has since been rewritten to adopt **Option C** (see `## Narrative implications` at the bottom of this document): only the *tool class* — MUMPS linear solver, exact AD derivatives, warm-start-from-reference philosophy — is claimed to carry over from P1 to P2. Tolerances, iteration caps, mesh schedule, transcription form, and control bounds are acknowledged as problem-specific and enumerated in the tables below. Under the Option C framing, the narrative and this audit are consistent; the line-by-line tables that follow are the supporting detail for the narrower claim.

---

## IPOPT options

Phase 1 options are set in two places:
1. `ipopt_collocation.py:294-302` — the `opts` dict inside `CR3BPBezierIPOPT.solve`.
2. `cr3bp_transfer.py:485-508` — the mesh-cascade loop in `solve_both`.

Phase 2 options are set in:
- `artemis2_ephemeris.py:572-581` (`solve_ipopt_collocation`)
- `artemis2_full_mission.py:410-417` (`solve_ipopt`)

| Option | P1 value | P2 Ephem value | P2 Full value | Inherited? | Notes |
|---|---|---|---|---|---|
| `tol` | `1e-8` (final), `1e-6` (coarse levels) | `1e-8` | `1e-6` | Partial | Final-level P1 value kept in Ephem; Full Mission slackens by 2 orders of magnitude. |
| `constr_viol_tol` | `tol * 0.1` (= `1e-9` final) | not set (IPOPT default `1e-4`) | not set (default) | **No** | P1 explicitly tightens constraint violation; P2 drops that and falls back to IPOPT default. |
| `acceptable_tol` | not set | `1e-6` | `1e-4` | **No** | Added in Phase 2. Enables early exit if primary `tol` not reached but trajectory is "good enough". |
| `dual_inf_tol`, `compl_inf_tol` | not set | not set | not set | OK | IPOPT defaults used in both phases. |
| `max_iter` | `3000` (final level), `1500` (coarse) | `3000` | `5000` | Partial | Ephem matches final-level P1. Full Mission raises to 5000 to accommodate TLI-burn nonlinearity. |
| `max_cpu_time` | not set | not set | not set | OK | |
| `linear_solver` | `'mumps'` | `'mumps'` | `'mumps'` | **Yes** | Only option carried over verbatim. |
| `nlp_scaling_method` | not set (default `'gradient-based'`) | not set | not set | OK | |
| `obj_scaling_factor` | not set | not set | not set | OK | |
| `jacobian_approximation` | not set (exact AD) | not set (exact AD) | not set (exact AD) | OK | Both phases rely on CasADi AD. |
| `hessian_approximation` | not set (exact AD) | not set (exact AD) | not set (exact AD) | OK | |
| `warm_start_init_point` | not set | `'yes'` | not set | **No** | Ephem turns on warm-start; Full Mission does not. Inconsistent. |
| `mu_strategy` | not set | not set | not set | OK | |
| `print_level` | `5` default, `3` in `mesh_refine_solve`, `0` in `cr3bp_transfer.solve_both` loop | `3` | `3` | Partial | Cosmetic; not a tuning change. |
| `print_time` | `False` | `False` | `False` | **Yes** | |
| `ipopt.sb` (banner) | `'yes'` | not set | not set | **No** | Minor. |

### Transcription (structural, not an "option" strictly — but narrative-relevant)

| Item | P1 | P2 Ephem | P2 Full | Notes |
|---|---|---|---|---|
| Transcription | Bézier global + Gauss-Legendre collocation | Multiple shooting + explicit RK4 (4 sub-steps) | Same as Ephem (RK4, 2 sub-steps inside short burn segments) | **Completely different method.** Not "reusing" P1 collocation code. |
| Control parameterization | Gauss-Legendre node values (`nc × d` per segment) | Piecewise-constant per segment | Piecewise-constant per segment | Different. |
| Control bounds | none | `±u_max` where `u_max = 3×` shooting max | Per-segment bounds: `1e-4 km/s²` coast, `burn_max_accel × 1.5` burn | Added in P2; absent in P1. |
| Waypoint (interior-state) constraints | none | none | Position waypoints every ~8h; burn-phase pinning every 5th node | Added in Full Mission; qualitatively different formulation. |

---

## Shooting (fsolve) options

P1 uses `solve_shooting` at `cr3bp_transfer.py:361-420`. P2 uses `solve_shooting` at `artemis2_ephemeris.py:325-409`. (Full Mission skips shooting entirely.)

| Option | P1 | P2 Ephem | Inherited? | Notes |
|---|---|---|---|---|
| `fsolve` method | default (hybrd / MINPACK) | default | Yes | |
| `xtol` | default (`1.49e-8`) | default | Yes | |
| `maxfev` | `2000` | `80` | **No** | 25× reduction — deliberate retuning to keep a 15-guess sweep tractable over 7.5-day arcs. |
| `full_output` | `True` | `True` | Yes | |
| Residual dimension | 4 (planar position+velocity) | 6 (3D position+velocity) | problem-specific | Expected. |
| Early-stop threshold | `res_norm < 1e-10` | `res_norm < 1e-4` | **No** | Loosened by 6 orders; necessary since 7.5-day 3D integration can't realistically reach 1e-10. |
| Guess count | 1 (physics-informed) + 1 (zero) + optional user + 8 random = **11** | 4 velocity-aligned + 2 position-aligned + 9 random = **15** | Partial | Similar multi-start philosophy, different seeding. |
| Random seed | `np.random.default_rng(42)` | `np.random.RandomState(42)` | Yes (same seed, different generator API) | |
| Residual scaling | none | none | Yes | |
| Inner ODE: method | `RK45` | `DOP853` | **No** | Changed integrator — DOP853 is higher order, better for long 3D arcs. |
| Inner ODE: rtol/atol (residual call) | `1e-12 / 1e-12` | `1e-11 / 1e-13` | **No** | Slightly relaxed rtol, tighter atol. |
| Inner ODE: final re-propagation rtol/atol | `1e-12 / 1e-12` | `1e-12 / 1e-14` | Partial | |
| Inner ODE: `max_step` | not set | `300s` (residual) / `120s` (final) | **No** | Added. |

---

## Dynamics constants

| Constant | P1 location | P2 location | Consistency |
|---|---|---|---|
| `MU` (Earth-Moon mass parameter) | `Planer/cr3bp_planar.py:17` → imported everywhere: `MU = 0.012150585609624` | N/A (P2 is dimensional, not CR3BP) | Single source of truth in P1. |
| `MU_EARTH` | N/A | `artemis2_ephemeris.py:58` = `398600.4418`; `artemis2_full_mission.py:52` = `398600.4418` | **Hardcoded in both P2 files**, values match but are duplicated — no common module. |
| `MU_MOON` | N/A | `artemis2_ephemeris.py:59` = `4902.800066`; `artemis2_full_mission.py:53` = `4902.800066` | Duplicated, values match. |
| `MU_SUN` | N/A | `artemis2_ephemeris.py:60` = `132712440041.93938`; `artemis2_full_mission.py:54` = `132712440041.93938` | Duplicated, values match. |
| `L_STAR`, `T_STAR`, `V_STAR` | N/A | `artemis2_ephemeris.py:63-65` (defined but unused downstream); absent from `artemis2_full_mission.py` | Dead constants in Ephem; missing altogether in Full. Nondimensionalization is declared but never applied — P2 code is fully dimensional. |

**Observation:** P2 duplicates all three gravitational parameters across its two files with no shared module. Values are numerically identical, so no correctness risk today, but any future edit (e.g., switching to DE440 values) must be done in both files. Recommend (non-blocking) consolidation to a shared `constants.py`.

---

## Mesh refinement cascade

**Phase 1 (`cr3bp_transfer.py:485-520`, also `ipopt_collocation.mesh_refine_solve`):**
4-level cascade on a common (degree=7, nc=12) sizing:

| Level | Segments | Tol | max_iter | Warm-start source |
|---|---|---|---|---|
| 1 (coarse) | 4 | `1e-6` | `3000` | Shooting solution |
| 2 (medium) | 8 | `1e-6` | `3000` | Level-1 Bézier output |
| 3 (fine) | 16 | `1e-6` | `3000` | Level-2 |
| 4 (very fine) | 32 | `1e-8` | `3000` | Level-3 |

**Phase 2 Ephem (`artemis2_ephemeris.py:416, 1162-1165`):** single level, `n_seg=100` (the call-site override of the function default 60). Warm-started from NASA OEM. No cascade.

**Phase 2 Full (`artemis2_full_mission.py:900-905`):** single level, `n_seg=120` nominal with non-uniform densification during burns (2-minute spacing during burns, ~coast spacing otherwise), ending at ~130-ish segments in practice. Warm-started from NASA OEM. No cascade.

**Verdict on cascade:** NARRATIVE says "mesh cascade ... carry[s] over". It does not. P2 uses a single-level solve in both files. The *idea* of coarse-to-fine is not present in any Artemis code path.

---

## Flagged retuning instances

Ranked by narrative impact (highest first):

1. **Mesh cascade dropped entirely in both P2 files.**
   - What: P1's 4-level (4→8→16→32) warm-started cascade replaced with a single-level solve (60 / 100 / 120 segments).
   - Legitimate or accidental: *Legitimate by design* — NASA OEM is already a good warm start, so the cascade's main benefit (bootstrapping a non-convex NLP) is less needed. But this directly contradicts a specific NARRATIVE claim.
   - Action: **Narrative must be corrected.** Either say "cascade not needed because OEM warm-start is high-quality" or run a cascade variant in P2.

2. **Transcription method changed: Bézier collocation → RK4 multiple-shooting.**
   - What: P1 uses `CR3BPBezierIPOPT` with Gauss-Legendre defect constraints and degree-7 Bézier control points. P2 uses `ca.Opti()` with explicit RK4 sub-steps and piecewise-constant controls.
   - Legitimate or accidental: *Legitimate* — simpler to code against an ephemeris lookup (can't put `astropy` calls inside CasADi AD), and multiple-shooting has better numerical conditioning for long 3D arcs. But this is *not* "the same IPOPT setup retargeted".
   - Action: NARRATIVE should describe P2 as "IPOPT multiple-shooting" (the NARRATIVE already does this in section 2.2, so the transcription change is acknowledged upstream — but the "without re-tuning" claim in section §Phase 2 is still wrong because the solver options themselves changed).

3. **`tol` loosened from 1e-8 to 1e-6 in `artemis2_full_mission.py`; `acceptable_tol=1e-4` added.**
   - What: Full-mission IPOPT uses `tol=1e-6`, `acceptable_tol=1e-4`, `max_iter=5000`.
   - Legitimate or accidental: *Legitimate* — the full mission includes the TLI burn with ~3 km/s Δv, a large magnitude discontinuity that stresses any solver. Relaxed tolerances with `acceptable_tol` make convergence robust.
   - Action: Document in Limitations or Phase 2 notes. Do not claim tolerances "carry over" from P1.

4. **Shooting `maxfev` cut from 2000 to 80.**
   - What: Per-guess Jacobian eval budget collapsed by 25×.
   - Legitimate or accidental: *Legitimate* — each residual call is a 7.5-day 3D integration, ~50× more expensive than planar CR3BP. Keeping `maxfev=2000` would make the 15-guess sweep intractable.
   - Action: Note in the fragility/robustness comparison that P2 shooting is "budget-constrained" and this is what produces the guess-fragility observation.

5. **Shooting early-stop threshold loosened: `1e-10 → 1e-4`.**
   - What: In `cr3bp_transfer.py:399` the shooting loop breaks when residual drops below `1e-10`; in `artemis2_ephemeris.py:376` it breaks below `1e-4`.
   - Legitimate or accidental: *Legitimate* given dimensional residuals (km/s²) and arc length, but this is a concrete retuning instance.
   - Action: Document as a per-problem residual-scale adjustment.

6. **`warm_start_init_point='yes'` set only in `artemis2_ephemeris.py`, not in `artemis2_full_mission.py`.**
   - What: Inconsistency *within* Phase 2.
   - Legitimate or accidental: *Likely accidental* — no evidence one was turned off on purpose.
   - Action: Minor — either remove it from Ephem or add it to Full. Doesn't impact NARRATIVE.

7. **`constr_viol_tol=tol*0.1` (tightening) present in P1, absent in both P2 files.**
   - What: P1 explicitly enforces constraint violation 10× tighter than the KKT tolerance; P2 falls back to IPOPT's default of `1e-4`.
   - Legitimate or accidental: *Ambiguous* — not obviously needed, but a silent change. For `artemis2_full_mission.py` with `tol=1e-6`, default `constr_viol_tol=1e-4` means the feasibility check is **looser than the optimality check**, which is mildly suspicious but typical for RK4-shooting.
   - Action: Note in Limitations; unlikely to change the trajectory meaningfully.

8. **Inner-ODE integrator changed: RK45 → DOP853; `rtol/atol` perturbed.**
   - What: P1 shooting uses `solve_ivp(method='RK45', rtol=1e-12, atol=1e-12)`; P2 uses `DOP853` with various tolerances.
   - Legitimate or accidental: *Legitimate* — DOP853 is the right choice for long, smooth 3D arcs. Worth a sentence in methodology.

9. **Dynamics constants (`MU_EARTH`, `MU_MOON`, `MU_SUN`) duplicated verbatim between `artemis2_ephemeris.py` and `artemis2_full_mission.py`.**
   - What: No common constants module.
   - Legitimate or accidental: Minor code-hygiene issue, not a tuning change.
   - Action: Low-priority refactor; not narrative-relevant.

---

## Narrative implications

Of the three options offered:

- **Option A (soften):** Rewrite NARRATIVE.md §Phase 2 to say "converges with minimal retuning; the IPOPT transcription was adapted from Bézier collocation to RK4 multiple-shooting to accommodate ephemeris-dependent dynamics that are not CasADi-AD-compatible. Linear-solver (MUMPS), objective form, and warm-start philosophy are preserved; tolerances and mesh schedule are problem-specific and documented in RETUNING_AUDIT.md."
- **Option B (revert):** Port the P1 Bézier collocation + mesh cascade onto the ephemeris dynamics. This is at least several hours of work (Bézier defects across an ephemeris ODE need either analytic Moon/Sun interpolation inside CasADi or parametric CasADi Functions wrapping the ephem cache — doable but non-trivial) and risks convergence regressions.
- **Option C (narrow):** Accept the retuning and narrow the claim to: "MUMPS linear solver, exact AD derivatives, and warm-start-from-reference philosophy carry over from P1 to P2. Tolerances, iteration caps, mesh schedule, and control bounds are problem-specific and retuned for dimensional 3D dynamics."

**Recommendation: Option C.** Option A is equivalent but wordier; Option B is out-of-scope for a course project (see `feedback_aae568_scope.md`). Option C preserves the honest pedagogical point — that IPOPT with AD is the right *class* of tool for both problems — while giving up the overstated "no retuning" claim that the code does not support.

**Concrete rewrite suggestion for NARRATIVE.md line 44:**

> IPOPT multi-shooting converges with tolerances and mesh sized for 3D dimensional dynamics; MUMPS linear solver, exact AD derivatives, and the warm-start-from-reference philosophy carry over unchanged from P1. Solver-option differences are enumerated in `Artemis2/RETUNING_AUDIT.md`.
