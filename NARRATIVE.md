# AAE 568 Course Project — Locked Narrative

> **Status:** Storyline frozen (three-phase arc P0 → P1 → P2). Numbers and framing are edited as artifacts and audits land; the pedagogical spine does not move.

## 1. Introduction
- **Problem.** Low-thrust trajectory optimization in cislunar regimes, where nonlinearity makes solver choice a first-class design decision.
- **Course focus.** Direct comparison of indirect (PMP + shooting) and direct (collocation + NLP) methods from AAE 568.
- **Thesis.** A Bézier-segment direct-transcription pipeline is validated against PMP on tractable problems, exposes a scalability limit that motivates IPOPT multi-shooting, and transfers to a mission-relevant Artemis II ephemeris problem with the same tool *class* (IPOPT + automatic differentiation + warm-start-from-reference) — tolerances, mesh schedule, and transcription form are retuned for dimensional 3D dynamics and enumerated in `Artemis2/RETUNING_AUDIT.md`.
- **Three-phase arc.** P0 Earth→Mars (pipeline check) → P1 Planar CR3BP (method tradeoff, solver pivot) → P2 Artemis II (mission-relevant validation).

## 2. Methodology

### 2.1 Common formulation
- Bolza min-energy objective, fixed $t_f$, fixed endpoints.
- PMP yields Hamiltonian $H$, adjoint ODE $\dot\lambda = -\partial H/\partial x$, optimal control $u^* = -\tfrac{1}{2}\lambda_v$ (AAE 568 notes §PMP).
- Direct transcription family: (a) Bézier collocation + IPOPT with exact AD (P1), (b) segmented Bézier + SLSQP with finite-difference gradients (P1 N-sweep) — hp-pseudospectral in spirit, (c) RK4 multiple-shooting + IPOPT with exact AD (P2).

### 2.2 Solvers and tools
- **Indirect:** `scipy.optimize.fsolve` on TPBVP in initial costates.
- **Direct (P0–1):** SciPy SLSQP driving global and segmented Bézier.
- **Direct (P1–2):** CasADi + IPOPT, analytic derivatives via AD. P1 uses Bézier collocation; P2 uses RK4 multiple-shooting (see §6.4 for why).
- **Mesh refinement cascade (P1 only):** coarse → medium → fine, warm-started between stages, on the P1 Bézier collocation. P2 uses a single-level solve warm-started from the NASA OEM; the cascade is not transferred (see `Artemis2/RETUNING_AUDIT.md` §Mesh refinement).
- **Warm-starting IPOPT:** runs are warm-started from a converged shooting trajectory (P1) or the NASA OEM (P2). Cold-start from a linear state interpolation was observed to land in a non-PMP basin on the CR3BP ($J=0.0878$ vs $J^*=0.04306$); warm-starting is made explicit here rather than hidden as an implementation detail.

## 3. Results

### Phase 0 — Two-body Earth→Mars (`Earth-Mars/`)
- **Purpose.** Verify the pipeline against a problem whose answer can be eyeballed.
- Indirect shooting and direct Bézier agree on trajectory and cost to nine significant figures ($|\Delta J|\approx 2.6\times 10^{-9}$; see §5.1).
- Control history matches $-\tfrac{1}{2}\lambda_v$ from PMP.
- **Next →** Pipeline trusted; move to nonlinear dynamics.

### Phase 1 — Planar CR3BP L1↔L2 Lyapunov (`Planer/`)
- **Purpose.** Stress methods on nonlinear dynamics; quantify accuracy-vs-cost tradeoff.
- **PMP reference:** $J^* = 0.04306$, 6.02 s — the anchor for the accuracy axis.
- **Bézier collocation + IPOPT (N=16 segments, degree 7, AD derivatives):** comparable accuracy (J=0.04323, +0.4% vs PMP), 5 IPOPT iterations, 2.0 s wall (warm-started from shooting; cold-start lands in a non-PMP basin at J=0.0878). The JSON method label `global_bezier_ipopt` is a legacy name for this segmented-collocation setup.
- **Segmented Bézier + SLSQP N-sweep:** matches PMP cost (N=16: $J=0.0432$, 0.4% error) but **~88× slower** than shooting (528.9 s vs 6.02 s).
- **Pareto plot:** $|J-J^*|$ vs wall time, overlaying SLSQP-segmented-Bézier and IPOPT on identical dynamics.
- **Jacobi-constant cross-check** (`Planer/jacobi_constant.png`): all three methods compute the same $\Delta C(t)$ trajectory, confirming they converge to a common controlled path. Note: under thrust $C$ is not conserved; the $O(10^{-2})$ trace is dominated by work done by $u^*(t)$, not method error. A finer residual-vs-reference diagnostic would be required to expose collocation's between-node drift; out of scope here.
- **Takeaway.** Segmented direct transcription is correct as a transcription family — the solver *pairing* is what breaks: SLSQP driven by dense finite-difference gradients scales as $\approx 10\times$ per doubling of $N$ on this problem and is impractical past $N=16$ on a laptop budget. Interior-point NLP with exact (AD) derivatives is the appropriate replacement; no claim is made about SLSQP with analytic gradients or sparse Jacobians.
- **Next →** IPOPT is the answer; P2 carries it to a harder problem.

### Phase 2 — Artemis II ephemeris validation (`Artemis2/`)
- **Purpose.** Show the P1 IPOPT approach transfers to mission-relevant 3D N-body dynamics.
- **Setup.** 3D Earth-Moon-Sun N-body via astropy's builtin analytic ephemeris (not JPL DE; see §6.6); truth is NASA post-flight OEM for Artemis II (flown **April 2–10, 2026**).
- IPOPT multi-shooting converges rapidly (2 iter / 0.81 s on the post-TLI arc, 4 iter / 1.68 s on full mission) — these iteration counts reflect the quality of the NASA-OEM warm start (IPOPT polishes a near-feasible trajectory rather than searching from scratch), not a generic IPOPT speed claim. The same solver class (MUMPS linear solver, exact AD derivatives, warm-start-from-reference philosophy) carries over from P1; tolerances, mesh schedule, and transcription form (RK4 multi-shooting rather than Bézier collocation) are problem-specific and documented in `Artemis2/RETUNING_AUDIT.md`. The pedagogical point is preserved: interior-point NLP with AD is the right tool class for both regimes.
- Deliverables: trajectory, control history, convergence history vs. OEM.
- Shooting baseline included; 15-seed sweep exposes its initial-guess fragility against IPOPT's robustness.
- **Takeaway.** IPOPT with AD is the right *class* of tool for both P1 and P2; the solver architecture (MUMPS, exact derivatives, warm-start from reference) carries over unchanged. Problem-specific tolerances, mesh schedule, transcription form, and control bounds are retuned for dimensional 3D dynamics — enumerated in `Artemis2/RETUNING_AUDIT.md`. Pedagogical circle closed on tool *class*, not verbatim settings.

## 4. Conclusion
- **Shown.** (P0) Pipeline produces PMP-consistent answers. (P1) Segmented Bézier matches PMP accuracy; SLSQP loses on cost, IPOPT recovers it. (P2) The same IPOPT+AD tool class transfers to Artemis II; solver architecture carries over while tolerances and mesh are problem-specific (see `Artemis2/RETUNING_AUDIT.md`).
- **Limitations.** Fixed-$t_f$ throughout; Phase 3 NRHO insertion cut for scope.
- **Future work.** Warm-starts for Moon-proximity arcs; free-$t_f$ via transversality; surrogate initial guesses via scimlstudio.

---

## 5. Key results (numerical summary)

All entries below are loaded from the authoritative `results_summary.json` and mirrored in `results_table.md`. Units: $J$ is the Bolza energy integral $\int |u|^2\,dt$; wall time is single-core on the laptop; "iterations" is NLP outer iters except where noted as `nfev` (fsolve function-evaluation count).

### 5.1 Phase 0 — Earth → Mars (pipeline check)
- Indirect shooting: $J = 1.49004478\times 10^{-2}$, residual $1.3\times 10^{-12}$, 0.85 s, 24 `nfev`.
- Bézier collocation + IPOPT (N=16 segments, warm-started from shooting): $J = 1.49004452\times 10^{-2}$, residual $2.2\times 10^{-15}$, 1.22 s, 2 iterations.
- **Agreement.** $|\Delta J| \approx 2.6\times 10^{-9}$ — i.e., PMP and direct transcription agree to the ninth decimal. This is the "machine-tolerance agreement" claim in §3 made concrete.

### 5.2 Phase 1 — Planar CR3BP L1↔L2 Lyapunov transfer
- **PMP anchor:** $J^{*} = 4.306\times 10^{-2}$, 6.0 s, 95 `nfev`.
- **Bézier collocation + IPOPT (N=16):** $J = 4.323\times 10^{-2}$ ($+0.4\%$ vs $J^{*}$), 2.0 s, 5 iterations. Cold-start cost was $0.0878$; the warm-start from shooting was required to land in the PMP basin.
- **Segmented Bézier + SLSQP N-sweep.** Cost vs $N$ (defect $<10^{-4}$ at every converged level): $J(N)=\{0.718,\,0.128,\,0.090,\,0.050,\,0.0432\}$ for $N\in\{1,2,4,8,16\}$; wall clock $\{0.10,\,3.17,\,15.2,\,49.5,\,529\}$ s; SLSQP iterations $\{21,\,169,\,198,\,140,\,294\}$. $N=32$ timed out beyond 900 s.
- **The 87× number.** $N=16$ SLSQP costs 529 s versus 6.0 s for shooting $\Rightarrow$ 88× slower (reported as "87×" in earlier drafts; updated number is from the re-instrumented run and should supersede the older figure in the report).
- **Cost scaling.** Wall clock grows $\approx 10\times$ per doubling of $N$; cost gap to $J^{*}$ contracts geometrically. Mesh-refinement convergence is observed but solver-stack cost is the dominant term — the quantitative motivation for the IPOPT pivot.
- **Jacobi-constant cross-check** (`Planer/jacobi_constant.png`): all three methods trace the same $\Delta C(t)$ to within $O(10^{-2})$, which is the *work done by the thrust*, not method error. Useful for the report as a "dynamics integrity" cue; not a precision diagnostic.

### 5.3 Phase 2 — Artemis II ephemeris validation
- **Shooting, post-TLI multi-start (15 seeds).** Only **1/15** met the $10^{-4}$ residual early-stop: seed 12 (random-normal, scale $10^{-6}$) with $J=6.21\times 10^{-7}$, residual $6.2\times 10^{-6}$, 130 s per-seed; total sweep 2033 s. The 14 failures ran against a per-seed `maxfev=80` ceiling — budget-constrained, not a basin proof, which is a subtle but honest point for the report.
- **IPOPT multi-shooting, post-TLI coast arc.** $J=5.30\times 10^{-12}$ (i.e., numerically zero — the OEM arc is a ballistic coast, so the min-energy control is $u\equiv 0$), residual $3.5\times 10^{-10}$, 906 NLP variables, 612 constraints. **Converges in 2 iterations, 0.81 s** — this count reflects a near-feasible warm start, not a from-scratch solve: the NASA OEM states are close enough to the NLP manifold that IPOPT polishes rather than searches. The speed is a property of the warm-start quality, not a generic IPOPT claim. Endpoint-position residual is 0.000 km by construction (the endpoint is a hard NLP equality constraint, not a measured validation quantity); the meaningful validation is that interior-state defects and the objective both drive to numerical zero.
- **IPOPT multi-shooting, full mission (day 0 → 8.9).** $J=9.44\times 10^{-4}$, 4 iterations in 1.68 s (again, 4 iterations reflects the quality of the OEM warm start), 2031 NLP vars, 1362 constraints, non-uniform mesh (225 segments, 2-min spacing across burns). The detection heuristic labels burns by epoch (`day_start < 2 → "TLI Burn"`, else "Entry Correction Burn"), but this OEM starts *Post-ICPS-Sep*, i.e., after TLI; neither detected burn is TLI. Relabel for the report:
    - **Burn 1** — days 0.874–0.945, likely the first Orion outbound trajectory correction (OTC-1 or equivalent).
    - **Burn 2** — days 8.88–8.91, a return-phase correction maneuver (not entry interface).
  The integrated-magnitude numbers $\sum|\Delta v_i|$ from the detection code (22.59 and 9.72 km/s) are *not* physical scalar ΔV. Each $|\Delta v_i| = |v_{i+1}-v_i|$ contains *all* velocity change over the sample interval — gravity plus thrust — and summing magnitudes rather than integrating the vector gives $\int|a_{\text{grav}}+a_{\text{thrust}}|\,dt$ instead of $\bigl|\int a_{\text{thrust}}\,dt\bigr|$. Gravity dominates: over a ~60–100 min burn window the integrated gravitational-acceleration magnitude alone is tens of km/s, which is the bulk of the overcount. Real Artemis II correction burns are tens of m/s. Treat the 22.59 / 9.72 numbers as "detected-burn window integrated $|a|\,dt$", not as reportable ΔV, until independently recomputed.
- **Shooting-vs-IPOPT headline.** On the same post-TLI arc, the 15-seed shooting sweep consumed 2033 s of compute and produced one marginal convergence (residual $6.2\times 10^{-6}$, below the $10^{-4}$ early-stop but well above the $\sim 10^{-12}$ P1 shooting residual); IPOPT solved the same problem in 0.81 s with 2 outer iterations to residual $3.5\times 10^{-10}$. The raw ratio is ~2500×, but it is a **budget-constrained** comparison, not a fundamental speedup: shooting's per-seed `maxfev` was capped at 80 (vs 2000 in P1) to keep the 15-seed sweep tractable on 7.5-day 3D arcs, and 14/15 seeds exhausted that budget. A budget-matched comparison (fixed total compute split across seeds) would give a smaller ratio. The qualitative claim — shooting is basin-fragile on long 3D arcs, IPOPT with AD is not — is robust to that framing; the specific 2500× number is not.

### 5.4 Claim-to-artifact traceability

| NARRATIVE claim | Backing artifact |
|---|---|
| "PMP and direct agree to nine significant figures" (§3 P0) | `results_summary.json` rows 0–1 ($|\Delta J|\approx 2.6\times 10^{-9}$) |
| "$J^{*}=0.04306$, 6.02 s anchor" (§3 P1) | `results_summary.json` CR3BP shooting row. (Older draft used 5.33 s from `nsweep_findings.md` §2 — same trajectory, different run; use JSON values.) |
| "N=16 matches PMP at 0.4%, ~88× slower" (§3 P1) | `results_summary.json` SLSQP N=16 row (528.9 s / 6.02 s = 87.9×). Older drafts said "87×" from the N-sweep-internal pairing (467.6 s / 5.33 s). |
| "Cold-start fell outside PMP basin" (§2.2) | `results_summary.json` Bézier-collocation IPOPT row (`method: global_bezier_ipopt`) — see `notes` |
| "15-seed shooting sweep exposes fragility" (§3 P2) | `results_summary.json` 15 shooting rows + best-of-15 rollup |
| "Tool *class* carries over, settings don't" (§3 P2) | `Artemis2/RETUNING_AUDIT.md` §Summary (Option C) — narrative now aligned. |

---

## 6. Control methodology and theory

This section works the methodology bottom-up: first the Bolza problem class (6.1), then the indirect branch (PMP + shooting, 6.2), then the Bernstein/Bézier algebra the direct branch rests on (6.3), then the direct transcription itself (defects, quadrature, NLP structure, 6.4), then solvers (6.5), dynamics (6.6), and the pedagogical rationale for the three-phase ladder (6.7). Every equation in 6.1–6.2 is a direct import from AAE 568 lecture notes (page numbers inline); 6.3–6.4 go beyond the lectures — the notes cover optimal control entirely via the indirect/PMP path and contain no slides on collocation, Bernstein basis, or direct transcription — so those subsections lean on the project's own `AAE568_Technical_Documentation.docx` and standard references (Betts 2010; Farouki's Bernstein survey).

### 6.1 Problem class (Bolza form, lecture notes p. 141)
The course defines the general optimal control problem in **Bolza** form:

$$\min_{u(\cdot)}\; J = \phi\bigl(x(t_f),t_f\bigr) + \int_{t_0}^{t_f} \mathcal{L}(x,u,t)\,dt, \qquad \dot x = f(x,u,t), \qquad x(t_0)=x_0.$$

"Mayer" and "Lagrange" are the special cases $\mathcal{L}\equiv 0$ and $\phi\equiv 0$ respectively; Bolza carries both terms. This project instantiates the Lagrange form with a fixed-endpoint, fixed-$t_f$ boundary setup:

$$\min_{u(\cdot)}\; J = \int_{t_0}^{t_f} u^{\top} R\, u \,dt, \qquad \dot x = f(x,t) + B(x,t)\,u, \qquad x(t_0)=x_0,\;x(t_f)=x_f.$$

For this project $R = I$ and $B = \begin{bmatrix}0\\I_{n/2}\end{bmatrix}$ (thrust enters additively through the velocity channel, $n=4$ in planar P1, $n=6$ in 3D P2), so the objective reduces to $J = \int |u|^2\,dt$ — the convention used throughout `results_summary.json`, with no $\tfrac12$ prefactor. The minimum-energy objective $\|u\|_2^2$ is used in place of the physically faithful $\|u\|_1$ (propellant-mass proxy) because the $L^2$ cost gives smooth PMP optimality conditions ($H_{uu}=2R\succ 0$, §6.2) and a well-conditioned NLP — classic trade, called out in AAE 568 notes before the first worked example. Minimum-time and minimum-fuel variants (discussed in `AAE568_Controls_Aspects.tex` §2.1) are scoped out of the locked narrative in favor of a single cost functional across P0–P2.

### 6.2 Indirect branch — PMP + shooting (lecture notes pp. 144–194)

**Step 1 — augmented cost and the Hamiltonian (pp. 144–145).** Adjoin the dynamics constraint with a costate $\lambda(t)\in\mathbb{R}^n$:

$$J_a = \phi(x(t_f),t_f) + \int_{t_0}^{t_f}\!\left[\mathcal{L}(x,u,t) + \lambda^{\top}\bigl(f(x,u,t)-\dot x\bigr)\right]dt.$$

Define the **Hamiltonian** $H(x,u,\lambda,t) := \mathcal{L}(x,u,t) + \lambda^{\top} f(x,u,t)$. Integration by parts moves the $\lambda^{\top}\dot x$ term onto $\dot\lambda$, yielding

$$J_a = \phi(x(t_f),t_f) + \bigl[\lambda^{\top}(t_0) x(t_0) - \lambda^{\top}(t_f) x(t_f)\bigr] + \int_{t_0}^{t_f}\bigl[H + \dot\lambda^{\top} x\bigr]dt.$$

For our Lagrange form ($\phi=0$) and quadratic running cost, $H = u^{\top}Ru + \lambda^{\top}(f + Bu)$.

**Step 2 — Euler–Lagrange necessary conditions (p. 147, boxed in lecture).**

$$\boxed{\quad\text{(a)}\;\dot\lambda = -\Bigl(\frac{\partial H}{\partial x}\Bigr)^{\!\top},\qquad \text{(b)}\;\lambda(t_f) = \Bigl(\frac{\partial\phi}{\partial x}\Bigr)^{\!\top}\bigg|_{t_f},\qquad \text{(c)}\;\frac{\partial H}{\partial u} = 0\quad}$$

For our setup:

- (a) **Costate ODE.** $\dot\lambda = -(\partial f/\partial x)^{\top}\lambda$ (since $\partial\mathcal{L}/\partial x = 0$ when the running cost is control-only). In the planar CR3BP the Jacobian $\partial f/\partial x$ is the variational system of the rotating-frame EOMs — a closed-form $6\times 6$ matrix whose velocity block is the $\pm 2$ Coriolis couple and whose position block is the $3\times 3$ Hessian $U_{xx}$ of the pseudo-potential.
- (b) **Transversality.** $\phi\equiv 0$ and $x(t_f)=x_f$ is fixed, so $\delta x(t_f)=0$ and no natural boundary on $\lambda$ is imposed — the unknowns are $\lambda(t_0)\in\mathbb{R}^n$, and shooting solves for them.
- (c) **Stationarity.** $\partial H/\partial u = 2Ru + B^{\top}\lambda = 0 \;\Rightarrow\; u^{*} = -\tfrac12 R^{-1}B^{\top}\lambda = -\tfrac12\lambda_v$ (with $R=I$, $B$ picking off the velocity channel). This is the same feedback structure that LQR produces for a quadratic cost (linear in $\lambda$) even though our dynamics are fully nonlinear.

**Step 3 — Legendre–Clebsch sufficient condition (pp. 152–153).** Check $H_{uu}(x^*,u^*,\lambda^*,t) = 2R \succ 0$ for all $t\in[t_0,t_f]$. Since $R=I$, $H_{uu}=2I$ everywhere, so the extremals are strict local minima (not saddles) — a one-line sanity check worth stating once, after which the ambiguity between minimum and maximum principle conventions is resolved for the whole project.

**Step 4 — Pontryagin Minimum Principle (pp. 154–157).** When the control is bounded, stationarity is replaced by the pointwise minimum condition

$$H(x^*,u^*,\lambda^*,t) \le H(x^*,u,\lambda^*,t) \qquad \forall u\in U.$$

For our unconstrained-thrust minimum-energy runs, PMP reduces to the stationarity form of step 2(c) and $u^* = -\tfrac12\lambda_v$ is the global minimizer. The bang-bang minimum-time law $u^* = -u_{\max}\lambda_v/\|\lambda_v\|$ (pp. 164–166 worked example) is available but not used in the locked narrative.

**Step 5 — TPBVP assembly (pp. 148–149, 186–194 stirred-tank example).** Substituting $u^*$ into $\dot x = f + Bu^*$ yields a coupled state–costate ODE of dimension $2n$ with split boundary data: $x(t_0)=x_0$ known, $x(t_f)=x_f$ required at the far end. **Shooting** collapses this into a root-finding problem in $\lambda(t_0)$: guess $\lambda(t_0)$, integrate $[x;\lambda]$ forward with $u^*$ substituted, read off $x(t_f)$, Newton-iterate on $\lambda(t_0)$ until $x(t_f)=x_f$. We drive the residual with `scipy.optimize.fsolve` (MINPACK `hybrd`); on planar P1 the shooting vector is $\lambda(t_0)\in\mathbb{R}^4$, and on 3D P2 it is $\mathbb{R}^6$.

The lectures (p. 196 summary) flag two failure modes: (i) sensitivity of the shooting map $\lambda(t_0)\mapsto x(t_f)$, whose Jacobian is the state-transition matrix $\Phi(t_f,t_0)$ and grows exponentially in hyperbolic CR3BP / cislunar regions — ill-conditioning `fsolve`'s Jacobian and shrinking the basin of convergence with arc length; and (ii) discontinuities in $u(t)$ (e.g. bang-bang) that break smoothness assumptions in `bvp4c`-class solvers. Both show up on P2 exactly as the lectures warn; the 15-seed sweep of §5.3 is the empirical characterization of failure mode (i).

### 6.3 Bézier curves and Bernstein polynomial math

The direct branch parameterizes the state trajectory on each segment as a **Bézier curve** in Bernstein basis. This subsection collects the algebra used by the collocation pipeline. It is not in the lecture notes — the notes cover optimal control via PMP/shooting only — so terminology and conventions follow `AAE568_Technical_Documentation.docx` §4.3 and standard references (Farouki 2012 Bernstein survey; Betts 2010 Ch. 4 for collocation).

**Bernstein basis.** For polynomial degree $d$ and parameter $\tau\in[0,1]$, the $d+1$ **Bernstein basis polynomials** are

$$B_i^d(\tau) = \binom{d}{i}\tau^i(1-\tau)^{d-i}, \qquad i=0,1,\dots,d.$$

They form a partition of unity ($\sum_i B_i^d(\tau) = 1$), are nonnegative on $[0,1]$, and are linearly independent on every sub-interval of positive length.

**Bézier curve on a segment.** Given $d+1$ vector-valued **control points** $\{P_i\}_{i=0}^{d}\subset\mathbb{R}^n$, the Bézier curve is

$$x_{\text{Bez}}(\tau) = \sum_{i=0}^{d} P_i\,B_i^d(\tau), \qquad \tau\in[0,1].$$

Three properties make this basis suitable for collocation:

1. **Endpoint interpolation.** $x_{\text{Bez}}(0) = P_0$ and $x_{\text{Bez}}(1) = P_d$. Segment-boundary states are exactly the first and last control points — no interior-node interpolation is needed to read off the endpoints, which simplifies boundary-condition enforcement and inter-segment $C^0$ matching (see "composite curve" below).
2. **Convex-hull property.** Because the basis is a partition of unity of nonnegative functions, $x_{\text{Bez}}(\tau)$ lies in the convex hull of $\{P_i\}$ for all $\tau\in[0,1]$. Tight geometric control over the curve's reach helps the warm-start heuristic stay physically sensible during IPOPT's first iterations and bounds the search neighborhood around the warm-start trajectory.
3. **Smooth analytical derivatives (hodograph).** The derivative of a degree-$d$ Bézier is itself a Bézier curve of degree $d-1$, with control points
   
   $$Q_i = d\,(P_{i+1} - P_i), \qquad i=0,\dots,d-1.$$
   
   So $\dot x_{\text{Bez}}(\tau) = \sum_{i=0}^{d-1} Q_i\,B_i^{d-1}(\tau)$. Applied twice, the second derivative drops to degree $d-2$ with control points $d(d-1)(P_{i+2}-2P_{i+1}+P_i)$. This closed-form hodograph is the reason defects can be evaluated analytically at any $\tau$ — no numerical differentiation inside the NLP.

**Composite (piecewise) Bézier.** Partition $[t_0,t_f]$ into $N$ segments $[t_k, t_{k+1}]$ with local parameter $\tau_k = (t-t_k)/\Delta t_k$. On each segment, one degree-$d$ Bézier gives $N(d+1)$ total control points if segments are unlinked. **$C^0$ continuity at interior knots** is enforced structurally by sharing the last control point of segment $k$ with the first control point of segment $k{+}1$ (i.e., identifying $P_d^{(k)} \equiv P_0^{(k+1)}$), reducing the unshared control-point count to $N(d+1) - (N-1) = Nd + 1$. $C^1$ continuity additionally would require matching hodograph endpoints $d(P_d^{(k)}-P_{d-1}^{(k)})/\Delta t_k = d(P_1^{(k+1)}-P_0^{(k+1)})/\Delta t_{k+1}$; the current implementation does not enforce this explicitly and relies on the NLP to find smooth solutions through the objective and defect structure (a deliberate simplification, documented in the Technical Doc §4.3.2).

**Evaluation.** `bezier.py` exposes `bezier_eval(cp, tau)` (direct basis sum), `bezier_derivative(cp, tau)` (via the hodograph formula), and `composite_bezier_eval(cps, breakpoints, t)` (locate the segment containing $t$, rescale, dispatch). De Casteljau-style recursive evaluation is available numerically but not used in the collocation loop, where the explicit basis-sum form is preferred so CasADi can symbolically differentiate through it.

### 6.4 Direct branch — collocation transcription and NLP structure

With the Bernstein algebra of §6.3 in hand, the direct transcription is mechanical. Discretize the state as a composite Bézier, keep the control as an explicit NLP decision variable at each collocation node, and enforce the dynamics as equality ("**defect**") constraints at those nodes.

**Defect constraint.** At each collocation point $\tau_k$ inside segment $s$ with duration $\Delta t_s$:

$$\boxed{\quad \text{def}_k \;:=\; \frac{1}{\Delta t_s}\frac{d x_{\text{Bez}}^{(s)}}{d\tau}\bigg|_{\tau_k} \;-\; f\bigl(x_{\text{Bez}}^{(s)}(\tau_k),\,u_k,\,t_k\bigr) \;=\; 0.\quad}$$

The left factor is the hodograph evaluated at $\tau_k$; the right is the dynamics at the Bézier-sampled state and the NLP control. The NLP makes the residual zero, so the Bézier curve's tangent matches the ODE at every collocation node. Between nodes the curve is interpolated by the polynomial, with accuracy governed by the standard collocation error bound $O(\Delta t^{2n_c})$ for Gauss–Legendre nodes of order $n_c$.

**Collocation nodes.** Gauss–Legendre nodes $\{\tau_k\}$ on $[0,1]$ give maximum-degree algebraic exactness for the quadrature used in the cost (see "objective" below), and the corresponding defect placement has well-studied superconvergence properties. The P1 collocation run uses $N=16$ segments and degree $d=7$ with 4 collocation nodes per segment; the N-sweep probes $N\in\{1,2,4,8,16\}$ at fixed degree.

**Objective as quadrature.** The running cost is approximated by Gauss–Legendre quadrature at the same nodes used for defects, with weights $\{w_k\}$:

$$J \;\approx\; \sum_{s=1}^{N}\sum_{k=1}^{n_c} w_k\,\|u_k^{(s)}\|^2\,\Delta t_s.$$

This is an $O(\Delta t^{2 n_c})$-accurate quadrature on smooth integrands and is consistent with the defect-constraint discretization order.

**NLP decision variables and constraint counts.** For a planar ($n=4$) P1 run with $N$ segments, degree $d$, $n_c$ collocation nodes per segment, and control dimension $m=2$:

- **State control points:** $N d + 1$ unique vectors in $\mathbb{R}^n$ after $C^0$-sharing at interior knots $\Rightarrow (Nd+1)\cdot n$ scalars. With fixed endpoints $x(t_0)=x_0$ and $x(t_f)=x_f$, the two boundary control points $P_0^{(1)}$ and $P_d^{(N)}$ are pinned, leaving $(Nd-1)\cdot n$ free state variables.
- **Controls:** $N\cdot n_c\cdot m$ scalars (piecewise constants at collocation nodes; no continuity enforced).
- **Defect equalities:** $N\cdot n_c\cdot n$ scalars. The full Earth–Mars reference ($N=8, d=7, n_c=12, n=4, m=2$) gives $\sim 420$ decision variables and $\sim 384$ equality constraints — consistent with the numbers reported in `AAE568_Technical_Documentation.docx` §4.3.2.
- **Boundary equalities:** $2n$ scalars (fixed endpoints).

**Parameterization lesson learned.** An earlier version parameterized only position $r(t)$ and back-solved control from $u=\ddot r - f_{\text{grav}}(r)$. Any smooth curve connecting the endpoints was "feasible" because the dynamics were satisfied by construction rather than enforced — so the NLP drove to trivially small costs by choosing unphysical curves. The fix (recorded in Tech Doc §6.1) is to parameterize the **full** state $[r,v]$ with Bézier control points, declare $u$ an explicit NLP variable, and enforce $\dot x = f(x,u)$ as hard defect constraints. This is the setup documented in §6.4 and used across P0–P1.

**Three concrete transcription flavors in this project.**

- **Bézier collocation + IPOPT (P1, $N=16$, degree 7, $n_c$ Gauss–Legendre nodes).** Solved by CasADi + IPOPT with exact AD derivatives. The JSON method label `global_bezier_ipopt` is a legacy name for this segmented setup — "global" describes the NLP (one monolithic sparse solve) rather than a single-polynomial transcription.
- **Segmented Bézier + SLSQP ($N$-sweep, P1).** Same transcription, but SLSQP driver with finite-difference gradients; the CFD **h-refinement** analogue (fix polynomial order, subdivide the mesh, watch cost and defect converge). The N-sweep is the mesh-convergence study (`nsweep_findings.md`).
- **RK4 multiple-shooting (P2 only).** Replace the Bézier state with an explicit RK4 step from segment-start to segment-end, with piecewise-constant control per segment as the NLP variable. Defect constraints become $x_{k+1}^{\mathrm{NLP}} = \text{RK4}(x_k^{\mathrm{NLP}}, u_k, \Delta t)$. Preferred in P2 because `astropy`'s ephemeris lookups are non-symbolic Python calls; embedding them inside CasADi's implicit collocation stencil would require either pre-tabulating Sun/Moon states as NLP parameters or wrapping the lookups in CasADi External Functions (which fall back to finite-difference Jacobian blocks and lose end-to-end AD). Explicit RK4 with ephemeris states sampled once per segment and passed in as segment parameters sidesteps both paths.

**Connection back to the course.** Although the Bernstein basis and collocation machinery are not in the lecture notes, their optimality structure is: at any IPOPT iterate satisfying first-order KKT, the Lagrange multipliers on the defect constraints $\text{def}_k=0$ are (up to sign and scaling) the **discrete costates** $\lambda(\tau_k)$ of the continuous PMP formulation in §6.2. The KKT stationarity on control decision variables reproduces $\partial H/\partial u = 0$ at the collocation nodes. So the direct branch is not a departure from PMP — it is PMP solved implicitly inside an NLP instead of explicitly via shooting, and the numerical agreement across methods observed in §5.1–§5.2 is precisely this equivalence in action.

### 6.5 NLP solvers
- **SLSQP** (sequential least-squares QP) on segmented Bézier (P1): finite-difference gradients, dense Jacobian, tens to a few hundred SLSQP iterations per solve (max observed: 294 at $N=16$). Scales as roughly $O(N^2 \cdot \text{iter})$ in wall time. Analytically, this is the direct descendant of Kuhn–Tucker static optimization covered in lecture (pp. 123–138) — the "augmented cost" slide set used for the PMP derivation is the same machinery, applied at a finite-dimensional NLP rather than an infinite-dimensional problem.
- **IPOPT** (interior-point) with MUMPS sparse linear solve and **CasADi automatic differentiation** (exact Jacobian and Hessian): 2–5 outer iterations on the same problems. This is the workhorse of modern direct-transcription optimal control (PSOPT, rockit, CasADi/Opti, etc.; the GPOPS-II family is a sibling that pairs Radau pseudospectral collocation with SNOPT rather than IPOPT), and the project treats "IPOPT + AD" as the tool of record.
- **Warm-start philosophy.** Cold-start IPOPT from a linear state-interpolation guess lands the CR3BP solve in a non-PMP basin ($J=0.0878$ instead of $0.0432$). Warm-starting from a converged shooting solution (P1) or from the NASA OEM (P2) is essential and is made explicit in the methodology rather than hidden. The bridge between branches is quantitative: if $\{u_k^*\}$ is the IPOPT-optimal control at collocation nodes, then $\lambda_v(t_0)\approx -2\,u_0^*$ seeds a subsequent shooting solve that would otherwise diverge from a random guess — a clean demonstration of the §6.4 "multipliers-as-costates" equivalence used for practical robustness.

### 6.6 Dynamics
- **P0.** Two-body Keplerian: $\ddot r = -\mu r/\|r\|^3 + u$.
- **P1.** Planar CR3BP in the Earth–Moon rotating frame: $\mu = 0.01215058561$; Lyapunov targets built with Ax=0.02 at L1 and L2. Boundary states are periodic-orbit samples.
- **P2.** Dimensional N-body with `astropy`'s builtin ephemeris: Earth + Moon + Sun point-mass gravity on the Orion state, propagated in Earth-centered inertial. The builtin is a low-precision analytical approximation from the *Explanatory Supplement to the Astronomical Almanac* — it is *not* JPL DE405 or DE432s, despite the label `ephem: astropy_builtin_DE405` that appears in `results_summary.json` (the label inherits an imprecise comment in the loader and should be read as "astropy builtin"). For a 10-day cislunar arc the accuracy is adequate; for pointing-grade work a JPL SPK kernel should be loaded explicitly via `solar_system_ephemeris.set('de432s')`. Truth is the NASA OEM for Artemis II, flown 2026-04-02 through 2026-04-10 UTC, extracted from `Artemis_II_OEM_2026_04_10_Post-ICPS-Sep-to-EI.asc`.

### 6.7 Why this specific ladder of three problems
The sequence P0 → P1 → P2 isolates three distinct failure modes of direct transcription so the report can attribute each observation to a single cause:

- P0 isolates *code correctness* (linear-ish dynamics, known analytic answer).
- P1 isolates *method tradeoff* (nonlinear but cheap dynamics, PMP reference available, mesh-convergence study tractable).
- P2 isolates *scale* (expensive dynamics, no analytic reference, NASA truth data, realistic mission profile).

---

## 7. Link to AAE 568 course content

Page numbers reference `AAE568-Lecture_Notes.pdf` (Prof. Hwang, Spring 2026). Every methodological claim the project makes traces to a specific lecture slide and is used for its named purpose, not extrapolated beyond.

| AAE 568 topic | Slides | How the project uses it |
|---|---|---|
| Calculus of variations | pp. 144 | Used to motivate the augmented-cost construction; referenced without re-derivation. |
| Hamiltonian and variation of $J_a$ | pp. 145–146, 150 | Direct import — $H=\|u\|^2+\lambda^{\top}(f+Bu)$ for each phase (no $\tfrac12$ prefactor; matches §6.1 convention and gives $u^*=-\tfrac12\lambda_v$). |
| Euler–Lagrange equations | p. 147 | State/costate ODEs in §6.2 are Euler–Lagrange applied to $J_a$. |
| TPBVP formulation | pp. 148–149 | P0 and P1 shooting code literally assembles the same TPBVP structure from the slide. |
| Pontryagin Minimum Principle | pp. 154–157 | Stationarity + minimum condition used to derive $u^{*}=-\tfrac12\lambda_v$ in closed form (unconstrained control; no bang–bang structure needed). |
| Step 1: TPBVP | pp. 186–189 (stirred-tank example) | Pedagogically the closest in-class parallel — same "write Hamiltonian, derive costate ODEs, assemble residual" workflow. Worth citing in the methods section as the template. |
| Step 2: Numerical TPBVP | pp. 190–191 | In-class example uses shooting; project uses the same tool (`fsolve`) then extends to direct transcription when shooting fragility becomes the story. |
| Constrained optimization (Lagrangian) | pp. 134–138 | Background for the NLP defect-constraint formulation; SLSQP and IPOPT both implement this. |
| Inequality constraints on control | pp. 161–166 | Relevant for P2 full-mission, where per-segment $|u|\le u_{\max}$ bounds were added (control-magnitude bounds enumerated in `RETUNING_AUDIT.md`). Not exercised in P0/P1. |
| Static optimization / KKT | pp. 123–133 | Background only — the NLP first-order conditions at every IPOPT iterate are the slide's KKT stationarity restated for sparse dynamic-system constraints. |

Topics covered in AAE 568 but **deliberately not used** in this project:

- *Dynamic programming / HJB* (pp. 195–217). The project is a two-point problem with smooth control, so PMP is the natural tool. DP would require state-space discretization and blow up in 6-D. An honest sentence in §4 Limitations or the report's conclusion says so.
- *LQR / ARE machinery* (pp. 232–266). Dynamics are nonlinear; LQR would be a local-linearization feedback law, not a trajectory design tool. Could be cited in the "future work" column for Moon-proximity stationkeeping, not for the Earth-to-Moon transfer.
- *MPC / receding-horizon control* (pp. 276–301). MPC is the closed-loop counterpart of the open-loop trajectory solved here. A final paragraph linking "open-loop optimal trajectory → reference for MPC tracking controller" is a legitimate bridge if scope allows.
- *Kalman filtering / LQE* (pp. 330–372). Not used; the project assumes full state is available.

---

## 8. Report & presentation construction notes

These are the things that will bite if forgotten — a checklist for whoever writes the final artifacts.

### 8.1 Framing decisions
- **Lead with the method comparison, not the mission.** The course is optimal control, not astrodynamics. The Artemis II phase is validation of the method, not a re-design of Artemis II. Open on "indirect vs direct transcription, and when does each fail?" and let the three phases illustrate it.
- **Keep the three-phase arc intact.** P0 → P1 → P2 is the pedagogical spine; reordering (e.g., leading with Artemis) sacrifices the "ladder of isolated failure modes" argument from §6.7.
- **Scope bar is pedagogical, not publication.** No need for Lawden's necessary conditions, primer vector theory, or free-$t_f$ transversality. Cite them as future work if asked; do not implement. (See `feedback_aae568_scope.md`.)

### 8.2 Claims already addressed in this file
Two contradictions between an earlier draft of the narrative and the artifacts have now been reconciled directly in §3 and §4:

1. **Wall-clock ratio.** "87× slower" updated to "~88× slower (528.9 s vs 6.02 s)" using the re-instrumented `results_summary.json` numbers for both terms, rather than mixing the N-sweep internal 5.33 s shooting timing with the re-run SLSQP wall-clock. If the final report uses the older N-sweep self-consistent pair (5.33 s, 87×), cite that figure from `nsweep_findings.md` rather than from the JSON; either pair is defensible as long as the numerator and denominator come from the same run.
2. **"Converges without re-tuning."** Rewritten per `RETUNING_AUDIT.md` Option C: the IPOPT+AD tool *class* (MUMPS, exact AD, warm-start-from-reference philosophy) carries over from P1 to P2; tolerances, iteration caps, mesh schedule, transcription form, and control bounds are problem-specific and enumerated in the audit. The report and slides should use the same framing verbatim to stay consistent with NARRATIVE §3 P2 and §4.

### 8.2b Claims still to watch when writing the report
- When restating the 88× number in the report, make sure the accompanying figure/caption uses the same (528.9 s, 6.02 s) pair drawn from `results_summary.json` rather than the earlier `nsweep_findings.md` (467.6 s, 5.33 s) pair. Either is correct, but mixing numerators and denominators across runs is not.
- The full RETUNING_AUDIT table (MUMPS kept, `tol` loosened in full-mission, `constr_viol_tol` dropped, mesh cascade removed, shooting `maxfev` cut 25×, integrator RK45→DOP853, etc.) belongs in an appendix of the report, not in the main text — too much granularity to read as methods-section prose. The one-sentence Option C summary is what §3 uses; the appendix is for reproducibility.

### 8.3 Numbers to sanity-check before inclusion
- **TLI Δv = 22.59 km/s and Entry-correction Δv = 9.72 km/s** in the full-mission record. Apollo's TLI was ~3.05 km/s; 22 km/s is ~7× too large. Plausible explanations: (a) `total_dv_km_s` in the burn-detection rollup is $\int |u|\,dt$ summed over *all* segments above a threshold (including coast leakage), (b) units or nondimensionalization issue between km/s and some scaled residual, (c) the piecewise-constant control is reconstructing a physical Δv by stacking acceleration values. Verify before showing on a slide — a 22 km/s TLI number will be the first thing an aerospace audience flags.
- **Post-TLI IPOPT cost $J=5.3\times 10^{-12}$.** This is numerically zero because the OEM arc is ballistic (no burn). Explain this in the report so it doesn't read as a suspiciously tight convergence.
- **SLSQP N=32 "timeout > 900 s"** in `nsweep_findings.md` is not in `results_summary.json`. Either add a `converged: false, reason: "timeout"` row for completeness or drop the N=32 column from the report table.

### 8.4 Figures that pull their weight
Prioritize these; cut others to stay under 10 minutes of slide time:

- `pareto_J_error_vs_wall_time.png` — the one chart that makes the method-comparison argument quantitatively. Should be slide 1 of Results.
- `cr3bp_transfer_comparison.png` — three methods overlaid on the same L1↔L2 trajectory in the rotating frame. Visual confirmation of agreement.
- `cr3bp_transfer_segmented_convergence.png` — monotone-decreasing cost vs $N$; pair with the 10× wall-clock growth caption.
- `jacobi_constant.png` — dynamics integrity cue; caveat that $C$ is not conserved under thrust.
- `convergence_history.png` (Artemis) — IPOPT per-iteration {obj, constr_viol, dual_inf} trace for the full mission; demonstrates interior-point's 4-iter convergence.
- `cr3bp_transfer_animation.gif` — for the presentation only (not the written report), closes the talk with motion.

### 8.5 Things to say, once, then move on
- Fixed-$t_f$ is a modeling choice, not an oversight. Free-$t_f$ adds a transversality condition $H(t_f)=0$ and an extra NLP variable; this is a clean extension but was scoped out.
- Segmented Bézier = hp-pseudospectral in the GPOPS-II sense. Stating this once lets the audience map the work onto a known tool family.
- "IPOPT with AD is the *class* of tool that scales" — this is the single defensible pedagogical claim of the project. Everything in P0–P2 is evidence for it. Say it at the start of the conclusion and let the results land.

### 8.6 Framing caveats — now carried in the body
The three "don't over-sell" reminders that earlier lived here have been folded into the narrative body so the report inherits them by default:

- Shooting-vs-IPOPT 2500× ratio is now stated with its budget-constrained caveat inline in §5.3 (shooting's `maxfev` was capped 25× below P1 to keep the 15-seed sweep tractable). Use the §5.3 phrasing verbatim; do not restate the raw 2500× without the caveat.
- N-sweep's SLSQP cost attribution is now in §3 P1 Takeaway: dense FD gradients on a growing NLP are what break, not SLSQP as an algorithm. Do not conflate.
- Jacobi-constant's diagnostic scope is already bounded in §3 P1: it confirms method agreement on a common controlled arc, not bounded between-node collocation error. Do not upgrade the figure's caption beyond that in the report.
