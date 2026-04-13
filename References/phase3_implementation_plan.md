# Phase 3 Implementation Plan: 3D LEO → 9:2 NRHO Transfer

**AAE 568 Course Project — Bézier Collocation vs Shooting for Cislunar Trajectory Design**
Drafted 2026-04-13

---

## Objective

Solve the minimum-fuel 3D transfer from a 185 km LEO (28.5° inclination) to the 9:2 L2 southern NRHO in the Earth-Moon CR3BP, comparing Bézier collocation (IPOPT) against indirect shooting.

---

## What We Already Have (from Phases 1–2)

| Capability | File | Status |
|---|---|---|
| 3D CR3BP EOM (uncontrolled + controlled) | `Earth-Mars/dynamics.py` | Ready (cr3bp_ode, cr3bp_controlled_ode) |
| Jacobi constant (3D) | `Earth-Mars/dynamics.py` | Ready |
| BezierCollocation with generic `pos_dim` | `Earth-Mars/bezier.py` | Ready (supports pos_dim=3) |
| CR3BPBezierCollocation (Coriolis handling) | `Planer/cr3bp_transfer.py` | Planar only — needs 3D extension |
| CasADi/IPOPT collocation for CR3BP | `Planer/ipopt_collocation.py` | Planar only — needs 3D extension |
| Indirect shooting (scipy fsolve) | `Earth-Mars/shooting.py` | 2-body only — needs CR3BP 3D version |
| Planar STM propagation (4×4) | `Planer/cr3bp_planar.py` | Needs 6×6 upgrade |
| Lyapunov orbit diff. correction | `Planer/cr3bp_planar.py` | Needs halo orbit extension |

---

## Implementation Steps

### Step 1 — 3D CR3BP Infrastructure (`cr3bp_3d.py`, new file)

Build the 3D foundation module:

**1a. Core dynamics (already in dynamics.py, refactor into this module)**
- `cr3bp_3d_ode(t, X, mu)` — 6D state: [x, y, z, vx, vy, vz]
- `cr3bp_3d_controlled_ode(t, X, mu)` — 12D state+costate with u* = −½λᵥ
- `cr3bp_gravity_3d(r, mu)` — Returns [Ux, Uy, Uz] (pseudo-potential gradient, excluding Coriolis)
- `jacobi_constant(state, mu)` — Energy integral for validation

**1b. STM propagation (6×6)**
- `cr3bp_3d_stm_ode(t, Y, mu)` — Propagate 6D state + 6×6 STM = 42 ODEs
- Requires the 6×6 variational matrix A(t), built from the pseudo-potential Hessian:
  ```
  Uxx, Uxy, Uxz
  Uxy, Uyy, Uyz
  Uxz, Uyz, Uzz
  ```
  Plus Coriolis contribution (off-diagonal 2Ω terms).
- `propagate_with_stm_3d(state0, t_span, mu)` — Wrapper with event detection

**1c. Differential correction for halo orbits**
- `compute_halo_orbit(xL2, Az, mu)` — Single-shooting differential correction
  - Exploits xz-plane symmetry of L2 southern halo family
  - At y=0 crossing: free variables [x₀, z₀, ẏ₀], targets [ẋ(T/2)=0, ż(T/2)=0]
  - 3×2 Jacobian from STM, pseudo-inverse correction
  - Richardson 3rd-order approximation for initial guess
- Purpose: verify we can independently reproduce the JPL 9:2 NRHO (not strictly needed since we have JPL data, but good for validation and we may want neighboring NRHOs)

**Validation gate:** Propagate the JPL 9:2 NRHO state for one full period. Check:
- Jacobi constant conserved to ~10⁻¹²  ✓ (already verified)
- State periodicity |x(T)−x(0)| < 10⁻⁷  ✓ (already verified)
- Perilune ≈ 2,931 km, apolune ≈ 71,395 km  ✓ (already verified)

### Step 2 — LEO State in the Rotating Frame

The LEO departure must be expressed in the CR3BP rotating frame:

- Earth center is at (−μ, 0, 0) in the rotating frame
- For a circular LEO at 185 km altitude, 28.5° inclination:
  - r_LEO = 6556 km = 0.01682 nondim
  - The orbit plane is tilted 28.5° from the Earth-Moon plane (xy-plane)
  - Parameterize departure point by true anomaly θ on the LEO

**Key decision: how to handle the LEO departure**

Option A (simplified, recommended for course project): Fix departure at a specific point on LEO. Choose the point that gives the best alignment with the transfer. This reduces the problem to a fixed-endpoint TPBVP.

Option B (full optimization): Include the LEO departure angle θ as an optimization variable. More realistic but adds complexity.

**Recommendation:** Start with Option A. Pick a departure point on the LEO in the xz-plane (θ chosen so the velocity vector points roughly toward the Moon). We can later upgrade to Option B if time permits.

LEO state at departure (nondimensional rotating frame):
```
x₀ = −μ + r_LEO·cos(θ)·cos(i_rot)
y₀ = r_LEO·cos(θ)·sin(i_rot)    [or adjust for rotating frame]
z₀ = r_LEO·sin(θ)·sin(28.5°)     [out-of-plane component]
```
Plus circular velocity + rotating-frame correction (−Ω×r term).

Note: The inclination in the rotating frame depends on the geometry at epoch. For a CR3BP study this is a free parameter — we pick a representative value.

### Step 3 — Transfer Problem Formulation

**Problem statement (minimum-energy, fixed-time):**
```
min  J = ∫₀ᵗᶠ |u(t)|² dt

s.t. ẋ = f_CR3BP(x) + B·u       (3D CR3BP + low-thrust control)
     x(0) = x_LEO                (LEO departure state)
     x(tf) ∈ NRHO               (arrive on the 9:2 NRHO)
     tf free or fixed            (transfer time)
```

**Boundary conditions:**
- Initial: Full state on LEO (6 conditions)
- Final: Match position and velocity on the NRHO at some phase φ
  - If we target apolune insertion: match the JPL apolune state (6 conditions)
  - If we allow free arrival phase: parameterize arrival point on NRHO by phase φ (reduces terminal conditions to a constraint manifold)

**Recommended approach for course project:** Target the NRHO apolune state directly (fixed endpoint). This gives a clean 6-condition terminal constraint matching our existing framework.

### Step 4 — Indirect Shooting Method (3D)

Extend the existing shooting infrastructure:

**4a. `cr3bp_3d_shooting.py`**
- 12D ODE: [x,y,z,vx,vy,vz, λx,λy,λz,λvx,λvy,λvz]
- Optimal control: u* = −½[λvx, λvy, λvz]
- Costate dynamics from Hamiltonian (6×6 variational equations of the augmented system)
- Shooting function: propagate from t₀ to t_f, return residual = x(tf) − x_target (6D)
- Free variables: λ(0) = [λx₀, λy₀, λz₀, λvx₀, λvy₀, λvz₀] (6 unknowns, 6 equations)

**4b. Initial guess strategy**
- This is the critical challenge for shooting in the CR3BP
- Strategy 1: Lambert arc in 2-body → rotate to CR3BP frame → use as initial guess for costate
- Strategy 2: Solve a sequence of easier problems via continuation
  - Start with low μ (nearly 2-body), solve, then increase μ toward Earth-Moon value
  - Or start with short transfer time and extend
- Strategy 3: Use the Bézier collocation solution to extract approximate costates (reverse-engineer λ from the converged u*)

### Step 5 — Direct Bézier Collocation with IPOPT (3D)

Extend `ipopt_collocation.py` to 3D:

**5a. `cr3bp_3d_ipopt_collocation.py`**
- State dimension: 6 (x, y, z, vx, vy, vz)
- Control dimension: 3 (ux, uy, uz)
- Dynamics defects now include z-component:
  ```
  ẍ − 2ẏ − Ux = ux
  ÿ + 2ẋ − Uy = uy
  z̈       − Uz = uz
  ```
- NLP variables per segment: 6×(degree+1) control points + 3×n_collocation controls
- Total NLP for N segments, degree d, n_c collocation points:
  - ~N × [6(d+1) + 3n_c] variables
  - ~N × 6n_c defect constraints + 6 boundary + 6(N−1) continuity

**5b. Mesh refinement cascade (key for convergence)**
```
Stage 1: N=4 segments, degree=3, n_c=4    → ~200 variables (coarse)
Stage 2: N=8 segments, degree=5, n_c=8    → ~800 variables (medium)
Stage 3: N=16 segments, degree=7, n_c=12  → ~2500 variables (fine)
```
Each stage warm-starts from the previous solution by interpolating the Bézier onto the finer mesh.

**5c. IPOPT settings**
- Sparse Jacobian via CasADi AD (block-diagonal structure)
- IPOPT options: max_iter=3000, tol=1e-8, acceptable_tol=1e-6
- MA27 or MUMPS linear solver

### Step 6 — Comparison and Analysis

Run both methods on the same problem and compare:

| Metric | Shooting | Bézier/IPOPT |
|--------|----------|--------------|
| Convergence rate | % of initial guesses that converge | % of mesh cascades that converge |
| Basin of convergence | Sensitivity to λ₀ perturbation | Sensitivity to initial trajectory guess |
| Computation time | Wall-clock (single solve) | Wall-clock (full cascade) |
| Solution quality | Δv, Jacobi conservation, constraint satisfaction | Same |
| Trajectory shape | Plot in rotating + inertial frames | Same |

### Step 7 — Visualization and Report

- 3D trajectory plots in rotating and inertial frames (matplotlib 3D)
- NRHO orbit with transfer arc overlay
- Control history u(t) for both methods
- Convergence comparison plots
- Update Technical Documentation (docx) with Phase 3 results

---

## Suggested File Structure

```
AAE 56800/Project/
├── Earth-Mars/          (unchanged)
├── Planer/              (unchanged)
├── ThreeD/              (NEW — Phase 3)
│   ├── cr3bp_3d.py              (3D dynamics, STM, halo orbit tools)
│   ├── cr3bp_3d_shooting.py     (indirect shooting, 3D)
│   ├── cr3bp_3d_collocation.py  (Bézier + IPOPT, 3D)
│   ├── leo_to_nrho_transfer.py  (main driver: problem setup + comparison)
│   └── plot_results.py          (3D visualization)
└── References/
    ├── NRHO_transfer_sources.md (done)
    └── phase3_implementation_plan.md (this file)
```

---

## Risk Assessment and Fallbacks

| Risk | Mitigation |
|------|-----------|
| Shooting fails to converge (sensitive to initial costate) | Use Bézier solution to warm-start costates; μ-continuation |
| IPOPT collocation NLP too large | Start very coarse (4 segments, degree 3); increase only as needed |
| Transfer time unknown a priori | Start with tf ≈ 4–6 days (direct transfer); can extend to low-energy (~100 days) |
| 3D LEO state in rotating frame is tricky | Start with a simplified coplanar departure (z₀=0), then add inclination |
| Project timeline pressure | Priority: get ONE method working end-to-end first (recommend collocation), then add shooting comparison |

---

## Recommended Execution Order

1. **cr3bp_3d.py** — 3D dynamics + STM + NRHO validation (half day)
2. **cr3bp_3d_collocation.py** — IPOPT Bézier for 3D CR3BP (1–2 days)
3. **leo_to_nrho_transfer.py** — Problem setup + run collocation (half day)
4. **cr3bp_3d_shooting.py** — Indirect shooting + warm-start from collocation (1 day)
5. **Comparison + plots + report update** (1 day)

**Total estimated effort: ~4–5 days**
