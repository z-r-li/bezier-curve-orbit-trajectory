# Reference Sources: LEO-to-NRHO Transfer Design

Compiled 2026-04-13 for AAE 568 Course Project — Phase 3 (3D LEO → 9:2 L2 Southern NRHO)

---

## Primary Data Source

1. **JPL Three-Body Periodic Orbits Database**
   - API: https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html
   - Interactive tool: https://ssd.jpl.nasa.gov/tools/periodic_orbits.html
   - Query used: `sys=earth-moon&family=halo&libr=2&branch=S`
   - Provides: 1,535-member L2 southern halo family with state vectors, periods, Jacobi constants, stability indices
   - **This is where our 9:2 NRHO initial conditions come from.**

## NASA Technical Reports (NTRS)

2. **"Gateway Destination Orbit Model: A Continuous 15-Year NRHO Reference Trajectory"**
   - Authors: D. C. Davis et al.
   - Ref: NASA/TM-2019-20190030294
   - URL: https://ntrs.nasa.gov/api/citations/20190030294/downloads/20190030294.pdf
   - Content: Full 15-year Gateway reference trajectory in ephemeris model; NRHO orbit parameters, eclipse avoidance, station-keeping budgets.

3. **"Targeting Cislunar Near Rectilinear Halo Orbits for Human Space Exploration" (AAS 17-267)**
   - Authors: E. M. Zimovan, K. C. Howell, D. C. Davis
   - Ref: NASA/TM-2017-20170001352
   - URL: https://ntrs.nasa.gov/api/citations/20170001352/downloads/20170001352.pdf
   - Content: NRHO targeting in CR3BP and ephemeris; transfer design methodology from Earth to NRHO.

4. **"Artemis I Trajectory Design and Optimization" (AAS 20-649)**
   - Authors: L. Burke et al.
   - Ref: NASA/TM-2020-20205005150
   - URL: https://ntrs.nasa.gov/api/citations/20205005150/downloads/AAS20649_Artemis1_Trajectory_Design_Optimization.pdf
   - Content: Artemis I mission design; TLI parameters, NRHO insertion, powered lunar flyby trajectory.

5. **"Generating Exploration Mission-3 Trajectories to a 9:2 NRHO"**
   - Authors: D. C. Davis et al.
   - Ref: NASA/TM-2019-20190000902
   - URL: https://ntrs.nasa.gov/api/citations/20190000902/downloads/20190000902.pdf
   - Content: EM-3 trajectory generation to the 9:2 NRHO; transfer design details.

6. **"Cloud Computing Methods for NRHO Trajectory Design"**
   - Authors: D. C. Davis et al.
   - Ref: NASA/TM-2019-20190029093
   - URL: https://ntrs.nasa.gov/api/citations/20190029093/downloads/20190029093.pdf
   - Content: Computational methods for NRHO trajectory design; orbit families and transfers.

7. **"Enabling Global Lunar Access for Human Landing Systems"**
   - Authors: various (NASA JSC)
   - Ref: NASA/TM-2020-20200002920
   - URL: https://ntrs.nasa.gov/api/citations/20200002920/downloads/20200002920.pdf
   - Content: NRHO-to-LLO transfers; surface access from Gateway.

## Purdue University (Howell Group)

8. **"Earth-Moon Near Rectilinear Halo and Butterfly Orbits" (AAS 18-406)**
   - Authors: R. Whitley, D. C. Davis, L. Burke, B. McCarthy, R. Power, K. McGuire, K. C. Howell
   - URL: https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Conferences/2018_AAS_WhiDavBurMcCPowMcGHow.pdf
   - Content: NRHO and butterfly orbit families in Earth-Moon CR3BP; transfers from Earth; Jacobi constant C ≈ 3.047, stability index ≈ 2.19 for 9:2 family.

9. **"Characteristics and Design Strategies for Near Rectilinear Halo Orbits"** (Zimovan MS Thesis, 2017)
   - Author: E. M. Zimovan; Advisor: K. C. Howell
   - URL: https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Masters/2017_Zimovan.pdf
   - Content: Comprehensive treatment of NRHO characteristics, stability, transfers; primary reference for 9:2 NRHO design.

10. **"Phase Control and Eclipse Avoidance in NRHOs" (AAS 20-xxx)**
    - Authors: D. C. Davis, S. Khoury, K. C. Howell, D. Sweeney
    - URL: https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Conferences/2020_AAS_DavKhoHowSwe.pdf
    - Content: 9:2 synodic resonance for eclipse avoidance; phase control strategies.

11. **"Low-Thrust Transfer Design Based on Collocation Techniques" (AAS 17-626)**
    - Authors: B. Pritchett, K. C. Howell, D. Grebow
    - URL: https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Conferences/2017_AAS_PriHowGre.pdf
    - Content: Collocation-based transfer design in CR3BP — **directly relevant to our Bézier collocation approach.**

## NASA White Papers & Overviews

12. **"How: NRHO — The Artemis Orbit"**
    - Author: Nujoud Merancy (NASA SAO)
    - URL: https://www.nasa.gov/wp-content/uploads/2023/10/nrho-artemis-orbit.pdf
    - Content: High-level overview of NRHO selection rationale, orbit parameters, Artemis architecture.

13. **"Why NRHO: The Artemis Orbit"**
    - Source: Lunar and Planetary Institute (LPI)
    - URL: https://www.lpi.usra.edu/lunar/artemis/resources/WhitePaper_2023_WhyNRHA-TheArtemisOrbit.pdf
    - Content: Rationale for 9:2 NRHO selection; comparison with other staging orbits.

14. **"A Lunar Orbit That's Just Right for the International Gateway"**
    - Source: NASA JSC
    - URL: https://www.nasa.gov/centers-and-facilities/johnson/lunar-near-rectilinear-halo-orbit-gateway/
    - Content: Public overview of Gateway NRHO; period ~7 days, perilune ~3,000 km, apolune ~70,000 km.

## Other References

15. **"Assessment of Cislunar Staging Orbits to Support the Artemis Program" (AAS 22-762)**
    - URL: https://ntrs.nasa.gov/api/citations/20220011365/downloads/AAS%2022-762.pdf
    - Content: Trade study of cislunar staging orbits including NRHO.

16. **"Optimal Outbound Transfer Windows for Artemis III and Beyond"**
    - Journal: Astrodynamics (Springer, 2024)
    - URL: https://link.springer.com/article/10.1007/s42064-024-0254-0
    - Content: Transfer window optimization for Artemis III missions to NRHO.

17. **Delta-v Budget** (Wikipedia, compiled from multiple NASA sources)
    - URL: https://en.wikipedia.org/wiki/Delta-v_budget
    - Content: LEO→TLI ≈ 3.20 km/s; TLI→Gateway ≈ 0.43 km/s; Gateway→LLO ≈ 0.73 km/s.

18. **"Near Rectilinear Halo Orbits"** (Degenerate Conic blog)
    - URL: https://degenerateconic.com/near-rectilinear-halo-orbits.html
    - Content: Tutorial with L2 southern NRHO initial conditions: x₀=1.0277926091, z₀=−0.1858044184, ẏ₀=−0.1154896637, T=1.5872714606; l*=384400 km, t*=375190.26 s.

---

## Verified 9:2 NRHO Data (from JPL database, propagation-verified 2026-04-13)

### Earth-Moon CR3BP System Parameters
| Parameter | Value |
|-----------|-------|
| Mass ratio μ | 0.01215058560962404 |
| Length unit l* | 389,703.265 km |
| Time unit t* | 382,981.289 s (4.433 days) |
| Velocity unit v* | 1.01755 km/s |

### 9:2 L2 Southern NRHO — Initial State at Apolune (y = 0 crossing)
| Component | Value (nondimensional) |
|-----------|----------------------|
| x₀ | 1.0196625817 |
| y₀ | 0.0 |
| z₀ | −0.1804191873 |
| ẋ₀ | 0.0 |
| ẏ₀ | −0.0980598247 |
| ż₀ | 0.0 |

### Orbital Properties
| Property | Value |
|----------|-------|
| Period (nondim) | 1.479980 |
| Period (days) | 6.560 |
| Jacobi constant | 3.04891 |
| Stability index | 1.255 |
| Perilune radius | 2,931 km (1,194 km altitude) |
| Apolune radius | 71,395 km (69,657 km altitude) |
| 9:2 resonance error | 0.03% |

### LEO Departure (Artemis Baseline)
| Parameter | Value |
|-----------|-------|
| Altitude | 185 km (circular) |
| Inclination | 28.5° (KSC) |
| Radius | 6,556 km |
| Circular velocity | 7.797 km/s |
| r_LEO (nondim) | 0.01682 |
| Earth center (nondim) | (−0.01215, 0, 0) |

### Delta-V Budget (High-Thrust)
| Maneuver | Δv |
|----------|-----|
| LEO → TLI | 3.13–3.20 km/s |
| TLI → NRHO insertion | ~0.43 km/s |
| Total | ~3.56–3.63 km/s |
| Transfer time | 4–6 days (direct) |
