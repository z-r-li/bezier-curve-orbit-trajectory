# ThreeD/ — Future-Work Exhibit

**Status:** Scoped out of the final course project. Kept in the repo as an exhibit, not an active line of work.

## What this folder is

Early 3D prototyping for what was originally planned as Phase 3: **LEO → NRHO insertion** in both a 3D CR3BP formulation and an astropy-ephemeris formulation. See `References/phase3_implementation_plan.md` for the original plan and `References/NRHO_transfer_sources.md` for the NRHO family references.

The directory contains two independent prototypes:

- **CR3BP 3D** (`cr3bp_3d.py`, `leo_to_nrho_cr3bp.py`) — rotating-frame 3D Circular Restricted Three-Body transfer from a LEO parking orbit to an L2 southern NRHO.
- **Ephemeris 3D** (`ephem_dynamics.py`, `ephem_boundaries.py`, `leo_to_nrho_ephem.py`) — same transfer under astropy's builtin Earth–Moon–Sun ephemeris.

Figures (`2d_projections.png`, `3d_trajectory.png`, `control_profile.png`, `ephem_nrho_3d.png`, `ephem_nrho_dv_comparison.png`, `ephem_nrho_primer.png`, `summary_stats.png`) are outputs of those two prototypes.

## Why it was scoped out

The course project's final three-phase arc (P0 → P1 → P2; see `NARRATIVE.md`) uses **Artemis II post-flight OEM validation** (Phase 2) as the "mission-relevant validation" capstone rather than a synthetic LEO→NRHO insertion. The NRHO work was informative during method development but is orthogonal to the locked report storyline:

- Report/presentation claims live in `NARRATIVE.md`.
- Backing artifacts live in `results_summary.json` and the Phase 0/1/2 folders.
- Nothing in `ThreeD/` feeds `results_summary.json` or any locked claim.

## If you reopen this later

- The CR3BP 3D transfer is cold-start basin-sensitive the same way P1 was; warm-start from a converged 2D transfer and step through increasing out-of-plane boundary conditions.
- The ephemeris prototype already uses the astropy-builtin path that carries over to `Artemis2/`; loading a JPL SPK kernel (`de432s`) would be the first precision upgrade. See `NARRATIVE.md §6.5` for the current ephemeris discussion.
- Primer-vector / Lawden-conditions analysis of the NRHO arrival is the natural extension; it is explicitly descoped from the course project but would make a reasonable continuation.

**Do not cite figures or numbers from this folder in the report or presentation** — they are not part of the locked narrative.
