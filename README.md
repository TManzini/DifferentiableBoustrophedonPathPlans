# Differentiable Boustrophedon Path Plans

This repository contains the source code associated with the ICRA'24 paper "[Differentiable Boustrophendon Path Plans](https://arxiv.org/abs/2309.09882)"

## Recent Extensions (Post-Paper)

This codebase has been extended beyond the original paper with:

1. **Closed-form transect-length solution (Eq. 6):** An exact, O(faces) geometric chord-length solution to the integral in Equation 6, derived via antisymmetric cancellation at polygon boundaries. This replaces the temperature-dependent trapezoidal approximation with an exact result valid as Temp→∞.

2. **Smoothed closed-form variant:** A C∞ smoothed variant using LogSumExp and softplus, with a tunable smoothness parameter (beta) that trades off between a melted (easy-to-optimize) landscape and the exact sharp chord. Enables gradient descent to escape local minima without relying on temperature annealing of the trapezoidal rule.

3. **Non-convex polygon support:** Ear-clip triangulation extends both closed-form variants (exact and smoothed) to arbitrary simple polygons (convex or concave). The chord through a concave polygon is the sum of per-triangle closed-form chords, exact by linearity of integration over disjoint interiors.

4. **Gradient-descent optimizer:** `optimize_gd.py` now supports:
   - `--use_closed_form`: exact closed-form chord (temperature-free)
   - `--smooth_closed_form`: smoothed closed-form with beta annealing
   - `--nonconvex`: triangulated (possibly concave) polygons

These extensions are present at commit **ef22b1c** and later. For research building on the closed-form machinery, please cite this work and reference the commit hash.

## ICRA 2024 Paper Replication

To replicate the results of the original ICRA 2024 paper, you can use the [replicate_icra2024.sh](https://github.com/TManzini/DifferentiableBoustrophedonPathPlans/blob/main/src/replicate_icra2024.sh) script. More explicitly, you should run the following commands...

    git clone https://github.com/TManzini/DifferentiableBoustrophedonPathPlans.git
    cd ./DifferentiableBoustrophedonPathPlans
    pip install -r requirements.txt
    cd ./src
    ./replicate_icra2024.sh

This will run several scripts to generate plots and data files, and output content to the console containing the results and plots in the paper. These results will be saved in a folder titled `replication_outputs` at the base directory of the repo. Should you encounter any issues, please raise them on this repository or contact the authors.

## Exploring the Extensions

To explore the closed-form and smoothing machinery:

- `src/validate_closed_form.py` — Validates the closed-form chord against the trapezoidal integral and discrete shapely scores.
- `src/demo_smooth_landscape.py` — Sweeps beta to visualize the loss landscape transition from smooth (melted) to sharp (exact).
- `src/validate_nonconvex_closed_form.py` — Validates non-convex support on concave shapes (L-shapes, stars, etc.).
- `src/generate_nonconvex_paper_figs.py` — Generates figures showing triangulation and multi-segment transects.

To run gradient descent on these surfaces:

    python src/optimize_gd.py --polygon_coordinates <corners.json> --use_closed_form --smooth_closed_form --nonconvex

Happy Path Planning!
