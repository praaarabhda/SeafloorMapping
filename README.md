# Mapping the Unseen Seafloor

**What do satellite-derived seafloor maps miss?** A comparison of seafloor measured by ship sonar with seafloor predicted from satellite gravity, in the Mariana Trench region, with a replication on the Mid-Atlantic Ridge.

## Findings
1. Satellite-only seafloor first looked **30% smoother**, but most of that was **selection bias**: ships survey rugged places. Fair comparisons leave a gap of about 4–13%.
2. Fourier power spectra show the difference is **scale-specific**: satellite-predicted seafloor keeps large features but loses **more than half the detail below a few km** (2.3–4.6 km across seven method variants). On the Mid-Atlantic Ridge the same pattern appears at ~11 km.
3. Satellite-only areas contain smooth **artificial lumps** (tested against two alternative explanations).
4. Blurring ship data to satellite quality shows **large seamounts survive**, while **22–33% of small knolls and cones disappear** or lose half their height.

![Power spectrum: ship vs satellite](figures/final/fig3_spectrum.png)

| | |
|---|---|
| ![](figures/final/fig2_selection_bias.png) | ![](figures/final/fig4_detector.png) |
| ![](figures/final/fig1_study_area.png) | ![](figures/final/fig5_what_is_lost.png) |

## Notebooks
| Notebook | Content |
|---|---|
| [`01_explore_the_trench`](01_explore_the_trench.ipynb) | Data download, sanity checks, maps, 3D model |
| [`02_ship_coverage_and_slope`](02_ship_coverage_and_slope.ipynb) | Grid resolution, ship-coverage mask, slope |
| [`03_cleaning_and_curvature`](03_cleaning_and_curvature.ipynb) | Artifact removal, Hessian curvature classes |
| [`04_seamount_detection`](04_seamount_detection.ipynb) | Seamount detector vs published catalog |
| [`05_clustering`](05_clustering.ipynb) | Terrain features, PCA, k-means / GMM |
| [`06_selection_bias`](06_selection_bias.ipynb) | Naive, stratified, border and spectral comparisons |
| [`07_small_bumps_hypotheses`](07_small_bumps_hypotheses.ipynb) | Artifacts vs geology vs detector |
| [`08_roundness_holdout`](08_roundness_holdout.ipynb) | Roundness filter, west/east holdout |
| [`09_robustness`](09_robustness.ipynb) | Sensitivity to method choices |
| [`10_replication`](10_replication.ipynb) | Mid-Atlantic Ridge, Magellan Seamounts |
| [`11_blurring_experiment`](11_blurring_experiment.ipynb) | Controlled blurring experiment |
| [`12_final_figures`](12_final_figures.ipynb) | Final figures |
| [`seafloor.py`](seafloor.py) | Shared functions |

## Run
```
python -m pip install -r requirements.txt
```
Run the notebooks in order (03, 05, 10 and 11 save files that later notebooks use). Data downloads automatically into `data/` on first run.

## Limitations
- One main region; the second region agrees in pattern but not in the exact scale, and the Magellan Seamounts had no fully ship-mapped areas to test.
- "Satellite-only" cells come from GMRT's global base grid (GEBCO), built mostly from depths predicted from satellite gravity, so results describe that product.
- The seamount detector is a measuring tool (held-out F1 0.56), not a finished product.

## Data
- [GMRT](https://www.gmrt.org), Global Multi-Resolution Topography (Lamont-Doherty Earth Observatory): depth grids and ship-coverage mask.
- Gevorgian et al. (2023), global seamount catalog ([Zenodo 7718512](https://zenodo.org/records/7718512)).
- Kim & Wessel (2011), seamount census.

## Use of AI
Code was written with the help of an AI coding assistant. I chose the questions, made or reviewed the analysis decisions, checked results against known facts, and interpreted the findings.
