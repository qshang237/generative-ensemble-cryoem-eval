# Generative Ensemble Model Evaluation Against Experimental Cryo-EM Density Maps

This project benchmarks three ensemble-generation approaches, **BioEmu**, **AlphaFlow**, and a shallow-MSA AlphaFold2 baseline (**AF2-MSA**), directly against experimentally determined cryo-EM density maps, rather than against molecular dynamics trajectories as most existing benchmarks do. Scoring generated ensembles against deposited density and analysing their population structure lets us separately assess three properties of a generated ensemble: whether it samples conformations compatible with an experimental state (**sampling**), whether unsupervised analysis detects distinct populations within it (**detection**), and whether those populations correspond to the experimentally resolved states (**recovery**).

## Why density maps instead of MD

Generative models of protein conformational ensembles are usually benchmarked against MD simulation. But MD itself only samples a force-field-dependent approximation of the true structural ensemble. A deposited cryo-EM map is a direct, if noisy and resolution-limited, observation of a real conformational state. This project uses that as ground truth instead.

## Dataset

5 multi-conformation membrane transport proteins, 15 EMDB depositions, curated from 58,906 EMDB entries by filtering for single-particle cryo-EM, protein-only samples under 4 Å resolution, then grouping by UniProt ID and requiring at least 2 entries per protein:

| Protein | Depositions | Role |
|---|---|---|
| GltPh | 3 | Multi-state transport cycle (Outward-open, Intermediate, Inward-open) |
| GPR4 | 2 | Same-state reproducibility control (same state, resolved at different pH) |
| AUX1 | 2 | Same-state reproducibility control (same apo state, two independent determinations) |
| SLC37A4 | 6 | Largest experimentally resolved transition in the dataset (outward-open vs. cytosol-open) |
| SPNS2 | 2 | Smallest experimentally resolved transition in the dataset (adjacent outward-facing sub-states) |

## Method

1. **Cross-validate the reference dataset first.** Before any generated ensemble is evaluated, every deposited structure within a protein is cross-compared against every other deposition's experimental map using masked cross-correlation (CCmask), superposed via a sequence-aligned common-Cα correspondence rather than by matching author-assigned residue numbers (robust to numbering offsets between structures). This establishes the experimentally anchored structural-difference scale that separates same-state controls from genuine transitions.
2. **Generate ensembles.** Up to 500 conformations per protein from each of BioEmu, AlphaFlow, and AF2-MSA (via localcolabfold, run on QMUL's Apocrita HPC).
3. **Cluster.** GROMOS/Daura neighbour-counting, gated by Hartigan's dip test for unimodality, so the pipeline doesn't fabricate spurious clusters on ensembles that are genuinely unimodal. The distance cutoff is chosen automatically from candidate valleys in a kernel density estimate of the pairwise-RMSD distribution, not fixed by hand.
4. **Score.** Every conformation is scored against every experimental map with a resolution-matched, Gaussian-smoothed density-fit metric, normalised against how well the deposited structure itself fits.

## Key findings

**Conformational sampling, population detection and experimental state recovery are separable properties of a generated ensemble, and do not necessarily covary.** GltPh is the case where all three align: BioEmu and AlphaFlow both produce subpopulations whose map preference matches the expected states. GPR4 and AUX1, included as same-state controls, correctly show no spurious second population under any method.

<p align="center">
  <img src="results/bestfit_chimeraX/SLC37A4.png" width="500"/>
  <br/>
  <em>Best-fitting BioEmu conformation (density_ratio 0.92) against SLC37A4's EMD-66194 density, vs. the deposited structure.</em>
</p>

**SLC37A4 and SPNS2 show the two complementary failure modes that motivate this framework.** SLC37A4 has the largest experimentally resolved difference in the dataset, yet neither BioEmu's nor AlphaFlow's ensemble organises into separable populations, even though both methods' best-scoring frames closely match the target density. SPNS2 has the smallest resolved transition, yet two of three methods produce a detectable population split that tracks overall fit quality rather than which state is preferred.

<p align="center">
  <img src="results/cross_comparison/SLC37A4_cross.png" width="420"/>
  <br/>
  <em>SLC37A4 cross-comparison: the six depositions resolve into two structurally distinct groups, the structural-difference scale against which the generated ensembles are evaluated.</em>
</p>

**No method is uniformly superior across sampling, detection and recovery.** BioEmu produces the highest-scoring individual frame in every system, but this does not translate into uniformly stronger population-level recovery — its SLC37A4 ensemble fails to separate into states despite a best-frame density ratio of 0.92. AF2-MSA's most structurally diverse ensemble (AUX1, pairwise RMSD up to ~70 Å) was also its worst-fitting, showing that a broad spread of generated structures is not by itself evidence of meaningful conformational sampling.

## Repository structure

```
01_dataset_curation.py     # EMDB query + UniProt-based multi-state filtering
02_build_dataset.py        # builds the final curated dataset table
03_cross_comparison.py     # CCmask cross-validation of the reference structures
04_fitting_bioemu.py       # BioEmu ensemble fitting
04_fitting_msa.py          # MSA subsampling ensemble fitting
04_fitting_alphaflow.py    # AlphaFlow ensemble fitting
06_per_residue_density_analysis.py  # per-residue backbone-density difference panels for best-scoring frames
shared_fitting.py          # shared clustering / scoring / figure-generation logic
results/                   # per-protein figures, cross-comparison matrices, best-fit structural renders
```

## Compute

All model generation and analysis ran on QMUL's Apocrita HPC (A100 80GB, SLURM), not any external cloud service.

## Author

Qi Shang ([github.com/qshang237](https://github.com/qshang237)) — PhD, Queen Mary University of London.
