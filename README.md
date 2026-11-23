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
2. **Generate ensembles.** 500 conformations are requested per protein from each of BioEmu (bioemu 1.4.0, checkpoint `bioemu-v1.1`), AlphaFlow (weights `alphaflow_md_base_202402.pt`) and AF2-MSA (ColabFold 1.6.1, AlphaFold2 model 1 with a subsampled MSA: `--max-seq 8 --max-extra-seq 16`, 500 seeds). BioEmu removes unphysical structures from its output, so its ensembles hold 335 to 490 frames. The commands and settings are in [`hpc/`](hpc/README.md).
3. **Cluster.** GROMOS/Daura neighbour-counting, gated by Hartigan's dip test for unimodality, so the pipeline doesn't fabricate spurious clusters on ensembles that are genuinely unimodal. The distance cutoff is chosen automatically from candidate valleys in a kernel density estimate of the pairwise-RMSD distribution, not fixed by hand.
4. **Score.** Every conformation is scored against every experimental map with a resolution-matched, Gaussian-smoothed density-fit metric, normalised against how well the deposited structure itself fits.
5. **Visualise.** Ensemble structure is shown by principal-component analysis of the Cα coordinates used for clustering (equivalent to classical MDS of the pairwise-RMSD matrix), coloured by cluster. PCA is used for visualisation only and does not influence cluster selection.
6. **Check the density ratio with other metrics.** Every conformation is also scored with three established metrics: Cα-RMSD to the deposited model (on the residues resolved in all deposited models of the protein), CC_mask (masked correlation between the map and a density simulated from the backbone at the map resolution, without B-factors) and the backbone Q-score (σ = 0.4 Å). The agreement of each metric with the density ratio is measured by the Spearman rank correlation over the frames of each ensemble and map.
7. **Quantify state specificity.** For every cluster and every map of its protein: precision (the fraction of the frames of the cluster with a density ratio of at least 0.75 with this map) and recall (the fraction of all frames with a density ratio of at least 0.75 with this map that are in the cluster). A cluster that is specific to one state has a high precision for one map and a low precision for the others.
8. **Validate the clustering externally.** The clustering procedure is applied, unchanged, to synthetic ensembles with known state labels. They are built on five states of the transporter XylE (three crystal structures, PDB 4GBY, 4JA3 and 4QIQ, and two designed open states), with three sets of state populations, a one-state control and three noise levels. The result is compared with the known labels by the adjusted Rand index.

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

**No method is uniformly superior across sampling, detection and recovery.** BioEmu produces the highest-scoring individual frame in four of the five systems, but this does not translate into uniformly stronger population-level recovery; its SLC37A4 ensemble fails to separate into states despite a best-frame density ratio of 0.92. AF2-MSA's most structurally diverse ensemble (AUX1, pairwise RMSD up to ~70 Å) was also its worst-fitting, showing that a broad spread of generated structures is not by itself evidence of meaningful conformational sampling.

## Repository structure

```
01_dataset_curation.py     # EMDB query + UniProt-based multi-state filtering
02_build_dataset.py        # builds the final curated dataset table
03_cross_comparison.py     # CCmask cross-validation of the reference structures
04_fitting_bioemu.py       # BioEmu ensemble fitting
04_fitting_msa.py          # AF2-MSA (shallow-MSA AlphaFold2) ensemble fitting
04_fitting_alphaflow.py    # AlphaFlow ensemble fitting
06_per_residue_density_analysis.py  # per-residue backbone-density difference panels for best-scoring frames
07_coordinate_pca.py       # PCA of Cα coordinates per ensemble, coloured by cluster (results/coordinate_pca/)
08_method_comparison_stats.py   # best frame, top 1 %, top 5 % and median density ratio per map and method, with bootstrap intervals
09_state_specificity.py         # state label of every map; density ratio of every deposited model with every map (reference)
14_xyle_states.py               # five XylE states for the external validation of the clustering
16_xyle_ensembles.py            # synthetic XylE ensembles with known state labels
17_rmsd_to_deposited.py         # Cα-RMSD of every frame to every deposited model
18_cc_mask.py                   # CC_mask of every frame with every map
20_q_score.py                   # backbone Q-score of every frame with every map
19_metric_summary.py            # all metrics in one table: rank agreement with the density ratio, best frame, frames per state
24_cluster_precision_recall.py  # precision and recall of every cluster with every map
25_fig_other_metrics.py         # figure: CC_mask and Q-score of each cluster, agreement with the density ratio
26_fig_precision_recall.py      # figures: precision and recall of the clusters (state specificity)
27_xyle_external_validation.py  # clustering of the XylE ensembles, compared with the known states
shared_fitting.py          # shared clustering / scoring / figure-generation logic
hpc/                       # job scripts and settings of the ensemble generation
requirements.txt           # Python packages of the analysis
results/                   # per-protein figures, cross-comparison matrices, coordinate PCA, best-fit structural renders, metric tables
```

Not stored in the repository: the generated ensembles and the EMDB maps (large files), the synthetic XylE ensembles (written by `16_xyle_ensembles.py`), and the three XylE structures, which `14_xyle_states.py` reads from `download/4GBY.pdb`, `download/4JA3.pdb` and `download/4QIQ.pdb` (RCSB PDB).

## Compute

The ensembles were generated on QMUL's Apocrita HPC (A100 80GB, SLURM). The multiple sequence alignments for BioEmu and AF2-MSA came from the public ColabFold MMseqs2 server; the structure generation itself ran on the cluster. The analysis scripts run on a workstation; the two slow steps (`18_cc_mask.py`, `20_q_score.py`) can be stopped and started again.

## Author

Qi Shang ([github.com/qshang237](https://github.com/qshang237)), PhD, Queen Mary University of London.
