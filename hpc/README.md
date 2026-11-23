# Ensemble generation

The ensembles were generated on a GPU cluster (NVIDIA A100 80 GB, SLURM). The three scripts
in this folder hold the commands that were used. Values that depend on the cluster (account,
partition, paths) are variables at the top of each script.

| Method | Script | Software and model | Input | Requested per protein |
|---|---|---|---|---|
| BioEmu | `run_bioemu.sh` | bioemu 1.4.0, checkpoint `bioemu-v1.1` (default) | `data/sequences_v1.txt` | 500 samples |
| AlphaFlow | `run_alphaflow.sh` | AlphaFlow `predict.py`, weights `alphaflow_md_base_202402.pt` | `data/alphaflow_input.csv` and one alignment file per protein | 500 samples |
| AF2-MSA | `run_af2msa.sh` | ColabFold 1.6.1 (localcolabfold), AlphaFold2 model 1 | `data/colabfold_input.csv` | 500 seeds |

## Settings

**BioEmu.** `python -m bioemu.sample --sequence <sequence> --num_samples 500 --output_dir <folder>`,
all other options at their default values: DPM solver with 50 steps, no steering, no explicit
seed (the base seed comes from the system time), and the filter for unphysical structures
(steric clashes, chain breaks) switched on. Because of this filter the output has fewer than
500 frames (335 to 490 for the five proteins). The multiple sequence alignment for the
embedding comes from the ColabFold MMseqs2 server (the BioEmu default).
Output: `samples.xtc` and `topology.pdb` (backbone and Cβ).

**AlphaFlow.** `python predict.py --mode alphafold --input_csv <csv> --msa_dir msa_dir/
--weights params/alphaflow_md_base_202402.pt --samples 500 --outpdb <folder>`, all other
options at their default values (10 steps). Output: one PDB file with 500 models per protein.

**AF2-MSA.** `colabfold_batch <csv> <folder> --max-seq 8 --max-extra-seq 16 --num-seeds 500
--num-models 1 --model-type alphafold2 --num-recycle 1`. Seeds 0 to 499, no dropout, no
relaxation. The input is the sequence, so ColabFold gets the alignment from the ColabFold
MMseqs2 server; the structure prediction itself runs on the cluster.
Output: one PDB file per seed.

## Run time

BioEmu: 41 to 87 min per protein. AlphaFlow: 6 to 10 h per protein. AF2-MSA: about 2 h for
the five proteins together.
