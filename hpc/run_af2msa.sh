#!/bin/bash
# AF2-MSA: AlphaFold2 with a subsampled MSA (ColabFold), 500 seeds per protein, one job for all proteins.
#
#   ACCOUNT=<slurm account> PARTITION=<partition> COLABFOLD_BIN=<folder with colabfold_batch> bash hpc/run_af2msa.sh
#
# INPUT_CSV   columns id,sequence   (default: data/colabfold_input.csv)
#             With sequences as input, ColabFold gets the MSA from the ColabFold MMseqs2 server.
# OUT_DIR     <id>_unrelaxed_rank_NNN_alphafold2_model_1_seed_SSS.pdb   (default: ./msa_results)
# ColabFold version used: 1.6.1 (localcolabfold).
set -euo pipefail
ACCOUNT=${ACCOUNT:?set ACCOUNT to the SLURM account}
PARTITION=${PARTITION:?set PARTITION to the SLURM partition}
COLABFOLD_BIN=${COLABFOLD_BIN:?set COLABFOLD_BIN to the folder that holds colabfold_batch}
INPUT_CSV=${INPUT_CSV:-$PWD/data/colabfold_input.csv}
OUT_DIR=${OUT_DIR:-$PWD/msa_results}
mkdir -p "$OUT_DIR"

sbatch << SLURM
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=af2msa
#SBATCH --output=${OUT_DIR}/af2msa_%j.out

export PATH="${COLABFOLD_BIN}:\$PATH"

colabfold_batch ${INPUT_CSV} ${OUT_DIR} \
    --max-seq 8 \
    --max-extra-seq 16 \
    --num-seeds 500 \
    --num-models 1 \
    --model-type alphafold2 \
    --num-recycle 1

echo "Done"
SLURM
