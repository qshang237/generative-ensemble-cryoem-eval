#!/bin/bash
# AlphaFlow: one SLURM job per protein, 500 samples per protein.
#
#   ACCOUNT=<slurm account> PARTITION=<partition> ALPHAFLOW_DIR=<clone of the AlphaFlow repository> bash hpc/run_alphaflow.sh
#
# ALPHAFLOW_DIR must hold predict.py, params/alphaflow_md_base_202402.pt and the alignments
# msa_dir/<name>/a3m/<name>.a3m of the five proteins.
# INPUT_CSV   columns name,seqres   (default: data/alphaflow_input.csv)
# OUT_DIR     one multi-model PDB file per protein (default: ./alphaflow_results)
# Needs a conda environment named "alphaflow".
set -euo pipefail
ACCOUNT=${ACCOUNT:?set ACCOUNT to the SLURM account}
PARTITION=${PARTITION:?set PARTITION to the SLURM partition}
ALPHAFLOW_DIR=${ALPHAFLOW_DIR:?set ALPHAFLOW_DIR to the AlphaFlow folder}
INPUT_CSV=${INPUT_CSV:-$PWD/data/alphaflow_input.csv}
OUT_DIR=${OUT_DIR:-$PWD/alphaflow_results}
mkdir -p "$OUT_DIR"

tail -n +2 "$INPUT_CSV" | cut -d, -f1 | while read -r PROTEIN; do
    [ -z "${PROTEIN:-}" ] && continue
    CSV="$OUT_DIR/alphaflow_input_${PROTEIN}.csv"       # one protein per job
    { echo "name,seqres"; grep "^${PROTEIN}," "$INPUT_CSV"; } > "$CSV"

    sbatch << SLURM
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=alphaflow_${PROTEIN}
#SBATCH --output=${OUT_DIR}/alphaflow_${PROTEIN}_%j.out

module load miniforge
module load cuda/11.8.0-gcc-12.2.0
mamba activate alphaflow

cd ${ALPHAFLOW_DIR}

python predict.py \
    --mode alphafold \
    --input_csv ${CSV} \
    --msa_dir msa_dir/ \
    --weights params/alphaflow_md_base_202402.pt \
    --samples 500 \
    --outpdb ${OUT_DIR}/

echo "Done: ${PROTEIN}"
SLURM

    echo "Submitted $PROTEIN"
done
