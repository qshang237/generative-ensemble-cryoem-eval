#!/bin/bash
# BioEmu: one SLURM job per protein, 500 samples requested per protein.
#
#   ACCOUNT=<slurm account> PARTITION=<partition> bash hpc/run_bioemu.sh
#
# SEQ_FILE   one protein per line: "<name> <sequence>"   (default: data/sequences_v1.txt)
# BASE_DIR   output folder; one sub-folder per protein     (default: ./bioemu_results)
# Needs a conda environment named "bioemu" with the bioemu package (version used: 1.4.0).
set -euo pipefail
ACCOUNT=${ACCOUNT:?set ACCOUNT to the SLURM account}
PARTITION=${PARTITION:?set PARTITION to the SLURM partition}
SEQ_FILE=${SEQ_FILE:-data/sequences_v1.txt}
BASE_DIR=${BASE_DIR:-$PWD/bioemu_results}

while read -r PROTEIN SEQ; do
    [ -z "${PROTEIN:-}" ] && continue
    OUT_DIR="$BASE_DIR/$PROTEIN"
    mkdir -p "$OUT_DIR"

    sbatch << SLURM
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=bioemu_${PROTEIN}
#SBATCH --output=${BASE_DIR}/${PROTEIN}_%j.out

module load miniforge
mamba activate bioemu
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python -m bioemu.sample \
    --sequence ${SEQ} \
    --num_samples 500 \
    --output_dir ${OUT_DIR}

echo "Done: ${PROTEIN}"
SLURM

    echo "Submitted $PROTEIN"
done < "$SEQ_FILE"
