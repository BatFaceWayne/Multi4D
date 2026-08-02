#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script. The SBATCH header and `module load` lines below are for the
# ETH Euler cluster - adapt them to your scheduler. All paths can be overridden
# with environment variables; the defaults are the authors' cluster layout.
# ---------------------------------------------------------------------------
#SBATCH -n 4
#SBATCH --mem-per-cpu=12000
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:23g
#SBATCH --time=0-10:00:00
#SBATCH --array=1-10
#SBATCH --job-name=tech_r23
#SBATCH --output=logs/tech_r23_%A_%a.out
#SBATCH --error=logs/tech_r23_%A_%a.err
module load eth_proxy stack/.2024-06-silent gcc/12.2.0 python_cuda/3.11.6 cuda/12.1.1 cudnn/9.2.0
# Paths: override with environment variables, or drop your own values into
# scripts/env.local.sh (gitignored, not shipped).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.local.sh" ] && source "$HERE/env.local.sh"
REPO=${MULTI4D_ROOT:-$(cd "$HERE/.." && pwd)}
DATA=${MULTI4D_DATA:-$REPO/data}
SCRATCH_OUT=${MULTI4D_OUT:-$REPO/output}
SCRATCH_TEST=${MULTI4D_RESULTS:-$REPO/results}
[ -n "${MULTI4D_VENV:-}" ] && source "$MULTI4D_VENV"
mkdir -p "$SCRATCH_OUT" "$SCRATCH_TEST" "$REPO/logs"
cd "$REPO"

# Technicolor replicates 2 & 3 (rep1 = array 5363754, expname technicolor_<Scene>_e308).
# 3 runs/scene for GPU-nondeterminism noise (single-scene +/-1.8). the shipped recipe.
SCENES=(Birthday Fabien Painter Theater Train)
T=$((SLURM_ARRAY_TASK_ID - 1))
SCENE=${SCENES[$((T % 5))]}
REP=$((2 + T / 5))          # tasks 1-5 -> rep2 ; tasks 6-10 -> rep3
EXPNAME=technicolor_${SCENE}_e308_rep${REP}
PORT=$((6560 + SLURM_ARRAY_TASK_ID))
echo "=== ${EXPNAME} (the shipped recipe, technicolor ${SCENE}/colmap_0) ==="
python3 train.py -s $DATA/${SCENE}/colmap_0 --port ${PORT} \
  --expname ${EXPNAME} --configs arguments/dynerf.py \
  --model_path ${SCRATCH_OUT}/${EXPNAME} \
  --saving_folder ${SCRATCH_TEST}/
wait
