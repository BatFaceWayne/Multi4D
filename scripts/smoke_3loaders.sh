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
#SBATCH --time=0-12:00:00
#SBATCH --job-name=smoke3
#SBATCH --array=0-2
#SBATCH --output=logs/smoke3_%A_%a.out
#SBATCH --error=logs/smoke3_%A_%a.err
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

# post-cleanup smoke validation: one scene per dataset loader
case $SLURM_ARRAY_TASK_ID in
  0) SRC=$DATA/dynerf_onlyv/cut_roasted_beef;      CFG=dynerf;                  EXP=smoke_crb ;;
  1) SRC=$DATA/nerfds/nerf-ds/bell_novel_view;     CFG=nerfds; EXP=smoke_bell ;;
  2) SRC=$DATA/technicolor/Painter/colmap_0;       CFG=dynerf;                  EXP=smoke_Painter ;;
esac
PORT=$((6900 + SLURM_ARRAY_TASK_ID))

echo "[smoke3] task=$SLURM_ARRAY_TASK_ID src=$SRC cfg=$CFG exp=$EXP"
python3 train.py -s $SRC --port $PORT \
  --expname $EXP --configs arguments/${CFG}.py \
  --model_path $SCRATCH_OUT/$EXP --saving_folder $SCRATCH_TEST/
wait
echo "[smoke3] DONE task=$SLURM_ARRAY_TASK_ID exp=$EXP"
