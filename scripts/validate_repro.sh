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
#SBATCH --time=0-08:00:00
#SBATCH --array=1-6
#SBATCH --job-name=valrepro
#SBATCH --output=logs/valrepro_%A_%a.out
#SBATCH --error=logs/valrepro_%A_%a.err
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

# Reproduction check: does the dynerf and dynerf_w configs
# still hit their pre-cleanup numbers? 3 runs each.
#   dynerf / cut_roasted_beef  -> expect peak ~33.85 (band 33.4-34)
#   e308_sky_deform / flame_salmon -> expect peak ~29.2-29.4
if [ $SLURM_ARRAY_TASK_ID -le 3 ]; then CFG=dynerf;            SCENE=cut_roasted_beef; else CFG=dynerf_w; SCENE=flame_salmon; fi
REP=$(( (($SLURM_ARRAY_TASK_ID - 1) % 3) + 1 ))
EXPNAME=valrepro_${CFG}_${SCENE}_rep${REP}
PORT=$((6500 + SLURM_ARRAY_TASK_ID))
echo "=== ${EXPNAME} (config ${CFG}.py) ==="
python3 train.py -s $DATA/dynerf_onlyv/${SCENE} --port ${PORT} \
  --expname ${EXPNAME} --configs arguments/${CFG}.py \
  --model_path ${SCRATCH_OUT}/${EXPNAME} \
  --saving_folder ${SCRATCH_TEST}/
wait
