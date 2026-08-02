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
#SBATCH --array=1-12
#SBATCH --job-name=valrepro2
#SBATCH --output=logs/valrepro2_%A_%a.out
#SBATCH --error=logs/valrepro2_%A_%a.err
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

# Reproduction check: remaining scenes (crb + flame_salmon covered by validate_repro.sh):
#   dynerf: flame_steak, sear_steak, cook_spinach x3 (easy)
#   golden (e308_sky_deform): coffee_martini x3 (hard)
T=$((SLURM_ARRAY_TASK_ID - 1))
if [ $T -lt 9 ]; then
  CFG=dynerf; EASY=(flame_steak sear_steak cook_spinach); SCENE=${EASY[$((T/3))]}; REP=$(( (T%3) + 1 ))
else
  CFG=dynerf_w; SCENE=coffee_martini; REP=$(( ((T-9)%3) + 1 ))
fi
EXPNAME=valrepro_${CFG}_${SCENE}_rep${REP}
PORT=$((6520 + SLURM_ARRAY_TASK_ID))
echo "=== ${EXPNAME} (config ${CFG}.py) ==="
python3 train.py -s $DATA/dynerf_onlyv/${SCENE} --port ${PORT} \
  --expname ${EXPNAME} --configs arguments/${CFG}.py \
  --model_path ${SCRATCH_OUT}/${EXPNAME} \
  --saving_folder ${SCRATCH_TEST}/
wait
