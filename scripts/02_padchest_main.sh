#!/usr/bin/env bash
# PadChest main experiment (Fig. 2a, Fig. 3, Table fig2_pad):
#   9 tau x 5 pragmatic seeds + 9 literal runs = 54 jobs, ~6 min each on one GPU.
# Main configuration: l=15, lambda_c=0.1, lambda_u=0.5, gamma_faith=100, beta=2,
#   T=60, b=6, sign attenuation, literal warm start, lambda_AE=0.1.
# GPUS="0 1 2 3" runs 4 jobs at a time; JOBS_PER_GPU=3 is safe on 24 GB cards.
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=${WANDB_MODE:-disabled}
read -ra GPUS <<< "${GPUS:-0}"
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
DEVS=(); for ((j=0; j<JOBS_PER_GPU; j++)); do DEVS+=("${GPUS[@]}"); done
WAVE=${#DEVS[@]}
TAUS=(${TAUS:-0 1 2 3 4 5 6 8 10})
SEEDS=(${SEEDS:-0 1 2 3 4})
LAMBDA_C=0.1
LAMBDA_U=0.5
L=15
CKPT_SUBDIR=speaker_listener_signmode
HYDRA_PARENT=outputs/sl_signmode
LOG_PARENT=logs/sl_signmode
mkdir -p "$LOG_PARENT"
i=0

for tau in "${TAUS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    d=${DEVS[$((i % WAVE))]}
    CUDA_VISIBLE_DEVICES=$d uv run experiments/speaker_listener/train.py \
      tau=$tau lambda_c=$LAMBDA_C lambda_u=$LAMBDA_U l=$L seed=$seed \
      ckpt_subdir=$CKPT_SUBDIR \
      run_name=l${L}_tau${tau}_lc${LAMBDA_C}_s${seed} \
      hydra.run.dir=$HYDRA_PARENT/dpo_tau${tau}_s${seed} \
      > $LOG_PARENT/dpo_tau${tau}_s${seed}.out 2>&1 &
    i=$((i + 1)); [ $((i % WAVE)) -eq 0 ] && wait
  done
done

for tau in "${TAUS[@]}"; do
  d=${DEVS[$((i % WAVE))]}
  CUDA_VISIBLE_DEVICES=$d uv run experiments/speaker_listener/train.py \
    literal=true tau=$tau lambda_c=$LAMBDA_C lambda_u=$LAMBDA_U l=$L seed=0 \
    ckpt_subdir=$CKPT_SUBDIR \
    run_name=l${L}_tau${tau}_literal_s0 \
    hydra.run.dir=$HYDRA_PARENT/lit_tau${tau} \
    > $LOG_PARENT/lit_tau${tau}.out 2>&1 &
  i=$((i + 1)); [ $((i % WAVE)) -eq 0 ] && wait
done
wait
echo "Done."
