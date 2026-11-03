#!/usr/bin/env bash
# CUB-50 main experiment (Fig. 2b, Fig. 4, Fig. 5, Table fig2_cub):
#   5 tau x (5 pragmatic + 5 literal seeds) = 50 jobs.
# Main configuration: l=30, lambda_c=0.1, lambda_u=0.9, gamma_faith=10, beta=2,
#   T=60, b=6, random_warmup=0.9, target_only attenuation, literal warm start,
#   init_listener_epochs=1000, lambda_AE=0.1 (all set in configs/speaker_listener_cub.yaml).
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=${WANDB_MODE:-disabled}
read -ra GPUS <<< "${GPUS:-0}"
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
DEVS=(); for ((j=0; j<JOBS_PER_GPU; j++)); do DEVS+=("${GPUS[@]}"); done
WAVE=${#DEVS[@]}
TAUS=(${TAUS:-0 5 10 15 20})
SEEDS=(${SEEDS:-0 1 2 3 4})
LAMBDA_C=0.1
L=30
CKPT_SUBDIR=speaker_listener_cub50_target_only_official
HYDRA_PARENT=outputs/sl_cub50_target_only_official
LOG_PARENT=logs/sl_cub50_target_only_official
mkdir -p "$LOG_PARENT"
i=0

for tau in "${TAUS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    d=${DEVS[$((i % WAVE))]}
    CUDA_VISIBLE_DEVICES=$d uv run experiments/cub/train_speaker_listener.py \
      tau=$tau seed=$seed ckpt_subdir=$CKPT_SUBDIR \
      run_name=cub50_to_l${L}_tau${tau}_lc${LAMBDA_C}_s${seed} \
      hydra.run.dir=$HYDRA_PARENT/dpo_tau${tau}_s${seed} \
      > $LOG_PARENT/dpo_tau${tau}_s${seed}.out 2>&1 &
    i=$((i + 1)); [ $((i % WAVE)) -eq 0 ] && wait
  done
done

# Literal speaker is deterministic; seeds only re-initialise the listener.
for tau in "${TAUS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    d=${DEVS[$((i % WAVE))]}
    CUDA_VISIBLE_DEVICES=$d uv run experiments/cub/train_speaker_listener.py \
      literal=true tau=$tau seed=$seed ckpt_subdir=$CKPT_SUBDIR \
      run_name=cub50_to_l${L}_tau${tau}_literal_s${seed} \
      hydra.run.dir=$HYDRA_PARENT/lit_tau${tau}_s${seed} \
      > $LOG_PARENT/lit_tau${tau}_s${seed}.out 2>&1 &
    i=$((i + 1)); [ $((i % WAVE)) -eq 0 ] && wait
  done
done
wait
echo "Done."
