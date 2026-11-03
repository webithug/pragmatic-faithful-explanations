#!/usr/bin/env bash
# Same-configuration speaker ablations (Tables abl_pad / abl_cub):
#   unfaithful pragmatic speaker (gamma_faith=0) and PCBM fixed speaker,
#   5 seeds each, at tau in {0, 10} (PadChest) and {0, 20} (CUB). 40 jobs.
# Requires the main runs (02/03) for the literal and faithful rows.
set -u
cd "$(dirname "$0")/.."
export WANDB_MODE=${WANDB_MODE:-disabled}
read -ra GPUS <<< "${GPUS:-0}"
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
DEVS=(); for ((j=0; j<JOBS_PER_GPU; j++)); do DEVS+=("${GPUS[@]}"); done
WAVE=${#DEVS[@]}
SEEDS=(${SEEDS:-0 1 2 3 4})
i=0

launch() {  # launch <script> <log_parent> <hydra_dir> <run_name> <overrides...>
  local script=$1 logp=$2 hdir=$3 name=$4; shift 4
  mkdir -p "$logp"
  local d=${DEVS[$((i % WAVE))]}
  CUDA_VISIBLE_DEVICES=$d uv run "$script" "$@" run_name=$name hydra.run.dir=$hdir \
    > "$logp/$(basename "$hdir").out" 2>&1 &
  i=$((i + 1)); [ $((i % WAVE)) -eq 0 ] && wait
}

# ---- PadChest: main config = l=15 lc=0.1 lu=0.5 sign (config defaults)
for tau in 0 10; do for s in "${SEEDS[@]}"; do
  launch experiments/speaker_listener/train.py logs/sl_signmode_gf0 \
    outputs/sl_signmode_gf0/dpo_tau${tau}_s${s} l15_tau${tau}_lc0.1_gf0_s${s} \
    tau=$tau seed=$s gamma_faith=0 ckpt_subdir=speaker_listener_signmode_gf0
  launch experiments/speaker_listener/train.py logs/sl_signmode_pcbm \
    outputs/sl_signmode_pcbm/dpo_tau${tau}_s${s} l15_tau${tau}_lc0.1_pcbm_s${s} \
    tau=$tau seed=$s literal=true +speaker_mode=pcbm ckpt_subdir=speaker_listener_signmode_pcbm
done; done

# ---- CUB-50: main config from configs/speaker_listener_cub.yaml
for tau in 0 20; do for s in "${SEEDS[@]}"; do
  launch experiments/cub/train_speaker_listener.py logs/sl_cub50_target_only_official_gf0 \
    outputs/sl_cub50_target_only_official_gf0/dpo_tau${tau}_s${s} cub50_to_l30_tau${tau}_lc0.1_gf0_s${s} \
    tau=$tau seed=$s gamma_faith=0 ckpt_subdir=speaker_listener_cub50_target_only_official_gf0
  launch experiments/cub/train_speaker_listener.py logs/sl_cub50_target_only_official_pcbm \
    outputs/sl_cub50_target_only_official_pcbm/dpo_tau${tau}_s${s} cub50_to_l30_tau${tau}_lc0.1_pcbm_s${s} \
    tau=$tau seed=$s literal=true +speaker_mode=pcbm ckpt_subdir=speaker_listener_cub50_target_only_official_pcbm
done; done
wait
echo "Done."
