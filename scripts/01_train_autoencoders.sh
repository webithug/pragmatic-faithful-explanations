#!/usr/bin/env bash
# Grounded autoencoders (Sec. 4 / App. "Grounded autoencoder").
#   PadChest : Linear(256 -> 78) on ViT embeddings,  lambda_AE = 0.1
#   CUB      : Linear(768 -> 312) on CLIP embeddings, lambda_AE = 0.1 (trained on all 200 classes)
# Produces the checkpoints shipped in the artifact bundle:
#   checkpoints/autoencoder_vit/ae_ae_vit_lam0.1.pt
#   checkpoints/autoencoder_cub/ae_ae_cub_lam0.1.pt
# Optional: SEEDS=1 retrains PadChest/CUB AEs with 5 seeds (concept-AUC table),
#           LAM0=1 also trains the classification-only AEs used in Table "faith vs AE".
set -eu
cd "$(dirname "$0")/.."
export WANDB_MODE=${WANDB_MODE:-disabled}
GPU=${GPU:-0}
mkdir -p logs/autoencoders

CUDA_VISIBLE_DEVICES=$GPU uv run experiments/padchest_gr/train_autoencoder.py \
  lam=0.1 run_name=ae_vit_lam0.1 > logs/autoencoders/ae_vit_lam0.1.out 2>&1
CUDA_VISIBLE_DEVICES=$GPU uv run experiments/cub/train_autoencoder.py \
  lam=0.1 run_name=ae_cub_lam0.1 > logs/autoencoders/ae_cub_lam0.1.out 2>&1

if [ "${SEEDS:-0}" = "1" ]; then
  for s in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$GPU uv run experiments/padchest_gr/train_autoencoder.py \
      lam=0.1 seed=$s run_name=ae_vit_lam0.1_s$s > logs/autoencoders/ae_vit_lam0.1_s$s.out 2>&1
    CUDA_VISIBLE_DEVICES=$GPU uv run experiments/cub/train_autoencoder.py \
      lam=0.1 seed=$s run_name=ae_cub_lam0.1_s$s > logs/autoencoders/ae_cub_lam0.1_s$s.out 2>&1
  done
fi

if [ "${LAM0:-0}" = "1" ]; then
  CUDA_VISIBLE_DEVICES=$GPU uv run experiments/padchest_gr/train_autoencoder.py \
    lam=0 epochs=2000 run_name=ae_lam0 > logs/autoencoders/ae_lam0.out 2>&1
  CUDA_VISIBLE_DEVICES=$GPU uv run experiments/cub/train_autoencoder.py \
    lam=0 run_name=ae_cub_lam0 > logs/autoencoders/ae_cub_lam0.out 2>&1
fi
echo "Done."
