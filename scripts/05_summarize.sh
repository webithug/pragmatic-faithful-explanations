#!/usr/bin/env bash
# Turn the run logs / checkpoints of 02-04 into the CSVs read by results/plots.ipynb
# and the tables in the paper. Safe to re-run; only the steps whose inputs exist run.
set -u
cd "$(dirname "$0")/.."
PAD=results/tau_summary_l15_signmode
CUB=results/tau_summary_cub50_target_only_official

# Fig. 2 / Tables fig2_pad, fig2_cub
ls outputs/sl_signmode/*/l15*.log >/dev/null 2>&1 && \
  uv run scripts/make_summary_csv.py padchest outputs/sl_signmode/*/l15*.log --out-dir $PAD
ls outputs/sl_cub50_target_only_official/*/cub50*.log >/dev/null 2>&1 && \
  uv run scripts/make_summary_csv.py cub outputs/sl_cub50_target_only_official/*/cub50*.log --out-dir $CUB

# Fig. 3: PadChest concept CDFs at tau in {6, 10}
[ -d checkpoints/speaker_listener_signmode ] && \
  uv run experiments/speaker_listener/plot_concept_dist.py \
    --ckpt-dir checkpoints/speaker_listener_signmode --taus 6 10 --seeds 0 1 2 3 4 \
    --cumulative --multi-tau --format pdf --out-dir $PAD

# Fig. 4 (CDF AUC vs tau) and Fig. 5 (attribute-group pies) on CUB, final iteration
[ -d checkpoints/speaker_listener_cub50_target_only_official ] && \
  uv run experiments/cub/concept_cdf_summary.py \
    --ckpt-dir checkpoints/speaker_listener_cub50_target_only_official --run-stem cub50_to_l30 \
    --taus 0 5 10 15 20 --seeds 0 1 2 3 4 --iter 60 --out-dir $CUB \
    --group-counts-dir results/cub_prior/final_iter

# Tables abl_pad / abl_cub (needs 04_ablations.sh)
[ -d logs/sl_signmode_gf0 ] && uv run experiments/speaker_listener/rebuttal_ablation.py

# Table faith_vs_lam (needs the lambda_AE=0 autoencoders: 01_train_autoencoders.sh LAM0=1 or the bundle)
[ -f checkpoints/autoencoder_vit/ae_ae_lam0.pt ] && uv run experiments/speaker_listener/faith_vs_lambda_ae.py

# Concept AUC of the 5-seed autoencoders (needs 01_train_autoencoders.sh SEEDS=1)
ls checkpoints/autoencoder_vit/ae_ae_vit_lam0.1_s*.pt >/dev/null 2>&1 && \
  uv run scripts/make_autoencoder_auc_csv.py checkpoints/autoencoder_vit/ae_ae_vit_lam0.1_s*.pt \
    --comment "PadChest grounded autoencoder (lam_AE=0.1) concept AUC over 5 seeds." --out $PAD/autoencoder_auc.csv
ls checkpoints/autoencoder_cub/ae_ae_cub_lam0.1_s*.pt >/dev/null 2>&1 && \
  uv run scripts/make_autoencoder_auc_csv.py checkpoints/autoencoder_cub/ae_ae_cub_lam0.1_s*.pt \
    --comment "CUB grounded autoencoder (lam_AE=0.1) concept AUC over 5 seeds." --out $CUB/autoencoder_auc.csv

echo "Done. Open results/plots.ipynb to draw the figures."
