# Tell Me What I Need to Know: Faithful and Pragmatic Explanations

Code and derived data for

> Wei-Po Wang, Jacopo Teneggi, Paul Yi, Tianmin Shu, Jeremias Sulam.
> *Tell Me What I Need to Know: Faithful and Pragmatic Explanations.*
> Transactions on Machine Learning Research, 2026. https://openreview.net/forum?id=GK9Gq9VVcU

A **speaker** selects a short list of concepts that explains a frozen black-box
classifier to a **listener**. The speaker is trained with DPO on listener
preferences (pragmatics) and regularised to stay **faithful** to the classifier
through a grounded autoencoder. Experiments: chest X-rays (PadChest / PadChest-GR,
4 diagnoses, 78 concepts) and birds (CUB-200-2011 restricted to 50 species,
312 attributes).

## Layout

```
configs/                 Hydra configs; defaults = the paper's main configuration
experiments/
  padchest/, padchest_gr/   PadChest preprocessing, ViT classifier, embeddings, grounded autoencoder
  cub/                      CUB preprocessing, CLIP embeddings, MLP classifier, autoencoder,
                            speaker-listener training and concept-distribution analysis
  speaker_listener/         PadChest speaker-listener training (DPO), PCBM baseline, analysis
pragmatixs/              small shared utilities (paths, logging, wandb monitor)
scripts/                 numbered launchers (01-05) and the summary-CSV writers
results/                 CSVs behind every figure/table of the paper + results/plots.ipynb
```

## Setup

```bash
uv sync                     # Python >= 3.12, PyTorch (CUDA 12.8 wheels on Linux)
cp .env.example .env        # the code refuses to start without a .env file
```

All paths are relative to the repository root: `data/`, `checkpoints/`,
`outputs/` (Hydra run dirs with the `.log` files that every analysis reads) and
`logs/`. Weights & Biases is disabled by the launch scripts (`WANDB_MODE=disabled`).

## Three levels of reproduction

### 1. Figures and tables from the released numbers (no GPU)

`results/plots.ipynb` redraws every figure from the CSVs in `results/`
(run it from inside `results/`). The tables of the paper are the CSVs themselves:

| Paper | File |
|---|---|
| Fig. 2 (pragmatic minus literal vs. tau), Tables "values plotted in Fig. 2" | `results/tau_summary_l15_signmode/acc_vs_tau*.csv`, `results/tau_summary_cub50_target_only_official/acc_vs_tau_top5*.csv` |
| Fig. 3 (PadChest concept CDFs, tau = 6, 10) | `results/tau_summary_l15_signmode/concept_cdf_*_tau6_tau10.csv` |
| Fig. 4 (CUB CDF-AUC gap vs. tau) | `results/tau_summary_cub50_target_only_official/cdf_auc.csv` |
| Fig. 5 (CUB attribute-group pies) | `results/cub_prior/` (`prior_text.csv`, `prior_mturk.csv`, `final_iter/*.csv`) |
| Tables "comparison of speakers" (literal / PCBM / unfaithful / ours) | `results/speaker_comparison/{padchest,cub_top5}_all_speakers.csv` |
| Table "faithfulness under two autoencoders" | `results/speaker_comparison/faith_vs_lambda_ae.csv` |
| Autoencoder concept AUC (5 seeds) | `results/*/autoencoder_auc.csv` |
| Appendix: base-classifier training curves | `results/appendix/base_classifier/*.csv` |

### 2. Retrain the speakers from the released embeddings and checkpoints (GPU)

Download `pragmatic-faithful-explanations-artifacts.tar.gz` from the GitHub
release and extract it in the repository root. It creates

```
data/padchest_final/   embeddings_vit_ln.pt (ViT embeddings), padchest_clean_final.csv, split.pt
data/PadChest-GR/      processed_labels_gr.csv (concept labels incl. LLM-extracted negations), gr_normal_sentence_concepts.json
data/CUB/              embeddings_clip{,_50}.pt (CLIP ViT-L/14), cub_class_attribute_prior{,_50}.pt,
                       cub_concepts.json, cub_classes.json, cub_text_attribute_mentions.npy, cub_text_caption_extractions.json
checkpoints/padchest/classifier_vit_b16.pt          frozen PadChest classifier (ViT head)
checkpoints/cub_mlp/classifier_mlp{,_50}.pt         frozen CUB classifier (MLP on CLIP features)
checkpoints/autoencoder_vit/ae_ae_vit_lam0.1.pt     grounded autoencoders (lambda_AE = 0.1) and
checkpoints/autoencoder_cub/ae_ae_cub_lam0.1.pt     the classification-only ones (ae_ae_lam0.pt, ae_ae_cub_lam0.pt)
```

No raw images are included. Then

```bash
scripts/01_train_autoencoders.sh   # optional: the bundle already contains these checkpoints
GPUS="0 1 2 3" JOBS_PER_GPU=2 scripts/02_padchest_main.sh   # 54 runs, ~6 min each on one 24 GB GPU
GPUS="0 1 2 3" JOBS_PER_GPU=2 scripts/03_cub_main.sh        # 50 runs
GPUS="0 1 2 3" JOBS_PER_GPU=2 scripts/04_ablations.sh       # 40 runs (unfaithful gamma=0 and PCBM speakers)
scripts/05_summarize.sh                                      # regenerates every CSV in results/
```

`TAUS="0 10" SEEDS="0"` on 02/03 runs a subset. Every run writes a Hydra
directory under `outputs/<family>/<run>/` with a `.log` file; all analysis
scripts read the logs and the per-iteration checkpoints under
`checkpoints/<ckpt_subdir>/`. Reported numbers are taken at the iteration that
maximises validation listener accuracy on the reconstructed embedding
(`val/listener_acc_recon`), mean +/- SEM over 5 seeds.

Main configuration (also the defaults in `configs/speaker_listener*.yaml`):

| | PadChest | CUB-50 |
|---|---|---|
| explanation length l | 15 | 30 |
| lambda_c, lambda_u | 0.1, 0.5 | 0.1, 0.9 |
| gamma_faith | 100 | 10 |
| beta (DPO), T (outer iters), b (candidates) | 2, 60, 6 | 2, 60, 6 |
| listener attenuation | sign | target_only |
| tau grid | 0, 1, 2, 3, 4, 5, 6, 8, 10 | 0, 5, 10, 15, 20 |
| lambda_AE | 0.1 | 0.1 |

Notes on reproducing the shipped CSVs:
- Rerunning the summaries on our own logs reproduces every pragmatic-speaker
  row and both speaker-comparison tables exactly (`scripts/05_summarize.sh`).
- The *literal* faithfulness in `acc_vs_tau*.csv` was averaged over five
  retrained autoencoders in the paper; `make_summary_csv.py` reports the
  single released autoencoder (0.898 instead of 0.911 on PadChest). Pragmatic
  rows are unaffected.
- Training is seeded but GPU non-determinism gives small run-to-run differences.

### 3. From raw data

The derived data above were produced by the scripts in `experiments/`, in this order.
Raw data must be obtained from the original sources (PadChest and PadChest-GR require
registration; CUB-200-2011 and the CUB captions of Reed et al. are public).

PadChest (`DATA_DIR/padchest/` holds the label CSV and PNGs; the GR report JSON goes to `data/PadChest-GR/`):
`padchest/preprocess_labels.py` -> `padchest/filter_physician.py` -> `padchest_gr/clean_gr.py`
(LLM extraction of negated findings, `ANTHROPIC_API_KEY`; prompts are in the script and the paper appendix)
-> `padchest_gr/merge_and_analyze.py` -> `padchest_gr/build_final_datasets.py` -> `padchest_gr/make_split.py`
-> `padchest/split_cls.py` -> `padchest/train_cls.py` (ViT classifier) -> `padchest_gr/extract_embeddings.py`.

CUB (`data/CUB/images`, `data/CUB/CUB_200_2011`, captions in `data/CUB_cvpr2016/text_c10`):
`cub/build_cub_dataset.py` -> `cub/extract_embeddings_clip.py` -> `cub/train_mlp_classifier.py`
-> `cub/extract_concepts_from_captions.py` (LLM, `OPENAI_API_KEY`) -> `cub/build_cub_prior.py`
-> `cub/build_subset.py --n-classes 50`.

Not included: the lambda_c and gamma_faith sweeps and the autoencoder sweep of
the appendix (same scripts, different overrides).

## Citation

```bibtex
@article{wang2026tellme,
  title   = {Tell Me What I Need to Know: Faithful and Pragmatic Explanations},
  author  = {Wang, Wei-Po and Teneggi, Jacopo and Yi, Paul and Shu, Tianmin and Sulam, Jeremias},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=GK9Gq9VVcU}
}
```

MIT license (see `LICENSE`).
