# Reproducing the experiments

## Repository structure

```
.
├── src/nemos/                      # Installable library (`pip install -e .`)
│   ├── algorithms/
│   │   ├── nemos.py                  # ★ NeMoS: nearest-neighbor UCB + active learning
│   │   ├── _ucb_math.py              # pure-Python UCB core (no torch)
│   │   ├── pakucb.py                 # PAK-UCB baseline (cubic-kernel UCB)
│   │   ├── linucb.py                 # LinUCB baseline
│   │   └── neuronal_s_nets.py        # neuronal-s baseline (EE-Net-style nets)
│   └── rewards/                      # Where an experiment's rewards come from
│       ├── precomputed.py            # replay scores from datasets/ (default)
│       ├── online.py                 # generate an image per step and score it
│       └── pipelines.py              # the six text-to-image pipelines
│
├── experiments/                    # Every experiment in the paper (run from inside this folder)
│   ├── compare_to_baselines.py       # Main comparison (Table 1 / main figure)
│   ├── compare_to_baselines_BERT.py  # BERT-embedding ablation
│   ├── compare_to_baselines_LLM.py   # Generalization to LLM selection
│   ├── model_addition.py             # Robustness: models added mid-stream
│   ├── model_removal.py              # Robustness: models removed mid-stream
│   ├── n_models_queries.py           # Models / query-budget sweep
│   ├── different_budgets.py          # Active-query budget sensitivity
│   ├── query_trigger_strategies.py   # Delta vs. other query triggers
│   ├── delta_analysis.py             # Sensitivity to the near-tie threshold delta
│   ├── theta.py                      # Sensitivity to the UCB parameter theta
│   └── intermediate_results.py       # Reward-estimation-error analysis
│
├── scripts/create_dataset/         # Build the offline evaluation datasets
├── notebooks/                      # plot_compare_to_baselines.ipynb — the main figure
└── assets/                         # Figure(s) used in the README
```

## Installation

```bash
git clone https://github.com/julesdamidaux/nemos-tmlr.git
cd nemos-tmlr
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                     # exposes the `nemos` package
```

`requirements.txt` uses permissive `>=` lower bounds, so pip may resolve newer versions of `torch`,
`transformers`, `diffusers`, `image-reward` or `hpsv2` than the ones used for the paper. Those
versions affect the *scores* the metrics produce, so an exact numerical match with the published
tables is not guaranteed; the qualitative comparisons between algorithms are unaffected, since every
algorithm sees the same rewards within a run.

A CUDA-capable GPU is recommended (experiments compute CLIP/BERT embeddings with PyTorch and fall
back to CPU automatically).

## Where rewards come from

Every experiment gets its rewards from a **reward source**, chosen by two lines in its config
block:

```python
metric      = "clip"          # "clip" | "imagereward" | "hpsv2"
reward_mode = "precomputed"   # "precomputed": replay datasets/; "online": generate images
```

| `reward_mode` | What a round does | Needs | Speed |
|---|---|---|---|
| `precomputed` *(default)* | replays scores from `datasets/<dataset>/metadata*.json` | the dataset files | seconds |
| `online` | generates `generations` images with **each** candidate model and scores them on the spot | a GPU and the model weights | hours |

`precomputed` is the protocol behind the paper's tables. `online` removes the offline dataset from
the loop entirely — prompts come from `datasets/prompts/<dataset>.json`, images are sampled fresh at
every step, and scores are cached per (model, prompt) in
`datasets/<dataset>/online_scores_<metric>.json` so a rerun does not regenerate them. Because the
samplers are stochastic and unseeded, `online` measures the same quantity under a different
sampling process: it does not reproduce the published numbers.

Cost, before switching: a round evaluates every candidate model (the oracle and the full-feedback
query both need all of them), so it costs `len(models) × generations` generations — 30 with the
paper's six models and `generations = 5`. A 2000-round run is tens of thousands of diffusion
samples. `OnlineRewards(keep_loaded=False)` trades speed for VRAM by freeing each pipeline after
use.

The implementation is in [`src/nemos/rewards/`](../src/nemos/rewards): `PrecomputedRewards`,
`OnlineRewards`, and the shared pipeline factory that `create_dataset_T2I.py` also uses, so images
are generated identically either way.

## Datasets

In `precomputed` mode the experiments never touch images: they replay **pre-computed
per-(prompt, model) scores** from a `datasets/` directory at the repository root (git-ignored):

```
datasets/
├── prompts/<dataset>.json            # the prompt set
├── <dataset>/metadata.json           # per-(prompt, model) CLIP scores
├── <dataset>/metadata_IR.json        # per-(prompt, model) ImageReward scores
└── <dataset>/metadata_HPS.json       # per-(prompt, model) HPSv2 scores
```

Datasets used in the paper: `flowers`, `ms-coco`, `carrot-bowl`, `flickr`. Models: `Sana`,
`Unidiffuser`, `LCM`, `Koala`, `SDXL-Turbo`, `SSD-1B`.

### Generating the datasets

The score files are **not distributed**: they are generated locally with the scripts in
`scripts/create_dataset/`, which requires a **CUDA GPU** and a few hours per dataset (six models ×
5 images × up to a few thousand prompts). Expect the resulting scores to differ slightly from the
published ones: the diffusion samplers are stochastic, the generation scripts set no RNG seed, and
model revisions are not pinned (see [Model versions](#model-versions)).

To generate a dataset (GPU required in practice):

```bash
cd scripts/create_dataset
python create_dataset_T2I.py                                   # images + CLIP scores -> metadata.json
python score_images.py --dataset flowers --metric imagereward   # -> metadata_IR.json
python score_images.py --dataset flowers --metric hpsv2         # -> metadata_HPS.json
```

Edit the config block at the top of `create_dataset_T2I.py` (prompt set, model list, output
directory) to target a specific dataset; `score_images.py` takes the dataset on the command line.
`create_dataset_LLM.py` builds the analogous dataset for the LLM selection task used by
`experiments/compare_to_baselines_LLM.py`.

### Scoring images: CLIPScore, ImageReward, HPSv2

CLIPScore is computed inline by `create_dataset_T2I.py` (into `metadata.json`). The two
preference-model metrics are computed afterwards, from the already-generated images, by a single
script — `scripts/create_dataset/score_images.py` — selected with `--metric`:

| `--metric` | Scorer | Output file | Score key |
|---|---|---|---|
| `imagereward` | `ImageReward-v1.0` (`image-reward` package) | `<dataset>/metadata_IR.json` | `image_reward_scores` |
| `hpsv2` | `hpsv2` package, variant from `--hps-version` | `<dataset>/metadata_HPS.json` | `hps_scores` |

```bash
cd scripts/create_dataset
python score_images.py --dataset carrot-bowl --metric hpsv2 --hps-version v2.1
python score_images.py --help          # --root / --input / --output for non-default layouts
```

Both metrics write the same schema as `metadata.json` — one entry per (prompt, model), with
`filenames` and the metric's per-image scores — so a dataset can be swapped between metrics by
pointing an experiment at a different metadata file. Images that fail to load, or prompts the
scorer raises on, are recorded as `null`; no score is ever substituted or interpolated.

#### HPSv2 scoring

`--metric hpsv2` is the HPSv2 code path, and running it is what produces `metadata_HPS.json`. Two
things to know:

- The `hpsv2` package downloads its own checkpoint on first use. The variant is `v2.1`, the default
  of `--hps-version`; pass the flag to score with another variant.
- To run an experiment on HPSv2 rewards, set `metric = "hpsv2"` in its config block — that resolves
  to `metadata_HPS.json` and the `hps_scores` key in `precomputed` mode, and to HPSv2 scoring of
  freshly generated images in `online` mode.

## Model versions

Exact identifiers as they appear in the code. None of the loading calls pins a Hugging Face
`revision=`: every model is fetched at the **latest** revision of its repository at download time,
which is what the "Revision/commit" column records below. Scores are therefore reproducible only up
to whatever the upstream repositories currently serve — if a model repository is updated, freshly
generated images (and their scores) can differ from the published ones.

### Text-to-image models (`scripts/create_dataset/create_dataset_T2I.py`)

| Name | Identifier | Revision/commit |
|---|---|---|
| Sana | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` | latest |
| Unidiffuser | `thu-ml/unidiffuser-v1` | latest |
| LCM | `SimianLuo/LCM_Dreamshaper_v7` | latest |
| Koala | `etri-vilab/koala-lightning-700m` (loaded with `variant="fp16"`) | latest |
| SDXL-Turbo | `stabilityai/sdxl-turbo` | latest |
| SSD-1B | `segmind/SSD-1B` | latest |

### Scoring and embedding models

| Name | Identifier | Revision/commit |
|---|---|---|
| CLIP (image–text scoring + prompt embeddings) | `openai/clip-vit-base-patch32` | latest |
| ImageReward | `ImageReward-v1.0` (via `ImageReward.load`) | latest |
| HPSv2 | see [HPSv2 scoring](#hpsv2-scoring) below | latest |
| Prompt embeddings — main experiments | `openai/clip-vit-base-patch32`, CLIP text tower (`get_text_features`) | latest |
| Prompt embeddings — BERT ablation (`compare_to_baselines_BERT.py`) | `bert-base-uncased` | latest |
| Question embeddings — LLM task (`compare_to_baselines_LLM.py`) | `roberta-base` | latest |

### LLM-selection task (`scripts/create_dataset/create_dataset_LLM.py`)

| Name | Identifier | Revision/commit |
|---|---|---|
| Gemma | `google/gemma-3-4b-it` | latest |
| Llama | `meta-llama/Llama-3.2-3B-Instruct` | latest |

Sampling settings that affect the generated images are in the code alongside each identifier
(inference steps, guidance scale, resolution, dtype) — e.g. SDXL-Turbo uses 2 steps at guidance 0.0,
Sana 10 steps at guidance 4.5 and 1024×1024. Each model generates `num_generations = 5` images per
prompt.

## Running the experiments

Each experiment is self-contained, with its configuration (dataset, models, horizon `T`, budgets,
hyperparameters) in a config block at the top. Run **from inside `experiments/`** so that the
`../datasets/...` paths resolve:

```bash
cd experiments
python compare_to_baselines.py        # main comparison (Table 1 / main figure)
python model_addition.py              # add-a-model robustness
python model_removal.py               # remove-a-model robustness
python n_models_queries.py            # models / query-budget sweep
python different_budgets.py           # active-learning budget sensitivity
python query_trigger_strategies.py    # query-trigger strategy comparison
```

Results are written under `experiments/results/<experiment>/data/` (pickled).

## Figures

Every experiment writes its own figure into `experiments/results/<experiment>/` when it finishes, so
no notebook is needed to see its results.

`notebooks/plot_compare_to_baselines.ipynb` shows how the multi-dataset main figure
(`assets/main_results.png`) is rebuilt from the saved pickles of `compare_to_baselines.py`. Its
parameter cell is **indicative**: those values (dataset, horizon `T`, number of runs, number of
models) also form the filename it loads, so point them at whichever run you actually have on disk.
