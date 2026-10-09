# Analyzing Controversies in Political News Articles

Does a simulated audience of language model personas detect controversial political news better than a single language model asked directly? This repository builds a gold set of 100 political news articles labeled by five annotators, and compares an audience of six LLM personas with a single LLM agent, an aggregator agent and four supervised baselines. It also measures how much the conclusions depend on the label source (human gold labels against LLM silver labels) and on the prompt wording and decoding seed.

It is the code of our paper, also included in this repository (`Analyzing_Controversies_in_Political_News_Articles.pdf`).

Authors: Kseniia Ostudina, Raphael Leonardi, Baptiste Pras, Martin Leiva and Vladimir Herrera Nativi (Université Paris-Saclay).

## Main results

PR-AUC with the minority class NO (not controversial) as the positive class, mean ± standard deviation over 9 runs (3 prompt paraphrases × 3 seeds) for the LLM systems and over 3 cross-validation repeats for the trained baselines.

| System | Backbone | Gold labels (24 NO) | Silver labels (9 NO) |
| --- | --- | --- | --- |
| Random scorer | | 0.240 | 0.090 |
| TF-IDF + logistic regression | | 0.416 ± 0.050 | 0.368 ± 0.032 |
| RoBERTa-base, fine-tuned | | 0.315 ± 0.017 | 0.212 ± 0.097 |
| DA-RoBERTa, fine-tuned | | 0.499 ± 0.033 | 0.423 ± 0.063 |
| DA-RoBERTa trained on BABE, used as is | | **0.665** | 0.406 |
| Single agent | Qwen2.5-7B | 0.566 ± 0.035 | 0.458 ± 0.083 |
| Audience, 6 personas | Qwen2.5-7B | 0.263 ± 0.045 | 0.104 ± 0.030 |
| Aggregator | Qwen2.5-7B | 0.372 ± 0.069 | 0.252 ± 0.088 |
| Single agent | GLM-4-9B | 0.454 ± 0.046 | **0.625 ± 0.138** |
| Audience, 6 personas | GLM-4-9B | 0.399 ± 0.045 | 0.144 ± 0.049 |
| Aggregator | GLM-4-9B | 0.450 ± 0.064 | 0.204 ± 0.069 |

The audience simulation never beats the single agent. Changing the label source changes the winner (Kendall's τ of 0.56 between the gold and silver rankings), and for most systems the prompt paraphrase moves the score more than the seed. The paper gives the ablation and the discussion. Its bootstrap intervals and figures were computed separately and are not produced by this pipeline.

## Repository content

| Path | Content |
| --- | --- |
| `run_all.py` | Main script: supervised baselines, LLM systems, ablation and report |
| `pipeline/` | Backends (MLX, Ollama), personas and prompts, LLM systems, baselines, metrics, data loading, report |
| `annotate_silver.py` | Silver labels from an LLM annotator that never sees the human labels |
| `merge_gold.py` | Majority vote over the annotator files, which gives the gold file |
| `annotator_stats.py` | Krippendorff's α, Cohen's κ, Gwet's AC1 and disagreements between annotators |
| `config.yaml` | Models, temperatures, baselines and run profiles |
| `tests/selftest.py` | Offline test of the metrics, the parsers and the whole pipeline with a fake model |
| `data/` | Gold labels, silver labels and sampling design, without article text |
| `outputs/` | Results of the final run, with the article text removed |
| `docs/annotation_guideline.md` | The annotation guideline used for the gold and silver labels |
| `gold_sampling/` | Pipeline that built the 100-article gold sample from misinfo-general (see its own README) |

## Data

The articles come from [misinfo-general](https://huggingface.co/datasets/ioverho/misinfo-general) (Verhoeven et al., 2024), distributed under CC BY-NC-SA 4.0. The authors ask that the article text not be redistributed openly, so this repository contains no title, no lead and no Reddit comment. Every file keeps two keys: `item_id` (our identifier, for example `W1-045`) and `article_id` (the identifier in misinfo-general, for example `2022-0023739`).

| File | Content |
| --- | --- |
| `data/articles.csv` | Gold labels: `label` (YES or NO, majority vote), `n_raters`, `n_yes`, `n_no`. 100 articles, 76 YES and 24 NO; 26 articles have 5 labels and 74 have 1 (204 labels in total) |
| `data/silver.csv` | Silver labels from Mistral-NeMo 12B (temperature 0, one pass): `label`, `confidence`, `agreement`, `reason`, `model`. 91 YES and 9 NO |
| `data/sample_design.csv` | Sampling design of each article: `wave`, `selection` (random or stratified), `stratum`, proxy `flag` (H or L), year, publisher leaning and reliability, `source`, `topic_id`, inclusion probabilities `pi_*`, sampling `weight` (1 / `pi_total`), and the annotators of the article (`overlap`, `n_raters`, `raters`) |

The silver `reason` column is written by the LLM annotator. Any run of consecutive words copied from the article (4 words or more from the title, 5 or more from the lead) was replaced with `[...]`.

### Rebuilding the article text

The pipeline needs the title and the first 150 words of each article. To run it, first request access to misinfo-general on Hugging Face and log in (`huggingface-cli login`). Then rebuild the text with the function that wrote the annotation files:

```python
import sys

import pandas as pd

sys.path.insert(0, "gold_sampling")
from gold.common import load_config
from gold.textfetch import fetch

cfg = load_config()
for name in ("articles", "silver"):
    df = pd.read_csv(f"data/{name}.csv", dtype=str, keep_default_na=False)
    text = fetch(cfg, df["article_id"].tolist())
    df.merge(text, on="article_id", how="left").to_csv(f"data/{name}_with_text.csv", index=False)
```

The `*_with_text.csv` files are listed in `.gitignore`, so they cannot be committed by mistake.

## Setup

Everything runs locally. The final run used a MacBook Air M5 with 24 GB of memory: Qwen2.5-7B-Instruct (4-bit) through [MLX](https://github.com/ml-explore/mlx), and GLM-4-9B and Mistral-NeMo through [Ollama](https://github.com/ollama/ollama).

```bash
mamba env create -f environment.yml     # or: pip install -r requirements.txt
mamba activate audience-pipeline
ollama pull glm4:9b
ollama pull mistral-nemo                # silver annotator, about 7 GB
python tests/selftest.py                # about 30 seconds, no model needed
```

Ollama must be running (the app, or `brew services start ollama`). `mlx-lm` only works on Apple Silicon. On another machine, remove it from the environment and pass `--backbones glm`.

## Usage

### Gold labels

`merge_gold.py` merges one annotated file per annotator (`item_id, title, lead_150w, label, confidence, reason`) by majority vote. Ties are written to `data/needs_adjudication.csv`. `annotator_stats.py` reads the same files and computes the agreement statistics. The individual annotator files are not part of this repository; their result is `data/articles.csv` and `outputs/annotator_stats.md`.

```bash
python merge_gold.py ann_1.csv ann_2.csv ... --out data/articles_with_text.csv
python annotator_stats.py ann_1.csv ann_2.csv ... --silver data/silver_with_text.csv --out outputs/annotator_stats_with_silver.md
```

### Silver labels

```bash
python annotate_silver.py --articles data/articles_with_text.csv --out data/silver_with_text.csv --estimate
python annotate_silver.py --articles data/articles_with_text.csv --out data/silver_with_text.csv
```

The script reads only `item_id`, `title` and `lead_150w`, so the silver labels cannot depend on the human labels. The model and its settings are in the `silver` section of `config.yaml`. Calls are cached in `outputs/silver/calls_silver.jsonl`, so an interrupted run resumes where it stopped.

### Experiment

Three profiles are defined in `config.yaml`: `smoke` (8 articles, to check the plumbing), `check` (a small file, 2 paraphrases and 1 seed) and `final` (3 paraphrases × 3 seeds = 9 runs). The final results were produced in stages with:

```bash
python run_all.py --profile final --gold data/articles_with_text.csv --silver data/silver_with_text.csv --ablation-articles all --stage baselines
python run_all.py --profile final --gold data/articles_with_text.csv --silver data/silver_with_text.csv --ablation-articles all --stage llm
python run_all.py --profile final --gold data/articles_with_text.csv --silver data/silver_with_text.csv --ablation-articles all --stage ablations
python run_all.py --profile final --gold data/articles_with_text.csv --silver data/silver_with_text.csv --ablation-articles all --stage report
```

`--stage all` runs the baselines, the LLM systems and the report; the ablation is always a separate stage. Every LLM result is cached by (system, paraphrase, seed, article), so a run can be stopped with Ctrl-C and restarted with the same command. Since the results of the final run are included in `outputs/final/full_logs/`, the `report` stage rebuilds every table without a single LLM call.

Useful options:

| Option | Effect |
| --- | --- |
| `--backbones qwen` | Run one backbone only |
| `--seeds 0` | Run seed 0 only; the other seeds can be added later, finished runs are reused |
| `--max-articles N` | Use the first N articles |
| `--ablations homogeneous deliberative selfconsistency` | Choose the ablations (only `homogeneous` by default) |
| `--estimate` | Print the number of calls and the hours that remain, then stop |
| `--call-stats` | Print the measured latency per call type, then stop |
| `--mock` | Use fake models, to test the plumbing |

The full LLM runs took about 23 hours on the laptop.

## Systems

| System | Name in the code | Calls per article | Score |
| --- | --- | --- | --- |
| Single agent | `llm_single` | 1 | Score in [0, 1] from the guideline, the title and the lead |
| Audience | `audience` | 6 | Normalized entropy of the stances (−1, 0, +1) of six personas: far left, center left, centrist, center right, far right, disengaged |
| Aggregator | `audience_direct` | 7 | A judge reads the guideline, the title and the six reactions and returns a score in [0, 1]; it reuses the persona calls |
| Homogeneous ablation | `ablation_homogeneous` | 6 | Entropy over six samples of the centrist persona |
| Supervised baselines | `tfidf`, `roberta_ft`, `da_roberta_ft` | | Stratified 5-fold cross-validation, repeated with 3 seeds |
| Bias detector used as is | `da_roberta_transfer` | | Probability of "biased" from `mediabiasgroup/da-roberta-babe-ft` |

Personas use temperature 0.6 and judges 0.3. An unreadable answer is retried up to twice with a new seed. Two more systems are implemented but were not evaluated in the paper: a deliberative audience (`audience_deliberative`) and a self-consistency ablation with a generic reader (`ablation_selfconsistency`). The pipeline also accepts a third label regime (`--distant FILE`), which was not used either.

## Outputs

| Path | Content |
| --- | --- |
| `outputs/final/readable/summary.md` | Full report: answers to the research questions, leaderboards per label regime, noise decomposition, ablation, memory and speed |
| `outputs/final/readable/metrics_by_run.csv` | Every metric for every system, regime and run |
| `outputs/final/readable/scores_by_article.csv` | Mean and standard deviation of each system's score per article |
| `outputs/final/full_logs/results_{qwen,glm}.jsonl` | One line per system, run and article: score, persona stances, generated reactions, reason or summary, latency |
| `outputs/final/full_logs/baseline_*.csv` | Scores and timings of the supervised baselines |
| `outputs/final/full_logs/backbone_info_*.json` | Checkpoint, memory and latency per call type for each backbone |
| `outputs/silver/silver_report.md` | Summary of the silver annotation |
| `outputs/annotator_stats*.md` | Agreement between annotators, and with the silver labels |

Some files of the run are not included because they contain the article text: the complete call logs (`calls_*.jsonl`, whose prompts include the lead) and the per-article cards (`readable/articles/`). In the files that are included, the generated texts (reactions, reasons and summaries) went through the same filter as the silver reasons: any run of consecutive words copied from the article (4 words or more from the title, 5 or more from the lead) was replaced with `[...]`. The titles were removed from the disagreement tables of `outputs/annotator_stats*.md`. Scores and metrics are unchanged: rebuilding the report from these files with the article text gives the same `summary.md`, `metrics_by_run.csv` and `scores_by_article.csv`.

## Gold sample

`gold_sampling/` contains the pipeline that drew the 100 articles: political filter on 4.2 million articles, Reddit and Wikipedia proxy signals, stratified sampling in two waves, and blind assignment to the five annotators. It cannot be rerun without large intermediate files, but its logic and settings are documented in [`gold_sampling/README.md`](gold_sampling/README.md).

## Citation

If you use the labels, please cite the dataset the articles come from:

> I. Verhoeven, P. Mishra, and E. Shutova. Yesterday's News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models. *arXiv preprint arXiv:2410.18122*, 2024. https://arxiv.org/abs/2410.18122
