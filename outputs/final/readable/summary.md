# Results summary (final profile)

Generated 2026-10-09 19:04. Label regimes evaluated: gold (100 articles), silver (100 articles). The LLM systems score each article once per run; each regime only changes which labels the scores are compared with.

Reading guide: PR-AUC and P@k use the minority class as positive (as in the protocol). 'F1 macro @best*' picks the best threshold on the same data, so it is optimistic. ECE reads each score as P(controversial). Mean ± std are over runs (LLM: paraphrase x seed; supervised baselines: cross-validation repeats). 'Beyond the noise' means the gap is larger than the larger within-system std over runs.

## A. Quick answers

### Gold labels: 100 articles (76 YES, 24 NO)

**1. Does the audience simulation beat a single agent?**

- glm, Audience, 6 personas (entropy): PR-AUC 0.399 ± 0.045 against 0.454 ± 0.046 for the single agent. Verdict: worse, beyond the noise. Better in 2 of 9 runs.
- glm, Audience + aggregator (direct): PR-AUC 0.450 ± 0.064 against 0.454 ± 0.046 for the single agent. Verdict: no difference beyond the noise. Better in 3 of 9 runs.
- qwen, Audience, 6 personas (entropy): PR-AUC 0.263 ± 0.045 against 0.566 ± 0.035 for the single agent. Verdict: worse, beyond the noise. Better in 0 of 9 runs.
- qwen, Audience + aggregator (direct): PR-AUC 0.372 ± 0.069 against 0.566 ± 0.035 for the single agent. Verdict: worse, beyond the noise. Better in 0 of 9 runs.

**2. How do the LLM systems compare with the supervised baselines?**

- Best LLM system: LLM single agent [qwen], PR-AUC 0.566 ± 0.035. Best supervised baseline: DA-RoBERTa BABE, zero-shot, PR-AUC 0.665 ± 0.000. Verdict for the best LLM system: too few runs to measure the noise.
- DA-RoBERTa BABE used as is (no training on our labels): PR-AUC 0.665 ± 0.000.

**3. Are the results stable across paraphrases and seeds (RQ2)?**

| System | Runs | PR-AUC mean | Std total | Std paraphrase | Std seed | Larger source |
|---|---|---|---|---|---|---|
| LLM single agent [glm] | 9 | 0.454 | 0.046 | 0.046 | 0.009 | paraphrase |
| LLM single agent [qwen] | 9 | 0.566 | 0.035 | 0.037 | 0.005 | paraphrase |
| Audience, 6 personas (entropy) [glm] | 9 | 0.399 | 0.045 | 0.017 | 0.024 | seed |
| Audience, 6 personas (entropy) [qwen] | 9 | 0.263 | 0.045 | 0.043 | 0.021 | paraphrase |
| Audience + aggregator (direct) [glm] | 9 | 0.450 | 0.064 | 0.017 | 0.063 | seed |
| Audience + aggregator (direct) [qwen] | 9 | 0.372 | 0.069 | 0.058 | 0.015 | paraphrase |

**4. Does persona diversity matter (heterogeneous audience against 6 draws of one centrist)?**

- glm: heterogeneous 0.399 ± 0.045 against homogeneous 0.285 ± 0.024. Verdict: better, beyond the noise. Heterogeneous is better in 9 of 9 runs.
- qwen: heterogeneous 0.263 ± 0.045 against homogeneous 0.263 ± 0.040. Verdict: no difference beyond the noise. Heterogeneous is better in 4 of 9 runs.
- Computed on 100 articles, of which 24 in the minority class.

**5. Do the two backbones agree?**

- Spearman correlation of the per-article audience score between glm and qwen: 0.19.
- Audience, 6 personas (entropy): PR-AUC 0.399 ± 0.045 on glm, 0.263 ± 0.045 on qwen.
- LLM single agent: PR-AUC 0.454 ± 0.046 on glm, 0.566 ± 0.035 on qwen.

### Silver labels: 100 articles (91 YES, 9 NO)

> **Fragile:** only 9 articles in the minority class NO (not controversial). PR-AUC and P@k rest on those few articles, so every ranking below can flip with one article. A random scorer gets 0.090.

**1. Does the audience simulation beat a single agent?**

- glm, Audience, 6 personas (entropy): PR-AUC 0.144 ± 0.049 against 0.625 ± 0.138 for the single agent. Verdict: worse, beyond the noise. Better in 0 of 9 runs.
- glm, Audience + aggregator (direct): PR-AUC 0.204 ± 0.069 against 0.625 ± 0.138 for the single agent. Verdict: worse, beyond the noise. Better in 0 of 9 runs.
- qwen, Audience, 6 personas (entropy): PR-AUC 0.104 ± 0.030 against 0.458 ± 0.083 for the single agent. Verdict: worse, beyond the noise. Better in 0 of 9 runs.
- qwen, Audience + aggregator (direct): PR-AUC 0.252 ± 0.088 against 0.458 ± 0.083 for the single agent. Verdict: worse, beyond the noise. Better in 0 of 9 runs.

**2. How do the LLM systems compare with the supervised baselines?**

- Best LLM system: LLM single agent [glm], PR-AUC 0.625 ± 0.138. Best supervised baseline: DA-RoBERTa FT (CV), PR-AUC 0.423 ± 0.063. Verdict for the best LLM system: better, beyond the noise.
- DA-RoBERTa BABE used as is (no training on our labels): PR-AUC 0.406 ± 0.000.

**3. Are the results stable across paraphrases and seeds (RQ2)?**

| System | Runs | PR-AUC mean | Std total | Std paraphrase | Std seed | Larger source |
|---|---|---|---|---|---|---|
| LLM single agent [glm] | 9 | 0.625 | 0.138 | 0.135 | 0.054 | paraphrase |
| LLM single agent [qwen] | 9 | 0.458 | 0.083 | 0.093 | 0.012 | paraphrase |
| Audience, 6 personas (entropy) [glm] | 9 | 0.144 | 0.049 | 0.023 | 0.021 | paraphrase |
| Audience, 6 personas (entropy) [qwen] | 9 | 0.104 | 0.030 | 0.023 | 0.020 | paraphrase |
| Audience + aggregator (direct) [glm] | 9 | 0.204 | 0.069 | 0.013 | 0.058 | seed |
| Audience + aggregator (direct) [qwen] | 9 | 0.252 | 0.088 | 0.086 | 0.029 | paraphrase |

**4. Does persona diversity matter (heterogeneous audience against 6 draws of one centrist)?**

- glm: heterogeneous 0.144 ± 0.049 against homogeneous 0.097 ± 0.012. Verdict: no difference beyond the noise. Heterogeneous is better in 9 of 9 runs.
- qwen: heterogeneous 0.104 ± 0.030 against homogeneous 0.111 ± 0.033. Verdict: no difference beyond the noise. Heterogeneous is better in 5 of 9 runs.
- Computed on 100 articles, of which 9 in the minority class.

**5. Do the two backbones agree?**

- Spearman correlation of the per-article audience score between glm and qwen: 0.19.
- Audience, 6 personas (entropy): PR-AUC 0.144 ± 0.049 on glm, 0.104 ± 0.030 on qwen.
- LLM single agent: PR-AUC 0.625 ± 0.138 on glm, 0.458 ± 0.083 on qwen.

### Do the label regimes agree? (RQ1)

Label agreement on the articles that both regimes label:

| Pair | Articles | Same label | Cohen's kappa |
|---|---|---|---|
| gold vs silver | 100 | 81% | 0.34 |

PR-AUC (minority) of each system under each regime, sorted by the first regime (only systems that have results under every regime):

| System | gold (rank) | silver (rank) |
|---|---|---|
| DA-RoBERTa BABE, zero-shot | 0.665 (#1) | 0.406 (#4) |
| LLM single agent [qwen] | 0.566 (#2) | 0.458 (#2) |
| DA-RoBERTa FT (CV) | 0.499 (#3) | 0.423 (#3) |
| LLM single agent [glm] | 0.454 (#4) | 0.625 (#1) |
| Audience + aggregator (direct) [glm] | 0.450 (#5) | 0.204 (#8) |
| TF-IDF + LR (CV) | 0.416 (#6) | 0.368 (#5) |
| Audience, 6 personas (entropy) [glm] | 0.399 (#7) | 0.144 (#9) |
| Audience + aggregator (direct) [qwen] | 0.372 (#8) | 0.252 (#6) |
| RoBERTa-base FT (CV) | 0.315 (#9) | 0.212 (#7) |
| Audience, 6 personas (entropy) [qwen] | 0.263 (#10) | 0.104 (#10) |

Kendall's tau between the system rankings of two regimes (1 = same order, 0 = unrelated, -1 = reversed):

| Pair | Systems | Kendall tau |
|---|---|---|
| gold vs silver | 10 | 0.56 |

> Silver labels come from another LLM. An LLM system can agree more with an LLM annotator than with humans, so read the silver ranking next to the gold ranking, not instead of it.

## B. Leaderboards (sorted by PR-AUC on the minority class)

### Gold labels (minority class: NO (not controversial), random scorer: 0.240)

| Rank | System | Runs | PR-AUC (minority) | Gain over random | P@10 | P@20 | F1 macro @0.5 | Versus #1 (noise-aware) |
|---|---|---|---|---|---|---|---|---|
| 1 | DA-RoBERTa BABE, zero-shot | 1 | 0.665 ± 0.000 | +0.425 | 0.700 ± 0.000 | 0.600 ± 0.000 | 0.581 ± 0.000 | best |
| 2 | LLM single agent [qwen] | 9 | 0.566 ± 0.035 | +0.326 | 0.736 ± 0.077 | 0.625 ± 0.043 | 0.717 ± 0.023 | behind #1 |
| 3 | DA-RoBERTa FT (CV) | 3 | 0.499 ± 0.033 | +0.259 | 0.567 ± 0.058 | 0.533 ± 0.029 | 0.678 ± 0.015 | behind #1 |
| 4 | LLM single agent [glm] | 9 | 0.454 ± 0.046 | +0.214 | 0.672 ± 0.051 | 0.516 ± 0.049 | 0.634 ± 0.026 | behind #1 |
| 5 | Audience + aggregator (direct) [glm] | 9 | 0.450 ± 0.064 | +0.210 | 0.615 ± 0.079 | 0.544 ± 0.076 | 0.690 ± 0.059 | behind #1 |
| 6 | TF-IDF + LR (CV) | 3 | 0.416 ± 0.050 | +0.176 | 0.433 ± 0.153 | 0.367 ± 0.029 | 0.551 ± 0.042 | behind #1 |
| 7 | Audience, 6 personas (entropy) [glm] | 9 | 0.399 ± 0.045 | +0.159 | 0.488 ± 0.124 | 0.423 ± 0.081 | 0.619 ± 0.055 | behind #1 |
| 8 | Audience + aggregator (direct) [qwen] | 9 | 0.372 ± 0.069 | +0.132 | 0.474 ± 0.165 | 0.431 ± 0.105 | 0.561 ± 0.068 | behind #1 |
| 9 | RoBERTa-base FT (CV) | 3 | 0.315 ± 0.017 | +0.075 | 0.267 ± 0.058 | 0.283 ± 0.104 | 0.525 ± 0.058 | behind #1 |
| 10 | Audience, 6 personas (entropy) [qwen] | 9 | 0.263 ± 0.045 | +0.023 | 0.278 ± 0.137 | 0.257 ± 0.096 | 0.513 ± 0.062 | behind #1 |

### Silver labels (minority class: NO (not controversial), random scorer: 0.090)

| Rank | System | Runs | PR-AUC (minority) | Gain over random | P@10 | P@20 | F1 macro @0.5 | Versus #1 (noise-aware) |
|---|---|---|---|---|---|---|---|---|
| 1 | LLM single agent [glm] | 9 | 0.625 ± 0.138 | +0.535 | 0.635 ± 0.065 | 0.396 ± 0.035 | 0.851 ± 0.044 | best |
| 2 | LLM single agent [qwen] | 9 | 0.458 ± 0.083 | +0.368 | 0.483 ± 0.126 | 0.401 ± 0.041 | 0.770 ± 0.049 | behind #1 |
| 3 | DA-RoBERTa FT (CV) | 3 | 0.423 ± 0.063 | +0.333 | 0.333 ± 0.058 | 0.250 ± 0.000 | 0.672 ± 0.020 | behind #1 |
| 4 | DA-RoBERTa BABE, zero-shot | 1 | 0.406 ± 0.000 | +0.316 | 0.400 ± 0.000 | 0.250 ± 0.000 | 0.429 ± 0.000 | behind #1 |
| 5 | TF-IDF + LR (CV) | 3 | 0.368 ± 0.032 | +0.278 | 0.300 ± 0.000 | 0.200 ± 0.000 | 0.476 ± 0.000 | behind #1 |
| 6 | Audience + aggregator (direct) [qwen] | 9 | 0.252 ± 0.088 | +0.162 | 0.284 ± 0.153 | 0.239 ± 0.076 | 0.597 ± 0.080 | behind #1 |
| 7 | RoBERTa-base FT (CV) | 3 | 0.212 ± 0.097 | +0.122 | 0.167 ± 0.058 | 0.083 ± 0.029 | 0.491 ± 0.013 | behind #1 |
| 8 | Audience + aggregator (direct) [glm] | 9 | 0.204 ± 0.069 | +0.114 | 0.242 ± 0.076 | 0.232 ± 0.070 | 0.620 ± 0.065 | behind #1 |
| 9 | Audience, 6 personas (entropy) [glm] | 9 | 0.144 ± 0.049 | +0.054 | 0.129 ± 0.090 | 0.130 ± 0.067 | 0.509 ± 0.042 | behind #1 |
| 10 | Audience, 6 personas (entropy) [qwen] | 9 | 0.104 ± 0.030 | +0.014 | 0.099 ± 0.083 | 0.090 ± 0.056 | 0.497 ± 0.056 | behind #1 |

# Details

## Gold labels in detail

Articles: 100 (76 YES, 24 NO, 76% YES). Minority class: NO (not controversial). A random scorer gets PR-AUC and P@k equal to the minority share, 0.240, and PR-AUC (YES) of 0.760.

### Main results, all 100 articles

| System | Runs | PR-AUC (minority) | P@10 (minority) | P@20 (minority) | PR-AUC (YES) | F1 macro @0.5 | F1 macro @best* | ECE | Calls/article | Sec/100 articles | Unreadable answers | No score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LR (CV) | 3 | 0.416 ± 0.050 | 0.433 ± 0.153 | 0.367 ± 0.029 | 0.842 ± 0.018 | 0.551 ± 0.042 | 0.620 ± 0.028 | 0.198 ± 0.009 | n/a | 0 | n/a | 0.0% |
| RoBERTa-base FT (CV) | 3 | 0.315 ± 0.017 | 0.267 ± 0.058 | 0.283 ± 0.104 | 0.805 ± 0.009 | 0.525 ± 0.058 | 0.565 ± 0.060 | 0.244 ± 0.003 | n/a | 254 | n/a | 0.0% |
| DA-RoBERTa FT (CV) | 3 | 0.499 ± 0.033 | 0.567 ± 0.058 | 0.533 ± 0.029 | 0.890 ± 0.011 | 0.678 ± 0.015 | 0.700 ± 0.008 | 0.166 ± 0.028 | n/a | 242 | n/a | 0.0% |
| DA-RoBERTa BABE, zero-shot | 1 | 0.665 ± 0.000 | 0.700 ± 0.000 | 0.600 ± 0.000 | 0.923 ± 0.000 | 0.581 ± 0.000 | 0.750 ± 0.000 | 0.333 ± 0.000 | n/a | 6 | n/a | 0.0% |
| LLM single agent [glm] | 9 | 0.454 ± 0.046 | 0.672 ± 0.051 | 0.516 ± 0.049 | 0.847 ± 0.011 | 0.634 ± 0.026 | 0.665 ± 0.030 | 0.153 ± 0.054 | 1.0 | 533 | 0.0% | 0.0% |
| LLM single agent [qwen] | 9 | 0.566 ± 0.035 | 0.736 ± 0.077 | 0.625 ± 0.043 | 0.898 ± 0.007 | 0.717 ± 0.023 | 0.745 ± 0.013 | 0.114 ± 0.033 | 1.0 | 404 | 0.0% | 0.0% |
| Audience, 6 personas (entropy) [glm] | 9 | 0.399 ± 0.045 | 0.488 ± 0.124 | 0.423 ± 0.081 | 0.847 ± 0.024 | 0.619 ± 0.055 | 0.642 ± 0.039 | 0.155 ± 0.014 | 6.0 | 2119 | 0.0% | 0.0% |
| Audience, 6 personas (entropy) [qwen] | 9 | 0.263 ± 0.045 | 0.278 ± 0.137 | 0.257 ± 0.096 | 0.754 ± 0.029 | 0.513 ± 0.062 | 0.527 ± 0.050 | 0.233 ± 0.052 | 6.0 | 1726 | 0.0% | 0.0% |
| Audience + aggregator (direct) [glm] | 9 | 0.450 ± 0.064 | 0.615 ± 0.079 | 0.544 ± 0.076 | 0.870 ± 0.027 | 0.690 ± 0.059 | 0.690 ± 0.059 | 0.247 ± 0.055 | 7.0 | 2697 | 0.0% | 0.0% |
| Audience + aggregator (direct) [qwen] | 9 | 0.372 ± 0.069 | 0.474 ± 0.165 | 0.431 ± 0.105 | 0.836 ± 0.024 | 0.561 ± 0.068 | 0.623 ± 0.049 | 0.109 ± 0.036 | 7.0 | 2186 | 0.0% | 0.0% |

### Ensemble view (per-article mean score over all runs)

| System | PR-AUC (minority) | P@10 (minority) | P@20 (minority) | PR-AUC (YES) | F1 macro @0.5 | F1 macro @best* | ECE |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR (CV) | 0.424 | 0.500 | 0.350 | 0.848 | 0.496 | 0.619 | 0.195 |
| RoBERTa-base FT (CV) | 0.318 | 0.300 | 0.250 | 0.825 | 0.465 | 0.561 | 0.244 |
| DA-RoBERTa FT (CV) | 0.499 | 0.600 | 0.500 | 0.891 | 0.678 | 0.696 | 0.155 |
| DA-RoBERTa BABE, zero-shot | 0.665 | 0.700 | 0.600 | 0.923 | 0.581 | 0.750 | 0.333 |
| LLM single agent [glm] | 0.583 | 0.700 | 0.550 | 0.871 | 0.643 | 0.680 | 0.119 |
| LLM single agent [qwen] | 0.673 | 0.800 | 0.650 | 0.921 | 0.719 | 0.740 | 0.080 |
| Audience, 6 personas (entropy) [glm] | 0.509 | 0.600 | 0.450 | 0.893 | 0.617 | 0.690 | 0.110 |
| Audience, 6 personas (entropy) [qwen] | 0.272 | 0.400 | 0.300 | 0.743 | 0.545 | 0.552 | 0.176 |
| Audience + aggregator (direct) [glm] | 0.654 | 0.733 | 0.675 | 0.940 | 0.740 | 0.791 | 0.241 |
| Audience + aggregator (direct) [qwen] | 0.531 | 0.600 | 0.500 | 0.880 | 0.583 | 0.707 | 0.097 |

### Pairwise reading of PR-AUC (minority)

'+' means the row system beats the column system by more than the larger within-system std, '-' the opposite, '~' means the gap is within the noise.

|  | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 |  | + | - | - | ~ | - | ~ | + | ~ | ~ |
| S2 | - |  | - | - | - | - | - | + | - | ~ |
| S3 | + | + |  | - | ~ | - | + | + | ~ | + |
| S4 | + | + | + |  | + | + | + | + | + | + |
| S5 | ~ | + | ~ | - |  | - | + | + | ~ | + |
| S6 | + | + | + | - | + |  | + | + | + | + |
| S7 | ~ | + | - | - | - | - |  | + | ~ | ~ |
| S8 | - | - | - | - | - | - | - |  | - | - |
| S9 | ~ | + | ~ | - | ~ | - | ~ | + |  | + |
| S10 | ~ | ~ | - | - | - | - | ~ | + | - |  |

S1 = TF-IDF + LR (CV); S2 = RoBERTa-base FT (CV); S3 = DA-RoBERTa FT (CV); S4 = DA-RoBERTa BABE, zero-shot; S5 = LLM single agent [glm]; S6 = LLM single agent [qwen]; S7 = Audience, 6 personas (entropy) [glm]; S8 = Audience, 6 personas (entropy) [qwen]; S9 = Audience + aggregator (direct) [glm]; S10 = Audience + aggregator (direct) [qwen]

### Ablation on 100 articles (24 in the minority class)

| System | Runs | PR-AUC (minority) | P@10 (minority) | P@20 (minority) | PR-AUC (YES) | F1 macro @0.5 | F1 macro @best* | ECE | Calls/article | Sec/100 articles | Unreadable answers | No score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM single agent [glm] | 9 | 0.454 ± 0.046 | 0.672 ± 0.051 | 0.516 ± 0.049 | 0.847 ± 0.011 | 0.634 ± 0.026 | 0.665 ± 0.030 | 0.153 ± 0.054 | 1.0 | 533 | 0.0% | 0.0% |
| LLM single agent [qwen] | 9 | 0.566 ± 0.035 | 0.736 ± 0.077 | 0.625 ± 0.043 | 0.898 ± 0.007 | 0.717 ± 0.023 | 0.745 ± 0.013 | 0.114 ± 0.033 | 1.0 | 404 | 0.0% | 0.0% |
| Audience, 6 personas (entropy) [glm] | 9 | 0.399 ± 0.045 | 0.488 ± 0.124 | 0.423 ± 0.081 | 0.847 ± 0.024 | 0.619 ± 0.055 | 0.642 ± 0.039 | 0.155 ± 0.014 | 6.0 | 2119 | 0.0% | 0.0% |
| Audience, 6 personas (entropy) [qwen] | 9 | 0.263 ± 0.045 | 0.278 ± 0.137 | 0.257 ± 0.096 | 0.754 ± 0.029 | 0.513 ± 0.062 | 0.527 ± 0.050 | 0.233 ± 0.052 | 6.0 | 1726 | 0.0% | 0.0% |
| Ablation: 1 persona x 6 draws [glm] | 9 | 0.285 ± 0.024 | 0.303 ± 0.039 | 0.303 ± 0.039 | 0.806 ± 0.022 | 0.502 ± 0.037 | 0.502 ± 0.037 | 0.348 ± 0.025 | 6.0 | 1545 | 0.0% | 0.0% |
| Ablation: 1 persona x 6 draws [qwen] | 9 | 0.263 ± 0.040 | 0.263 ± 0.074 | 0.266 ± 0.070 | 0.775 ± 0.043 | 0.471 ± 0.055 | 0.479 ± 0.044 | 0.387 ± 0.052 | 6.0 | 1810 | 0.0% | 0.0% |

Verdicts (a difference larger than the larger std counts as real):

- qwen, heterogeneous against homogeneous personas: PR-AUC 0.263 ± 0.045 against 0.263 ± 0.040, no difference beyond the noise.
- qwen, 6 personas against a single agent: PR-AUC 0.263 ± 0.045 against 0.566 ± 0.035, worse, beyond the noise.
- glm, heterogeneous against homogeneous personas: PR-AUC 0.399 ± 0.045 against 0.285 ± 0.024, better, beyond the noise.
- glm, 6 personas against a single agent: PR-AUC 0.399 ± 0.045 against 0.454 ± 0.046, worse, beyond the noise.

## Silver labels in detail

Articles: 100 (91 YES, 9 NO, 91% YES). Minority class: NO (not controversial). A random scorer gets PR-AUC and P@k equal to the minority share, 0.090, and PR-AUC (YES) of 0.910.

### Main results, all 100 articles

| System | Runs | PR-AUC (minority) | P@10 (minority) | P@20 (minority) | PR-AUC (YES) | F1 macro @0.5 | F1 macro @best* | ECE | Calls/article | Sec/100 articles | Unreadable answers | No score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LR (CV) | 3 | 0.368 ± 0.032 | 0.300 ± 0.000 | 0.200 ± 0.000 | 0.947 ± 0.015 | 0.476 ± 0.000 | 0.673 ± 0.005 | 0.253 ± 0.008 | n/a | 0 | n/a | 0.0% |
| RoBERTa-base FT (CV) | 3 | 0.212 ± 0.097 | 0.167 ± 0.058 | 0.083 ± 0.029 | 0.942 ± 0.002 | 0.491 ± 0.013 | 0.592 ± 0.046 | 0.370 ± 0.013 | n/a | 240 | n/a | 0.0% |
| DA-RoBERTa FT (CV) | 3 | 0.423 ± 0.063 | 0.333 ± 0.058 | 0.250 ± 0.000 | 0.965 ± 0.012 | 0.672 ± 0.020 | 0.701 ± 0.029 | 0.119 ± 0.032 | n/a | 221 | n/a | 0.0% |
| DA-RoBERTa BABE, zero-shot | 1 | 0.406 ± 0.000 | 0.400 ± 0.000 | 0.250 ± 0.000 | 0.974 ± 0.000 | 0.429 ± 0.000 | 0.748 ± 0.000 | 0.471 ± 0.000 | n/a | 5 | n/a | 0.0% |
| LLM single agent [glm] | 9 | 0.625 ± 0.138 | 0.635 ± 0.065 | 0.396 ± 0.035 | 0.990 ± 0.005 | 0.851 ± 0.044 | 0.851 ± 0.044 | 0.204 ± 0.065 | 1.0 | 533 | 0.0% | 0.0% |
| LLM single agent [qwen] | 9 | 0.458 ± 0.083 | 0.483 ± 0.126 | 0.401 ± 0.041 | 0.987 ± 0.004 | 0.770 ± 0.049 | 0.785 ± 0.045 | 0.194 ± 0.071 | 1.0 | 404 | 0.0% | 0.0% |
| Audience, 6 personas (entropy) [glm] | 9 | 0.144 ± 0.049 | 0.129 ± 0.090 | 0.130 ± 0.067 | 0.938 ± 0.017 | 0.509 ± 0.042 | 0.518 ± 0.032 | 0.235 ± 0.029 | 6.0 | 2119 | 0.0% | 0.0% |
| Audience, 6 personas (entropy) [qwen] | 9 | 0.104 ± 0.030 | 0.099 ± 0.083 | 0.090 ± 0.056 | 0.904 ± 0.023 | 0.497 ± 0.056 | 0.509 ± 0.043 | 0.253 ± 0.033 | 6.0 | 1726 | 0.0% | 0.0% |
| Audience + aggregator (direct) [glm] | 9 | 0.204 ± 0.069 | 0.242 ± 0.076 | 0.232 ± 0.070 | 0.959 ± 0.014 | 0.620 ± 0.065 | 0.620 ± 0.065 | 0.373 ± 0.055 | 7.0 | 2697 | 0.0% | 0.0% |
| Audience + aggregator (direct) [qwen] | 9 | 0.252 ± 0.088 | 0.284 ± 0.153 | 0.239 ± 0.076 | 0.962 ± 0.017 | 0.597 ± 0.080 | 0.654 ± 0.064 | 0.209 ± 0.034 | 7.0 | 2186 | 0.0% | 0.0% |

### Ensemble view (per-article mean score over all runs)

| System | PR-AUC (minority) | P@10 (minority) | P@20 (minority) | PR-AUC (YES) | F1 macro @0.5 | F1 macro @best* | ECE |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR (CV) | 0.381 | 0.400 | 0.200 | 0.948 | 0.476 | 0.680 | 0.252 |
| RoBERTa-base FT (CV) | 0.176 | 0.200 | 0.150 | 0.931 | 0.476 | 0.585 | 0.364 |
| DA-RoBERTa FT (CV) | 0.462 | 0.300 | 0.250 | 0.970 | 0.694 | 0.707 | 0.125 |
| DA-RoBERTa BABE, zero-shot | 0.406 | 0.400 | 0.250 | 0.974 | 0.429 | 0.748 | 0.471 |
| LLM single agent [glm] | 0.871 | 0.700 | 0.450 | 0.998 | 0.828 | 0.878 | 0.206 |
| LLM single agent [qwen] | 0.580 | 0.600 | 0.400 | 0.995 | 0.751 | 0.833 | 0.202 |
| Audience, 6 personas (entropy) [glm] | 0.176 | 0.100 | 0.150 | 0.955 | 0.558 | 0.599 | 0.240 |
| Audience, 6 personas (entropy) [qwen] | 0.099 | 0.100 | 0.100 | 0.900 | 0.533 | 0.540 | 0.220 |
| Audience + aggregator (direct) [glm] | 0.305 | 0.300 | 0.350 | 0.986 | 0.652 | 0.702 | 0.372 |
| Audience + aggregator (direct) [qwen] | 0.486 | 0.300 | 0.300 | 0.984 | 0.634 | 0.694 | 0.203 |

### Pairwise reading of PR-AUC (minority)

'+' means the row system beats the column system by more than the larger within-system std, '-' the opposite, '~' means the gap is within the noise.

|  | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 |  | + | ~ | - | - | - | + | + | + | + |
| S2 | - |  | - | - | - | - | ~ | + | ~ | ~ |
| S3 | ~ | + |  | ~ | - | ~ | + | + | + | + |
| S4 | + | + | ~ |  | - | ~ | + | + | + | + |
| S5 | + | + | + | + |  | + | + | + | + | + |
| S6 | + | + | ~ | ~ | - |  | + | + | + | + |
| S7 | - | ~ | - | - | - | - |  | ~ | ~ | - |
| S8 | - | - | - | - | - | - | ~ |  | - | - |
| S9 | - | ~ | - | - | - | - | ~ | + |  | ~ |
| S10 | - | ~ | - | - | - | - | + | + | ~ |  |

S1 = TF-IDF + LR (CV); S2 = RoBERTa-base FT (CV); S3 = DA-RoBERTa FT (CV); S4 = DA-RoBERTa BABE, zero-shot; S5 = LLM single agent [glm]; S6 = LLM single agent [qwen]; S7 = Audience, 6 personas (entropy) [glm]; S8 = Audience, 6 personas (entropy) [qwen]; S9 = Audience + aggregator (direct) [glm]; S10 = Audience + aggregator (direct) [qwen]

### Ablation on 100 articles (9 in the minority class)

| System | Runs | PR-AUC (minority) | P@10 (minority) | P@20 (minority) | PR-AUC (YES) | F1 macro @0.5 | F1 macro @best* | ECE | Calls/article | Sec/100 articles | Unreadable answers | No score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM single agent [glm] | 9 | 0.625 ± 0.138 | 0.635 ± 0.065 | 0.396 ± 0.035 | 0.990 ± 0.005 | 0.851 ± 0.044 | 0.851 ± 0.044 | 0.204 ± 0.065 | 1.0 | 533 | 0.0% | 0.0% |
| LLM single agent [qwen] | 9 | 0.458 ± 0.083 | 0.483 ± 0.126 | 0.401 ± 0.041 | 0.987 ± 0.004 | 0.770 ± 0.049 | 0.785 ± 0.045 | 0.194 ± 0.071 | 1.0 | 404 | 0.0% | 0.0% |
| Audience, 6 personas (entropy) [glm] | 9 | 0.144 ± 0.049 | 0.129 ± 0.090 | 0.130 ± 0.067 | 0.938 ± 0.017 | 0.509 ± 0.042 | 0.518 ± 0.032 | 0.235 ± 0.029 | 6.0 | 2119 | 0.0% | 0.0% |
| Audience, 6 personas (entropy) [qwen] | 9 | 0.104 ± 0.030 | 0.099 ± 0.083 | 0.090 ± 0.056 | 0.904 ± 0.023 | 0.497 ± 0.056 | 0.509 ± 0.043 | 0.253 ± 0.033 | 6.0 | 1726 | 0.0% | 0.0% |
| Ablation: 1 persona x 6 draws [glm] | 9 | 0.097 ± 0.012 | 0.071 ± 0.039 | 0.071 ± 0.039 | 0.923 ± 0.007 | 0.383 ± 0.029 | 0.476 ± 0.000 | 0.481 ± 0.014 | 6.0 | 1545 | 0.0% | 0.0% |
| Ablation: 1 persona x 6 draws [qwen] | 9 | 0.111 ± 0.033 | 0.094 ± 0.061 | 0.099 ± 0.060 | 0.919 ± 0.020 | 0.384 ± 0.045 | 0.477 ± 0.001 | 0.496 ± 0.056 | 6.0 | 1810 | 0.0% | 0.0% |

Verdicts (a difference larger than the larger std counts as real):

- qwen, heterogeneous against homogeneous personas: PR-AUC 0.104 ± 0.030 against 0.111 ± 0.033, no difference beyond the noise.
- qwen, 6 personas against a single agent: PR-AUC 0.104 ± 0.030 against 0.458 ± 0.083, worse, beyond the noise.
- glm, heterogeneous against homogeneous personas: PR-AUC 0.144 ± 0.049 against 0.097 ± 0.012, no difference beyond the noise.
- glm, 6 personas against a single agent: PR-AUC 0.144 ± 0.049 against 0.625 ± 0.138, worse, beyond the noise.

## Memory and speed

| Model | Backend | Checkpoint | Peak memory (GB) | Mean s per call |
|---|---|---|---|---|
| qwen | mlx | mlx-community/Qwen2.5-7B-Instruct-4bit | 4.7 | 3.14 |
| glm | ollama | glm4:9b | 5.3 | 3.41 |
| DA-RoBERTa FT (CV) (gold) | torch (MPS or CPU) | n/a | 10.8 | n/a |
| DA-RoBERTa BABE, zero-shot (gold) | torch (MPS or CPU) | n/a | 10.8 | n/a |
| RoBERTa-base FT (CV) (gold) | torch (MPS or CPU) | n/a | 8.7 | n/a |
| TF-IDF + LR (CV) (gold) | sklearn (CPU) | n/a | 0.0 | n/a |
| DA-RoBERTa FT (CV) (silver) | torch (MPS or CPU) | n/a | 10.8 | n/a |
| DA-RoBERTa BABE, zero-shot (silver) | torch (MPS or CPU) | n/a | 10.8 | n/a |
| RoBERTa-base FT (CV) (silver) | torch (MPS or CPU) | n/a | 11.9 | n/a |
| TF-IDF + LR (CV) (silver) | sklearn (CPU) | n/a | 10.8 | n/a |

Unreadable answers = share of calls whose answer could not be parsed (they are retried up to twice). No score = share of articles that still had no usable score after the retries; those get the neutral value 0.5 in the metrics, so a high value means the row is not trustworthy. Sec/100 articles for LLM systems is the sum of call latencies (sequential-equivalent time). For supervised baselines it is the full cross-validation time (training included) divided by the number of articles.

Latency by call type (baseline = single agent, persona = persona calls, aggregate = aggregator, homog = homogeneous ablation draws, delib = deliberation, selfcons = self-consistency draws):

| Backbone | Call type | Calls | Mean s | Mean prompt tokens | Mean generated tokens | Retried |
|---|---|---|---|---|---|---|
| qwen | aggregate | 900 | 4.59 | 603 | 42 | 0.0% |
| qwen | baseline | 900 | 4.04 | 603 | 34 | 0.0% |
| qwen | homog | 5400 | 3.02 | 371 | 22 | 0.0% |
| qwen | persona | 5400 | 2.88 | 382 | 25 | 0.0% |
| glm | aggregate | 900 | 5.78 | 571 | 53 | 0.0% |
| glm | baseline | 900 | 5.33 | 577 | 47 | 0.0% |
| glm | homog | 5402 | 2.57 | 362 | 26 | 0.0% |
| glm | persona | 5402 | 3.53 | 373 | 29 | 0.0% |

## Do the backbones read the personas the same way?

| Persona | qwen: mean stance (share -1 / 0 / +1) | glm: mean stance (share -1 / 0 / +1) |
|---|---|---|
| persona_center_left | -0.05 (37% / 30% / 32%) | -0.01 (44% / 14% / 43%) |
| persona_center_right | -0.07 (39% / 28% / 32%) | +0.17 (35% / 13% / 52%) |
| persona_centrist | -0.16 (26% / 64% / 10%) | -0.06 (30% / 46% / 24%) |
| persona_disengaged | -0.04 (16% / 72% / 12%) | +0.08 (11% / 70% / 19%) |
| persona_far_left | -0.23 (60% / 3% / 37%) | +0.34 (32% / 1% / 67%) |
| persona_far_right | -0.38 (65% / 9% / 26%) | +0.48 (25% / 2% / 73%) |

| Persona (qwen vs glm) | Same majority stance per article | Articles |
|---|---|---|
| persona_center_left | 68% | 100 |
| persona_center_right | 61% | 100 |
| persona_centrist | 64% | 100 |
| persona_disengaged | 82% | 100 |
| persona_far_left | 52% | 100 |
| persona_far_right | 32% | 100 |

## What this run does not cover

- No temporal split and no Horvitz-Thompson reweighting: metrics describe this sample, not the natural feed. Supervised baselines use cross-validation on the same articles, which can leak topic and favor them.
- BABE, Reddit and DELPHI side experiments are not included.
- Kendall's tau (RQ1) needs at least two regimes; with one regime it is not computed.
