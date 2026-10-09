# Gold sample

This folder contains the pipeline that drew the 100 political news articles of the gold set from [misinfo-general](https://huggingface.co/datasets/ioverho/misinfo-general) (about 4.2 million articles from 2017 to 2022, with publisher labels from Media Bias/Fact Check and BERTopic topics in `metadata.db`). Its result is `../data/sample_design.csv`.

The code is provided to document the sampling. It **cannot be rerun as is**: it needs the Hugging Face dataset and about 230 MB of intermediate files (`candidates.parquet`, `proxies.parquet`, `metadata.db`, `data/`), which are not included because they would expose the article text or are too large.

## Steps

Every setting is in `config.yaml`. Every random draw uses a seed derived from the master seed `20261005` (`sha256("<seed>:<step>")`, see `gold/common.py`), so adding a step never changes the random stream of another one. Each step logs to `logs/` with a snapshot of the resolved config.

| Step | Command | Code | Output |
| --- | --- | --- | --- |
| 1. Content statistics | `python extract_pilot_articles.py content-stats` | `gold/content_stats.py` | Words and masked tokens per article (no text kept) |
| 2. Reddit posts | `python extract_pilot_articles.py reddit-fetch` | `gold/reddit_fetch.py` | Submissions from r/politics, r/news, r/worldnews and r/Conservative, from the [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) API |
| 3. Wikipedia lists | `python extract_pilot_articles.py wiki` | `gold/wiki.py` | Pages of Wikipedia's [List of controversial issues](https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues) and their redirects |
| 4. Candidates | `python extract_pilot_articles.py candidates` | `gold/candidates.py`, `gold/political.py` | `candidates.parquet`: 1,801,181 articles |
| 5. Proxies | `python extract_pilot_articles.py proxies` | `gold/proxies.py`, `gold/urlnorm.py` | `proxies.parquet`: Reddit and Wikipedia signals per article |
| 6. Strata | `python extract_pilot_articles.py strata` | `gold/strata.py` | Proxy flag, strata and allocation plan |
| 7. Draw a wave | `python extract_pilot_articles.py wave K` | `gold/waves.py`, `gold/assign.py`, `gold/textfetch.py` | Sample with weights, and blind annotation files |
| 8. Rebalance | `python rebalance.py --after-wave K` | `rebalance.py` | Stratified quota of wave K+1 |
| 9. Export | `python export_gold_annotations.py` | `export_gold_annotations.py` | Labels joined with the design and the Reddit posts |

`python extract_pilot_articles.py catchup K name` deals again the wave-K items whose only annotator is absent.

### Candidates

An article is kept if its BERTopic topic is political: at least 30% of the topic's titles match a political lexicon (`gold/political.py`), or at least 15% when one of its 5 keywords is political. Satire publishers are excluded, titles must have at least 20 characters, articles must have 150 to 2,000 words (masked tokens not counted) with at most 50% masked tokens, and exact or near-duplicate titles are removed (the earliest article is kept).

### Proxies and flag

A Reddit post matches an article when its normalized URL is the article's URL and it was created between 2 days before and 30 days after publication. Year 2018 has no URLs in the dataset, so it is absent. The sampling pool is the set of Reddit-matched articles (190,059).

An article is flagged **H** (likely controversial) if any rule of `config.yaml` (`proxy_flag`) fires, and **L** otherwise:

* the title or the topic keywords match an issue page of the Wikipedia list;
* the most commented post has an upvote ratio of at most 0.80 and at least 20 comments (from 2020 on, when ratios are available);
* the upvote ratios of different subreddits differ by at least 0.20;
* the ratio of comments to score is at least 2, with at least 20 comments.

### Sampling

The sample has 2 waves of 50 articles. Each wave has 10 articles drawn uniformly at random from the pool, to estimate prevalence, and 40 stratified articles. A stratum is `flag | year | bias | reliability | matched` (for example `H|2020|Right|unreliable|m1`), where bias is Left, Center or Right and reliability is reliable or unreliable (Questionable Source or Conspiracy-Pseudoscience in MBFC). A publisher appears at most 5 times, and a (topic, month, publisher) combination at most twice.

Every article keeps its inclusion probability: `pi_random` for the random part, `pi_strat` for the stratified part (summed over the waves), and `pi_total = 1 - (1 - pi_random) × (1 - pi_strat)`. The sampling weight is `1 / pi_total`, so weighted estimates describe the whole pool.

After wave 1 was labeled, `rebalance.py` lowered the share of H articles in the stratified part from 0.50 to 0.43, to raise the share of the minority class (NO).

### Assignment

In each wave, 13 articles (26 in total) go to all five annotators, to measure agreement. They are picked at a regular step in the wave sorted by stratum, so they cover every stratum and the random part. The other 37 articles of each wave go to one annotator each, sorted by flag, bias, year and stratum and dealt in turn, starting one annotator later in each wave, so nobody gets mostly H or mostly L articles. Each annotator labeled 40 to 42 articles.

Annotation files are blind: they contain only `item_id, title, lead_150w, label, confidence, reason`, in random order. The lead is the first 150 words of the article (`gold/textfetch.py`). Nobody knows which items are shared, nor their stratum or source.

## Pilot

`extract_pilot_articles.py` also keeps the commands of an earlier pilot (`inspect`, `labels`, `extract`, `assign`, `merge`). The pilot took the first 200 political titles of the 2020 split and assigned 30 of them to the 5 annotators, under a first version of the guideline. **The pilot is not the final sample** and none of its labels is used in the paper. Its code stays here because the final assignment (`gold/assign.py`) reuses its functions.

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests
```

The tests run the draw and assignment logic on a synthetic frame, with no network and no data. The checks on real outputs are skipped when those files are absent. The code needs pandas 2: with pandas 3, the strata code fails on read-only arrays.
