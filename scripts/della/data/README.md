# Della copy of the July Baseline training data

What the July rungs (`experiments/grug/moe/della_july_baseline.py`) read on Della, and which files have already been
used, so a later expansion can draw disjoint files. All paths are under `/scratch/gpfs/KARTHIKN/wc9403/marin_store/`;
`marin_store_big` symlinks to the same raw and tokenized directories.

## Current corpus (from 2026-09-14, runs with the `_codestrat` id suffix)

| component | on Della | July reference | how it was chosen |
|---|---|---|---|
| Nemotron-CC (7 tiers) | 317 of 31,279 files, 108 GiB compressed, ~65B tokens | full dump (~10.4 TiB) | 1% of each quality/kind/kind2 group, uniform over files, seed 20260914 (`nemotron_cc_sample1pct_20260914.txt`) |
| starcoderdata | full: 863 parquet, 289.5 GiB | full | `bigcode/starcoderdata@9fc30b5` snapshot_download |
| proof-pile-2 | full: 482 jsonl.zst, 47.6 GiB | full | `EleutherAI/proof-pile-2@901a927` snapshot_download |
| Paloma, uncheatable_eval | full | full | Marin steps |

Nemotron-CC could not be taken whole: 10.4 TiB compressed, ~25 TiB tokenized, against ~13 TiB of free quota and
weeks of login-node tokenization. The 1% draw is uniform over files within each group, so document statistics match
the full tier in expectation; it covers 9-60 CC snapshots per group, not every snapshot. Mixture weights come from
`nemotron_mix` in `experiments/pretraining_datasets/nemotron.py`, not from file counts. Per-tier tokens are
~7x the d1024 need, so no tier repeats within a run.

Downloader: `scripts/della/nemotron_sample.py`. Its input `nemotron_paths.txt` is the S3 listing of
`contrib/Nemotron/Nemotron-CC/data-jsonl/` (31,279 files, 99 snapshots); a copy is kept at
`marin_store/raw/nemotron_paths.txt`.

## Files already used or downloaded (exclude when expanding)

| manifest | files | status |
|---|---|---|
| `nemotron_cc_sample1pct_20260914.txt` | 317 | tokenized, in use by every `_codestrat` run |
| `nemotron_cc_subset17_used_by_d512_run1.txt` | 17 | first d512 run (Paloma 3.596, discarded); subset of the 53 |
| `nemotron_cc_subset53_used_by_failed_d768seg1.txt` | 53 | first d768 segment (5011 steps, discarded); raw kept at `raw/nemotro-cc-eeb783-subset-20260914` in both stores |
| `nemotron_cc_reserve172_downloaded_unused.txt` | 172 | downloaded 2026-09-13 to `raw/nemotro-cc-eeb783-reserve`, never tokenized; 36 overlap the 53, 2 overlap the 1% sample |

The three sets overlap as noted; a disjoint expansion passes all four manifests to `--exclude`. The discarded runs
saw 17/53 files for at most 11k/5k steps, so reusing those files is a purity question, not a contamination one: keep
them excluded for the July reproduction, feel free to reuse them for unrelated work.

Old code samples, kept for provenance only: `raw/starcoderdata-720c8c-random9` (9 files, no python/cpp; caused the
+0.03 Paloma gap of the first d512 run) and `raw/starcoderdata-720c8c-subset-20260914` (92-language stratified
2.6 GB sample, never trained on); `raw/proof-pile-2-f1b1d8-subset-20260914` (6 train files). Their tokenized caches
carry the same suffixes under `tokenized/`.

## Expanding later

```bash
D=scripts/della/data
python scripts/della/nemotron_sample.py nemotron_paths.txt /scratch/gpfs/KARTHIKN/wc9403/marin_store/raw/nemotron-expansion \
    --fraction 0.01 --seed 20261001 --exclude $D/nemotron_cc_sample1pct_20260914.txt \
    --exclude $D/nemotron_cc_subset53_used_by_failed_d768seg1.txt --exclude $D/nemotron_cc_reserve172_downloaded_unused.txt
```

Then move the new files under `raw/nemotro-cc-eeb783/contrib/...` next to the existing ones, move the seven
`tokenized/nemotron_cc/*` caches aside, and retokenize with `ZEPHYR_MAX_WORKERS=16` one component at a time
(the tokenize step globs the whole directory; a cache is not incremental). Append the new manifest to this table.
