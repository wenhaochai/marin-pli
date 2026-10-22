# Byte-level and character-level LMs under repeated data: what prior work shows

Scope: does any prior work show or test that byte-level, character-level or tokenizer-free LMs are more (or less) robust than BPE models to multi-epoch repetition at a fixed amount of text? Our comparison point: 300m Qwen3-style decoder, MuonH, fineweb-edu, fixed text; 8 passes over 1/8 of the text cost bytes (vocabulary 256) +0.0003 Paloma bpb and the 128K BPE 0.08 bpb, with 64K/32K/16K/8K BPE falling monotonically in between; bytes lose 0.027 at 520m; bytes take 4.74x more steps and tokens for the same text.

Bottom line across all questions: I found **no paper that directly tests byte-level vs BPE robustness to multi-epoch repetition at fixed text**. The closest direct test is Kharitonov et al. 2021 (vocabulary size vs memorization in multi-epoch training, BPE vocabularies from ~600 to ~20K tokens, plus 0 merges on random strings), which points the same way as our finding. Everything else is indirect.

## Q1. Do the main byte/char architectures report multi-epoch training, repetition, overfitting, memorization or data efficiency vs BPE?

### Takeaway
None of ByT5, CANINE, Charformer, MegaByte, MambaByte, SpaceByte, BLT, EvaByte, MBLM, AU-Net, H-Net or Al-Rfou 2019 runs a repetition ablation (unique vs repeated text) against BPE. Several make **data-efficiency claims at fixed or smaller text** (ByT5 "4x less text", EvaByte "5x less data", H-Net "3.6x less data" on DNA, "better scaling with data"), and some byte models were in fact trained for many epochs (MambaByte ~14 epochs of PG-19; Al-Rfou many passes over 90M characters with dropout 0.55), but none of these isolates repetition.

### Cited Findings

**ByT5 (Xue et al., 2021/TACL 2022, arXiv 2105.13626) — indirect (data efficiency at fewer bytes, single pass).**
- Pretrained 1M steps on batches of 2^20 tokens, same token budget as mT5; so ByT5 was "pre-trained on 4× less text than mT5" ("roughly 4× less actual text"). The paper states that "a byte-level model trained on the same number of tokens as a word- or subword-level model will have been trained on less text data." — [ByT5 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2105.13626)
- Parameter-matched (d_model and d_ff adjusted), ~1.2x FLOPs per forward pass, ~+33% pretraining wall time. — [ByT5](https://ar5iv.labs.arxiv.org/html/2105.13626)
- ByT5 beats mT5 at Small and Base: GLUE Small 80.5 vs 75.6, Base 85.3 vs 83.0; SuperGLUE Small 67.8 vs 60.2, Base 74.0 vs 72.5; TyDiQA-GoldP Small 82.6/73.6 vs 75.9/64.8. Mixed at larger sizes on translate-train. — [ByT5](https://ar5iv.labs.arxiv.org/html/2105.13626)
- Conclusion: gains "suggest that byte-level models may be more data efficient learners." On Dakshina, ByT5-Large reaches peak in 2.5x fewer fine-tuning steps. They saw "clear saturation or overfitting on validation set metrics" in fine-tuning for some tasks (both models), and "did not observe any robust trends" with data availability (high vs low resource languages). — [ByT5](https://ar5iv.labs.arxiv.org/html/2105.13626)
- Status: indirect. Compares at fixed tokens (so bytes see 4x less text), single epoch; no repetition test, no memorization analysis.

**CANINE (Clark et al., TACL 2022, arXiv 2103.06874) — no repetition evidence.** Character encoder; +2.8 F1 over mBERT on TyDi QA with 28% fewer parameters; abstract does not discuss data volume, epochs or overfitting. — [CANINE](https://arxiv.org/abs/2103.06874)

**Charformer (Tay et al., ICLR 2022, arXiv 2106.12672) — no repetition evidence.** GBST learned soft subwords over bytes; on par with subword models, 28%–100% faster than vanilla byte/subword Transformers; abstract silent on data efficiency or epochs. — [Charformer](https://arxiv.org/abs/2106.12672)

**MegaByte (Yu et al., 2023, arXiv 2305.07185) — no repetition evidence.** Claims byte models "competitively with subword models on long context language modeling"; abstract gives no data/epoch analysis. — [MegaByte](https://arxiv.org/abs/2305.07185). MegaByte-758M+262M trained on 80B bytes and 1.3B+350M on 400B bytes of PG-19-scale corpora (as tabulated in MambaByte). — [MambaByte HTML](https://arxiv.org/html/2401.13660)

**MambaByte (Wang et al., COLM 2024, arXiv 2401.13660) — indirect, in fact multi-epoch.**
- MambaByte-353M ~30B bytes; MambaByte-972M ~150B bytes (380K steps); subword Mamba-1.03B listed at 600B bytes; Transformer-XL baselines 400B bytes. PG-19 train split is 11.74 GB. — [MambaByte HTML](https://arxiv.org/html/2401.13660)
- SpaceByte's paper describes MambaByte as trained for "14 epochs of PG-19." — [SpaceByte HTML](https://arxiv.org/html/2404.14408)
- Claim: "MambaByte outperforms MegaByte with 0.63× less compute and training data" (note: Table 2 numbers 30B vs 80B bytes imply 0.375x; the 0.63x is internally inconsistent per the fetch). No claim of data efficiency vs subword models; no epoch/overfitting analysis. — [MambaByte HTML](https://arxiv.org/html/2401.13660)
- Status: byte models (and the subword baselines they cite) were trained for many passes over PG-19, but no unique-vs-repeated ablation exists.

**SpaceByte (Slagle, NeurIPS 2024, arXiv 2404.14408) — compute-matched, no repetition.**
- Fixed training budgets of 10^18 and 10^19 FLOPs; bytes trained = training FLOPs / training FLOPs-per-byte; Pareto frontier of bpb vs inference FLOPs-per-byte. No epoch counts, no overfitting analysis. — [SpaceByte HTML](https://arxiv.org/html/2404.14408)
- At 10^19 FLOPs (bpb, PG-19/arXiv/GitHub): subword SentencePiece 0.989/0.768/0.508; GPT-2 tokenizer 1.013/0.796/0.554; plain byte Transformer 1.138/0.909/0.655; MegaByte 1.083/0.822/0.570; SpaceByte 1.009/0.748/0.500. — [SpaceByte HTML](https://arxiv.org/html/2404.14408)
- At ~30B bytes, ~728M FLOPs-per-byte: subword Transformer-1B PG-19 0.908, Stories 0.809, arXiv 0.666, GitHub 0.400; SpaceByte 0.918/0.833/0.663/0.411. "SpaceByte outperforms MegaByte, even though MegaByte was trained using 2.7 times as much compute and data." — [SpaceByte HTML](https://arxiv.org/html/2404.14408)

**BLT, Byte Latent Transformer (Pagnoni et al., ACL 2025, arXiv 2412.09871) — FLOP-matched, no repetition.**
- Comparisons are FLOP-controlled; batches hold 16M bytes on average; BPE models use Llama 3 compute-optimal ratios; BLT latent transformer same size as BPE counterpart, same data. 8B comparison: Llama 3 BPE on 1T tokens vs BLT-Entropy on 4.5T bytes (equivalent FLOPs). — [BLT HTML](https://arxiv.org/html/2412.09871)
- BPE "perform[s] better with small training budgets and [is] quickly surpassed by BLT, not far beyond the compute-optimal regime": at ~3.1E8 inference FLOPs/byte, BPE compute-optimal at 50B bytes, crossover at 150B bytes; at ~2.1E9 FLOPs/byte, BPE compute-optimal 400B bytes, crossover 1T bytes. — [BLT HTML](https://arxiv.org/html/2412.09871)
- No discussion of epochs, repetition, overfitting or memorization; only a remark that BLT on CUTE was trained on "16x less data than Llama 3.1." — [BLT HTML](https://arxiv.org/html/2412.09871)
- Status: indirect. Note the regime: BLT wins when more bytes per parameter are seen (over-trained), i.e. the advantage grows with data, not with scarcity.

**EvaByte (Zheng et al., HKU NLP blog, Jan 2025) — self-reported data efficiency, uncontrolled.**
- 6.5B byte model; 1.5T bytes (1.2T pretraining + annealing runs of 100B and 200B bytes); 1.22T bytes ≈ 0.4T tokens; "rivaling modern open-source tokenizer-based LMs despite using 5x less training data." The 5x is not derived on the page; comparisons are to OLMo/Qwen2.5 trained on different data. — [EvaByte blog](https://hkunlp.github.io/blog/2025/evabyte/); [model card](https://huggingface.co/evabyte/EvaByte-Phase1)
- "Consistent upward trend with no signs of plateauing"; beats BLT "even with 3-4x fewer training bytes." No epochs/repetition/overfitting discussion for pretraining; no controlled byte-vs-BPE at same data. — [EvaByte blog](https://hkunlp.github.io/blog/2025/evabyte/)

**H-Net, dynamic chunking (Hwang, Wang, Gu, 2025, arXiv 2507.07955) — data- and compute-matched, single pass; claims better data scaling.**
- "When compute- and data- matched, an H-Net with one stage of hierarchy operating at the byte level outperforms a strong Transformer language model operating over BPE tokens"; 2-stage H-Net matches the token Transformer "of twice its size" while "demonstrating significantly better scaling with data"; DNA "nearly 4x improvement in data efficiency." — [H-Net abstract](https://arxiv.org/abs/2507.07955)
- Setup: 100B-token FineWeb-Edu subset; byte models 8192 bytes/sequence vs BPE 1792 tokens; batch 256; models matched in bytes per batch and FLOPs per byte, "so every model sees the same underlying data in each minibatch." H-Net variants cross over the BPE Transformer after ~200B bytes (space-chunked), ~100B (1-stage), ~30B (2-stage). XL final FineWeb-Edu bpb: Transformer 0.730, H-Net 1-stage 0.728, 2-stage 0.715. DNA: each H-Net reaches the isotropic model's pre-decay perplexity with "approximately 3.6× less data" (final perplexity Transformer 2.769, Mamba-2 2.757, H-Net 2.705/2.697). — [H-Net HTML](https://arxiv.org/html/2507.07955v2)
- Status: the cleanest fixed-text (fixed bytes) comparison methodology in this list, and the closest analogue to our fixed-text design; but it is single-epoch (no repetition), and its "data efficiency" means lower loss per unique byte seen, not robustness to repeats.

**Multiscale Byte LM (Egli, Manica, Born, 2025, arXiv 2502.14553) — no evidence.** Hierarchical byte decoder for 5M-byte contexts; abstract silent on data efficiency, epochs or BPE at fixed data. — [MBLM](https://arxiv.org/abs/2502.14553)

**AU-Net (Videau et al., 2025, arXiv 2506.14761) — compute-matched, no repetition.** "Shallow hierarchies tie strong BPE baselines" under controlled pretraining compute; no data-efficiency or epoch claims in the abstract. — [AU-Net](https://arxiv.org/abs/2506.14761)

**Al-Rfou et al. 2019 character Transformer (arXiv 1808.04444) — many epochs, heavy regularization, no BPE comparison.**
- 64-layer char Transformer: 1.13 bpc text8, 1.06 enwik8. Datasets: 100M characters, 90M train. T64 trained 4M steps (best checkpoint ~2.5M steps), 16 sequences per step; model size "necessitated aggressive dropout rates of 0.55"; T12 dropout 0.2; enwik9 dropout 0.1 chosen on dev; auxiliary losses argued to "serve as an additional regularizer." On lm1b (large corpus) all dropout set to 0. Byte T64 on lm1b: 1.03 bpb ≈ 40.6 word perplexity vs a word-level model at 23.7. — [Al-Rfou (ar5iv)](https://ar5iv.labs.arxiv.org/html/1808.04444)
- Status: indirect counter-evidence. A char model with ~235M params on 90M characters, trained for many passes (my arithmetic, assuming 512-char contexts: 2.5M × 16 × 512 ≈ 20B chars ≈ 230 passes), does overfit enough to need dropout 0.55. Bytes are not immune to repetition at extreme ratios.

**The Efficiency Gap in Byte Modeling (Lee et al., Google DeepMind/Cornell, 2026, arXiv 2605.12928) — compute-matched, no repetition.**
- SlimPajama-627B; 48M–1.23B non-embedding params; bytes vocab 256 with 8192-byte context vs Llama 2 BPE (32K) with 1792-token context; FLOP budgets 6e18–2e21. AR: at bpb 1.0 bytes need ~7.9x compute (2.3e20 vs 2.9e19); at bpb 0.8 ~2.3x (2.7e21 vs 1.2e21); extrapolated AR parity ~1.3e22 FLOPs. Masked diffusion: ~34.3x at bpb 1.2, parity ~4.2e26. No epochs/repetition/memorization analysis. — [Efficiency Gap HTML](https://arxiv.org/html/2605.12928)

### Inferences
- The field compares bytes vs BPE almost exclusively at **matched FLOPs** (SpaceByte, BLT, AU-Net, Efficiency Gap) or **matched bytes per batch and FLOPs per byte** (H-Net). Under FLOP matching the byte model sees fewer bytes; under fixed tokens (ByT5) it sees ~4x fewer bytes. Our design (fixed text, unequal FLOPs, 4.74x more byte steps) matches H-Net's data axis but not its compute axis.
- The repeated "bytes are more data efficient" claims (ByT5, EvaByte, H-Net DNA, BLT CUTE) concern **loss per unique byte in one pass**, which is a different quantity from **loss lost to repetition**. Our +0.0003 bpb result is about the second quantity and appears to be new.
- BLT's crossover (BPE wins at small budgets, BLT wins at 2.5–3x compute-optimal bytes) and H-Net's widening gap with data are consistent with a picture where byte models are under-fit relative to BPE at equal text, which would also make them less sensitive to repeats. This is a plausible link, not a tested one.
- Al-Rfou shows the limit: at ~200+ passes over 90M chars with a 235M model, char models overfit and need very strong dropout. So our result should be stated as robustness at 8 passes and 300m, shrinking at 520m (0.027), not as immunity.

### Gaps
- No paper found that trains a byte/char LM and a BPE LM on the same text for k passes over 1/k of the data. I searched for byte-level + epochs/repetition/data-constrained and found none (searches in Oct 2026).
- MambaByte, MegaByte and the PG-19 subword baselines were multi-epoch, but no paper reports their overfitting curves; their train/val gaps are not tabulated.
- I could not read the full text of CANINE, Charformer, MegaByte, MBLM, AU-Net; claims of "no repetition analysis" for these rest on abstracts plus the absence of such results in later papers that cite them.
- Bolmo (byteified OLMo, [arXiv 2512.15586](https://arxiv.org/abs/2512.15586)) and "Decoupling the Benefits of Subword Tokenization ... via Byte-level Simulation" (Gigant, Peng, Quesnelle, [arXiv 2604.27263](https://arxiv.org/abs/2604.27263), attributes subword gains to throughput and boundary priors) were found but their abstracts do not mention repetition; full texts not checked.

## Q2. Do any compare at fixed text (bytes), and how is compute matched?

### Takeaway
Yes, a few: H-Net is the clearest fixed-bytes, FLOPs-per-byte-matched comparison; SpaceByte reports a ~30B-byte matched table; BLT uses constant bytes per batch but matches total FLOPs (so bytes differ). None of them vary the number of passes.

### Cited Findings
- H-Net: models matched "in bytes per batch and FLOPs per byte"; every model sees the same data each minibatch; 100B-token FineWeb-Edu; byte context 8192 vs BPE 1792 tokens. — [H-Net HTML](https://arxiv.org/html/2507.07955v2)
- SpaceByte: fixed training FLOPs (10^18, 10^19), then a ~30B-byte table matched on ~728M inference FLOPs-per-byte; subword Transformer-1B layer counts set to match FLOPs-per-byte. — [SpaceByte HTML](https://arxiv.org/html/2404.14408)
- BLT: constant 16M bytes per batch; FLOP-matched totals (Llama 3 1T tokens vs BLT 4.5T bytes at 8B). — [BLT HTML](https://arxiv.org/html/2412.09871)
- Efficiency Gap: fixed FLOP budgets; e.g. BPE baseline at 45B BPE tokens vs byte models at 188B byte tokens in Table 1. — [Efficiency Gap HTML](https://arxiv.org/html/2605.12928)
- ByT5: fixed tokens (same steps and batch as mT5), therefore 4x less text. — [ByT5](https://ar5iv.labs.arxiv.org/html/2105.13626)

### Inferences
- The common methodological move to make byte models affordable (patching/chunking to ~4–6 bytes per latent step, H-Net chunks ≈ GPT-2 token length) gives the main network a BPE-like sequence length. If the Kharitonov sequence-length mechanism (Q4) is right, these hierarchical byte models may lose part of the plain byte model's resistance to memorization. This is untested.
- Our setup is closest to "plain byte Transformer" rows (SpaceByte Table 1, Efficiency Gap), which are the most compute-hungry; the repetition advantage we see may not transfer to BLT/H-Net-style models.

### Gaps
- No fixed-text study reports what happens to the byte–BPE gap when the same text is repeated.

## Q3. Character/byte models in low-resource or small-data settings (MT, BabyLM)

### Takeaway
Fine-tuning evidence (MT) shows byte models gain most when data is small: ByT5 beats mT5 by up to 10 chrF++ at 400 sentence pairs, with the gap shrinking to ~1.2 at 4.5M pairs. BabyLM character entries report parity with subword models, not a clear win. None measure overfitting under pretraining repetition.

### Cited Findings
- Edman, Sarti, Toral, van Noord, Bisazza (TACL 2024, arXiv 2302.14220): ByT5 vs mT5 at 300M/582M/1.23B; fine-tuning on 0.4k, 2k, 10k, 50k, 250k NewsCommentary pairs plus WMT14 De→En 1.25M and 4.5M. ByT5 is higher in chrF++ at every size; gap "up to 10 chrF++ points" at 400 sentences (large). De→En large: 250k: 54.72 vs 56.83; 1.25M: 58.38 vs 59.78; 4.5M: 61.51 vs 62.73 chrF++; COMET roughly tied (0.843/0.841, 0.857/0.856, 0.867/0.874). Max epochs scaled from 6250 at 0.4k down to 10 at 250k, early-stopping patience 5. — [Edman et al. HTML](https://arxiv.org/html/2302.14220v3)
- Same paper: character models needed fewer epochs to converge. ByT5 zero-shot (unseen related languages) chrF++ fell from 37.7 (10k pairs) to 23.4 (250k pairs) while mT5 stayed at 34.4; the authors attribute this to number of training steps and read it as ByT5 memorizing source–target co-occurrences; freezing the first 25% of encoder layers largely prevented it. — [Edman et al. HTML](https://arxiv.org/html/2302.14220v3)
- Title-level conclusion: "in less time-sensitive, or low-resource settings, character-level translations *are* worth the wait"; char models ≥4x slower in training and inference. — [Edman et al. (arXiv)](https://arxiv.org/abs/2302.14220)
- Kharitonov et al. MT appendix (En–Nl, 8M OPUS pairs, BPE 10k/20k/40k/80k merges): BLEU 24.52/24.95/24.81/25.12; epochs to early stop 100/80/71/65 (smaller vocab trains longer before early stopping); membership-inference AUC rises with vocab size. — [Kharitonov et al. HTML](https://arxiv.org/html/2110.02782v2)
- BabyLM 2024: Bunzeck, Duran, Schade, Zarrieß, "Graphemes vs. phonemes: battling it out in character-based language models": grapheme-llama is "comparable to subword-based models on grammatical benchmarks"; phoneme-llama lower. — [ACL Anthology 2024.conll-babylm.5](https://aclanthology.org/2024.conll-babylm.5/)
- Same group, COLING 2025 (arXiv 2410.01487): small Llama models with character vocabularies "perform well" on syntactic and new lexical/phonetic tests; no epoch/repetition analysis visible in the abstract. — [Bunzeck et al.](https://arxiv.org/abs/2410.01487)
- BabyLM 2025 tokenizer entries found (morpheme tokenizers, syllable tokenizer "SylliTok", sub-token embeddings) are not byte-level; official strict-small GPT-BERT baselines use 16,384 BPE. I found no byte-level 2025 entry. — [BabyLM 2025 proceedings](https://aclanthology.org/events/babylm-2025/); [2025.babylm-main.35](https://aclanthology.org/2025.babylm-main.35.pdf); [2025.babylm-main.21](https://aclanthology.org/2025.babylm-main.21.pdf)
- A word-segmentation probing study on BabyLM data (arXiv 2504.03338) discusses byte-level latent patching scoring above BPE on BabyLM syntactic and lexical benchmarks (search-snippet level, not verified in full text). — [BabyLM's First Words](https://arxiv.org/pdf/2504.03338)
- Older/mixed evidence (search summary, not verified in full): on small IWSLT MT, character and byte models usually underperformed BPE (Shaham and Levy 2021); Libovický et al. review char-level MT. — [Why don't people use character-level MT? (arXiv 2110.08191)](https://arxiv.org/pdf/2110.08191)

### Inferences
- MT small-data advantage for bytes is real but measured at **fine-tuning** time, on top of a byte-pretrained model, and confounded by pretraining differences (ByT5 saw 4x less text than mT5).
- Edman's zero-shot collapse with longer training is a case where the byte model **memorizes more** with more steps, a counterexample to "bytes never overfit." It concerns transfer to unseen languages, not in-distribution loss.
- BabyLM (BabyLM data is 10M/100M words and entries routinely train many epochs) would be a natural testbed, but character entries report benchmark parity, not overfitting curves.

### Gaps
- No BabyLM report found that plots validation loss vs epoch for char/byte vs BPE.
- Exact BabyLM grapheme-llama numbers (vocab size, epochs, scores) not extracted; would need the PDF ([2024.conll-babylm.5.pdf](https://aclanthology.org/2024.conll-babylm.5.pdf)).

## Q4. Memorization/extraction rates of byte-level vs subword models

### Takeaway
No study compares extraction rates of byte-level vs subword LMs directly. The one direct test of tokenization granularity vs memorization (Kharitonov, Baroni, Hupkes 2021) finds that **larger BPE vocabularies memorize more** under multi-epoch training, at fixed parameter count, and attributes it to shorter sequences. This is the closest prior result to our monotone 128K > 64K > ... > 8K > bytes ordering.

### Cited Findings
- Kharitonov, Baroni, Hupkes, "How BPE Affects Memorization in Transformers" (arXiv 2110.02782): "with a large subword vocabulary size, Transformer models fit random mappings more easily and are more vulnerable to membership inference attacks"; language models with large vocabularies reproduce training data more often when prompted. — [abstract](https://arxiv.org/abs/2110.02782)
- Setups (all multi-epoch): (1) random-label SNLI (367,388 examples), vocab 611/1,097/4,943/9,574/18,336 tokens, up to 500 epochs; 1-layer Encoder fits ~87% of random labels at 611 tokens vs >98% at 18,336; LM and MLM go from ~94%/97% to 100%. (2) Membership inference on SNLI, 100 epochs, loss-threshold attack: AUC rises monotonically with vocab for all three architectures. (3) PAQ QA recovery, 14M pairs, 70 epochs, vocab 1,280–20,776: exact-match answer recovery rises with vocab. (4) Parameter-controlled variant (embedding + FC, ~18.2–18.3M params) keeps the trend. — [Kharitonov et al. HTML](https://arxiv.org/html/2110.02782v2)
- Mechanism tests: random incompressible strings still show the effect (rules out redundancy/compression); doubling vocabulary with synonym splits at fixed length does not reproduce it (rules out pure vocab size); uniform-frequency shortening keeps it (rules out frequency). Conclusion: shorter sequences drive memorization (conjecture by elimination; more attention heads help memorization on 8-merge random strings). — [Kharitonov et al. HTML](https://arxiv.org/html/2110.02782v2)
- Limitation: smallest real-text vocabulary is ~611 tokens; character-level/0-merge only for random strings; small models (4–12 layers); not a pretraining-scale LM.
- Tokenization-as-augmentation under repetition: Cosma et al. 2026 "Training Language Models with homotokens Leads to Delayed Overfitting" (arXiv 2601.02867): 88M and 244M models on FineWeb-Edu, GPT-2 tokenizer, ~20 tokens/param, repetitions R ∈ {16, 24, 32, 64}; for R > 16 homotokens (alternative segmentations) delay overfitting and reach lower best loss; no effect for R < 16; no byte/char experiment. — [homotokens HTML](https://arxiv.org/html/2601.02867v1)
- Standard extraction metrics (k-extractable with k prior tokens) are tokenizer-dependent, which makes byte-vs-subword extraction comparisons non-trivial. — [Carlini et al. 2021, arXiv 2012.07805](https://arxiv.org/pdf/2012.07805)
- Background (BPE only): repeated data up to ~4 epochs is nearly as good as unique, with returns decaying to zero (Muennighoff et al., "Scaling Data-Constrained Language Models"). — [JMLR version](https://jmlr.org/papers/volume26/24-1000/24-1000.pdf); Xue et al. 2023 "To Repeat or Not To Repeat" (token crisis). — [NeurIPS 2023 poster](https://neurips.cc/virtual/2023/poster/72533)

### Inferences
- Kharitonov's sequence-length mechanism predicts exactly our ordering: 128K BPE (shortest sequences) loses most to repetition, smaller BPE vocabularies less, bytes (4.74x longer) least. Our result extends their finding from memorization probes on small models to Paloma bpb under 8-pass pretraining at 300m, and adds the byte endpoint.
- A competing explanation is per-parameter exposure: bytes take 4.74x more optimizer steps for the same text, and the model is relatively under-fit per byte (consistent with BLT/Efficiency-Gap showing bytes behind BPE at equal FLOPs). Under-fit models lose less to repetition (Muennighoff's regime). The two explanations make different predictions: the under-fitting account predicts the byte advantage shrinks as model size grows relative to data (consistent with our 520m loss of 0.027); the sequence-length account predicts a persistent ordering at fixed size.
- Homotokens suggest that multiple segmentations of the same text act as augmentation; bytes have a single segmentation, so this mechanism does not explain the byte result.

### Gaps
- No direct extraction/membership-inference comparison of byte-level vs BPE LMs found (searches Oct 2026, including 2025–2026 extraction literature).
- No replication of Kharitonov at LLM pretraining scale found.
