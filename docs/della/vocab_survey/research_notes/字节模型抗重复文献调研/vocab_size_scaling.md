# Vocabulary size / tokenization vs memorization, overfitting under repetition, and data-constrained scaling

Scope note: searched 2026-10-22 (arXiv abs/html, Semantic Scholar citation API, ACL Anthology, NeurIPS pages). "Direct" = the paper varies vocabulary/tokenization AND measures memorization or repetition cost. "Indirect" = touches one side only. Our own page (wenhaochai.com/exps/vocab-overfitting.html) came up in web search; it is our own work and is NOT independent evidence.

## Q1. Kharitonov, Baroni & Hupkes 2021 — exact claims, setup, follow-ups

### Takeaway
Kharitonov et al. is still the only controlled study that varies BPE vocabulary size at fixed parameter count and measures memorization; larger vocab => more memorization (random-label fitting, membership-inference AUC, verbatim QA recovery), attributed (by exclusion) to shorter token sequences. None of its 38 citing papers re-tests it on LM pretraining or on multi-epoch held-out loss; they only restate "bigger vocab -> more memorization".

### Cited Findings
- Paper: "How BPE Affects Memorization in Transformers", arXiv 2110.02782 (v1 6 Oct 2021, v2 2 Dec 2021); arXiv lists no peer-reviewed venue. Abstract claim: BPE vocab size "strongly affects" ability and tendency to memorize, even at fixed parameter count; larger vocab -> fits random mappings better, more membership-inference vulnerable, regurgitates training data more often; conjectured cause: shorter sequences. — [arXiv abs](https://arxiv.org/abs/2110.02782)
- (DIRECT, memorization) Random-label memorization on SNLI (neutral removed, 367,388 examples, labels random ±1). LM/MLM 4 layers, 4 heads, d=512, FFN 1024; Encoder 1 layer d=256. Adam lr 5e-4, batch 512, up to 500 epochs, stop at train acc ≥0.999. BPE merges 0.5/1/5/10/20 K -> vocab 611 / 1,097 / 4,943 / 9,574 / 18,336. Encoder train acc ~87% at vocab 611 -> >98% at 18,336; LM/MLM ~94% / ~97% -> 100%. Holds with parameters held roughly fixed (embedding dim shrunk as vocab grows, e.g. LM 500 merges d=4400, 18,311,749 params vs 20K merges d=260, 18,231,474 params). — [arXiv html v2](https://arxiv.org/html/2110.02782v2)
- (DIRECT) Membership inference (SNLI true labels, half train/half hold-out, Yeom et al. loss-threshold, 100 epochs): AUC rises monotonically with vocab size for all three architectures (values only in Fig. 2); hold-out test accuracy > 0.78 for all models. — [arXiv html v2](https://arxiv.org/html/2110.02782v2)
- (DIRECT) Verbatim extraction / QA recovery on PAQ L1 (~14M QA pairs, 4M probed): vocab 1,280 / 1,784 / 5,784 / 10,784 / 15,784 / 20,776; LM-base (6L, 8H, d=512) and LM-large (12L, 16H, d=1024), dropout 0.1, 70 epochs, no early stopping; top-1 and top-5 exact-match recovery increase consistently with vocab (Fig. 3 only), also with parameters controlled; LM-large > LM-base. — [arXiv html v2](https://arxiv.org/html/2110.02782v2)
- Mechanism controls: (i) incompressible random binary strings, merges {0..256}: memorization still rises with vocab -> not about removing redundancy; (ii) "vocabulary growth without shortening" (each token randomly split into two synonym tokens, same sequence length): memorization drops at first — vocab size per se is not the cause; (iii) uniform-frequency idealized merging (vocab 2/4/16/256): effect persists -> not about skewed frequency. Authors conclude sequence length is the main factor "by exclusion". — [arXiv html v2](https://arxiv.org/html/2110.02782v2)
- MT appendix (En–Nl, 8M OPUS pairs; BPE 10k/20k/40k/80k): val BLEU 24.52 / 24.95 / 24.81 / 25.12; epochs before stopping 100 / 80 / 71 / 65 (smaller vocab trained longer before stopping); MI AUC rises with vocab. Authors: memorization and generalization "not directly at odds". No direct character-vs-BPE LM comparison; no held-out loss under repeated data. — [arXiv html v2](https://arxiv.org/html/2110.02782v2)
- Citing works (Semantic Scholar lists 38). Those that cite it about vocab/memorization, all as restatements, no new test: "Undesirable Memorization in LLMs: A Survey" 2410.02650 ("Bigger tokenizer vocabulary leads to more memorization"); "Mitigating Memorization in LLMs using Activation Steering" 2503.06040 (same sentence); "The Landscape of Memorization in LLMs" 2507.05578; ROME 2403.00510; "Optimized Tokenization for Transcribed Error Correction" 2310.10704; "Between words and characters" survey 2112.10508 ("larger subword units allow for and lead to more memorization"); stroke-sequence NMT 2211.12781; "Memorization Dynamics of Fill-in-the-Middle Pretraining" 2605.22981 (lists tokenization as a factor); Tirumala et al. 2205.10770 (grouped cite only). — [Semantic Scholar API citations](https://api.semanticscholar.org/graph/v1/paper/arXiv:2110.02782/citations?fields=title,year,externalIds,contexts&limit=200)
- Closest follow-up on vocabulary *structure*: Schäfer, Hofmann, Schlag, Pimentel, "On the Effect of (Near) Duplicate Subwords in Language Modelling" (2404.06508) cites Kharitonov's duplication control; fully duplicating the vocabulary makes LMs need ~17% more data; merging natural near-duplicates hurts. (Indirect: sample efficiency, single pass, no memorization metric.) — [arXiv](https://arxiv.org/abs/2404.06508)
- Tokenizer-level (not model-level) privacy: "Membership Inference Attacks on Tokenizers of LLMs" (2510.05699): attack success rises with tokenizer vocab 80K -> 200K, best AUC 0.771 at 200K (attack on the merge list, not on the trained model). — [arXiv](https://arxiv.org/html/2510.05699v4)

### Inferences
- Kharitonov's sequence-length mechanism predicts our ordering (bytes << 8K << 128K): at fixed text, a smaller vocab gives longer sequences, which per their controls makes memorization harder. Their duplication control also says vocab size alone (more rows, same length) does not raise memorization; our page's rare-row ablations (rare rows explain ~half the gap) partly cut against a pure length story, so the two mechanisms may coexist.
- Their MT appendix (smaller vocab ran more epochs before stopping) is weak, indirect support that small vocabularies overfit later in wall-epochs.
- Kharitonov measured training-set memorization in small models trained to convergence over many epochs; it never measured held-out loss cost of repetition, never used bytes, and is not peer-reviewed.

### Gaps
- Exact MI AUC and PAQ recovery numbers are only in figures; not extracted.
- Stopping criterion for the MT epoch counts not confirmed from text.
- No citing paper found that replicates Kharitonov at modern LM pretraining scale.

## Q2. Data-constrained scaling (Muennighoff 2023 and 2025–2026 follow-ups): tokenizer varied? which regularizers help?

### Takeaway
None of the data-constrained scaling papers varies vocabulary size; each fixes one tokenizer (GPT-2 50K, Llama-2 32K, Qwen3 151,670). Regularizers that help under repetition: strong weight decay, masked-input regularization, dropout (Xue et al.), low-entropy token dropout, code mixing, and — the only tokenization-side lever — BPE-dropout re-tokenization of repeats, which helps only at ≥8 epochs.

### Cited Findings
- Muennighoff et al. 2023, "Scaling Data-Constrained Language Models" (2305.16264): GPT-2 architecture and tokenizer, not varied. Up to 4 epochs "negligible changes to loss" (8.7B: 4 epochs ends only 0.5% higher val loss than single epoch); fitted R_D* = 15.39 ("15 repetitions (or 16 epochs)"), R_N* = 5.31; value of compute "eventually decays to zero"; mixing in code gives ~2x effective tokens; perplexity filtering helps, dedup does not on their benchmarks; no early stopping; dropout/weight decay not tested in main text. — [arXiv html](https://arxiv.org/html/2305.16264)
- Xue, Fu, Zhou, Zheng, You, "To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis" (2305.13230, NeurIPS 2023): multi-epoch degradation; key factors dataset size, model size, objective; most regularizers do not help except dropout, which is "highly effective" but needs tuning at scale. Tokenizer not varied. — [arXiv](https://arxiv.org/abs/2305.13230); [NeurIPS](https://neurips.cc/virtual/2023/poster/72533)
- Lovelace, Belardi, Kundurthy, Sudhakar, Weinberger, "Prescriptive Scaling Laws for Data Constrained Training" (2605.01640, 2 May 2026): Llama-2 tokenizer, vocab 32,000, not varied; 15M–1B params, FineWeb sample-10BT, unique budgets 50M–6B tokens, up to 16 epochs (64 in Muennighoff validation). Chinchilla + additive penalty P·R_D·(N/U_D) (also 2- and 4-param forms). Weight decay λ=0.1 vs λ=1.0: 1-param P 0.02305 -> 0.00681 (~70% smaller overfitting coefficient). Downstream metric is BPB averaged over 19 OLMES tasks; training metric is val loss. Only weight decay tested. — [arXiv abs](https://arxiv.org/abs/2605.01640); [html](https://arxiv.org/html/2605.01640v1)
- Xu, Wu, Cho, Hu, Wang, "Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws" (2606.06888, June 2026): GPT-2 tokenizer + [MASK] (50,258), not varied; no BPB. 72M–1.4B on DCLM unique budgets 100M–400M, epochs tuned (16–32 best at 100M). Tuned weight decay 0.4 (72M) to 3.2 (1.4B) vs standard 0.1; dropout fixed 0.1. Masked-input regularization (MIR) beats strong-WD AR: 100M-token val loss 3.471->3.468 (140M), 3.422->3.405 (257M), 3.367->3.333 (664M), 3.347->3.317 (1.4B); worth ~1.3x unique data. SoftQ law L = E + (A/N^ρ + B/U^(ρ/(1+α)))^(α/ρ); argues additive Chinchilla forms are misspecified under repetition. — [arXiv abs](https://arxiv.org/abs/2606.06888); [html](https://arxiv.org/html/2606.06888v2)
- Chudnovsky, Kazdan, Levi, Schaeffer, Denisov-Blanch, He, Donmez, Koyejo, Donoho, "Internal Data Repetition Destroys Language Models" (2606.24998, 23 Jun 2026): Qwen3 tokenizer, vocab 151,670, explicitly "one tokenizer" (Limitations); Qwen3-style 34M–344M, FineWeb-Edu-Dedup, AdamW wd 0.01; repeated docs = 10% of tokens. Loss peaks at an intermediate repeat count; peak location is a power law in model size; at 344M the worst repeat count matches a no-repetition run with 67% of the FLOPs (~1/3 of compute lost). Toy linear-regression reproduces it; limitation notes the toy lacks "discrete tokens". Loss in nats, no BPB. — [arXiv abs](https://arxiv.org/abs/2606.24998); [html](https://arxiv.org/html/2606.24998v1)
- Chai & Xiong, "What Is a Repeated Token Worth? The Scaling Geometry of Multi-Epoch Pretraining" (2610.05591, 4 Oct 2026): GPT-2 byte-level BPE, vocab not varied; metric bpb. Second epoch worth 0.87–1.02 of fresh token; half-value epoch R_c ≈ 2.7(D/N)^0.24 (~4–5 epochs at ≤20 TPP, ~10 at 300 TPP); excess loss ΔL = 0.0203·z^1.07 with z=(R−1)N_total/U (R²=0.96); at D/N=20, 8x repeats of a subset cost 0.032 bpb at 45M and 0.022 bpb at 1.2B; consecutive replay up to +0.46 bpb vs shuffling <0.01; source unigram entropy correlates r=−0.89/−0.91 with loss rise (GitHub worst, +0.16 bpb at 16 epochs). — [arXiv abs](https://arxiv.org/abs/2610.05591); [html](https://arxiv.org/html/2610.05591v1)
- (DIRECT on tokenization-side regularizer, NOT on vocab size) Chai & Xiong §6.3: BPE-dropout 0.05 re-segments each repeat (same bytes, same order; eval deterministic), 45M on CommonCrawl, R∈{1,2,4,8,16}. Always-on minus always-off: +0.021 bpb at R=1, +0.010 at R=4, −0.022 at R=8, −0.154 at R=16; loss rise from 4->16 epochs 0.218 -> 0.054 bpb under always-on. Switching off for final 20% wins by 0.015 at R≤4 but loses 0.038 at R=16. "Re-tokenization helps only when reuse is heavy." — [arXiv html](https://arxiv.org/html/2610.05591v1)
- Cosma, Ruseti, Radoi, Dascalu, "Training Language Models with homotokens Leads to Delayed Overfitting" (2601.02867, Jan 2026): alternative segmentations ("homotokens") of the same word via a small causal encoder; 88M/244M, FineWeb-Edu subsets repeated R∈{16,24,32,64}, GPT-2 tokenizer (~50K); delays overfitting for R>16, no change for R<16; with aya-23 tokenizer (~255K) homotokens help multilingual 32-epoch fine-tuning but not with GPT-2 tokenizer (authors: canonical GPT-2 tokens already over-fragmented). Numbers only in figures. — [arXiv html](https://arxiv.org/html/2601.02867v1)
- Wang et al., "Entropy-Guided Token Dropout" (EntroDrop, 2512.23422): multi-epoch on scarce domain data; low-entropy tokens learned fast and dominate; masking them with a curriculum beats standard regularizers, 0.6B–8B (abstract gives no numbers; no tokenizer angle). — [arXiv](https://arxiv.org/abs/2512.23422)

### Inferences
- The data-constrained literature treats the token as the unit of repetition with a fixed tokenizer; its laws (R_D*, R_c, ΔL) are therefore conditioned on the tokenizer and cannot say how vocabulary moves them.
- Chai & Xiong's BPE-dropout and homotokens results are the closest published evidence that *token-level diversity over the same bytes* lowers repetition cost, with the same threshold shape as our result (little effect at ≤4 passes, large at ≥8). Our 2.4-pass "no difference" vs 8-pass "large difference" matches their R=4 vs R=8 crossover.
- Magnitude check: Chai & Xiong's 8x-subset cost (0.032 bpb at 45M, 0.022 at 1.2B, GPT-2 50K) sits between our 64K (0.075) and 8K (0.026) values; differences in model, optimizer and data block a tight comparison.
- Chung & Kim's frequent/rare split (Q3) and Chai & Xiong's entropy correlation both suggest the cost concentrates on low-entropy, frequent-pattern data / rare rows; a larger vocab turns more frequent byte strings into single rare-ish tokens.

### Gaps
- Could not confirm whether any data-constrained paper reports a vocabulary ablation in an appendix not reached (2606.06888 html truncated at 100K chars; Muennighoff html truncated).
- Chai & Xiong Appendix L (compute overhead of re-tokenized tokens) not read.

## Q3. Vocabulary scaling laws and tokenizer-choice studies: any repetition or memorization results?

### Takeaway
Vocabulary-scaling and tokenizer-choice papers are single-pass (or confound passes with vocab) and do not measure memorization. One indirect hint: Tao et al. find the optimal vocab shrinks when data is the bottleneck (16K -> 10K at 302M) and that a smaller vocab wins in their "insufficient data" setting.

### Cited Findings
- Tao et al., "Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies" (2407.13623, NeurIPS 2024): 33M–3B, up to ~500B characters; normalizes loss across vocab with unigram-normalized loss L_u (correlation with BPC ρ=0.9888). Optimal vocab grows with compute (Llama2-70B should be ≥216K). Table 3 (N_nv=2.87B): "insufficient data/undertraining" 2.8e20 FLOPs, V=24K beats 32K (avg 43.9 vs 43.2); overtraining 2.3e21 FLOPs, V=43K beats 32K (51.6 vs 50.3; ARC-C 29.1 -> 32.0). Fig. 7 (N_nv=302M): when data is the bottleneck the optimal vocab shrinks 16K -> 10K, described as "a mechanism to prevent over-fitting"; rare-token under-fitting risk rises with vocab size when data is limited. "Insufficient data" is a smaller token budget, NOT explicitly repeated epochs; no memorization analysis. — [arXiv html](https://arxiv.org/html/2407.13623v3); [NeurIPS](https://neurips.cc/virtual/2024/poster/93395)
- Huang et al. (ByteDance Seed), "Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling" (2501.16975, ICML 2025 poster): input n-gram vocab up to 12.8M (OLMo2 151M–1B, 400B–1T tokens; OLMoE 1.3B/7B, 500B); log-linear input-vocab vs loss; 5.7x training-loss speedup at 1B. Single pass except the synthetic CFG (fixed 20M sentences, 10 epochs, no overfitting analysis); notes larger output vocab can hurt "large models prone to overfitting" and large input vocab "requires longer training". No repeated-data study. — [arXiv html](https://arxiv.org/html/2501.16975); [ICML](https://icml.cc/virtual/2025/poster/44467)
- Takase, Ri, Kiyono, Kato, "Large Vocabulary Size Improves Large Language Models" (2406.16508; ACL 2025 Findings): unigram SentencePiece 5K/10K/50K/100K/500K, ~680M non-embedding, SlimPajama (En) / CC (Ja). Two conditions: fixed 1T tokens (500K model then sees ~1.5 epochs En, ~2 epochs Ja) and fixed 1 epoch (En 830B tokens at 5K vs 640B at 500K). Avg score En 1T: 47.8 (5K) -> 50.3 (500K); En 1-epoch: 47.9 -> 49.4; Ja 1T: 49.4 -> 58.6; Ja 1-epoch: 49.8 -> 56.2. Authors flag that fixed-token comparisons favor large vocab because it "has an advantage of seeing more data"; no val loss, overfitting or memorization measured. Counter-evidence direction only at ≤2 epochs. — [arXiv html](https://arxiv.org/html/2406.16508); [ACL Anthology](https://aclanthology.org/2025.findings-acl.57.pdf)
- Chung & Kim, "Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training" (2508.15390, NeurIPS 2025): vocab 24K–196K at fixed data/compute; gains come from ~2,500 most frequent words while rare-word loss rises; larger vocab mostly increases frequency imbalance. No repetition/overfitting. — [arXiv](https://arxiv.org/abs/2508.15390)
- Ali et al., "Tokenizer Choice For LLM Training: Negligible or Crucial?" (2310.08754): 24 LLMs at 2.6B; English-centric tokenizers cost up to 68% extra training on multilingual data; no epochs/memorization. — [arXiv](https://arxiv.org/abs/2310.08754)
- Dagan et al., "Getting the most out of your tokenizer for pre-training and domain adaptation" (2402.01035): code models; vocab size has little effect on coding performance (per search summary); no memorization/repetition found. — [arXiv html](https://arxiv.org/html/2402.01035v2)
- Schmidt et al., "Tokenization Is More Than Compression" (2402.18376, EMNLP 2024): 64 LMs 350M–2.4B, PathPiece vs BPE; no epochs/memorization in abstract. — [arXiv](https://arxiv.org/abs/2402.18376)
- Goldman et al., "Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance" (2403.06265, EMNLP 2024 Findings): tokenizer trained on 1M docs down to character-level; compression correlates with downstream performance, more for generation and smaller models; no epochs/memorization. — [arXiv](https://arxiv.org/abs/2403.06265)
- Koutsiaris, "Nested Byte-Level Vocabularies ... A Pre-Registered Negative Result" (2608.28151): 30 tiny models (3.1M/10.6M bodies, 200M tokens), BPB; shared multi-cap model trails specialists 3.64% BPB at 32K, 2.96% at 8K; no repetition/memorization. — [arXiv](https://arxiv.org/abs/2608.28151)
- BabyLM (data-limited, often multi-epoch) tokenizer papers vary vocab (e.g. 4K vs 8K LSTMs; 52K -> 80K bilingual) but none studies epoch count × vocab. — [BabyLM 2025 proc.](https://aclanthology.org/2025.babylm-main.16.pdf); [Bilingual BabyLM](https://arxiv.org/html/2603.29552v2)

### Inferences
- Tao et al.'s Fig. 7 / Table 3 is the strongest published (indirect) support: in a data-bottlenecked regime the loss-optimal vocabulary is smaller and the authors attribute it to over-fitting. It is per-token-budget, not per-byte repetition, and it is not a memorization measurement.
- Takase's design is the closest to "fixed text" (1-epoch condition) but goes the other way (bigger vocab wins) at ≤2 epochs, consistent with our finding that vocab barely matters at 2.4 passes.
- Chung & Kim's rare-word degradation with larger vocab matches our page's rare-row finding mechanistically.

### Gaps
- Did not read full texts of Ali/Dagan/Schmidt/Goldman; claims of "no repetition/memorization result" rest on abstracts and search summaries.

## Q4. Memorization studies (Carlini 2023, Tirumala 2022, Biderman 2023, Positional fragility): do they vary tokenization?

### Takeaway
No. All use one fixed tokenizer per model family; only Tirumala varies dictionary size incidentally (added identifier tokens) and only to control for it.

### Cited Findings
- Carlini et al., "Quantifying Memorization Across Neural Language Models" (2202.07646; ICLR 2023): three log-linear relations — model capacity, duplication count, prompt context length (in tokens). No tokenization variation. — [arXiv](https://arxiv.org/abs/2202.07646)
- Tirumala et al., "Memorization Without Overfitting" (2205.10770, NeurIPS 2022): 125M–13B; larger models memorize faster, including before overfitting (first epoch where val perplexity rises); forgetting baseline rises with scale and with repetition frequency (~1e-2), spaced repetition ~1e-3 effect. Tokenizer not varied; dictionary-size increase only as a control in the unique-identifier experiment (§4.3); Kharitonov cited only in a group. — [arXiv html](https://arxiv.org/html/2205.10770)
- Biderman et al., "Emergent and Predictable Memorization in LLMs" (2304.11158): Pythia suite, single tokenizer; predicting memorization from smaller runs. — [arXiv](https://arxiv.org/abs/2304.11158)
- Xu et al., "Positional Fragility in LLMs" (2505.13171, NeurIPS 2025): 1B/3B/8B on 83B tokens; verbatim recall strongest for short prefixes at context start and drops sharply when the prefix is offset; no tokenizer variation found in abstract/search. Apertus (2509.14233) with Goldfish loss reports no memorization at ≤128 exposures. — [arXiv](https://arxiv.org/abs/2505.13171v2); [Apertus](https://arxiv.org/pdf/2509.14233)

### Inferences
- Carlini's context-length factor is in tokens; at fixed bytes a smaller vocab gives more tokens per prompt but also more tokens to reproduce, so comparisons across tokenizers need byte-normalized extraction metrics that these papers do not define.

### Gaps
- Did not read Biderman/Positional Fragility full text for any tokenizer appendix.

## Q5. Byte-normalized (bpb) cost of repetition across tokenizers at fixed text; per-token vs per-byte compute

### Takeaway
I found no paper that measures bits-per-byte cost of repetition across tokenizers at fixed text. Bpb is standard for cross-tokenizer comparison (SpaceByte, BLT, Lovelace et al. downstream, Chai & Xiong), and FLOP-per-byte matching is discussed in byte-model work, but none combines it with repeated data.

### Cited Findings
- SpaceByte (2404.14408, NeurIPS 2024) compares byte and subword models at fixed compute in bpb and uses FLOPs-per-byte as the cost proxy; no repetition. — [arXiv](https://arxiv.org/pdf/2404.14408)
- Byte Latent Transformer (2412.09871, ACL 2025): FLOP-controlled byte-level scaling to 8B / 4T bytes in bpb; argues perplexity is only meaningful for a fixed tokenizer; no repetition. — [arXiv](https://arxiv.org/pdf/2412.09871)
- Lee et al., "The Efficiency Gap in Byte Modeling" (2605.12928): compute-matched byte vs BPE (AR and masked diffusion); byte overhead worse for MDM ("context fragility"); abstract has no data-repetition claims. — [arXiv](https://arxiv.org/abs/2605.12928)
- Gigant, Peng, Quesnelle, "Decoupling the Benefits of Subword Tokenization ... via Byte-level Simulation" (2604.27263): subword benefits come from higher training throughput (samples per compute) and boundary priors; no repetition or memorization. — [arXiv](https://arxiv.org/abs/2604.27263)
- Chai & Xiong report all repetition costs in bpb but with a single GPT-2 tokenizer; their BPE-dropout arm produces more tokens per byte and they report the compute overhead (Appendix L, not read). — [arXiv html](https://arxiv.org/html/2610.05591v1)
- Takase et al. explicitly separate "same tokens" vs "same epoch (text)" and note that token-matched comparisons give large vocabs more passes. — [arXiv html](https://arxiv.org/html/2406.16508)

### Inferences
- The per-token vs per-byte compute confound in our design (8K 1.35x and bytes 4.74x more tokens/steps per byte) is unaddressed in published repetition work. Gigant et al.'s throughput explanation implies the opposite confound direction for unique-data quality (more bytes per FLOP helps large vocabs), which makes a fixed-text, fixed-FLOP control the natural next experiment.

### Gaps
- No source found; this cell of the literature appears empty as of 2026-10-22.

## Q6. Does any work directly test "smaller vocabularies lose less to multi-epoch repetition at fixed text"?

### Takeaway
No published or preprint work found that directly tests it. Closest evidence: (a) Kharitonov 2021 (larger vocab memorizes more; training-set metrics only); (b) Tao et al. 2024 (optimal vocab shrinks in data-bottlenecked regime, framed as preventing over-fitting); (c) Chai & Xiong 2026 and homotokens 2026 (stochastic re-segmentation of the same bytes lowers repetition cost only at heavy reuse, ≥8 / >16 repeats); (d) Takase 2025 (fixed-epoch condition, bigger vocab still wins at 1 epoch, no repetition test). The only direct test located is our own page.

### Cited Findings
- Our own page "How Vocabulary Size Shapes Overfitting" (MuonH Qwen3, Marin speedrun, 128K Marin tokenizer truncated to 64K/32K/16K/8K plus 256 bytes; fineweb-edu-10B; 2.4 and 8 passes; Paloma bpb) is the only search hit that directly tests the claim; it is our work, not independent evidence. — [wenhaochai.com](https://wenhaochai.com/exps/vocab-overfitting.html)
- A general web search for this question surfaced only the homotokens paper, Xue et al., Muennighoff, and tokenizer explainers; none tests vocab × epochs. — [homotokens](https://arxiv.org/html/2601.02867v1); [Xue et al.](https://openreview.net/pdf?id=Af5GvIj3T5)

### Inferences
- Our result appears novel as a direct, byte-normalized, fixed-text measurement. Prior work predicts its direction from three angles: shorter sequences ease memorization (Kharitonov), rare large-vocab rows under-train/overfit (Tao; Chung & Kim), and diverse segmentations of the same bytes act as augmentation under heavy reuse (Chai & Xiong; Cosma et al.).
- Expected reviewer objections, from this literature: compute per byte (smaller vocab gets more steps), parameter count (untied embeddings shrink with vocab), and unique-data quality (Tao/Takase say large vocab wins with fresh data).

### Gaps
- Semantic Scholar citation lists of Tao 2024, Muennighoff 2023 and Chai & Xiong 2026 not swept for a vocab × repetition paper; a hidden 2026 preprint remains possible.
