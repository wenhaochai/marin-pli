# Remedies for multi-epoch / data-constrained degradation in LM pretraining (general methods)

Scope note for the report writer. The target regime is a 300M decoder-only LM (Marin speedrun MuonH Qwen3 recipe), about 6B training tokens (~20 tokens/param), and 8 passes over a random 1/8 subset (about 750M unique tokens, about 2.5 unique tokens/param). This is a *mild* repetition regime. Most 2025-2026 "data-constrained" papers study a far more extreme regime (100M-200M unique tokens for 150M-1.4B models, i.e. 0.1-1 unique tokens/param, 16-800 epochs). Optimal hyperparameters from that regime (weight decay 16-32x standard, dropout, 16+ epochs) should not be copied at face value. The closest-regime paper found is Min 2026 (arXiv 2610.15965: 1B-token pool, 153M-1.5B models).

Local code fact (checked in `lib/levanter/src/levanter/optim/muonh.py`, 2026-10-21): MuonH keeps every linear weight matrix at exactly its initial Frobenius norm (update, then renormalize); `lm_head` uses AdamH (also norm-preserving); embeddings and vector parameters use plain `optax.scale_by_adam` with **no weight decay term at all**. The docstring says the MuonH learning rate corresponds to `sqrt(lr * weight_decay)` of a Muon+WD configuration. Consequence: the "large weight decay" lever from the literature has no direct knob in this recipe for linear layers, and the input embedding table is completely unregularized.

## Q1. Scaling laws for repeated data: what epoch count is "free", and which remedies did the classic papers test?

### Takeaway
Up to ~4 epochs is near-free and value decays with a fitted half-life of ~15 repetitions (Muennighoff 2023, AR, up to 8.7B); a later fit at more extreme repetition gives ~31 for AR. At 8 passes the Muennighoff law predicts ~83% of the value of fresh data, so a small but non-zero gap is expected even with a perfect recipe. Of classic regularizers, only dropout clearly helped (Xue 2023); weight decay was unstable there, label smoothing slight.

### Cited Findings
- Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel, "Scaling Data-Constrained Language Models", 2023, arXiv 2305.16264. "Training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data." Models up to 8.7B params, 900B total tokens. — [arXiv 2305.16264](https://arxiv.org/html/2305.16264)
- Same paper: 8.7B model on 178B tokens, built from 44B unique tokens repeated 4 epochs, "finishes training with only 0.5% higher validation loss" than all-unique. — [arXiv 2305.16264](https://arxiv.org/html/2305.16264)
- Same paper: fitted R_D* ≈ 15.39 (≈ 15 repetitions / 16 epochs); "meaningful gains from repeating data can be made up to around 16 epochs", then returns diminish very fast; 44-epoch runs can show rising loss mid-training. — [arXiv 2305.16264](https://arxiv.org/html/2305.16264)
- Same paper, remedies tested: mixing in code gives "a 2× increase in effective tokens even when evaluating only natural language tasks"; up to 50% code showed no deterioration, beyond that NL performance falls quickly. Perplexity filtering (keep lowest-25% perplexity, ~2 epochs) helped; deduplication (100-char overlaps, 4 epochs) "does not help". The portion of the paper I read did not report dropout/weight-decay ablations. — [arXiv 2305.16264](https://arxiv.org/html/2305.16264)
- Same paper, allocation: in the data-constrained regime allocate extra compute "to more epochs rather than more parameters"; a model with 27% fewer params under their allocation beat the Chinchilla-suggested one. — [arXiv 2305.16264](https://arxiv.org/html/2305.16264)
- Hernandez et al. (Anthropic), "Scaling Laws and Interpretability of Learning from Repeated Data", 2022, arXiv 2205.10487: repeating 0.1% of the data 100 times (90% of tokens still unique) degrades an 800M model to the performance of a ~400M model (half the size). This is about a *small repeated slice*, not uniform epoching. — [arXiv 2205.10487](https://arxiv.org/abs/2205.10487)
- Xue, Fu, Zhou, Zheng, You, "To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis", NeurIPS 2023, arXiv 2305.13230. Setup: T5-style encoder-decoder on C4, 78M-2.8B params, 2^27 tokens repeated up to 256 epochs. Main degradation example: T5-Base, 2^27 tokens × 256 epochs vs 2^35 tokens × 1 epoch, C4 val accuracy 64.6 → 61.7 (−2.9). — [arXiv 2305.13230](https://arxiv.org/html/2305.13230)
- Xue et al.: dataset size, parameter count and objective matter; data quality and FLOPs matter less (Wikipedia subset degraded similarly). Larger models overfit more: T5-XL on 2^27 tokens was worse than T5-Large on 2^29 tokens. Parameter-shared models (fewer params, same FLOPs) degraded less; MoE (more params) degraded more. — [arXiv 2305.13230](https://arxiv.org/html/2305.13230)
- Xue et al., regularizers: "Dropout alone was the most effective"; stacking DropPath, label smoothing or weight decay gave no further gain; weight decay caused instability (NaN) in their setup; label smoothing gave a slight improvement. Dropout 0.1 worked at Base/Large; at XL a slight late-training drop remained; MoE/XL sweeps peaked at dropout 0.2-0.3. — [arXiv 2305.13230](https://arxiv.org/html/2305.13230)
- Xue et al., delayed dropout: dropout off for the first ~32K steps (of 2^19), then 0.1 — final performance comparable to dropout-from-start, and better early in training. — [arXiv 2305.13230](https://arxiv.org/html/2305.13230)
- Prabhudesai et al. 2025 (arXiv 2507.15857) re-fit the AR half-life in a more extreme regime (7M-2.5B params, 25M-100M unique tokens, up to 800 epochs): R_D* ≈ 31-32 for AR (vs ≈ 15 in Muennighoff) and ≈ 494-513 for masked diffusion. — [arXiv 2507.15857](https://arxiv.org/html/2507.15857)

### Inferences
- Muennighoff's effective-data formula D' = U·(1 + R*·(1 − e^(−R/R*))) with R = 7 extra passes and R* = 15.4 gives D' ≈ 6.63U, i.e. ~83% of 8U fresh tokens. Plugging that into a Chinchilla-style data term (B = 410.7, β = 0.28, Hoffmann approach-3 constants) at D = 6B gives ≈ 0.04 nats/token ≈ 0.013 bpb (assuming ~4.3 bytes/token). This is a rough estimate with borrowed constants, but it suggests the observed 0.026-0.082 bpb penalty is 2-6× larger than "inherent" repetition cost, i.e. a large part is recipe-specific overfitting that regularization/data-order fixes could remove. It also suggests that matching fresh data *exactly* may be impossible in principle at 8 passes; the realistic goal is shrinking the gap toward ~0.01 bpb.
- Xue's "more parameters overfit more" plus the observation that the penalty grows with vocabulary are consistent with the embedding table (whose size scales with vocab, and which has no WD in MuonH) being a major memorization site.

### Gaps
- I did not read the remaining half of Muennighoff 2023 (appendix); I could not confirm whether it contains dropout/weight-decay ablations.
- Galactica's (Taylor et al. 2022) ~4 epochs without degradation is often cited; not verified in this session.

## Q2. Regularization in data-constrained pretraining: weight decay, dropout (incl. early/late), label smoothing, z-loss, input noise

### Takeaway
In AdamW settings, much larger weight decay (0.8-3.2, i.e. 8-32× the 0.1 default) is the single strongest tuned lever in the extreme regime (Kim 2025; Xu 2026). Dropout helps in multi-epoch training but hurts single-epoch training, so "late" (delayed) dropout is the principled form. Input-token noise is contested: random replacement gave large gains at 150M/100 epochs (Chen 2026), but masking/attention dropout gave none at 140M/50 epochs (Prabhudesai 2025). A masked-copy auxiliary loss (MIR) is worth ~1.3× data but doubles forward cost.

### Cited Findings
- Kim, Kotha, Liang, Hashimoto, "Pre-training under infinite compute", 2025 (ICLR 2026 per search snippet), arXiv 2509.14786. 200M DCLM tokens, 150M/300M/600M/1.4B Llama-style models, AdamW, cosine. Untuned (WD 0.1) recipe: best epochs 8/8/4/4 and loss rises with params (1.4B worse than 600M). Tuned WD: 0.8 (150M), 1.6 (300M), 3.2 (600M, 1.4B); tuned epochs 16/16/8/8; LRs 3e-3/3e-3/1e-3/1e-3. With this, loss follows a power law in params with asymptote 3.43 (unregularized asymptote 3.75). — [arXiv 2509.14786](https://arxiv.org/html/2509.14786)
- Kim et al.: the regularized recipe alone is 2.29× more data efficient than the standard recipe at 200M tokens; the full recipe (regularization + params + ensembles) 5.17×. The authors call their asymptote estimation "unreliable" in places. Dropout was not tested. — [arXiv 2509.14786](https://arxiv.org/html/2509.14786)
- Xu, Wu, Cho, Hu, Wang, "Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws", June 2026, arXiv 2606.06888. 72M-1.4B models, 100M-400M unique DCLM tokens, AdamW + WSD, dropout 0.1. Strong-WD settings at 100M tokens: 257M → 16 epochs, WD 1.6, LR 1e-3; 1.4B → 16 epochs, WD 3.2. At 257M/100M tokens: multi-epoch AR with WD 0.1 = 3.880 vs strong WD = 3.422 (−0.46 nats). — [arXiv 2606.06888](https://arxiv.org/html/2606.06888v2)
- Xu et al., MIR (masked-input regularization): loss = NTP(x) + λ·NTP(x̃), x̃ = copy of x with each token replaced by an extra [MASK] token with probability r ~ U(r_min, r_max); no architecture change; two forward passes per step. On top of strong WD: 257M 3.422 → 3.405 (−0.017); 664M 3.367 → 3.333 (−0.034); 1.4B 3.347 → 3.317 (−0.030); gain ~0.006 at 72M. Worth ~1.28-1.34× more unique data at 200M-400M tokens. 1.4B BoolQ 45.1 → 55.3. — [arXiv 2606.06888](https://arxiv.org/html/2606.06888v2)
- Chen, Zhang, Wang, "Data Augmentations for Data-Constrained Language Model Pretraining", June 2026, arXiv 2606.16246. 150M Llama, 75M DCLM tokens, AdamW (WD 0.033), WSD, 100 epochs. Baseline min val loss 4.015 at epoch 16, then rises. Random token replacement 15%: 3.841 (−0.174) at epoch 28; replacement 30%: −0.150; masking 15%: −0.105; right-to-left 50%: −0.105; offset (predict token i+k, k≤5, exponential): −0.145; FIM 50%: −0.068. Best combo (random 5% + R2L 50% + offset): 3.805 (−0.210); after WSD decay 3.792 vs 4.000. Downstream mean 41.0% → 42.9%. Token noise + offset interfere (3.995). No fresh-data reference, no dropout/WD interaction study, single model size. — [arXiv 2606.16246](https://arxiv.org/html/2606.16246v1)
- Contradiction: Prabhudesai et al. 2025 tested AR with attention dropout (25-75%) and token masking at 140M params / 50 epochs: "neither approach improved validation loss"; AR still overfit. — [arXiv 2507.15857](https://arxiv.org/html/2507.15857)
- Ni et al., "Diffusion Language Models are Super Data Learners", Nov 2025, arXiv 2511.03276: at 1B params, 1B unique tokens, 96 epochs, AR with input masking 10-90%: modest masking (best 10%) helped but "falls well short of diffusion"; high masking collapses. Dropout 10-90% raised AR performance in data-constrained settings but did not remove the diffusion advantage (numbers only in figures). — [arXiv 2511.03276](https://arxiv.org/html/2511.03276v1)
- Wang et al., "Entropy-Guided Token Dropout" (EntroDrop), Dec 2025, arXiv 2512.23422: masks low-entropy tokens with a curriculum on strength during multi-epoch domain adaptation, 0.6B-8B; abstract reports it beats standard regularization baselines but gives no numbers. Domain-adaptation setting, not pretraining from scratch. — [arXiv 2512.23422](https://arxiv.org/abs/2512.23422)
- Liu, Bauer, Manning, "Drop Dropout on Single-Epoch Language Model Pretraining", 2025, arXiv 2505.24788 (ACL Findings): Pythia 160M/1.4B and BERT, single epoch: no dropout beats dropout on LM, BLiMP, SQuAD, MNLI; early dropout was worse than none. — [arXiv 2505.24788](https://arxiv.org/abs/2505.24788)
- Liu, Xu, Jin, Shen, Darrell, "Dropout Reduces Underfitting", ICML 2023, arXiv 2303.01500 (vision): early dropout helps underfitting models; *late dropout* (off early, on later) is proposed for overfitting models; both improve ImageNet generalization. — [arXiv 2303.01500](https://arxiv.org/abs/2303.01500)
- Slowrun (Q Labs) baselines all use heavy regularization with Muon: Limited track 2.7B model, dropout 0.1, weight decay 1.6; Tiny track 300M model, WD 0.8, dropout 0.1; 100M FineWeb tokens. A record "weight decay schedule" gave −0.009 (tiny: 3.374 → 3.365; 2-hour: 3.188 → 3.150 together with LR schedule change). Stochastic depth −0.003. — [qlabs-eng/slowrun](https://github.com/qlabs-eng/slowrun)
- Min, "Scaling Repeated-Data Pretraining: Batch Size, Regularization, and GPU-Time Tradeoffs", Oct 2026, arXiv 2610.15965 (closest regime: 1B unique tokens, 153M-1.5B): WD swept {0.011…3.0}; WD matters most at large batch; at batch 64 sequences WD 0.3 stayed optimal. Tied embeddings alone gave little; tied + positive weight decay on the embedding gave the best result (1.5B: 2.971 → 2.946 at B=64; 3.064 → 3.030 at B=512). — [arXiv 2610.15965](https://arxiv.org/html/2610.15965)
- Xue et al. 2023: label smoothing slight improvement; weight decay unstable in their T5 setup. — [arXiv 2305.13230](https://arxiv.org/html/2305.13230)

### Inferences
- MuonH has no WD knob on linear layers. In a scale-invariant (pre-norm, QK-norm) network, the main effect of large decoupled WD is a larger equilibrium angular learning rate (∝ sqrt(lr·wd)) and smaller weight norms. The MuonH analog of "16× WD" is therefore roughly a larger MuonH LR and/or a smaller fixed weight-norm radius, not a new term. This is my inference from the docstring and standard equilibrium arguments; it is untested here.
- The input embedding table (Adam, zero WD) is the one place where classical WD applies directly in this recipe, and Min 2026 found embedding decay was the best recipe add-on. Because the embedding table grows with vocabulary, embedding regularization (WD, embedding dropout, token replacement) is the most plausible fix for the *vocab-dependent* part of the penalty.
- Dropout evidence fits a "late" schedule: it hurts in single-epoch training and helps once data repeat. The queued "residual dropout from pass 2" matches Xue's delayed dropout and Liu 2023's late dropout. At our mild regime, rates 0.1 are more plausible than 0.2 (Xue's 0.2-0.3 optima were at 256 epochs).
- Token-noise results conflict across papers (strong gain at 100 epochs/75M tokens; none at 50 epochs/140M; modest at 96 epochs/1B). Effects measured at 30-100 epochs probably shrink at 8 passes.

### Gaps
- No source found on z-loss as a remedy for repetition.
- No study of input-embedding dropout specifically in multi-epoch LM pretraining was found.
- No source tested Muon/MuonH's effective-LR or weight-norm radius as a regularizer under repetition.
- MIR's r_min, r_max and λ were in a truncated appendix; not extracted.

## Q3. Ensembling, distillation, self-distillation, weight averaging (EMA/SWA)

### Takeaway
Ensembling gives the largest single gains in the extreme regime, and distilling the ensemble keeps most of it (Kim 2025; Slowrun Unlimited track; q0). These cost several times the compute, so they do not fit a same-compute goal. Weight averaging (EMA, SWA, averaging last-epoch checkpoints) is near-free and gave 0.004-0.018 nats per record in Slowrun.

### Cited Findings
- Kim et al. 2025: regularized 300M = 3.57; 300M-member ensemble asymptote 3.34 (3.27 with large-K-tuned HPs); even K=3 beats the regularized asymptote; joint param+ensemble asymptote 3.17. 8×300M ensemble (3.32) distilled into one 300M student gives 3.36, retaining 83% of the ensemble gain and beating the regularized asymptote (3.43). Self-distillation 300M → 300M with real+synthetic token mix: student "vastly outperforms its teacher", matching ~3.43. Downstream: best ensemble +9% avg (PIQA/SciQ/ARC-E) over best unregularized; distilled +7%. CPT: 4B tokens with an 8-member ensemble beat default CPT on 73B tokens (17.5×). — [arXiv 2509.14786](https://arxiv.org/html/2509.14786)
- Slowrun Unlimited track: single 2.7B 3.402 → 8-model logit-averaged ensemble 3.264 (−0.138); members distilling the previous model + CE: 3.166 → 3.126 (−0.040); probability averaging instead of logit averaging, 20 models: 3.024; snapshot ensembles: 2.987. — [qlabs-eng/slowrun](https://github.com/qlabs-eng/slowrun)
- Slowrun Limited track (single model, 1 h on 8×H100): EMA of weights + HP tuning 3.270 → 3.252 (−0.018); weighted average of last 3 epoch checkpoints 3.252 → 3.248 (−0.004); SWA 3.248 → 3.236 (−0.012). Tiny (300M) track: EMA 3.365 → 3.353 (−0.012); SWA 3.353 → 3.345 (−0.008). — [qlabs-eng/slowrun](https://github.com/qlabs-eng/slowrun)
- Mandal, Berman, Vegesna, Dahal, "q0: Primitives for Hyper-Epoch Pretraining", June 2026, arXiv 2606.03938 (Slowrun setup: 1.8B, 100M FineWeb tokens): cyclic schedule with anti-correlated LR and WD to collect diverse models, chain distillation (each model distills its predecessor), and a learned prior to weight members; matches a 256-epoch ensemble baseline with ~56 epochs (4.6× fewer); cumulative data efficiency ~12.9×. — [arXiv 2606.03938](https://arxiv.org/abs/2606.03938)
- Secondary write-ups of the Q Labs March 2026 post report ~10× data efficiency (ensemble of 1.8B models on 100M tokens matching a ~1B-token baseline) and earlier 2.4×-5.5×; I could not open the primary post. — [ubos.tech summary](https://ubos.tech/?p=68304); [sesamedisk summary](https://sesamedisk.com/nanogpt-slowrun-data-efficient-modeling/)

### Inferences
- Under a strict same-compute rule, ensembling/distillation are out of scope, except as an upper-bound reference. EMA/SWA over the last 1-2 passes costs almost nothing and is tokenizer-agnostic; expected effect at 300M/8 passes is likely in the 0.005-0.015 nats range (Slowrun deltas were at 100M tokens, far more epochs).
- Note: Slowrun EMA/SWA gains were measured on top of a WSD-like decay; with a cosine schedule that already anneals to ~0, averaging may give less.

### Gaps
- No paper found that measures EMA/SWA gain as a function of epoch count against a fresh-data control.

## Q4. Data-side remedies: rephrasing/synthetic data, token-level augmentation, ordering, fresh-data mixing

### Takeaway
Free data-order fixes have solid evidence: reshuffling every epoch and *repacking documents into new sequence boundaries* each epoch gave 0.03-0.14 nats at 1.5B/1B tokens (Min 2026) and 0.026 nats in Slowrun. Rephrasing (WRAP) and code mixing add effective data but change the training distribution or need a generator model.

### Cited Findings
- Min 2026 (1.5B, 1B-token pool, ~20+ passes): val CE by loader — fixed order 3.011 (B=64) / 3.205 (B=512); shuffle 2.977 / 3.124; repack documents into new sequences each epoch 2.971 / 3.064; repack + tied + embedding decay 2.946 / 3.030. The repetition-aware recipe also moved the best epoch count up (1.5B: 20 → 32; 474M: 32 → 80; 153M: 112 → 240). — [arXiv 2610.15965](https://arxiv.org/html/2610.15965)
- Slowrun: "shuffle data every epoch" was record #2, 3.402 → 3.376 (−0.026); "document-level shuffling" 3.211 → 3.204 (−0.007) in Limited, 3.345 → 3.332 (−0.013) in Tiny; per-epoch document-level shuffling −0.005 in the 2-hour track. — [qlabs-eng/slowrun](https://github.com/qlabs-eng/slowrun)
- Chen et al. 2026: token-level augmentations (random replacement, masking, R2L, offset prediction) — numbers in Q2. — [arXiv 2606.16246](https://arxiv.org/html/2606.16246v1)
- Maini, Seto, Bai, Grangier, Zhang, Jaitly, "Rephrasing the Web" (WRAP), 2024, arXiv 2401.16380: rephrasing C4 with an instruction model speeds pretraining ~3×; >10% average perplexity improvement on Pile subsets at equal compute; >2% zero-shot QA gain over 13 tasks. — [arXiv 2401.16380](https://arxiv.org/abs/2401.16380)
- Muennighoff 2023: up to 50% code in the mix doubles effective tokens with no NL loss; perplexity filtering helps for noisy data; dedup did not help. — [arXiv 2305.16264](https://arxiv.org/html/2305.16264)
- Kim et al. 2025: self-distillation with generated (synthetic) tokens mixed with real tokens lets a same-size student beat its teacher. — [arXiv 2509.14786](https://arxiv.org/html/2509.14786)

### Inferences
- First action item: check whether the current Levanter/Marin loader reuses the same permutation and the same sequence packing in every pass. If packing boundaries are fixed, repacking per pass is free and Min's data show it beats plain reshuffling, most of all at large batch.
- WRAP/code mixing change what "held-out Paloma bpb at same compute" means (different data distribution, generator compute), so they fit the goal poorly.

### Gaps
- Synthetic Continued Pretraining (EntiGraph), REWIRE, BeyondWeb, Synthetic Bootstrapped Pretraining, D4 and Charton & Kempe "repeated examples" were not verified in this session.
- No evidence found on curricula (e.g., easy→hard, or unique-first) specifically for multi-epoch pretraining.

## Q5. Objective-side remedies: diffusion LMs, any-order, multi-token prediction, masking objectives

### Takeaway
Masked diffusion LMs beat AR only after many epochs (hundreds) and with weak AR baselines; at 8 passes AR is better, and strong weight decay largely erases the diffusion advantage. Cheap AR-compatible pieces (random orderings, offset/multi-token targets, masked-copy auxiliary loss) give partial gains.

### Cited Findings
- Prabhudesai, Wu, Zadeh, Fragkiadaki, Pathak, "Diffusion Beats Autoregressive in Data-Constrained Settings", 2025, arXiv 2507.15857: at 100M unique tokens, best AR 3.71 (at ~50 epochs) vs best diffusion 3.55 (at 500 epochs); at single-epoch compute-optimal settings diffusion is far worse (10.65 vs 7.07). Critical compute where diffusion overtakes AR: C_crit(U) = 2.12×10^1.956 · U^2.174. AR trained on N random token orderings improved with N; at N=16 (278M, 100 epochs) AR loss approached diffusion. Attention dropout and token masking did not help AR. — [arXiv 2507.15857](https://arxiv.org/html/2507.15857)
- Ni et al. 2025, arXiv 2511.03276: diffusion advantage attributed to any-order modeling, iterative-denoising compute and Monte-Carlo augmentation (no quantitative split). A 1B DLM trained 480 epochs on 1B tokens: HellaSwag 56% / MMLU 33% vs AR 41% / 29%. Crossover moves later with more unique data and earlier with bigger models. AR validation loss rising while downstream accuracy still improves is reported (1B AR, 1.5B unique tokens, 64 epochs). The paper has an internal inconsistency (1B vs 11B tokens for the 480-epoch run). — [arXiv 2511.03276](https://arxiv.org/html/2511.03276v1)
- Xu et al. 2026: with WD 0.1, dLLM (3.603) beats multi-epoch AR (3.880) at 257M/100M tokens; with strong WD, AR 3.422 vs dLLM 3.484, "comparable across sizes"; dLLM loss is an ELBO bound, not directly comparable. — [arXiv 2606.06888](https://arxiv.org/html/2606.06888v2)
- Slowrun: multi-token prediction loss record 3.227 → 3.222 (−0.005). — [qlabs-eng/slowrun](https://github.com/qlabs-eng/slowrun)
- Chen et al. 2026: offset (predict token i+k) −0.145, R2L 50% −0.105 at 150M/75M tokens/100 epochs. — [arXiv 2606.16246](https://arxiv.org/html/2606.16246v1)

### Inferences
- At 8 passes and ~2.5 unique tokens/param, we are far below C_crit, so switching objectives is not worth testing for this goal. Part of the reported diffusion advantage came from comparing against under-regularized AR.
- Ni et al.'s observation (val NLL rises while downstream ranks still improve) is a reminder that the Paloma bpb gap may overstate the capability gap, but the stated goal is bpb, so this does not change the target.

### Gaps
- No evidence at ≤10 epochs for any-order or MTP objectives as repetition remedies.

## Q6. Optimizer, schedule and batch size under repetition

### Takeaway
Batch size is an under-appreciated lever: with repeated data the optimal batch is 3-5× smaller than single-pass laws predict (Min 2026). No paper directly measures whether Muon/Shampoo/SOAP change the multi-epoch penalty; Slowrun uses Muon plus very large WD, and Muon variants (MuonEq-R) gave small gains there.

### Cited Findings
- Min 2026: U-shaped batch-size curve with an interior optimum near 64 sequences × 4096 tokens (~0.26M tokens) for 153M-1.5B; B=64 vs B=512 val CE: 1.5B 3.124 → 2.977, 474M 3.136 → 3.093, 153M 3.416 → 3.332. For 1.5B at 20 passes, single-pass batch laws predicted 1.23M/0.75M tokens, 2.9-4.7× too large. Claimed: batch-tuned repeated training "matches roughly 8× more unique data" (as summarized; the exact reference baseline was not extracted). AdamW + WSD. — [arXiv 2610.15965](https://arxiv.org/html/2610.15965)
- Slowrun: "MuonEq-R" record −0.003 (Limited), −0.009 (2-hour); "LR and warmdown tuning" −0.006; "weight decay schedule; LR schedule adjustment" −0.009 (Limited) and −0.038 (2-hour); repeating middle layers over the last 3 epochs −0.010; "first-order meta-gradient on MLP matrices" −0.012. — [qlabs-eng/slowrun](https://github.com/qlabs-eng/slowrun)
- q0 (arXiv 2606.03938): cyclic schedule with anti-correlated LR and weight decay to create diverse models (used for ensembling). — [arXiv 2606.03938](https://arxiv.org/abs/2606.03938)
- A May 2026 study of Muon in adversarial (CIFAR-10) training reports Muon does not prevent late-stage overfitting; not LM pretraining. — [arXiv 2605.26929](https://arxiv.org/html/2605.26929v2)

### Inferences
- Check the speedrun recipe's batch size. If it is ≥1M tokens, a smaller batch (more steps, same tokens) is a same-FLOPs change with strong evidence of reducing repetition overfitting; the cost is wall-clock/throughput, not FLOPs. It must be tested on the fresh-data control too, since smaller batch may also help unique-data training.
- For MuonH, the LR is an angular step size on a fixed-norm sphere; per the docstring it already corresponds to sqrt(lr·wd). A higher LR (or slower decay) in later passes is the closest analog of Kim's strong-WD recipe. Untested.

### Gaps
- No paper found comparing Muon/Shampoo/SOAP vs AdamW on the *size* of the multi-epoch penalty.
- No source found on lowering LR in later epochs specifically for repetition; WSD vs cosine under repetition was not directly compared.

## Q7. Model size and parameter allocation under repetition

### Takeaway
Without regularization, more parameters overfit more under repetition (Xue 2023; Kim 2025: 1.4B worse than 600M). With strong regularization, bigger models win again and loss follows a power law in params (Kim 2025). The repetition gap grows with model size (Xu 2026), so a fixed model size (300M here) cannot be "fixed" by scaling alone under same compute.

### Cited Findings
- Kim et al. 2025: untuned 1.4B worse than 600M at 200M tokens; regularized recipe restores monotone power-law scaling (asymptote 3.43). — [arXiv 2509.14786](https://arxiv.org/html/2509.14786)
- Xue et al. 2023: T5-XL on 2^27 tokens worse than T5-Large on 4× more data; parameter count matters more than FLOPs. — [arXiv 2305.13230](https://arxiv.org/html/2305.13230)
- Xu et al. 2026: the loss gap between data budgets grows with model size, which a Chinchilla additive law cannot express; their SoftQ law L = E + (A/N^ρ + B/U^(ρ/(1+α)))^(α/ρ) fits best (RMSE 0.008 vs 0.023 Muennighoff, 0.027 Chinchilla). — [arXiv 2606.06888](https://arxiv.org/html/2606.06888v2)
- Muennighoff 2023: in data-constrained settings put extra compute into epochs rather than params. — [arXiv 2305.16264](https://arxiv.org/html/2305.16264)

### Inferences
- The goal fixes model size and compute, so allocation is not a lever. But the "more params overfit more" result implies the larger embedding tables of large vocabularies add memorization capacity, consistent with the vocab-dependent penalty.

### Gaps
- No source separates embedding parameters from non-embedding parameters in repetition overfitting.

## Q8. Ranking by expected value for this goal (300M, 8 passes, ~8 GPU-hours per test, every tokenizer, same compute)

### Takeaway
Highest expected value: (1) per-pass reshuffle *and repack* of documents; (2) weight decay on the input embedding table (currently zero in MuonH); (3) late dropout / embedding dropout / low-rate random token replacement (all queued, all tokenizer-agnostic); (4) EMA/SWA over the last passes; (5) a smaller batch. Ensembling, distillation, diffusion objectives and rephrasing either break the same-compute rule or only pay off at far more epochs. No verified source shows any method fully closing the gap to fresh data at 8 passes; Muennighoff's law implies a residual of roughly 17% effective data.

### Cited Findings
- Repack + shuffle per epoch: −0.040 (B=64) to −0.141 (B=512) nats vs fixed order at 1.5B/1B tokens; zero compute cost. — [arXiv 2610.15965](https://arxiv.org/html/2610.15965); Slowrun per-epoch shuffle −0.026. — [qlabs-eng/slowrun](https://github.com/qlabs-eng/slowrun)
- Embedding weight decay with tied embeddings: −0.025 to −0.034 nats on top of repacking at 1.5B. — [arXiv 2610.15965](https://arxiv.org/html/2610.15965)
- Delayed dropout ≈ dropout-from-start in final quality (Xue 2023); dropout hurts single-epoch training (Liu, Bauer, Manning 2025). — [arXiv 2305.13230](https://arxiv.org/html/2305.13230); [arXiv 2505.24788](https://arxiv.org/abs/2505.24788)
- Random token replacement 15%: −0.174 at 150M/100 epochs; contradicted (masking: no gain) at 140M/50 epochs. — [arXiv 2606.16246](https://arxiv.org/html/2606.16246v1); [arXiv 2507.15857](https://arxiv.org/html/2507.15857)
- EMA −0.012 to −0.018, SWA −0.008 to −0.012 per record. — [qlabs-eng/slowrun](https://github.com/qlabs-eng/slowrun)
- Small batch: −0.04 to −0.15 nats (B=64 vs 512 sequences). — [arXiv 2610.15965](https://arxiv.org/html/2610.15965)
- MIR auxiliary masked loss: −0.017 (257M) to −0.034 (664M) on top of strong WD, ~1.3× data, but 2× forward cost. — [arXiv 2606.06888](https://arxiv.org/html/2606.06888v2)

### Inferences
Ranked list (my judgment; "cost" = extra compute per run; "fit" = applies to every tokenizer):
1. **Per-pass reshuffle + repack documents** (the queued "new data order each pass", extended to new packing boundaries). Zero cost, every tokenizer, strongest evidence in the closest regime. Verify first what the loader does now.
2. **Decoupled WD on input embeddings** (Adam group; try 0.1 and 0.3 × lr-scaled), optionally also on vectors/biases. Zero cost. Directly targets the only unregularized, vocab-sized parameter block. Not yet in the queued list; recommended addition.
3. **Input-embedding dropout / random input-token replacement at 5-10%, from pass 2** (queued). Cheap. Expect smaller gains than Chen 2026 because 8 passes ≪ 100 epochs; avoid ≥15% masking (collapse/interference reported).
4. **Residual dropout 0.1 from pass 2** (queued). Prefer 0.1 over 0.2 at this mild regime; keep it off in pass 1 to avoid the single-epoch penalty.
5. **EMA or SWA over the last 1-2 passes.** Near-zero cost; effect may be smaller under cosine decay to 0.
6. **Smaller global batch at fixed tokens** (if current batch ≥1M tokens). Same FLOPs, slower wall-clock; must be applied to the fresh-data control too.
7. **MuonH LR increase in later passes** (MuonH analog of strong WD). Speculative; no direct evidence.
8. **MIR** — only if the same-compute rule allows counting the second forward pass; otherwise out.
9. Not recommended for this goal: ensembling/distillation (multi-× compute), diffusion/any-order objectives (need ≫50 epochs), WRAP/synthetic rephrasing and code mixing (change data distribution, need generator compute), label smoothing (slight gain only; likely raises bpb), z-loss (no evidence).

Fairness note for the report: every change that also helps unique-data training (e.g., smaller batch, EMA) must be applied to the fresh-data baseline too, or the "gap closed" claim is inflated.

### Gaps
- No verified paper measures any remedy at exactly ~8 epochs and ~2.5 unique tokens/param against a fresh-data control with the same tuned recipe.
- The Min 2026 "8× more unique data" equivalence figure needs reading in full to know its reference baseline.
- Whether the vocab-dependent penalty lives in the input embeddings, the lm_head (AdamH, norm-preserving), or rare-token contexts in the body is untested; a cheap diagnostic is per-token-frequency-bucket bpb on Paloma for repeated vs fresh runs.
