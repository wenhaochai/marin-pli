# How optimizer, parametrization, normalization and numerics shape layer internals in deep transformers (survey for the "pls" project, 2017–2026)

Verification legend (applies to every citation below):
- **[full]**: I read the paper's HTML or PDF text in this session.
- **[abs]**: I read the arXiv abstract page or the abstract from the arXiv API in this session.
- **[meta]**: I confirmed only the arXiv id, title, authors and date through the arXiv API in this session. Content claims next to a [meta] tag come from my prior knowledge and are marked "(unverified detail)".
- **[blog]**: I read a non-arXiv page (blog, docs, code) in this session.
- kexue.fm posts were read from the local kexue.fm SQLite copy (`kexue-fm:kexue` skill).

Project facts used for mapping (from the assignment, not from sources): MuonH on every linear weight (Frobenius norm fixed at init), AdamH on heads, plain Adam on norm gains / embeddings / routing params, weight decay configured but not applied, no warmup at 130m, bf16 compute, 48-layer models, ten residual designs.

---

## Q1. Muon / norm-constrained / hyperspherical training: effects on residual growth, layer specialization, depth; what happens to norm gains under Adam when weights are norm-fixed

### Takeaway
Hyperball (MuonH/AdamH) fixes ‖W‖_F at its init value and normalizes each update, so the relative (angular) step equals the learning rate from step 0. The paper puts gains, embeddings and scalars on plain Adam and gives no analysis of what those gains do. With weight matrices norm-locked, every change in block output scale must come from the Adam-trained gains (and embeddings), so gains become the only scale channel. No source I found studies gain dynamics under norm-fixed weights. That gap is exactly the LNS-vs-Pre-LN finding.

### Cited Findings
**Hyperball / MuonH / AdamH (Marin)**
- Hyperball wraps a base optimizer: W_{t+1} = R·Normalize(W_t − η_t·R·Normalize(u_t)), with R = ‖W_0‖_F fixed at init (entries init with std 1/√d_in). η_t "directly sets the relative update length". MuonH uses u_t = s_μ·msign(M_t), s_μ = max(1, √(d_out/d_in)). [full] — [Wen, Dang, Lyu, Ma, Liang 2026, arXiv:2606.16899](https://arxiv.org/html/2606.16899v1)
- Hyperball is applied only to attention and MLP weight matrices of a pre-norm Transformer. Embeddings, normalization gains and "other scalar parameters" use plain Adam "because their norm can carry semantic information". The paper does not analyze gain behavior, residual-stream growth or activation growth. [full] — [arXiv:2606.16899](https://arxiv.org/html/2606.16899v1)
- Learning-rate transfer across depth (d=128, L=4→512): optimal-LR drift ≈1.4× for AdamH and MuonH vs ≈3× for AdamW and ≈4× for MuonW. [full] — [arXiv:2606.16899](https://arxiv.org/html/2606.16899v1)
- MuonH: 20–30% token-equivalent speedup over the AdamW scaling law at 1.2B (Qwen3-style), MuonW ≈10%. Marin speedrun: AdamH/MuonH match weight-decay baselines ≈10% larger. A Marin MuonH run ends 0.04 loss below the AdamW baseline over 159B tokens (the paper text gives "88B"; a secondary summary says 8B — treat the model size as unconfirmed). 130M overtrained fit: L∞ = 3.065 (MuonH) vs 3.079 (MuonW). [full] — [arXiv:2606.16899](https://arxiv.org/html/2606.16899v1); [Modelwire summary](https://themodelwire.com/article/fantastic-pretraining-optimizers-and-where-to-find-them-ii-hyperball-optimizatio-01KV77X7SP050113R8XZG2JTE1)
- Under AdamW the equilibrium angular step is (η^ang)² = 2(1−α)(1−β₁)/(1+αβ₁) with α = 1 − ηλ; weight decay sets the angular LR through an equilibrium norm. Runs with (η, λ) = (0.002, 0.2) and (0.004, 0.1) reach nearly the same loss, but the larger-η run has ≈2× larger Q/K norms. [full] — [arXiv:2606.16899](https://arxiv.org/html/2606.16899v1)
- A follow-up called "Hyperball May Not Be a Free Lunch" reportedly argues LR scheduling stays essential. I could not find it on arXiv; only a BAAI Hub summary surfaced. **[unconfirmed]** — [BAAI Hub](https://hub.baai.ac.cn/view/56966)
- Prior Marin optimizer benchmark (context for MuonH): [Wen, Hall, Ma, Liang 2025, "Fantastic Pretraining Optimizers and Where to Find Them", arXiv:2509.02046](https://arxiv.org/abs/2509.02046) [meta]

**Su Jianlin (kexue.fm) on norm-fixing**
- Su names Hyperball and Nemotron-Flash as "renormalize to τ after each step" schemes. He argues that fixing a norm exactly "破坏原本的训练动力学" (changes the original training dynamics) and proposes only a bound ‖W‖ ≤ τ ("Post Clip", or "Pre Decay", which reduces to ordinary weight decay for the RMS/Frobenius norm). — [kexue.fm/archives/11729, MuP之上：4. 坚守参数的稳定性 (2026-04-24)](https://kexue.fm/archives/11729); [Nemotron-Flash, Fu et al. 2025, arXiv:2511.18890](https://arxiv.org/abs/2511.18890) [meta]
- Steepest descent on a spectral sphere (fixed spectral norm, not Frobenius): [kexue.fm/archives/11241](https://kexue.fm/archives/11241). The Spectral Sphere Optimizer (SSO) constrains both weights and updates spectrally. It reports wins over AdamW and Muon on Dense 1.7B, MoE 8B-A1B and **200-layer DeepNet** models, plus "suppressed outliers, and strictly bounded activations". [abs] — [Xie et al. 2026, arXiv:2601.08393](https://arxiv.org/abs/2601.08393)

**Muon at scale (Moonshot)**
- Without weight decay, Muon weight RMS and layer-output RMS grew "exceeding the high-precision range of bf16". Moonlight scales updates to 0.2·√max(A,B) to match AdamW update RMS. AdamW trains RMSNorm gains, the LM head and embeddings. **Weight decay on RMSNorm γ is "crucial for stability" because it prevents excessively high per-layer output RMS.** Muon weights have higher SVD entropy than AdamW weights in >90% of matrices at 1.2T tokens, with the largest gap on MoE router weights. [full] — [Liu et al. (Moonshot) 2025, "Muon is Scalable for LLM Training", arXiv:2502.16982](https://arxiv.org/html/2502.16982)
- Kimi K2: with vanilla Muon (9B active / 53B total), max attention logits "quickly exceed a magnitude of 1000". The report says this happens more often with Muon than with AdamW. QK-Clip rescales W_q, W_k per head by γ_h = min(1, τ/S_max^h) with τ = 100. Logits stay capped and decay back after ≈30% of training. [full] — [Kimi Team 2025, Kimi K2, arXiv:2507.20534](https://arxiv.org/html/2507.20534); [kexue.fm/archives/11126](https://kexue.fm/archives/11126)

**nGPT (hyperspherical)**
- All hidden states and weight rows live on the unit sphere: h ← Norm(h + α(h_block − h)). There is no weight decay, no warmup and no final norm. The paper reports 4–20× fewer steps (1k–8k context) at 0.5B/1B. [full] — [Loshchilov, Hsieh, Sun, Ginsburg 2024, arXiv:2410.01131](https://arxiv.org/html/2410.01131)
- **nGPT separates init from Adam step size explicitly.** Each scaling vector is stored at s_scale and multiplied by s_init/s_scale in the forward pass, so its effective value is s_init and its effective Adam step is η·s_init/s_scale. Eigen LRs: α_init = 0.05 ("order of 1/n_layers"), α_scale = 1/√d_model. s_qk and s_z: init 1, scale 1/√d_model. Learned α_A falls from ≈0.25 (24 layers) to ≈0.20 (36 layers). (Appendix A.7 states a different α default; the paper does not reconcile the two.) [full] — [arXiv:2410.01131](https://arxiv.org/html/2410.01131)

**Norm-geometry optimizers**
- Scion: LMO over a norm ball; one set of weights and grads, storable in half precision, no Adam; hyperparameter transfer across sizes. [abs] — [Pethick et al. 2025, arXiv:2502.07529](https://arxiv.org/abs/2502.07529)
- Modular norm: a normalization wrapper makes the optimal LR roughly invariant to width and depth. [abs, via search snippet] — [Large et al. 2024, arXiv:2405.14813](https://arxiv.org/html/2405.14813); [Bernstein & Newhouse 2024, "Old Optimizer, New Norm", arXiv:2409.20325](https://arxiv.org/abs/2409.20325) [meta]
- Lipschitz transformers trained with Muon plus weight constraints run with no LayerNorm, no QK-norm and no logit softcap. A 10-Lipschitz 145M model reaches 21% accuracy, while matching NanoGPT's 39.4% needs a Lipschitz bound of 10^264. [abs] — [Newhouse et al. 2025, arXiv:2507.13338](https://arxiv.org/abs/2507.13338)
- "Optimal Scaling Needs Optimal Norm": for Adam and Scion, the optimal (LR, batch) pair keeps the same output-layer operator norm across scale (up to 1.3B, 138B tokens). Per-layer-group LR tuning helps; the output layer is most sensitive. [abs] — [Filatov et al. 2025, arXiv:2510.03871](https://arxiv.org/abs/2510.03871)

**Scale-invariance, rotation, warmup**
- Weight decay drives scale-invariant weights to a "rotational equilibrium" in which angular updates become homogeneous across layers and neurons. Controlling rotation explicitly reduces the need for warmup. [abs] — [Kosson, Messmer, Jaggi 2024 (ICML), arXiv:2305.17212](https://arxiv.org/abs/2305.17212)
- Warmup mainly counteracts large early angular updates. Normalizing the update removes most of the need for warmup. [abs] — [Kosson et al. 2024, arXiv:2410.23922](https://arxiv.org/abs/2410.23922)
- For scale-invariant weights, the effective step is the nominal step × 1/‖w‖ (AdamP, citing Hoffer et al. 2018). [search snippet] — [Heo et al., AdamP, arXiv:2006.08217](https://arxiv.org/pdf/2006.08217)
- EDM2 redesigns layers to preserve activation, weight and update magnitudes on expectation, with forced weight normalization. This removes magnitude drifts and imbalances (FID 2.41→1.81 on ImageNet-512). [abs] — [Karras et al. 2023, arXiv:2312.02696](https://arxiv.org/abs/2312.02696)
- Weight normalization (direction/magnitude split) is the classical precedent. [meta] — [Salimans & Kingma 2016, arXiv:1602.07868](https://arxiv.org/abs/1602.07868)

**Muon and outliers / internals**
- Outlier-Safe Pre-training combines Muon, single-scale RMSNorm (one scalar gain instead of per-channel gains) and a learnable embedding projection. 1.4B / 1T tokens: excess kurtosis 0.04 vs 1818.56 for standard training; 4-bit average 35.7 vs 26.5 (Adam). The authors call channel-wise operations and adaptive (diagonal) gradient scaling recognized causes of outliers and say Muon "eliminates privileged bases". [abs] — [Park et al. 2025 (ACL), arXiv:2506.19697](https://arxiv.org/abs/2506.19697)
- Non-diagonal preconditioners (SOAP) plus an unnormalized "Outlier Protected" block reduce outlier features. OPT-125m int8 W/A perplexity: 14.87 (OP+SOAP) vs 63.4 (Pre-Norm+Adam). [abs] — [He, Noci, Paliotta, Schlag, Hofmann 2024 (NeurIPS), arXiv:2405.19279](https://arxiv.org/abs/2405.19279)
- "Muon as a Residual Connection" (1 author, controlled linear settings): orthogonalized updates fit local targets more slowly but give representations that later layers use more easily. [abs] — [Huang 2026, arXiv:2607.01124](https://arxiv.org/abs/2607.01124)

### Inferences
- **Gains are the only scale knob under MuonH.** In a pre-norm block y = W·(γ ⊙ x̂), W's Frobenius norm is locked, so block output RMS can change only through γ (and through W's alignment with the input). The 2.1–3.5× larger trained Pre-LN gains (vs LNS) are therefore the model's only route to bigger writes in deep layers, where the residual stream is large. Moonlight's statement that WD on γ is "crucial" to stop per-layer output RMS growth is the closest published evidence. Our config applies no WD to gains, so nothing restrains that channel.
- **Init vs step size, in the nGPT form.** Adam's per-coordinate step is ≈η whatever the parameter scale. LNS (effective gain g = γ/√ℓ, γ init 1) therefore has init g = 1/√ℓ and step η/√ℓ. Pre-LN has init 1 and step η. The queued run (Pre-LN, γ init 1/√ℓ) has init 1/√ℓ and step η. The missing fourth cell is init 1 and step η/√ℓ (Pre-LN with gain LR ×1/√ℓ, or nGPT-style s_init = 1, s_scale = √ℓ). With all four cells the init and step effects separate cleanly (2×2 factorial).
- **MuonH makes init-based depth schemes permanent.** DeepNet's β-scaled init (β = (8M)^{-1/4} ≈ 0.227 at M = 48) and any 1/√L weight init stay fixed for the whole run, because R = ‖W_0‖_F never changes. Under AdamW they would drift. So DeepNorm's down-weighting of branch weights is exact for the whole MuonH run, and the gains alone can undo it.
- **No warmup is consistent with Hyperball and Kosson.** The angular step is fixed from step 0. But plain-Adam gains, embeddings and routing parameters get no such normalization, and Kosson's analysis says those early Adam steps are the ones warmup protects. Logging per-group relative update ‖Δγ‖/‖γ‖ over the first 1k steps is cheap and would show whether gains take outsized early steps.
- **QK logit growth is a likely Muon-family risk.** MuonH fixes ‖W_q‖_F and ‖W_k‖_F, but alignment can still raise q·k (Kimi K2 saw >1000 logits with Muon). Pre-norm gains feed q and k quadratically, so a growing γ raises logits by γ². Log max attention logit per layer for Pre-LN vs LNS.
- **Prediction to test.** OSP and He et al. tie outliers to per-channel gains plus diagonal Adam. In our setup the linear weights use Muon (non-diagonal), but gains are per-channel and Adam-trained. That combination is the remaining outlier source. Experiment: a single-scale (scalar) gain per norm (OSP's "single-scale RMSNorm") in Pre-LN and in LNS. Then measure the 637×-RMS max activation and the final-norm zero gains.

### Gaps
- Hyperball paper: no measurement of gain trajectories, residual-stream norm or massive activations under MuonH. The LM head and bias assignment is unstated ([arXiv:2606.16899](https://arxiv.org/html/2606.16899v1)).
- I found no paper that studies Adam-trained norm gains when all matrices are norm-fixed. This looks open.
- "Hyperball May Not Be a Free Lunch": not found on arXiv; unconfirmed.
- The Marin repo and GitHub issues were not checked directly for MuonH implementation details (e.g., whether gains get WD, Adam ε).

---

## Q2. Depth-dependent parametrization: depth-μP, 1/√L branch scaling, LNS vs Pre-LN, Admin, DeepNet init; evidence that init vs per-step scale matters for gains

### Takeaway
Depth-μP and CompleteP treat the branch multiplier and the per-parameter Adam LR as separate depth knobs. CompleteP gives an explicit rule: the LayerNorm-parameter LR scales as L^{α−1} (so ∝ L^{−1/2} for 1/√L branches, and unscaled for 1/L). LNS changes only the forward multiplier on the gain, which under Adam also shrinks the absolute gain step by 1/√ℓ. No paper I found isolates init from step size for gains. nGPT's s_init/s_scale trick is the standard tool to do it.

### Cited Findings
- **LayerNorm Scaling (LNS)**: scales the LN output by 1/√ℓ to counter Pre-LN variance growth. Claim: nearly half the layers of Llama, Mistral, DeepSeek and Qwen are "less effective than expected", because deep Pre-LN blocks' Jacobians approach identity. Tested 130M–7B; NeurIPS 2025. [abs] — [Sun, Song, Li, Yin, Zheng, Liu 2025, "The Curse of Depth in LLMs", arXiv:2502.05795](https://arxiv.org/abs/2502.05795)
- **CompleteP** (α = 1, branch multiplier m_L^{−1}): the table gives the Pre-LN (LayerNorm) LR depth exponent as α−1 (0 for CompleteP). The hidden-weight AdamW LR has no depth term at α = 1. AdamW ε (residual blocks) scales as m_N^{−1}·m_L^{−1}, and WD scales with width only. Compute savings over μP are 12–34% (11.8% at N=1984, L=32; 34.4% at N=832, L=179). Only α = 1 gives "complete" (non-lazy) feature learning; for α = 1/2 blocks approach their linearization at rate 1/√L. [full] — [Dey et al. 2025, arXiv:2505.01618](https://arxiv.org/html/2505.01618)
- **Depth-μP**: classifies block multipliers and LRs; finds a unique optimal parametrization for one-layer blocks. It identifies "fundamental limitations" for deeper blocks such as transformer blocks. [abs] — [Yang, Yu, Zhu, Hayou 2023, Tensor Programs VI, arXiv:2310.02244](https://arxiv.org/abs/2310.02244)
- 1/√depth residual branches plus μP transfer hyperparameters across width and depth (ResNets, ViTs). [abs] — [Bordelon, Noci, Li, Hanin, Pehlevan 2023, arXiv:2309.16620](https://arxiv.org/abs/2309.16620)
- **DeepNet / DeepNorm**: x_{l+1} = LN(α·x_l + f(x_l)), with init gain β on ffn, v_proj and out_proj (q, k gain 1). Decoder-only with M layers: α = (2M)^{1/4}, β = (8M)^{−1/4}. Encoder-decoder: encoder α = 0.81(N⁴M)^{1/16}, β = 0.87(N⁴M)^{−1/16}; decoder α = (3M)^{1/4}, β = (12M)^{−1/4}. Scales to 1,000 layers. [full: Figure 2 table extracted from PDF] — [Wang et al. 2022, arXiv:2203.00555](https://arxiv.org/abs/2203.00555). Su's derivation of the same "增量爆炸" (update explosion) argument: [kexue.fm/archives/8978](https://kexue.fm/archives/8978)
- **Admin** (adaptive model initialization): Post-LN's heavy reliance on its residual branch amplifies parameter perturbations. [meta] — [Liu et al. 2020, arXiv:2004.08249](https://arxiv.org/abs/2004.08249); summarized in [kexue.fm/archives/8978](https://kexue.fm/archives/8978)
- Pre-LN puts well-behaved gradients at init and removes the need for warmup. [meta] — [Xiong et al. 2020, arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- **Mix-LN**: Pre-LN gives diminished gradient norms in deep layers; Post-LN in early layers, Pre-LN in deep layers, 70M–7B. [abs] — [Li, Yin, Liu 2024, arXiv:2412.13795](https://arxiv.org/abs/2412.13795)
- **Peri-LN** (= Sandwich placement with an output-LN): Pre-LN hidden-state variance grows ~exponentially with depth and training, Peri-LN ~linearly. **Freezing the output-LN gain at 1 still reduces gradient spikes**; a learnable gain is slightly better and "adjusts with depth". 10× WD (0.033→0.33) rescued one diverging Pre-LN run. Five-seed benchmark avg Pre-LN → Peri-LN: 49.69→51.57 (400M), 53.71→56.55 (1.5B), 56.69→58.56 (3.2B). Gemma and OLMo use this placement. [full] — [Kim et al. 2025 (ICML), arXiv:2502.02732](https://arxiv.org/html/2502.02732v3)
- **Keel** (the "KEEL" design): Post-LN with a Highway-style connection replacing the ResNet residual; trains past 1000 layers with no special init; beats Pre-LN in perplexity and depth scaling (abstract gives no numbers). [abs] — [Chen & Wei 2026, "Post-LayerNorm Is Back: Stable, ExpressivE, and Deep", arXiv:2601.19895](https://arxiv.org/abs/2601.19895)
- ResiDual (Pre-Post-LN) argues Pre-LN causes "representation collapse" and Post-LN gradient vanishing. [abs] — [Xie et al. 2023, arXiv:2304.14802](https://arxiv.org/abs/2304.14802)
- Sub-LN / Magneto: [meta] — [Wang et al. 2022, "Foundation Transformers", arXiv:2210.06423](https://arxiv.org/abs/2210.06423)
- ReZero (zero-init scalar per residual branch): [meta] — [Bachlechner et al. 2020, arXiv:2003.04887](https://arxiv.org/abs/2003.04887)
- Su's explanation of why Pre-LN underperforms: for large t, x_{t+1} ≈ x_t, so stacking adds width rather than depth ("Pre Norm的深度有水分", Pre-LN's depth is partly hollow). — [kexue.fm/archives/9009](https://kexue.fm/archives/9009)
- Per-layer LR and Adam ε matter across parametrizations (Adam-atan2 removes ε). [meta] — [Everett et al. 2024, arXiv:2407.05872](https://arxiv.org/abs/2407.05872)
- Unit scaling / u-μP: unit-variance activations, weights and grads at init for low precision, with fixed depth multipliers. [meta] — [Blake, Orr, Luschi 2023, arXiv:2303.11257](https://arxiv.org/abs/2303.11257); [Blake et al. 2024, arXiv:2407.17465](https://arxiv.org/abs/2407.17465)

### Inferences
- **LNS is not a branch multiplier in modern blocks.** Scaling the pre-norm gain by c scales attention logits by c² (q and k both carry c) and V by c. A SwiGLU MLP output scales by c² (gate × up). So LNS's 1/√ℓ is ≈1/ℓ on the MLP write at init and also cools attention by 1/ℓ. Depth-μP/CompleteP results for branch multipliers do not carry over directly. Experiment: compare LNS against an explicit output-side branch multiplier 1/√ℓ (or 1/ℓ, CompleteP-style) with gains left at 1.
- **CompleteP's rule predicts the gain-LR gap matters.** CompleteP sets the LN-gain LR exponent to α−1. If LNS is viewed as a per-layer multiplier of order ℓ^{−1/2}, the matched depth-μP-style gain LR would shrink as well. LNS does shrink the absolute step by 1/√ℓ automatically, but its relative step stays η. The 2×2 factorial (init × step) from Q1 tests whether the trained gains follow init (gap closes only partly) or step size.
- **Prediction for the queued run.** If the queued run (init 1/√ℓ, step η) ends with gains near Pre-LN's, then step size dominates, and the "init gap only partly closes" result reflects the √ℓ-smaller absolute steps of LNS. If it ends near LNS's, then init dominates (e.g., through early-training path dependence). Either outcome is informative. Also log the relative step ‖Δγ‖/‖γ‖ per layer: Adam makes it ≈η/|γ| per coordinate, so √ℓ larger for the queued run than for LNS.
- **Adam ε becomes relevant in deep LNS layers.** LNS shrinks γ's gradient by 1/√ℓ (and SwiGLU paths shrink it further). If √v̂ approaches ε for deep-layer gains, Adam's step falls below η and LNS gains move even less. CompleteP scales ε by m_L^{−1} for this reason. Check: log √v̂/ε for gains at ℓ = 1, 24, 48.
- **DeepNorm under MuonH**: α = 96^{1/4} ≈ 3.13 and β = 384^{−1/4} ≈ 0.227 for M = 48. Because MuonH keeps ‖W‖_F at its β-scaled init, the branch-to-skip ratio stays ≈β/α at the weight level for the whole run. Only gains and alignment can rebalance it. This makes DeepNorm's small branch signal relative to α·x permanent, which feeds the bf16 cost in Q5.

### Gaps
- The LNS paper's exact per-layer numbers (variance growth, perplexity deltas) were not on the abstract page; full text not read.
- Depth-μP's exact Adam LR exponent for gains was read only through CompleteP's generalized table (α−1), not from Tensor Programs VI itself.
- No paper found that trains Pre-LN with gains initialized at 1/√ℓ, or that separates gain init from gain step size. The project's queued run appears novel.
- Admin's ω formula and Xiong's numbers are from memory (marked [meta]).

---

## Q3. Massive activations, attention sinks, outlier features, super weights, entropy neurons, final-norm gain zeros: how they arise, their role (bias/temperature), how losses or architecture change them

### Takeaway
Massive activations form abruptly in one early layer, at a few fixed dimensions, on sink-like tokens (BOS, delimiters). They act as input-independent biases: setting them to their mean is harmless, setting them to zero is catastrophic. They exist mainly to let softmax attention "do nothing" (sinks/no-op heads). They dominate the RMSNorm denominator, so they act as a per-token temperature/scale signal, and they live in directions the unembedding ignores ("dark signals"). Removing the need for them (attention gates, KV biases, sigmoid attention, scalar gains with Muon) removes them. A separate readout channel does not remove them: they reappear in the readout stream.

### Cited Findings
**Massive activations: formation and role**
- LLaMA2-7B top hidden values 2622, 1547, 802, 477 vs median 0.2 (≈10,000×; the abstract says up to "100,000 times"). Fixed dims 1415, 2533 (7B); 2100, 4743 (13B); Mixtral 7100 at dims 2070, 3398. They appear abruptly at layer 2 (7B) or layer 4 (13B), stay nearly constant through middle layers, and **diminish in the last few layers (disappear at the final layer, App. A.4)**. Tokens: start token, first "." or "\n". [full] — [Sun, Chen, Kolter, Liu 2024 (COLM), arXiv:2402.17762](https://arxiv.org/html/2402.17762v2)
- Setting them to zero: LLaMA2-7B perplexity → ∞, accuracy 68.95% → 36.75%. Setting them to their mean: no change (68.94%). Zeroing equally many median activations: no drop. → They act as **fixed biases**. [full] — [arXiv:2402.17762](https://arxiv.org/html/2402.17762v2)
- **Massive values dominate the LayerNorm/RMSNorm denominator**, pushing other coordinates of that token toward zero and making normalized features at those tokens nearly identical. [full] — [arXiv:2402.17762](https://arxiv.org/html/2402.17762v2)
- GPT-2 with explicit learnable per-head key and value biases has **no massive activations** at equal perplexity (3.04). A sink-token variant still has them. [full] — [arXiv:2402.17762](https://arxiv.org/html/2402.17762v2)
- ViTs: CLIP and DINOv2 show massive activations; in DINOv2-reg they sit in a register, and fixing registers to their mean keeps 87.0% accuracy (registers = learned biases). [full] — [arXiv:2402.17762](https://arxiv.org/html/2402.17762v2); [Darcet et al. 2023, "Vision Transformers Need Registers", arXiv:2309.16588](https://arxiv.org/abs/2309.16588) [meta]
- A "Massive Emergence Layer" is consistent across families; **RMSNorm and FFN parameters in that layer jointly create** the massive activation. The resulting rigid token reduces hidden-state diversity, and their fix also weakens attention sinks. [abs] — [Shi et al. 2026, arXiv:2605.08504](https://arxiv.org/abs/2605.08504)
- Not all massive activations are harmful (suppressing some does not explode perplexity). KV-bias mitigation is model-specific. TVR + KV bias or DyT balance mitigation and quality. [abs] — [Owen et al. 2025, arXiv:2503.22329](https://arxiv.org/abs/2503.22329)
- **Splitting the residual stream into a scratch stream and a protected readout ("commitment") stream does not remove massive activations.** The model rebuilds the fixed-dimension start-token outlier inside the protected channel: smaller, but more concentrated. A stronger sparsity penalty makes it more persistent (160M, 290M). [abs] — [Vemula 2026, arXiv:2606.20743](https://arxiv.org/abs/2606.20743)
- Massive values in Q and K (not V) come from RoPE and matter for contextual knowledge. [abs] — [Jin et al. 2025, arXiv:2502.01563](https://arxiv.org/abs/2502.01563)

**Attention sinks**
- StreamingLLM: keeping initial-token KV restores streaming quality. [meta] — [Xiao et al. 2023, arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- When sinks emerge (60M default model): they appear between ≈1k and 2k steps. Sink₁ vs weight decay: 15.2% (γ = 0) → 18.2% (0.1) → 41.1% (0.5, peak) → 6.1% (2.0) → 0.01% (5.0). Smaller LR delays and weakens sinks; less data removes them. Learnable K/V biases move the sink there (Sink₁ → 0.00%). **Sigmoid attention without normalization has no sink and no massive activations** (1B: 45.11% → 2.46%). Post-LN keeps the sink (13.5%), with massive activations appearing before the post-LN. Sinks act like key biases. [full] — [Gu et al. 2025 (ICLR spotlight), arXiv:2410.10781](https://arxiv.org/html/2410.10781v2)
- Sinks prevent over-mixing (over-squashing) and depend on context length, depth and packing. [abs] — [Barbero et al. 2025, arXiv:2504.02732](https://arxiv.org/abs/2504.02732)
- Output-gated attention (head-specific sigmoid gate after SDPA): first-token attention 46.7% → 4.8%; max hidden activation 1053 → 94 (layer average). It removes loss spikes and tolerates higher LR (48-layer, 400B tokens, LR 8e-3: PPL 9.195 baseline vs 7.325 gated); RULER at 128k 31.65 → 58.82. [full] — [Qiu et al. (Qwen) 2025, arXiv:2505.06708](https://arxiv.org/html/2505.06708)
- Outliers come from heads learning "no-op" updates: softmax inputs are pushed ever larger to get exact zeros. Clipped softmax and gated attention fix this. [abs] — [Bondarenko, Nagel, Blankevoort 2023, arXiv:2306.12929](https://arxiv.org/abs/2306.12929)
- "Systematic outliers" (activation, weight, attention) arise from softmax and act as **implicit context-aware scaling factors**. Eliminating them structurally speeds convergence. [abs] — [An et al. 2025, arXiv:2502.06415](https://arxiv.org/abs/2502.06415)
- QK-Clip / Muon logit explosion: see Q1. — [arXiv:2507.20534](https://arxiv.org/html/2507.20534)

**Outlier features, super weights, quantization**
- LLM.int8(): systematic outlier feature dims appear at scale. [meta] — [Dettmers et al. 2022, arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- BERT outlier dimensions tied to LayerNorm parameters. [meta] — [Kovaleva et al. 2021, arXiv:2105.06990](https://arxiv.org/abs/2105.06990)
- Super weights: pruning one parameter raises perplexity by three orders of magnitude and drops zero-shot accuracy to chance. Super weights create "super activations". [abs] — [Yu et al. 2024, arXiv:2411.07191](https://arxiv.org/abs/2411.07191). (Location in an early mlp.down_proj is my recollection; unverified detail.)
- SwiGLU weight alignment amplifies outliers only over long training (2T tokens FP8); Smooth-SwiGLU fixes it. [abs] — [Fishman et al. 2024, arXiv:2409.12517](https://arxiv.org/abs/2409.12517)
- Outlier features vs optimizer/architecture: OP block + SOAP; Muon + scalar RMSNorm (OSP). See Q1. — [arXiv:2405.19279](https://arxiv.org/abs/2405.19279); [arXiv:2506.19697](https://arxiv.org/abs/2506.19697)

**Entropy neurons, final norm, dark signals**
- **Entropy neurons** have large weight norms, write into the **unembedding null space**, and act through the **final LayerNorm scale** to shrink logits (raise entropy), with little direct logit effect. Found in models up to 7B. Token-frequency neurons shift output toward or away from the unigram distribution. [abs] — [Stolfo et al. 2024 (NeurIPS), arXiv:2406.16254](https://arxiv.org/abs/2406.16254); universal neurons in GPT-2: [Gurnee et al. 2024, arXiv:2401.12181](https://arxiv.org/abs/2401.12181) [meta]
- **Dark signals**: the tail of the (un)embedding spectrum carries the signals responsible for attention sinking. Sink-receiving tokens project strongly on that tail, and suppressing parts of the spectrum keeps loss low only if sinking is preserved. [abs] — [Cancedda 2024, arXiv:2402.09221](https://arxiv.org/abs/2402.09221)
- **Final-LayerNorm collapse**: in an in-context linear-regression transformer, a large fraction of final (unembedding) LN weights go to zero in late stages, and the collapse spreads to earlier LNs. Interpretation: the model reads predictions from a few privileged residual dims. [full, partial] — [Hoogland et al. 2024/2025, arXiv:2402.02364](https://arxiv.org/html/2402.02364)
- GPT-2's LN layers can be removed by fine-tuning on 500M tokens with −0.05 CE change. [abs] — [Heimersheim 2024, arXiv:2409.13710](https://arxiv.org/abs/2409.13710)
- AttnRes uses a ∝ exp(w·RMSNorm(y_s)) over sources, with RMSNorm on keys. Each block f has an input RMSNorm, so Su notes weighted mean and weighted sum are equivalent (RMSNorm(cx) = RMSNorm(x)). — [kexue.fm/archives/11664](https://kexue.fm/archives/11664); [Kimi Team 2026, "Attention Residuals", arXiv:2603.15031](https://arxiv.org/abs/2603.15031) [meta]

### Inferences
- **Final-norm zero gains on massive dims are the expected configuration, not a bug.** Massive dims carry an input-independent bias whose job is in attention (sinks), and Sun et al. find they fade by the last layer. A zeroed final gain on those dims makes them "dark" to the unembedding (Cancedda), while their size still enters the final RMS denominator. That denominator path is the entropy-neuron mechanism (Stolfo): a per-token temperature. Test: (a) regress per-token logit entropy on the massive-dim magnitude at the final norm input; (b) recompute the final RMS excluding the massive dims and measure Δloss. If Δloss is large, the dims serve as a temperature knob; if not, they are pure attention bias.
- **Separate heads should move, not remove, massive activations** (Vemula's result). Check whether the separate-heads designs rebuild a start-token outlier in the stream the last MLP reads. That would also explain why separate heads change the last MLP's direct logit contribution.
- **AttnRes(Full) 53,000× sources follow from scale-invariance.** Source keys pass through RMSNorm, and the consumer f has an input RMSNorm. So nothing in the forward pass penalizes a source's magnitude, and under bf16 nothing bounds it except numerics. The 53,000× source is likely a sink/bias carrier, as in LLaMA2-7B's ≈13,000×. Tests: (i) value-side RMSNorm or a per-source bound; (ii) add learnable KV biases or an output gate (Gu, Sun, Qiu) and check whether the 53,000× source and the 637×-RMS activation fall.
- **Expect sinks even without weight decay.** Gu et al. find a 15% sink at γ = 0, so our "WD never applied" setting still predicts sinks. Their peak at γ = 0.5 suggests that adding WD to gains could raise sink strength before lowering it.
- **Cheapest causal experiment: per-head sigmoid output gate** (Qiu et al.) in Pre-LN and LNS 48-layer models. Prediction: max activation drops ~10×, final-norm zero gains disappear or shrink, and the Pre-LN vs LNS gain gap narrows (if gains were growing partly to feed massive dims).

### Gaps
- Super-weight location and stopword details not re-verified.
- No source shows directly that final-norm gains are zero exactly on massive-activation dims in LLMs. Hoogland et al. show final LN collapse only in a toy regression transformer. Search found nothing in LLMs; this may be a new observation.
- Shi et al. (ME layer) and Owen et al. models and numbers are abstract-level only.
- No paper studies massive activations under Muon/MuonH specifically, beyond OSP's kurtosis result (which also changes the gains).

---

## Q4. Effective rank / representation collapse across depth; anisotropy

### Takeaway
Pure attention loses rank doubly exponentially with depth. Residual branches with depth-dependent scaling prevent that at init. Trained LLMs are anisotropic in most layers, and intermediate layers often compress then re-expand information. Pre-LN makes late-layer states redundant because the growing stream drowns block writes. Muon weights have flatter singular spectra (higher SVD entropy) than AdamW weights. None of these sources measures stream-sum rank in multi-stream (HC/mHC) designs.

### Cited Findings
- Pure self-attention loses rank doubly exponentially with depth; skip connections and MLPs counteract it. [meta] — [Dong, Cordonnier, Loukas 2021, arXiv:2103.03404](https://arxiv.org/abs/2103.03404)
- Rank collapse at init makes query/key gradients vanish. Depth-dependent residual scaling prevents it, and queries/values get disproportionate gradient norms (a motivation for adaptive optimizers). [abs] — [Noci et al. 2022, arXiv:2206.03126](https://arxiv.org/abs/2206.03126)
- Pre-LN hidden states are more redundant (angular distance) by the end of training, which the authors attribute to the growing residual drowning sublayers. [full] — [Kim et al. 2025, Peri-LN, arXiv:2502.02732](https://arxiv.org/html/2502.02732v3)
- Anisotropy appears across objectives and modalities, so it is "inherent to Transformers". [abs] — [Godey, de la Clergerie, Sagot 2024, arXiv:2401.12143](https://arxiv.org/abs/2401.12143)
- Anisotropy and intrinsic dimension change systematically across layers and training. [meta, title-level] — [Razzhigaev et al. 2023, arXiv:2311.05928](https://arxiv.org/abs/2311.05928)
- Intermediate layers often give better features than the last; layers balance compression and signal preservation (32 embedding tasks). [abs] — [Skean et al. 2025, arXiv:2502.02013](https://arxiv.org/abs/2502.02013)
- "Guess-then-Refine": early layers propose high-frequency tokens, which later layers refine. [abs] — [Gupta et al. 2025, arXiv:2510.18871](https://arxiv.org/abs/2510.18871)
- Deeper layers of LLMs can be pruned with little loss. [meta] — [Gromov et al. 2024, arXiv:2403.17887](https://arxiv.org/abs/2403.17887); Mix-LN frames this as a Pre-LN training shortfall [abs] — [arXiv:2412.13795](https://arxiv.org/abs/2412.13795)
- The massive-activation token stays unchanged across layers and reduces hidden-state diversity. [abs] — [Shi et al. 2026, arXiv:2605.08504](https://arxiv.org/abs/2605.08504)
- Muon gives higher SVD entropy than AdamW for >90% of weight matrices. [full] — [arXiv:2502.16982](https://arxiv.org/html/2502.16982)
- HC widens the stream to k copies (classic k = 4) with H^pre, H^post and H^res. Su notes the product of tanh-generated H^res risks explosion or collapse. mHC constrains these matrices. — [kexue.fm/archives/11664](https://kexue.fm/archives/11664); [Zhu et al. 2024, Hyper-Connections, arXiv:2409.19606](https://arxiv.org/abs/2409.19606) [meta]; [Xie et al. (DeepSeek) 2025, mHC, arXiv:2512.24880](https://arxiv.org/abs/2512.24880) [meta]

### Inferences
- **Separate-heads HC stream-sum rank 64.** Low rank in a summed multi-stream state is the multi-stream analogue of Pre-LN redundancy: if the k streams converge to near-copies, or if H^res products collapse (Su's warning), the sum spans few directions. Diagnostics: (a) effective rank of each stream and of the sum separately; (b) cross-stream cosine similarity across depth; (c) effective rank of the products of H^res. Also check whether 64 equals an architectural number (head dim, stream count × something). If it does, the cause is structural, not learned.
- **Muon and per-layer rank.** Muon's flatter weight spectra predict higher per-layer write rank. The stream rank then depends mainly on the residual design. Comparing write effective rank vs stream effective rank per layer across the ten designs separates the optimizer effect from the architecture effect.
- Pre-LN vs LNS: LNS reduces late-layer redundancy in the "curse of depth" story. Measure the angular distance between consecutive layers' states (Peri-LN's metric) in Pre-LN, LNS and the queued run. This tells whether the gain differences translate into representation differences.

### Gaps
- No paper found on effective rank of HC/mHC stream sums, or on rank under MuonH.
- Dong et al., Gromov et al. and Razzhigaev et al. details are [meta] only (not re-read in this session).

---

## Q5. Low-precision effects in LayerNorm / Post-LN (bf16 statistics) and known fixes

### Takeaway
The standard fix is to compute normalization statistics and the gain multiply in fp32 and to keep the residual stream in fp32. PyTorch autocast already runs layer_norm in fp32, but not rms_norm. HF's Llama RMSNorm casts back to bf16 before the gain multiply, and Gemma's (1+w) gain had to move to fp32. I found no paper that quantifies the bf16 cost of DeepNorm's α·x residual. The project's 1–2e-3 nat cost and the 75–92% recovery from fp32 statistics appear new as numbers.

### Cited Findings
- PyTorch autocast "CUDA Ops that can autocast to float32" includes **layer_norm, group_norm, softmax, log_softmax, norm, normalize, rsqrt, sum**. **rms_norm is not on the list.** [blog] — [PyTorch 2.14 AMP docs](https://docs.pytorch.org/docs/2.14/amp.html)
- HF LlamaRMSNorm: upcasts to float32, computes variance and rsqrt in fp32, then `return self.weight * hidden_states.to(input_dtype)`. **The gain multiply runs in the input dtype (bf16).** [blog] — [transformers modeling_llama.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py)
- Gemma bug list: "Layernorm (w+1) should be done in float32". vLLM's Gemma norm computes variance, normalization and the (1+w) scale in fp32 before casting (HF PR 29402). Gemma 3 activations exceed fp16 max 65504. [blog] — [Unsloth, "Fixing All Gemma Bugs" (2024)](https://unsloth.ai/blog/gemma-bugs); [Unsloth Gemma 3 docs](https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune)
- Megatron-LM `--fp32-residual-connection` ("move residual connections to fp32"; originally asserted fp16) and `--apply-residual-connection-post-layernorm` (original BERT ordering). [blog, mirror commits] — [ORNL Megatron-LM mirror commit](https://code.ornl.gov/candle/Megatron-LM/-/commit/deffcb6a41c1017ff09a6d4c4d3b8817dae386fd); [commit 62632d39](https://code.ornl.gov/candle/Megatron-LM/-/commit/62632d39364e499bbfe9e143d2856dbe3f3a6802)
- Variance as E[x²]−E[x]² in bf16 statistics collapses to 0 or goes negative when mean ≫ std (outputs ~±10⁶), whereas torch's bf16 layer_norm gives ±1. [blog] — [tt-metal issue #57861](https://github.com/tenstorrent/tt-metal/issues/57861)
- Saving LN mean and rstd in the input dtype (computed in fp32) corrupts backward for near-constant inputs. The fix stores them in fp32. [blog] — [Liger-Kernel PR #1498](https://github.com/linkedin/Liger-Kernel/pull/1498)
- Peri-LN: Pre-LN top-100 hidden values exceed fp16 max early in training (3.2B), while Peri-LN stays below. The authors suggest this overflow explains OPT (fp16) instabilities. [full] — [arXiv:2502.02732](https://arxiv.org/html/2502.02732v3)
- Moonlight: without WD, Muon weights and layer outputs exceed "the high-precision range of bf16". [full] — [arXiv:2502.16982](https://arxiv.org/html/2502.16982)
- CogView introduced Sandwich-LN and "precision bottleneck relaxation" (PB-relax) to remove NaN losses in fp16. [abs]. The detail that PB-relax divides inputs by their max before LayerNorm and rescales attention logits before softmax is my recollection; the 10 MB PDF could not be fetched (unverified detail). — [Ding et al. 2021, arXiv:2105.13290](https://arxiv.org/abs/2105.13290)
- bf16 training studies: nearest rounding of weight updates cancels small updates; stochastic rounding or Kahan summation recovers fp32 quality. [meta] — [Zamirai et al. 2020, arXiv:2010.06192](https://arxiv.org/abs/2010.06192); [Kalamkar et al. 2019, arXiv:1905.12322](https://arxiv.org/abs/1905.12322) [meta]
- Low-precision stability: [Wortsman et al. 2023, arXiv:2304.13013](https://arxiv.org/abs/2304.13013) [meta]; attention-logit growth and output-logit divergence reproduce at small scale with high LR, and qk-layernorm and z-loss fix them [abs] — [Wortsman et al. 2023, arXiv:2309.14322](https://arxiv.org/abs/2309.14322)
- Unit scaling for low precision. [meta] — [arXiv:2303.11257](https://arxiv.org/abs/2303.11257)
- SwiGLU outlier amplification in FP8 over long runs. [abs] — [arXiv:2409.12517](https://arxiv.org/abs/2409.12517)
- Keeping the residual add in fp32 fixed a ~30× amplification of an fp16 layer-0 rounding error (one small project's report; anecdotal). [blog] — [GitHub issue](https://github.com/lecuong1502/Semantic-Aware_Adaptive_KV_Cache_Precision/issues/36)

### Inferences
- **Size of the DeepNorm bf16 error (back-of-envelope).** bf16 has an 8-bit significand, so unit roundoff is 2⁻⁸ ≈ 3.9e-3. For M = 48, α ≈ 3.13 and the skip term α·x has RMS ≈ 3.13 (x is an LN output). Rounding α·x and then α·x + f(x) to bf16 gives absolute error ≈ 3.9e-3 × 3.13 ≈ 0.012 per element per layer. The branch write f(x) is held near β-scale by MuonH (Q2), so it is about 10× smaller than α·x. The per-layer rounding noise is then about 1–4% of the branch signal. Summed over 48 Post-LN layers, a ~1e-3 nat cost is plausible. Two fixes to compare: (a) fp32 residual sum (Megatron-style) with LN on the fp32 sum; (b) fp32 LN statistics only (already 75–92%). The remaining 8–25% should be the rounded α·h product, and (a) should remove it.
- **Where else bf16 bites in our setup.** (i) RMSNorm gain multiply in bf16 (HF Llama pattern, Gemma bug): for Pre-LN gains of order 1, the bf16 spacing near 1 is 2⁻⁷ ≈ 7.8e-3. That is coarser than typical per-step Adam gain changes, so if gains are stored or applied in bf16, small updates are lost (Zamirai). Check that the gain parameter is fp32 and the multiply is in fp32. (ii) Massive activations at 637× RMS or 53,000× median dominate the norm denominator. Sum-of-squares in bf16 then loses all non-massive coordinates' contribution, so fp32 statistics matter most on sink tokens. Measure the bf16-vs-fp32 output difference per token, split into massive-carrying vs other tokens.
- **Diagnostic.** If the f32-statistics fix recovers most of the DeepNorm cost, the same fix in Pre-LN, LNS and AttnRes should recover nearly nothing, unless those designs have large-mean residuals (mean ≫ std triggers the E[x²]−E[x]² cancellation). Running f32 statistics across all ten designs is a cheap control.

### Gaps
- No paper found that measures the bf16 loss cost of DeepNorm or Post-LN statistics. The project's 1–2e-3 nat measurement looks new.
- Not checked: whether the project's framework (e.g., JAX/Levanter in Marin, or PyTorch without autocast) runs norm statistics in bf16 by default. PyTorch autocast covers layer_norm but not rms_norm.
- CogView PB-relax details and Megatron flag behavior come from secondary or mirror sources; the official NVIDIA/Megatron-LM `arguments.py` was not read.
