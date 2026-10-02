# pls (per-layer supervision) — reading notes, 2026-09-29

Sources read in full: blog llal/ (+78 figures in img1/), blog llal-change/ (+4 figures), both Notion originals
(notion_llal.txt, notion_change.txt; only content diff: the 2T-run limitation sentence was updated — "second MoE layer
is about to receive signals at the end of training; the first MoE layer keeps collapsing"), arXiv 2504.15471 (Bigram
Subnetworks), 2309.04827 (Neurons in LLMs: Dead, N-gram, Positional), 2601.07372 (Engram). Texts under this dir.

## LLAL recipe (as published)
- L = CE(softmax(W_lm h^(L)), y) + λ_aux · CE(softmax(W_lm h^(ℓ)), y). W_lm shared (no new head). Same targets.
- h^(ℓ) = hidden state output by layer ℓ (residual stream after the block). Final-norm use NOT specified (formula shows
  W_lm h^(ℓ) directly; their Appendix G decoding-CE probe does use final norm + LM head).
- No stop-gradient: "its error can be reduced only by the network prefix and the shared LM head".
- ONE layer only in every reported run. Main: ℓ=1 (first MoE layer; L0 dense). PoC: ℓ=5, ℓ=3, constant λ=0.1, 11k steps.
- Schedule: λ starts 0.1, linear anneal to 0 over window from step 0, then removed. Windows: 40k/90k (Exp6-1),
  10k/90k (Exp6-2), 4k/135k (180B Exp7; 2T Exp10b), 2k/90k (stopped at 40k, = 10k run). Insensitive to window.
- Cost: <0.3% extra train compute (their claim). Hurts pipeline parallelism (cross-stage comm) -> reason to shorten.
- 60B setup: 32L, MLA 16 heads, hidden 1536, L0 dense + 31 MoE layers (768 routed + 1 shared, top-8, sigmoid,
  aux-free bias 1e-3), ~1B active, AdamW wd 0.1 eps 1e-8, cosine 90k steps, warmup 2k, seq 8K, batch 384, ~280B tok.

## Numbers
- 60B AdamW: val 1.725 -> 1.701 (Δ −0.0235 10k / −0.0241 40k). MMLU/MMLU-Pro 0.5162/0.1899 -> 0.5555/0.2306 (10k),
  0.5214/0.2119 (40k: same val loss, smaller MMLU gain).
- 60B Muon: Muon alone −0.0206 / −0.0194 vs AdamW; +LLAL −0.0440 / −0.0445 => LLAL adds ≈ −0.024 on Muon too;
  MMLU +0.014, MMLU-Pro +0.006/+0.007.
- 180B (40L, WSD 135k, ~850B tok, 4k window): val ≈ 1.557 -> 1.536; MMLU +0.028, MMLU-Pro +0.048.
- 2T (60B, eps 1e-12, 4k window): val gap −0.0294 at 300k steps; MMLU 0.625 -> 0.656, MMLU-Pro 0.280 -> 0.337.
- Masking: baseline L1-L3 routed masks do nothing; LLAL Exp6-2 mask L1-L2 -> MMLU 0.258 (chance).
- Load balance (MaxVio/MinVio) much better with LLAL.
- Baseline alternatives: 0.1x router LR (+0.0095 val, MMLU up), eps 1e-12 (norms rescued, val flat, function not),
  Muon (norms healthy, masking shows early layers still unused), RMT-4x (−0.0098), AttnRes (unstable).

## Mechanism (their hypothesis)
- Critical window (~first 2% of steps) decides layer-wise dynamics. Lower layers first learn real function, then
  function migrates upward (reallocation); lower pathway loses causal role; weight decay + AdamW eps finish it.
- Aux = "protected local demand" upper layers cannot satisfy; lower layers become suppliers of contextual features;
  once upper layers depend on them, main loss keeps the pathway after removal. After removal: decoding CE rebounds
  0.5–0.9 nats, direct contribution +4.6 -> +2.0 nats, but ablation ΔNLL keeps rising (0.33 -> 0.39).
- Follow-up: rescued early experts carry rare knowledge (PopQA rare-name ΔNLL at L1/L2: 0.944/2.158 vs common
  0.025/0.108, AdamW LLAL; Muon L1 0.012 -> 1.169). Engram before L1 = healthier early experts; before L14 = worse.
- Their own open question: "weak signal issue inherent to both MoE and dense... validity on dense warrants
  investigation" (inline comment). Muon/eps fixes rescue norms but not function.

## Kaiyue dense baseline facts relevant to per-layer LLAL
- Qwen3 dense, hybrid_norm (pre + post norm on each sublayer), untied lm_head, final RMSNorm (adam, free gain).
- MuonH: every hnn.Linear norm-pinned at init + orthogonalized normalized updates; lm_head AdamH (norm-pinned);
  embeddings + norm gains Adam (eps 1e-20/1e-15). => LLAL's MoE failure mode (eps-capped updates + WD shrink ->
  norm collapse) is structurally impossible here; a dense test isolates the functional/critical-window claim.
- Existing hooks: objective_qwen3.forward_with_aux (scan_via returns every layer's output), aux_gate (loss-level
  anneal), levanter.trainer.current_train_step() (step-based schedule, used by sampled softmax).
- Sizes: 130m 6L (57 min), 300m 12L (~5 h), 520m 24L (~18 h), 1_2b 16L (~51 h) on 4xH100.
- 130m noise: 8-run baseline pool c4_en bpb mean 1.16327, sd 0.00035; macro sd 0.007–0.010. Verdict needs n>=4/arm.
- Hill-climb lessons: early-gated aux mostly "free but useless"; MTP via shared lm_head hurt even gated (+0.0014).

## Per-layer cost (fused CE = one lm_head matmul per aux layer; FLOPs/token incl. fwd+bwd)
| size | L | aux | per-step x in window | +total @10% window | @2.5% |
| 130m | 6 | 5 | 3.9x | +29% | +7% |
| 300m | 12 | 11 | 4.9x | +39% | +10% |
| 520m | 24 | 23 | 5.5x | +45% | +11% |
| 1_2b | 16 | 15 | 3.7x | +27% | +7% |

## Open decisions (to user)
weights (0.1 each vs 0.1 total), window (10% vs 2.5%), controls (single-layer ℓ=1), ladder 130m -> 300m -> 520m,
final-norm sharing, metric (Paloma macro + c4_en bpb).
