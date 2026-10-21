# Multi-token prediction (MTP) and what LLM layers know about future tokens

Scope: survey for the "pls" project (NTP vs NTP + one DeepSeek-V3-style MTP module at weight 0.3, at 130m/300m/520m/1.2b; detached RMSNorm + vocab-head probes on every frozen layer for x_{t+1} and x_{t+2}).
Verification status: every arXiv id below was checked against the arXiv API or the paper page on 2026-10-21. Numbers marked **[verified]** were read from the paper text (HTML or PDF). Numbers marked **[secondary]** come only from a search summary or an abstract and were not checked against tables. Items marked **[UNCONFIRMED]** could not be checked.

---

## Q1. MTP papers and scale dependence: what helps, what hurts, where the future information lives

### Takeaway
Parallel-head MTP (Gloeckle-style: heads predict x_{t+2..t+n} from the same trunk state, without seeing x_{t+1}) hurts small models and helps only from about 1–3B parameters, mostly on generative code benchmarks. DeepSeek-V3's sequential MTP module, which also reads the embedding of the true x_{t+1}, leaves held-out BPB unchanged (0.729 vs 0.729; 0.658 vs 0.657) yet improves downstream benchmarks. That matches the pls finding: NTP loss is unchanged within 0.001 nats at 130m/300m/1.2b. The literature agrees that NTP-trained models place after-next ("t+2") information in middle layers. The last layers specialize for the next token, so after-next readability stalls or falls there.

### Cited Findings

**Foundational MTP objectives**
- ProphetNet (Qi, Yan, Gong, Liu, Duan, Chen, Zhang, Zhou, 2020, arXiv 2001.04063) introduced future n-gram prediction for seq2seq pretraining. It reported new SOTA on summarization and question generation at the same pretraining scale. — [arXiv 2001.04063](https://arxiv.org/abs/2001.04063)
- Blockwise parallel decoding (Stern, Shazeer, Uszkoreit, 2018, arXiv 1811.03115) predicted several future steps in parallel and verified them. It used up to 2x fewer decoding iterations at no quality loss. This is the ancestor of Medusa/MTP-for-speed. — [arXiv 1811.03115](https://arxiv.org/abs/1811.03115)
- TeaForN (Goodman et al., 2020, arXiv 2010.03494) re-feeds output vectors through the decoder N times, so h_t is trained against x_{t+1..t+N}. 苏剑林 frames it as "giving teacher forcing foresight". He describes a single h_t under teacher forcing as having no direct connection to y_{>t}. — [kexue.fm/archives/7818](https://kexue.fm/archives/7818) (TeaForN paper itself not fetched; id from the blog link)

**Gloeckle et al. 2024: "Better & Faster Large Language Models via Multi-token Prediction" (arXiv 2404.19737, ICML 2024)** — [arXiv](https://arxiv.org/abs/2404.19737), [HTML](https://arxiv.org/html/2404.19737v1)
- Design: n independent output heads on a shared trunk. To keep parameter count fixed, the model removes n−1 trunk layers when it adds n−1 head layers. **[verified]**
- Scale sweep (code, n=4 vs n=1, pass@1, Table S7) **[verified]**:
  - MBPP: 0.3B 1.8→1.0; 0.6B 4.7→3.0; 1.3B 6.8→7.4; 3B 11.1→12.7; 6.7B 23.9→26.0; 13B 26.0→30.5.
  - HumanEval: 0.3B 1.9→1.2; 0.6B 2.9→2.7; 1.3B 4.6→4.8; 3B 7.2→7.2; 6.7B 12.8→13.8; 13B 14.1→15.8.
  - So MTP is worse at 0.3B and 0.6B and crosses over around 1.3B–3B.
- Headline: at 13B, 12% more HumanEval and 17% more MBPP problems solved. 4-token models decode up to 3x faster with self-speculation. **[verified]**
- Natural language: on six multiple-choice NLP benchmarks, n=2 matches the NTP baseline throughout training and n=4 regresses slightly. Summarization ROUGE-L improves with n=2/n=4. On GSM8K, n=2 leads at 200B tokens, but n=4 is worse at 500B. **[verified]**
- Head variants (n=4, MBPP/HumanEval pass@1): parallel 33.8/24.0, linear heads 33.6/21.9, causal 31.9/20.9, anticausal 30.8/20.9. No variant clearly beat the parallel heads. **[verified]**
- Induction: on children's stories, 2-token prediction "vastly improves" induction for models up to about 30M non-embedding parameters. The advantage disappears at ≥100M. **[verified]**
- Information-theoretic argument (labelled "speculation" by the authors): H(X)+H(Y) = H(X|Y) + 2I(X;Y) + H(Y|X). 2-token prediction therefore doubles the weight on I(X;Y) between next and after-next tokens. n-token prediction weights "choice points" by n(n+1)/2. **[verified]**
- The paper contains no probing of its own representations. **[verified]**

**DeepSeek-V3 (DeepSeek-AI, 2024, arXiv 2412.19437)** — [HTML](https://arxiv.org/html/2412.19437v1)
- MTP module k: shared embedding and output head, one transformer block TRM_k, and projection M_k ∈ R^{d×2d} applied to [RMSNorm(h_i^{k−1}); RMSNorm(Emb(t_{i+k}))]. The module therefore sees the true next token. Depth D=1. λ=0.3 for the first 10T tokens, then 0.1 for the last 4.8T. The module is discarded at inference. **[verified]**
- Ablation (Table 4; baseline → w/ MTP) **[verified]**:
  - Small MoE (2.4B active/15.7B total, 1.33T tokens): Pile-test BPB 0.729→0.729; BBH 39.0→41.4; MMLU 50.0→53.3; DROP 39.2→41.3; TriviaQA 56.9→57.7; NQ 22.7→22.3; HumanEval 20.7→26.8; MBPP 35.8→36.8; GSM8K 25.4→31.4; MATH 10.7→12.6.
  - Large MoE (20.9B active/228.7B total, 540B tokens): BPB 0.658→0.657; BBH 70.0→70.7; MMLU 67.5→66.6; DROP 68.5→70.6; TriviaQA 67.0→67.3; NQ 27.2→28.5; HumanEval 44.5→53.7; MBPP 61.6→62.2; GSM8K 72.3→74.0; MATH 38.6→39.8.
- The MTP module predicts the second token with an 85–90% acceptance rate, which gives 1.8x TPS. **[verified]**
- The stated reasons are hedged by the authors: MTP "densifies the training signals" and "may enable the model to pre-plan its representations". **[verified]**

**MTP at small scale: negative and mixed results**
- KORMo (Kim et al., 2025, arXiv 2510.09426), Sec. 2.4: a 1.33B proxy trained on 60B tokens with only the objective changed. 9-task average: NTP 43.38 vs MTP 41.35. MTP is worse on ARC-E (65.24→49.12), HellaSwag (37.86→29.57) and PIQA (70.24→61.53), and better on BoolQ, OBQA and TruthfulQA. The authors conclude: "NTP is a more stable and efficient training objective under the conditions of a small-scale model and limited data." The MTP variant (parallel vs sequential) was not extracted. **[verified numbers; MTP architecture UNCONFIRMED]** — [arXiv HTML](https://arxiv.org/html/2510.09426)
- Token Order Prediction (Zuhri, Fuadi, Aji, arXiv 2508.19228, v2 Feb 2026). They trained NTP vs MTP at 340M/1.8B/7B. Lambada PPL: 30.34→35.31, 11.38→13.69 and 7.97→8.99, so MTP is worse at all three sizes. Lambada accuracy: 36.35→35.32, 49.58→47.93 and 55.89→53.13. TOP adds one extra unembedding head with a ListNet ranking loss over "how soon each token appears" and beats both: 7B Lambada PPL 7.64, accuracy 57.03. The paper also reports a DeepSeek-style MTP baseline ("DS-MTP"), but those numbers were not extracted. **[verified for NTP/MTP/TOP; DS-MTP numbers not read]** — [arXiv HTML](https://arxiv.org/html/2508.19228v2)
- Pre-training curriculum for MTP (Aynetdinov & Akbik, 2025, arXiv 2505.22757): 1.3B/3B models on MiniPile (~1.7B tokens). Static MTP trails NTP on the main head, e.g. 1.3B 4-token MiniPile BPB 1.18 vs 1.08. A forward curriculum (NTP→MTP) recovers 2.39% BPB. A reverse curriculum (MTP→NTP) recovers 5.52% BPB and 9.85% LAMBADA but loses the self-speculative speedup. Their hypothesis: small models have limited capacity to hold future-token information at each step. **[verified]** — [arXiv HTML](https://arxiv.org/html/2505.22757)
- AdaMTP (Cui et al., arXiv 2608.00434, Aug 2026). Abstract only: fixed-horizon MTP "injects noisy, conflicting training signals" across high-entropy boundaries. Because the heads share the backbone, the gradients "interfere with the model's core capabilities". AdaMTP masks MTP loss across detected boundaries. 7–12B fine-tuning only. **[secondary: abstract]** — [arXiv](https://arxiv.org/abs/2608.00434)

**Fine-tuning-time MTP / draft heads**
- Medusa (Cai, Li, Geng, Peng, Lee, Chen, Dao, 2024, arXiv 2401.10774): each head is softmax(W2·(SiLU(W1·h_t)+h_t)) on the last hidden state. W2 is initialized from the LM head and W1 from zero. Medusa-1 (frozen backbone) gives 2.18x speedup at MT-bench 6.23. Direct joint fine-tuning degrades quality to 5.925 vs 6.17. Medusa-2 recovers to 6.18 at 2.83x by keeping the LM loss, setting λ0=0.2, using a 4x head learning rate and warming up the heads. **[verified]** — [arXiv HTML](https://arxiv.org/html/2401.10774)
  - Per-head top-1 accuracy of about 60% for "next-next", top-5 >80%: from a survey only. **[secondary/UNCONFIRMED against Medusa tables]**
- EAGLE (Li, Wei, Zhang, Zhang, 2024, arXiv 2401.15077) uses feature-level autoregression with the token sequence shifted by one. It gives 2.7–3.5x on LLaMA2-Chat-70B. — [arXiv](https://arxiv.org/abs/2401.15077)
- EAGLE-3 (Li, Wei, Zhang, Zhang, 2025, arXiv 2503.01840, NeurIPS 2025) says top-layer features "correspond to the logits of the next token" and are "inherently limited to the next token". This makes next-next prediction from them "a significant challenge", so EAGLE-3 fuses low-, mid- and high-level features. Ablation on LLaMA-3.1-8B (MT-bench speedup / acceptance length τ), cumulative: EAGLE-2 3.16x/4.05 → remove feature constraint 3.82x/5.37 → + fused multi-layer features 4.40x/6.13. GSM8K: 3.39x/4.24 → 3.77x/5.22 → 4.48x/6.23. **[verified]** — [arXiv HTML](https://arxiv.org/html/2503.01840)
- On multi-token prediction for efficient LLM inference (Mehra, Alonso Garcia, Mauch, 2025, arXiv 2502.09419) **[verified]** — [arXiv HTML](https://arxiv.org/html/2502.09419):
  - NTP models can predict x_{t+2} by marginalizing over x_{t+1}. This works better with scale because larger models have sparser (lower-entropy) next-token distributions.
  - Intermediate layers of large models already closely match the final output by KL. They are specialized for NTP, so MTP heads on a frozen backbone do poorly.
  - Translation, 2nd-token top-5 accuracy: Pythia 410M baseline 0.682 / heads-only 0.500 / joint diff-LR+WHS 0.590; 1B 0.730/0.590/0.623; 1.4B 0.760/0.573/0.642; 2.8B 0.777/0.607/0.659. Next-token baseline accuracy: 0.827/0.847/0.866/0.881.
- Your LLM Knows the Future (Samragh et al., Apple, 2025, arXiv 2507.11851): appends k mask tokens and fine-tunes Tulu3-8B with gated LoRA, so NTP-token behaviour is unchanged, plus a sampler head. Reported ~5x faster code/math and ~2.5x for chat, "without any loss in quality". **[secondary]** — [arXiv](https://arxiv.org/abs/2507.11851)
- MuToR, "Multi-Token Prediction Needs Registers" (Gerontopoulos, Gidaris, Komodakis, 2025, arXiv 2505.10518, NeurIPS 2025) **[verified]** — [arXiv HTML](https://arxiv.org/html/2505.10518):
  - Registers are interleaved only at training time. Regular tokens cannot attend to registers, so NTP representations are untouched, and registers are discarded at inference.
  - Gloeckle-style heads in fine-tuning are mixed. Gemma-2B: GSM8K 38.87→40.66, 2M-GSM 69.02→69.02, AQUA-RAT 40.16→38.45. Llama-3-8B: 2M-GSM 87.33→86.35.
  - The Gloeckle-style head result degrades with horizon on 1M-GSM: dmax 2/3/4/6 → 66.69/66.36/65.53/65.55 vs NTP 66.09. MuToR scores 67.15–68.33.

**Other "beyond next token" objectives**
- Semformer (Yin, Ding, Song, Zhang, EMNLP 2024, arXiv 2409.11143): planning tokens in the prefix regress autoencoder latents of the future response. Near-perfect graph path-finding. A 125M OpenWebText pretraining was also run. **[secondary: abstract]** — [arXiv](https://arxiv.org/abs/2409.11143v1)
- TRELAWNEY, "Looking beyond the next token" (Thankaraj, Jiang, Kolter, Bisk, 2025, arXiv 2504.11336): rearranges training data by inserting lookahead spans in special tokens, with no architecture change. Gains on star-graph, CLRS-Text and TinyStories. **[secondary: abstract; numbers UNCONFIRMED]** — [arXiv](https://export.arxiv.org/abs/2504.11336)
- JTP, "Efficient Joint Prediction of Multiple Future Tokens" (Ahn, Lamb, Langford, 2025, arXiv 2503.21801): teacher-forces future tokens through a representation bottleneck. It claims to learn a short-horizon belief state that parallel MTP does not. Preliminary star-graph results only. **[secondary: abstract]** — [arXiv](https://arxiv.org/abs/2503.21801v1)
- Future Summary Prediction (Mahajan, Goyal, Youbi Idrissi, Pezeshki, Mitliagkas, Lopez-Paz, Ahuja, 2025, arXiv 2510.14751, ICLR 2026): predicts a summary of the long future (bag-of-words, or a reverse-LM embedding). Pretrained at 3B and 8B; beats NTP and MTP on math, reasoning and code. **[secondary: abstract; numbers not read]** — [arXiv](https://arxiv.org/abs/2510.14751)
- Pitfalls of next-token prediction (Bachmann & Nagarajan, 2024, arXiv 2403.06963): teacher forcing can fail to learn star-graph path-finding ("Clever Hans"), and teacherless multi-token training can fix it. — [arXiv](https://arxiv.org/abs/2403.06963)
- Planning via MTP (Zhong et al., 2025, arXiv 2509.23186) uses 100-node DAG path-finding. 1-token 89.31%, Meta-style 2-token 88.65%, DeepSeek-style 2-token 90.90%, and 2-token + Next-Token Injection (adding the true next-token embedding before the transfer layer) 92.65%. On degree-0/1 cases (next-token-like), Meta-style MTP is slightly below baseline (92.03 vs 92.58; 85.54 vs 86.57). **[verified]** — [arXiv HTML](https://arxiv.org/html/2509.23186v1)

### Inferences
- Gloeckle-style MTP hurts small models and DS-style MTP is neutral on BPB. The key difference is whether the auxiliary head sees x_{t+1}. A Gloeckle head must model the marginal p(x_{t+2}|x_{≤t}) from h_t, which competes with next-token capacity. A DeepSeek module receives Emb(x_{t+1}), so the trunk only needs to supply features for p(x_{t+2}|x_{≤t+1}). Many of these features are "breadcrumbs" the trunk computes anyway. Three results support this: the pls finding of NTP loss within 0.001 nats, DeepSeek's BPB 0.729→0.729, and Zhong et al.'s gain from injecting x_{t+1}. Isolating this needs a matched Gloeckle-head arm (see E5).
- At 130m–1.2b, pls sits below the Gloeckle code crossover (1.3–3B) and above the induction-benefit regime (<100M non-embedding). Benchmark gains from MTP are therefore not expected at pls sizes, and their absence is not a contradiction. DeepSeek's gains were at ≥2.4B active.
- Ordinary models "improve the after-next probe little over the last layers". This agrees with EAGLE-3 (top-layer features are next-token logits), Mehra et al. (late/intermediate layers are specialized for NTP) and Future Lens (t+2 information peaks mid-depth in GPT-J; see Q2).
- MTP adds after-next information only past ~60% depth, and the amount grows with size (0.18/0.26/0.38 nats at the last layer). One explanation: the MTP module reads only the last-layer state, so the gradient pressure toward linearly readable t+2 features is strongest in the top layers. Wu et al. (Q3) find that pre-caching grows with scale, which matches the growth with size.

### Gaps
- No paper found that runs a matched-data, matched-seed size sweep of DeepSeek-style (sequential, x_{t+1}-conditioned) MTP below 2B. DeepSeek ablated only at 15.7B/228.7B total. The DS-MTP rows of TOP (arXiv 2508.19228) may hold this but were not extracted.
- Medusa per-head accuracies were not confirmed from the paper's tables.
- No public per-layer analysis of a DeepSeek-V3 MTP-trained model was found.

---

## Q2. Measuring future-token information per layer: methods and pitfalls

### Takeaway
Three method families measure future-token information per layer: linear/affine probes (logit lens, tuned lens, the pls probes), learned-prompt causal interventions (Future Lens), and synthetic/behavioural probes (myopia gap, NextLat offset probes). Probe class strongly changes the answer. In GPT-J at t+2, linear probes reach 27.5–29.2% precision@1, a learned prompt reaches 48.4%, and a bigram reaches 20.1%. Under the data-processing inequality, every layer holds at most the information of its prefix. A layer-wise probe curve therefore measures V-usable information (Xu et al. 2020), not Shannon information. This matches the pls Lean result.

### Cited Findings

**Future Lens (Pal, Sun, Yuan, Wallace, Bau, CoNLL 2023, arXiv 2311.04897)** **[verified from PDF]** — [arXiv](https://arxiv.org/abs/2311.04897)
- Model GPT-J-6B (28 layers). Probes predict GPT-J's own output N tokens ahead (N=0 next, N=1 after-next, …) from a single hidden state h^l_T.
- Methods:
  - Linear "HS" maps h^l_T to the final-layer state at T+N.
  - Linear "VOCAB" maps directly to vocabulary logits.
  - Fixed-prompt causal intervention transplants h^l_T into a generic prompt.
  - Learned-prompt intervention trains a soft prefix through the frozen model with a KL loss.
- Data: probes trained on 100,000 Pile tokens (10,000 for learned prompts) and tested on 1,000 tokens. Train and test were drawn **only from positions where GPT-J's prediction was correct**, and accuracy is measured against GPT-J's own greedy output.
- Best precision@1 at N=1 (after-next): learned prompt 48.4%, fixed prompt 20.8%, linear HS 29.2%, linear VOCAB 27.5%. The Pile bigram baseline at N=1 is 20.1%. Learned prompts beat the best baseline by 24.8% (p@1) on average.
  - Caveat: the PDF text extraction scrambled Table 2's columns. The 48.4% figure matches the abstract ("more than 48%"), but the mapping of the other cells to N=2/N=3 is uncertain.
- "Predictive accuracy of the learned prompt model peaks at the middle-layer hidden states … very different from the immediate next-token N=0, in which accuracy peaks at the last layer". The best learned-prompt layer for N=1 is l=14 of 28.
- Calibration: N=1 accuracy is 26%/57%/77%/95% for model confidence bins of 0–30/30–60/60–90/90–100%. Named-entity subset: 44/42/37% for N=1/2/3.
- Probes trained for N=1 generalize to other N. Probes trained for other N do poorly at N=1.

**Wu, Morris, Levine (COLM 2024, arXiv 2404.00859), "Do Language Models Plan for Future Tokens?"** **[verified from PDF]** — [arXiv](https://arxiv.org/abs/2404.00859)
- Two hypotheses:
  - Pre-caching: off-diagonal gradients make h_t compute features useful only for later positions.
  - Breadcrumbs: features useful at t are also the ones useful at t+τ.
- "Myopic descent" removes gradients from loss ℓ_j to hidden states at earlier positions i<j. The myopia gap measures pre-caching.
- GPT-2 124M from scratch: vanilla 3.28 CE, myopic 3.40 CE (gap 0.12), local-myopic 3.26 (local myopia bonus 0.02), transformer-bigram 5.33. On PG-19, aggregate CE is 4.67 vs 4.77. The myopic model is better at the start of sequences and falls behind as context grows.
- Pythia 14M–2.8B (fine-tuned from pretrained checkpoints, 10M sequences of 64 tokens): "the importance of pre-caching increases with scale, becoming non-negligible with larger models, e.g. Pythia 2.8B". Exact per-size numbers are only in figures (not extracted).
- Linear probes for meaningfulness require "no pre-existing linear relationship between the inputs and the quantities we are probing for", citing V-information (Xu et al. 2020).

**Probing theory and pitfalls**
- Xu, Zhao, Song, Stewart, Ermon (ICLR 2020, arXiv 2002.10689), "A Theory of Usable Information Under Computational Constraints": predictive V-information depends on the observer's function class V. Unlike Shannon MI, and in violation of the data-processing inequality, it "can be created through computation". This formally reconciles "no layer adds information" (the pls Lean proof) with "probe loss improves with depth". — [arXiv](https://arxiv.org/abs/2002.10689)
- Pimentel, Valvoda, Hall Maudslay, Zmigrod, Williams, Cotterell (ACL 2020, arXiv 2004.03061): if probing estimates MI, the most powerful probe should be used. By the DPI, contextual representations cannot hold more information about the target than the sentence itself, so the interesting quantity is ease of extraction. — [arXiv](https://arxiv.org/abs/2004.03061)
- Hewitt & Liang (EMNLP 2019, arXiv 1909.03368), control tasks: high probe accuracy alone does not show that a representation encodes structure. They report selectivity (task accuracy minus control-task accuracy) and favour low-capacity probes. — [arXiv](https://arxiv.org/abs/1909.03368)
- Hewitt, Ethayarajh, Liang, Manning (EMNLP 2021, arXiv 2109.09234), conditional probing: measures V-information beyond a baseline representation. After conditioning on non-contextual embeddings, properties appear at deeper layers than plain probes show. — [arXiv](https://arxiv.org/abs/2109.09234)
- Tuned lens (Belrose et al., 2023, arXiv 2303.08112): per-layer trained affine translators decode intermediate states more reliably and with less bias than the logit lens. The pls RMSNorm + vocab-head probe is a tuned-lens-like affine probe per layer. — [arXiv](https://arxiv.org/abs/2303.08112)
- Next-Latent Prediction (Teoh, Tomar, Ahn, Hu, Pearce, Sharma, Krishnamurthy, Islam, Lamb, Langford; arXiv 2511.05963, v4 Jun 2026) **[verified from HTML text; no exact numbers in text]** — [arXiv HTML](https://arxiv.org/html/2511.05963):
  - Setup: TinyStories (1,000-token vocabulary, length 256, 100k steps). 20 linear probes, one per offset 1..20, on the frozen **final-layer** state, comparing GPT (NTP), MTP, JTP, BST and NextLat.
  - "Extra token-level objectives in BST, MTP and JTP noticeably degraded next-token prediction (offset 1)". MTP/JTP probe quality drops sharply with offset.
  - NextLat (latent-dynamics regression plus KL to the frozen head) matches GPT at offset 1 and is best up to 20 tokens ahead.
- Unlocking the Future (Men, Cao, Jin, Chen, Liu, Zhao, 2024, arXiv 2406.16033), on planning tasks: "middle and upper layers encode a few short-term future decisions" when planning succeeds. **[secondary: abstract]** — [arXiv](https://arxiv.org/abs/2406.16033)
- Emergent Response Planning (Dong et al., 2025, arXiv 2502.06258): small (MLP) probes on prompt-time hidden states predict global attributes of the response (length, reasoning steps, answer choice, confidence). The paper also studies scaling with model size. **[secondary: no numbers extracted]** — [arXiv](https://arxiv.org/abs/2502.06258)
- Anthropic, "Tracing the thoughts of a large language model" / "On the Biology of a LLM" (2025): in Claude 3.5 Haiku, the model selects a rhyme word ("rabbit") before writing the line. Suppressing that concept, or injecting "green", changes the line. This is causal evidence of multi-token lookahead in features. **[secondary: blog summary]** — [Anthropic](https://www.anthropic.com/research/tracing-thoughts-language-model)
- Mehra et al. (arXiv 2502.09419, above): the marginalization baseline Σ_{x_{t+1}} p(x_{t+1}|x_{≤t}) p(x_{t+2}|x_{≤t+1}) gives the model's own optimal t+2 prediction. Frozen-backbone heads fall well short of it (e.g. 2.8B: 0.607 vs 0.777 top-5). — [arXiv HTML](https://arxiv.org/html/2502.09419)

### Inferences
- **Probe capacity.** In GPT-J, the gap between linear probes (~28–29%) and learned prompts (48.4%) at t+2 shows that linear readability underestimates t+2 information by a large factor. The pls statement "MTP adds after-next information only past ~60% depth" is a statement about affine-readable information. It could partly reflect where MTP makes t+2 information linearly readable, not where it exists. The Lean result already implies this. Writing claims as "linearly decodable (V-information with V = RMSNorm∘linear)" puts them at the right level.
- **Selection and target pitfalls.** Future Lens trains and tests only on positions where the model is correct, and scores against the model's own output, not the corpus. Both inflate accuracy. The pls probes use corpus targets on held-out text, which is the cleaner choice; keep it.
- **Single-position pitfall.** Wu et al.'s breadcrumbs mean that t+2 information can sit in earlier positions' states and be gathered later by attention. A single-position probe at t misses that. This is fine for comparing NTP vs MTP at the same position, but it is not a measure of "what the model knows".
- **The MTP module is outside the probe family.** The DeepSeek module conditions on Emb(x_{t+1}). The pls after-next probe estimates p(x_{t+2}|h_t) ≈ the marginal over x_{t+1}. The two targets differ in entropy by I(x_{t+1}; x_{t+2} | x_{≤t}). A conditional probe (Hewitt et al. 2021) that also sees Emb(x_{t+1}) is the matched family (E2).
- **Ceiling reference.** The marginalization baseline (Mehra et al.) per pls model gives an upper reference for the after-next probe in nats. The gap "probe − ceiling" per layer is more interpretable than raw probe loss across sizes, because the ceiling itself improves with size.

### Gaps
- No paper found that does a layer-by-layer t+2 probe comparison between NTP and MTP-trained models from scratch at matched data. NextLat probes only the final layer, and Future Lens studies only an NTP model. The pls result looks new in this respect.
- Exact Pythia myopia-gap numbers per size (Wu et al. Fig. 5/7) were not extractable from the text.
- Future Lens per-layer curves for the linear probes (the closest analogue to pls) are only in figures. The text gives no peak layer for linear probes at N=1.

---

## Q3. Do auxiliary future losses hurt or help intermediate next-token representations?

### Takeaway
The evidence is mixed and mostly indirect. Shared-backbone future losses can hurt next-token quality:
- NextLat: MTP/JTP degrade the final-layer offset-1 probe.
- Medusa: naive joint fine-tuning drops MT-bench from 6.17 to 5.925.
- KORMo and TOP: parallel MTP raises PPL at 340M–7B.
- AdaMTP: claims gradient interference.

Methods that protect the NTP path, or that condition on x_{t+1}, avoid the harm: DeepSeek BPB is unchanged, and so are MuToR registers, gated LoRA and Medusa-2. No paper found measures the effect on *mid-depth* next-token probes, so the pls 1.2b result (mid-depth next probes worse by up to 0.15 nats under MTP, final loss unchanged) has no direct precedent.

### Cited Findings
- NextLat (arXiv 2511.05963): MTP, JTP and BST "noticeably degraded next-token prediction (offset 1)" for a linear probe on the frozen final layer (TinyStories). NextLat, which regresses latents and adds a KL term to the frozen head, does not. **[verified text; no numbers]** — [arXiv HTML](https://arxiv.org/html/2511.05963)
- Medusa (arXiv 2401.10774): direct joint fine-tuning with heads gives MT-bench 5.925 vs 6.17. Medusa-2 avoids this with the LM loss kept, λ0=0.2, a 4x head learning rate and head warm-up (6.18). **[verified]** — [arXiv HTML](https://arxiv.org/html/2401.10774)
- Mehra et al. (arXiv 2502.09419): joint fine-tuning beats heads-only at every size but stays below marginalization. Weighted hidden states (WHS: heads read a learned mix of layers) with differential learning rates is best. **[verified]** — [arXiv HTML](https://arxiv.org/html/2502.09419)
- AdaMTP (arXiv 2608.00434): shared-backbone gradients from fixed-horizon MTP "interfere with the model's core capabilities". The remedy is loss masking at boundaries. **[secondary: abstract]** — [arXiv](https://arxiv.org/abs/2608.00434)
- Curriculum (arXiv 2505.22757): reverse curriculum (MTP→NTP) gives the best NTP metrics (5.52% BPB, 9.85% LAMBADA over static MTP). Ending training on pure NTP undoes most of the harm. **[verified]** — [arXiv HTML](https://arxiv.org/html/2505.22757)
- DeepSeek-V3 (arXiv 2412.19437): λ is lowered from 0.3 to 0.1 for the last 4.8T of 14.8T tokens, with no change in BPB at either ablation scale. **[verified]** — [arXiv HTML](https://arxiv.org/html/2412.19437v1)
- MuToR (arXiv 2505.10518): register tokens isolate the MTP signal, since regular tokens never attend to registers. It beats Gloeckle-style heads, which fall below NTP at dmax≥4 on 1M-GSM. **[verified]** — [arXiv HTML](https://arxiv.org/html/2505.10518)
- Gloeckle et al.: the n=4 model regresses slightly on NLP multiple-choice while n=2 matches. The trunk loses n−1 layers to the heads at fixed parameters, which confounds "MTP hurts" with "shallower trunk". **[verified]** — [arXiv HTML](https://arxiv.org/html/2404.19737v1)
- Zhong et al. (arXiv 2509.23186): in theory, the second-step loss updates the backbone weights and induces "spurious adjacency". Injecting the true next-token embedding before the transfer layer gives a residual shortcut and better results. **[verified]** — [arXiv HTML](https://arxiv.org/html/2509.23186v1)
- LayerSkip (Elhoushi et al., 2024, arXiv 2404.16710): early-exit losses on intermediate layers plus layer dropout enable self-speculation up to 2.16x. This is a counterpart that *trains* mid-depth layers to be next-token readable. — [arXiv](https://arxiv.org/abs/2404.16710)
- Final-layer mechanisms that write next-token-specific structure:
  - Lad, Gurnee, Tegmark et al. (arXiv 2406.19384) propose four inference stages, ending in "residual sharpening", where final layers suppress features that add noise to the next-token prediction. Deleting or swapping middle layers is benign and final layers are fragile. — [arXiv](https://arxiv.org/abs/2406.19384)
  - Stolfo et al. (NeurIPS 2024, arXiv 2406.16254): entropy neurons write into an unembedding null space and act through the final LayerNorm scale. Token-frequency neurons push the output toward or away from the unigram distribution. — [arXiv](https://arxiv.org/abs/2406.16254v2)
  - Gurnee et al. (2024, arXiv 2401.12181) find universal neurons in GPT-2, including prediction and suppression neuron families in late layers. — [arXiv](https://arxiv.org/abs/2401.12181)
  - The details of prediction/suppression neurons are from memory of the paper; only the abstract was verified. **[UNCONFIRMED in this session]**

### Inferences
- **The 1.2b mid-depth harm** (next probes worse by ≤0.15 nats, final loss unchanged) fits a capacity-reallocation story. MTP pushes mid-to-late layers toward t+2-useful features. Mid-depth states then become less affine-readable for x_{t+1}, while the final layers still finish the next-token computation. Under V-information this is a change in *where* x_{t+1} becomes readable, not a loss of information. A useful precise claim: "MTP delays linear next-token readability at mid depth at 1.2b without changing final next-token loss." It grows with size because pre-caching pressure grows with scale (Wu et al.) and DeepSeek-style MTP rewards it.
- **"Last layers write away from after-next directions"** (the queued 130m test) has two mechanistic candidates: residual sharpening (Lad et al.) and suppression/entropy neurons (Stolfo et al.; Gurnee et al.). EAGLE-3's claim that top-layer features "correspond to the logits of the next token" is the speculative-decoding version of the same idea. A clean test: project each final-layer block's write Δh onto the after-next probe's row space (W_probe rows for the top-k t+2 tokens) and compare with the next probe's row space.
- **Gloeckle-style negatives at small scale are partly confounded** by the trunk losing n−1 layers. DeepSeek-style MTP (and pls) adds a module without removing trunk depth, which is another reason pls sees no NTP loss change.

### Gaps
- No paper found measures intermediate-layer next-token probe loss as a function of an auxiliary future loss. The pls 1.2b observation has no direct precedent in this search.
- No paper found tests whether the harm reverses with λ annealing (DeepSeek's 0.3→0.1) at the level of mid-depth probes.

### Concrete next experiments (mapped to pls findings)
- **E1. Marginalization ceiling** (Mehra et al.). Per model, compute the after-next loss of Σ_{top-K x_{t+1}} p(x_{t+1}|·) p(x_{t+2}|·, x_{t+1}) (K≈20, renormalized) on the probe test set. Report each layer's after-next probe as a gap to this ceiling. This separates "MTP adds info" from "MTP makes the marginal more readable", and normalizes across sizes.
- **E2. Conditional (DeepSeek-family) probe** (Hewitt et al. 2021). On every frozen layer, train a probe on [RMSNorm(h_l); RMSNorm(Emb(x_{t+1}))] → linear → vocab for x_{t+2}, for NTP and MTP models. If the MTP advantage over NTP shrinks under this probe, MTP mainly built features *complementary to x_{t+1}* (what its module needs), not marginal t+2 information.
- **E3. Probe-capacity control** (Future Lens; Hewitt & Liang). At 3–4 layers (30%, 60%, 80%, 100% depth) of 300m and 1.2b, add a 2-layer MLP probe (or a learned-prompt probe) plus a control task (shuffled-target selectivity). Check that "after-next gain only past ~60% depth" survives higher capacity.
- **E4. Myopia gap** (Wu et al.). Train myopic-descent NTP at 130m/300m, same data. Report (myopic − vanilla) NTP loss and the per-layer after-next probe. Test whether MTP's late-layer after-next gain lies in the same subspace that myopic training removes (pre-caching), using CCA or probe-direction overlap.
- **E5. Head-type arm.** One Gloeckle parallel head (no x_{t+1} input, no trunk depth removed) and/or one TOP head at 300m with the same data and seed. Prediction: NTP loss rises, as in TOP and KORMo, and mid-depth next probes degrade more than with DS-style MTP. This isolates "conditioning on x_{t+1}" as the reason pls sees no NTP cost.
- **E6. λ and stop-gradient ablations for the 1.2b mid-depth harm.** Three variants: (a) λ=0.1, DeepSeek's late value; (b) λ 0.3→0 reverse curriculum for the last 10% of tokens (Aynetdinov & Akbik); (c) stop the MTP gradient below layer ⌊0.6L⌋. Measure mid-depth next-probe loss and last-layer after-next gain. Also log per-layer cosine between ∇NTP and ∇MTP on trunk parameters (AdaMTP's interference claim).
- **E7. Final-layer write-away** (the queued 130m run, extended). Decompose final-k blocks' residual writes onto after-next vs next probe directions, and run zero/mean-ablation of the last block's MLP (Lad et al.-style layer deletion). If the after-next probe loss improves when the final MLP is ablated, the last layers actively suppress t+2-readable directions (residual sharpening / suppression neurons).
- **E8. Multi-offset probes** (NextLat). Probe offsets 1..8 on the final and mid layers for NTP vs MTP at 1.2b. NextLat predicts that MTP's advantage collapses beyond offset 2. If it does not, MTP built belief-like state.
- **E9. Downstream check at the largest size only.** HumanEval/MBPP/GSM8K at 1.2b. Gloeckle's crossover (1.3–3B) and DeepSeek's code/math gains predict at most small gains. This would make the negative result complete: MTP changes internals at 1.2b without changing loss, and probably not benchmarks.
