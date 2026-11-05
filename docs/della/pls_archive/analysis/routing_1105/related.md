# Related work check: learned routing between residual paths

Our claim. In 48-layer language models trained with MuonH on FineWeb-Edu, several residual paths lower held-out loss. Removing the learned routing between the paths does not raise the loss. HC with identity mixing, or with one learned number, is better than HC. mHC with identity mixing instead of Sinkhorn mixing is the same as mHC. AttnRes with input-independent weights is better than AttnRes.

Method. Every source below was fetched on della-vis1 on 2026-11-04 (arXiv abstract page, then arXiv HTML, or the PDF when the HTML was broken). Text is in `txt/`. Every number below comes from that text. A source marked NOT OPENED was not read.

## 1. Hyper-Connections (arXiv 2409.19606)

- Title: "Hyper-Connections".
- Authors: Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou.
- Venue and year: arXiv, v1 2024-09-29, last version 2025-03-18. The arXiv page and the HTML text give no venue.
- Setting: OLMo-1B dense (16 layers), OLMo-7B, OLMoE-1B-7B. 500B tokens. Expansion rate n = 4 by default. The optimizer is the OLMo recipe (Sec. 4, "We employ the experimental setup outlined by OLMo"); the text names no optimizer.
- Static vs dynamic (Table 2, OLMo-1B, V2/V3 eval loss):
  - Baseline 2.811 / 2.544.
  - SHC×2 2.799 / 2.538; DHC×2 2.802 / 2.534; DHC×2 without tanh 2.792 / 2.529.
  - SHC×4 2.791 / 2.528; DHC×4 2.781 / 2.515; DHC×4 without tanh 2.779 / 2.516.
  - Text: "At an expansion rate of 2, the improvements of DHC and SHC are similar. However, at an expansion rate of 4, DHC performs notably better than SHC." The gap at n = 4 is 0.010 (V2) and 0.013 (V3). Static HC already takes 67% (V2) and 55% (V3) of the DHC×4 gain over the baseline (my arithmetic).
- Frozen mixing (Table 3, OLMo-1B-DHC×4; "✗" means "not trainable from initialization"). With tanh, all trainable: 2.781 / 2.515. Width connections WC frozen: 2.802 / 2.532. B frozen: 2.783 / 2.520. Text: "not training WC leads to significant performance declines, with the V2 loss increasing by 0.021 and the V3 loss by 0.017 ... the impact is less pronounced when B is not trained."
  - Note: frozen WC here is the Eq. 14 initialization. A_r is the identity, and A_m picks stream k mod n. So this row is close to our "identity mixing" arm. The paper does not say whether the dynamic (tanh) part of WC is also frozen. In their run it is worse than learned mixing, though it still beats the baseline on V2 (2.802 vs 2.811) and V3 (2.532 vs 2.544).
- No mixing at all: the paper has no n > 1 run without A_r. n = 1 (Table 1) is worse than the baseline: DHC×1 2.819 / 2.556 vs 2.811 / 2.544.
- What mixing does, per the paper: HC can "adjust the strength of connections between features at different depths and dynamically rearrange layers" (abstract). Sec. 3 calls this "sequential-parallel duality". Appendix: "SHC shares exactly the same connection patterns" as DHC (Fig. 13).
- Versus our claim: it disagrees. Freezing the width connections at identity-like values costs 0.021 V2 loss in their run, and static mixing is worse than dynamic mixing at n = 4. Setting difference: 16 layers rather than 48, the OLMo recipe rather than MuonH (the paper does not name the optimizer), Dolma rather than FineWeb-Edu, and 500B tokens.

## 2. mHC: Manifold-Constrained Hyper-Connections (arXiv 2512.24880)

- Title: "mHC: Manifold-Constrained Hyper-Connections".
- Authors: Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyuan Yuan, Lean Wang, Wenfeng Liang (DeepSeek).
- Venue and year: arXiv, v1 2025-12-31, last version 2026-01-05. No venue given.
- Setting (Appendix A.1, Table 5): DeepSeek-V3-style MoE models of 3B, 9B and 27B total parameters, with 12, 18 and 30 layers, trained on 39.3B, 105B and 262B tokens. A separate 3B model is trained on 1.05T tokens. Optimizer: AdamW, betas (0.9, 0.95). n = 4. Sinkhorn-Knopp t_max = 20.
- What the constraint does, per the paper: it projects H_res onto doubly stochastic matrices "to restore the identity mapping property" (abstract). The spectral norm is ≤ 1, and the product of doubly stochastic matrices stays doubly stochastic, so the composite map cannot blow up (Sec. 4.1). Composite gain falls from "nearly 3000 in HC" to "approximately 1.6" (Sec. 5.4, Fig. 7).
- Why not the identity, per the paper (Sec. 4.1): "While the original identity mapping ensures stability by enforcing H_res = I, it fundamentally precludes information exchange within the residual stream, which is critical for maximizing the potential of multi-stream architectures." This sentence is the paper's argument; no experiment in the paper sits next to it.
- Identity-mixing evidence (Table 1, "Ablation Study of HC Components", called "preliminary experiments"; model size not stated). A disabled mapping is replaced by a fixed one: 1/n for H_pre, ones for H_post, "the identity matrix for H_res". Absolute loss gap: all fixed 0.0; H_res learned −0.022; H_res and H_pre learned −0.025; all three learned −0.027. Text: "the residual mapping H_res yields the most significant performance gain."
  - Note: this ablation is on unconstrained HC, not on mHC. The paper has no run of mHC with the Sinkhorn mixing replaced by the identity.
- Compared with: the residual baseline and HC only. 27B: mHC final loss 0.021 lower than the baseline (Sec. 5.2, Fig. 5a). HC shows "an unexpected loss surge around the 12k step" (Sec. 3.1, Fig. 2). Downstream (Table 4, 27B; baseline / HC / mHC): BBH 43.8 / 48.9 / 51.0, DROP 47.0 / 51.6 / 53.9, GSM8K 46.7 / 53.2 / 53.8, MMLU 59.0 / 63.0 / 63.4.
- Versus our claim: it disagrees. Their Table 1 says that learned H_res beats identity H_res by 0.022 loss in HC. That contradicts our HC result directly. It does not test our mHC arm (Sinkhorn mixing replaced by the identity). Setting difference: MoE, AdamW, at most 30 layers rather than 48 dense layers with MuonH; the Table 1 scale and token count are not stated.

## 3. mHC-lite (arXiv 2601.05732)

- Title: "mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations".
- Authors: Yongyi Yang, Jianyang Gao.
- Venue and year: arXiv, 2026-01-09. The HTML has the keyword line "Machine Learning, ICML" from the template. No acceptance is stated.
- Setting (Appendix, Tables 2 and 3): nanoGPT, AdamW (β 0.9, 0.95), cosine schedule, 10,000 iterations, sequence length 1024, batch 16 per GPU on 8 A100s. S / M / L models have 6 / 12 / 24 layers and widths 512 / 768 / 1024. Data: OpenWebText and FineWeb-Edu. Tokens are not stated. Without gradient accumulation they would be 16 × 8 × 1024 × 10,000 ≈ 1.3B; the paper does not mention accumulation.
- Claim (abstract): mHC-lite builds H_res as a convex combination of the n! permutation matrices, so it is exactly doubly stochastic. It "matches or exceeds mHC in performance while achieving higher training throughput with a naive implementation and eliminating the residual instabilities observed in both HC and mHC."
- Numbers (Table 1, final validation loss; Residual / HC / mHC / mHC-lite):
  - OpenWebText S 3.562 / 3.471 / 3.469 / 3.467; M 3.336 / 3.264 / 3.259 / 3.255; L 3.242 / 3.248 / 3.198 / 3.198.
  - FineWeb-Edu S 3.536 / 3.473 / 3.473 / 3.477; M 3.321 / 3.273 / 3.243 / 3.249; L 3.240 / 3.244 / 3.204 / 3.185.
  - At L (24 layers), HC is no better than the residual baseline on either dataset (3.248 vs 3.242; 3.244 vs 3.240). mHC and mHC-lite are about 0.04 to 0.055 better.
- Stability (Sec. 3.1, Fig. 3): column sums of the composite mHC matrix "may deviate from 1 by up to 220% in a 24-layer network".
- Identity mixing: the paper has no identity-mixing or static arm. It compares only Residual, HC, mHC and mHC-lite.
- Versus our claim: it neither supports nor contradicts our claim, because it has no identity arm. Two facts are close to ours: unconstrained HC loses its gain at 24 layers, and two different ways to build a doubly stochastic H_res give the same loss. Setting difference: at most 24 layers, AdamW, about 1B tokens (estimated), FineWeb-Edu as in our runs.

## 4. Attention Residuals (arXiv 2603.15031)

- Title: "Attention Residuals".
- Authors: Kimi Team; Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo Chen, Bohong Yin, Yutian Chen, Junjie Yan, Ming Wei, Y. Zhang, Fanqing Meng, Chao Hong, Xiaotong Xie, Shaowei Liu, Enzhe Lu, Yunpeng Tai, Yanru Chen, Xin Men, Haiqing Guo, Y. Charles, Haoyu Lu, Lin Sui, Jinguo Zhu, Zaida Zhou, Weiran He, Weixiao Huang, Xinran Xu, Yuzhi Wang, Guokun Lai, Yulun Du, Yuxin Wu, Zhilin Yang, Xinyu Zhou (36 named authors after "Kimi Team", in arXiv order).
- Venue and year: arXiv technical report ("attnres tech report"), 2026-03-16. The arXiv HTML did not render, so I read the PDF.
- Method (Sec. 3, Eqs. 2–4): h_l = Σ_i α_{i→l} v_i, with α = softmax over ϕ(w_l, k_i). The query w_l is "a layer-specific learnable vector". The keys k_i = v_i are the earlier layer outputs, passed through RMSNorm. The weights are input-dependent through the keys only. Pseudo-queries start at zero, so AttnRes starts as an equal-weight average (Sec. 5).
- Ablation (Table 4, Sec. 5.3). Model: the 436M-activated MoE row of Table 2: 16 Transformer blocks (Fig. 8 caption: "16 attention and 16 MLP layers"; Table 4 calls it the "16-layer model"), d_model 1168, 87.9B tokens, context 8192. "All models share identical hyperparameters and compute budget." Validation loss:
  - Baseline (PreNorm) 1.766; DenseFormer 1.767; mHC 1.747.
  - Full AttnRes 1.737; with input-dependent query 1.731; **with input-independent mixing 1.749**; with sigmoid 1.741; without RMSNorm 1.743.
  - SWA (W = 1 + 8) 1.764; Block (S = 4) 1.746; Block with multihead (H = 16) 1.752; Block without RMSNorm 1.750.
  - Text: "We removed the query and key and replaced them with learnable, input-independent scalars to weigh previous layers, which hurts performance (1.749 vs. 1.737)."
  - Text on DenseFormer: it combines outputs "with fixed, input-independent scalar coefficients; it shows no gain over the baseline (1.767), highlighting the importance of input-dependent weighting."
- Scaling (Table 2, five sizes, 12 to 17 blocks, 38.7B to 119B tokens; Baseline / Block / Full / mHC(-lite)): at 436M 1.766 / 1.746 / 1.737 / 1.747; at 528M 1.719 / 1.693 / 1.692 / 1.694. Fitted curves: baseline L = 1.891 C^−0.057, Block 1.870 C^−0.058, Full 1.865 C^−0.057; "equivalent to a 1.25× compute advantage" (Sec. 5.1).
- Large run (Sec. 5.2): Kimi Linear 48B total / 3B active, 54 layers, Block AttnRes with 9 blocks, 1T WSD tokens plus about 400B mid-training tokens, "the Muon optimizer". Downstream (Table 3, baseline → AttnRes): GPQA-Diamond 36.9 → 44.4, Math 53.5 → 57.1, HumanEval 59.1 → 62.2, MMLU 73.5 → 74.6.
- Optimizer for the Table 2 and Table 4 runs: not stated. Muon is named only for the 48B run.
- A sentence on identity mixing in HC (Sec. 6, "Prior Residuals as Depth-Wise Linear Attention"): "[58] show that replacing A^×_{i+1→l} with the identity matrix still yields competitive performance, highlighting the role of state expansion." Reference [58] is "Tian Xie. Your DeepSeek mHC Might Not Need the 'm'. Zhihu blog post. 2026." See item 11.
- Versus our claim: it disagrees on AttnRes. In their run, input-independent weights cost 0.012 loss (1.749 vs 1.737), and the paper concludes that input dependence matters. Their static arm still beats the baseline by 0.017 and DenseFormer by 0.018. Setting difference: MoE with KDA/MLA hybrid attention, 16 blocks (32 sublayers) rather than 48 layers, 87.9B tokens, optimizer for this table not stated, data not named.

## 5. DepthBench (arXiv 2609.32534)

- Title: "DepthBench: Measuring How Residual Connections Enable More Computational Depth".
- Authors: Keyu Wang, Yangyi Huang, Jiale Kang, David González-Martínez, Weiyang Liu, Shiwei Liu.
- Venue and year: arXiv, 2026-09-26. No venue stated.
- Recipe (Sec. 2.3): OLMo-core, FineWeb-Edu, 20 tokens per parameter ("8B tokens for 400M models and 32B for 1.6B models"), sequence length 2048, about 1M tokens per step. AdamW (β 0.9, 0.95, weight decay 0.1), cosine decay to 10%. The learning rate is swept per architecture at 400M (Table 4: HC, mHC, Full and Block AttnRes all pick 2e-3). LLaMA-like dense backbone. HC and mHC use n = 4.
- Shapes: 400M main suite with L = 16, 20, 24, 28, 32, 42, 70 (Table 3). The 1.6B suite uses L = 28, 40, 54. The 200M–500M suite uses aspect ratios near 28, 43 and 76. **No run has 48 layers.** The nearest are L = 42 (400M, d = 800) and L = 54 (1.6B, d = 1504).
- Numbers written in the text (Sec. 3, 400M, validation loss): Pre-LN 2.759 at L = 16 → 2.782 at L = 32. Full AttnRes 2.751 → 2.718. HC 2.729 → 2.699. Block AttnRes best 2.715 at L = 20, then 2.730 at L = 32. mHC is best at L = 24 "but shows no consistent gain with increasing depth". The values at L = 42 and L = 70 are only in Fig. 2, and the HTML gives no numbers for them.
- HC at scale (Sec. 3): "the 500M run at the largest aspect ratio encounters gradient explosion, while the 1.6B results do not show the same clear improvement with depth."
- mHC mechanism (Sec. 4.3): the composite residual products have effective rank 1.44–1.65 in mHC versus 2.53–2.81 in HC (Fig. 10a). Doubly stochastic maps share the fixed point U = (1/m)11ᵀ, so "residual products approach U". They conclude: "stabilizing residual transport can reduce the diversity of long-range routes".
- Static or identity variants: none. HC is run as static plus dynamic (tanh) routing, as in the original paper (Appendix A). mHC starts from "near-identity residual mixing" (biases 0 on the diagonal, −8 off it), but the mixing is learned. No arm freezes the mixing or makes the AttnRes weights input-independent.
- Versus our claim: it partly agrees and does not test the key point. It agrees that multi-stream and dense paths help in deep FineWeb-Edu models and that the mHC constraint loses HC's depth gains. It does not test whether the learned routing is needed. Its own explanation (diverse "long-range routes") assumes the routing matters. Setting difference: AdamW rather than MuonH, 8B–32B tokens, at most 32 layers with numbers in the text (42, 54 and 70 only in figures).

## 6. Identity Mappings in Deep Residual Networks (arXiv 1603.05027)

- Title: "Identity Mappings in Deep Residual Networks".
- Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
- Venue and year: ECCV 2016 (arXiv comment: "ECCV 2016 camera-ready").
- Setting: ResNet-110 on CIFAR-10 (54 two-layer residual units), image classification.
- Numbers (Table 1, test error; "fail" means error above 20%):
  - Identity shortcut (original): 6.61%.
  - Constant scaling 0.5 on the shortcut, F unscaled: fail. Shortcut 0.5 and F 0.5 ("frozen gating"): 12.35%.
  - Exclusive (Highway-style) gating 1−g(x) / g(x): fail for b_g init 0 to −5; 8.70% at −6; 9.81% at −7.
  - Shortcut-only gating: 12.86% at b_g = 0; 6.91% at b_g = −6.
  - 1×1 conv shortcut: 12.22%. Dropout 0.5 on the shortcut: fail.
- Claim (Sec. 3.2): "Multiplicative manipulations (scaling, gating, 1×1 convolutions, and dropout) on the shortcuts can hamper information propagation and lead to optimization problems." And: gating and 1×1 shortcuts "cover the solution space of identity shortcuts ... However, their training error is higher than that of identity shortcuts, indicating that the degradation of these models is caused by optimization issues, instead of representational abilities."
- Versus our claim: it agrees in spirit and is the closest classic precedent. A learned, input-dependent gate on the path is never better than the plain identity, and it is better the closer it starts to the identity. Setting difference: one residual path rather than several, a CNN on CIFAR-10 with SGD rather than a 48-layer language model with MuonH, and no learned mixing between paths.

## 7. Highway Networks (arXiv 1505.00387) and Training Very Deep Networks (arXiv 1507.06228)

- Titles: "Highway Networks"; "Training Very Deep Networks".
- Authors (both): Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber.
- Venues: Highway Networks was "Presented at ICML 2015 Deep Learning workshop" (arXiv comment). Training Very Deep Networks is NeurIPS 2015 ("in Advances in Neural Information Processing Systems 2015"). The NeurIPS paper is the better citation: it is peer reviewed, and Sec. 4.1, "Routing of Information", asks our question directly.
- Mechanism: y = H(x)·T(x) + x·(1 − T(x)), with T(x) = σ(W_T x + b_T), an input-dependent gate. The bias starts negative (−2 and −4 in the analysis).
- Claim, with the routing question asked directly (1507.06228, Sec. 4.1): "One possible advantage of the highway architecture over hard-wired shortcut connections is that the network can learn to dynamically adjust the routing of the information based on the current input. This begs the question: does this behaviour manifest itself in trained networks or do they just learn a static routing that applies to all inputs similarly." Their answer comes from gate visualizations (Figs. 2–3), not from an ablation: "The gating system acts not just as a mechanism to ease training, but also as an important part of the computation in a trained network." Conclusion: the gates help "routing information ... responding differently to different inputs, unlike fixed 'skip' connections."
- Numbers: the comparisons are against plain networks and Fitnets (CIFAR-10, MNIST), not against fixed shortcuts. Neither paper trains a highway net with the gate frozen or replaced by a constant.
- Versus our claim: it disagrees in spirit. It argues that input-dependent routing is part of the computation, but it shows this only by visualization. He et al. (item 6) later trained the comparison and found that the identity is better. Setting difference: fully connected and CNN nets on MNIST and CIFAR trained with SGD; no language models.

## 8. DenseNet (arXiv 1608.06993)

- Title: "Densely Connected Convolutional Networks".
- Authors: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
- Venue and year: CVPR 2017.
- The one claim that bears on extra fixed paths (Sec. 1): "Each layer has direct access to the gradients from the loss function and the original input signal, leading to an implicit deep supervision [20]. This helps training of deeper network architectures." Sec. 5 restates it: "a single classifier on top of the network provides direct supervision to all layers through at most two or three transition layers."
- Note: DenseNet's paths are fixed (concatenation), and the "routing" is the next convolution's ordinary weights. Table 5 of the AttnRes paper labels DenseNet "Static". The feature-reuse heat map (Fig. 5) shows that trained layers "spread their weights over many inputs", but there is no ablation of learned versus fixed path weights.
- Versus our claim: it agrees in direction. Its benefit is attributed to the existence of short, fixed paths (gradient access and feature reuse), not to any input-dependent choice among them. Setting difference: CNNs on CIFAR, SVHN and ImageNet trained with SGD, and concatenation rather than summed streams.

## 9a. DenseFormer (arXiv 2402.02622)

- Title: "DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging".
- Authors: Matteo Pagliardini, Amirkeivan Mohtashami, Francois Fleuret, Martin Jaggi.
- Venue and year: arXiv, v1 2024-02-04, v2 2024-03-21. Neither the arXiv page nor the HTML states a venue.
- Mechanism (Sec. 3): after each block, a Depth Weighted Average (DWA) Y_i = Σ_j α_{i,j} X_j over the embedding and all earlier block outputs. The α are learned scalars, input-independent ("static (as opposed to dynamic) attention weights", Sec. 2). They start at the identity (α_{i,i} = 1).
- Setting (Sec. 4): OpenWebText2 (about 17B tokens), 40k steps, batch 400 × 256 tokens (about 4.1B tokens seen), AdamW (β 0.9, 0.95, weight decay 0.1), lr 0.001 cosine. 8 heads of 64 dimensions.
- Numbers (Table 1, perplexity, 3 seeds, standard error in parentheses):
  - 48 blocks: Transformer 18.61 (0.02); "Skips With Gains" (one learned scale per skip) 18.45 (0.03); DenseFormer 1×1 17.84 (0.00); 4×1 17.86; 4×5 17.87.
  - 72 blocks: Transformer 17.82 (0.04); DenseFormer 17.12 (0.02). A 90-block Transformer gets 17.44.
  - Text: "the perplexity of a 48 block DenseFormer is only matched by a 72 block Transformer baseline."
- Static vs learned (input-dependent) weights: **no direct comparison.** Sec. 2 notes that Depth-wise Attention (ElNokrashy et al., 2022) computes the weights by a dot product, but no run compares the two. The learned static weights form "a surprisingly stable pattern ... across random seeds" (Fig. 5). Dropping more than 15% of the smallest weights at inference "leads to a significant increase in perplexity" (Fig. 7).
- Versus our claim: it agrees, and it is the closest language-model precedent. Input-independent mixing over all earlier layers beats the 48-block baseline by 0.77 perplexity, about the gain of 24 extra blocks. Its "Skips With Gains" arm is close to our "one learned number" arm and also gains (18.61 → 18.45). Setting difference: 48 blocks as in ours, but AdamW, OpenWebText2, about 4B tokens, sequence length 256, and no learned-routing arm to compare against. Note the conflict: AttnRes Table 4 (item 4) finds DenseFormer no better than its baseline (1.767 vs 1.766) at 16 blocks.

## 9b. MUDDFormer (arXiv 2502.12170)

- Title: "MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections".
- Authors: Da Xiao, Qingye Meng, Shengping Li, Xingyuan Yuan.
- Venue and year: ICML 2025 ("Accepted to the 42nd International Conference on Machine Learning (ICML'25)").
- Setting: Pile, context 2048, Chinchilla-sized token budgets (Table 2: 405M / 24 layers / 7B tokens; 834M / 24 / 15B; 1.4B / 24 / 26B; DeepNarrow 797M / 34 / 15B and 1.34B / 42 / 26B). AdamW (β 0.9, 0.95, weight decay 0.1), cosine schedule (Appendix D.1). Large runs: MUDDPythia 1.4B and 2.8B on 300B tokens.
- Static vs dynamic (Table 5, 405M, 24 layers, Pile perplexity; components added one at a time):
  - Transformer++ 11.68.
  - + Static Dense (= DenseFormer) 11.44.
  - + Dynamic Dense 11.09.
  - + Multiway Static Dense 11.27.
  - + Multiway Dynamic Dense 10.83.
  - + Multiway Dynamic Dense + re-allocation (MUDDFormer) 10.77.
  - Text: "All three ingredients of dense connections, i.e. static, dynamic and multiway, make contributions."
- Scaling (Sec. 3.1, Fig. 3): "While DenseFormer and Hyper-Connections show clear advantage over Transformer++, DDFormer outperforms them by adding dynamicity to dense connections." "MUDDFormer-834M matches the loss of Transformer++ trained with 1.89× compute." Depth (Fig. 4): Transformer++ "shows diminishing returns beyond 24 layers", while MUDDFormer DeepNarrow "maintains gains up to 42 layers".
- Versus our claim: it disagrees. Dynamic weights beat static weights by 0.35 perplexity (single-way) and 0.44 (multiway) at 405M. The paper's whole thesis is that input dependence matters. Static dense connections still beat the baseline (11.68 → 11.44). Setting difference: 24 layers rather than 48 for the ablation, AdamW, Pile, 7B tokens; no MuonH.

## 10. kexue.fm (苏剑林), archives 11664 and 9889

Source: the local kexue-fm SQLite (plugin 0.1.5, 1,344 posts, latest dated 2026-10-27), queried on della-vis1.

### 11664: "Attention Residuals 回忆录" (2026-03-19)

- Author: 苏剑林 (Jianlin Su), one of the AttnRes authors (item 4).
- On HC: "比较有意思的变化是HC，它开始走扩大残差流的路线，但也许是效果上的不稳定，并没有引起太多反响。" He derives that HC is a form of inter-layer attention, with weight a_{t+1,s} = H^pre_{t+1} H^res_{t←s+1} H^post_s: "HC其实相当于'旋转90度'的DeltaNet。实践中，三个H矩阵由tanh激活的简单线性层计算而来，这导致连乘起来的H^res_{t←s}有爆炸或坍缩的风险，也无法保证a_{t+1,s}的非负性。"
- On mHC and identity mixing: "后来mHC做了改进……由双随机矩阵对乘法的封闭性保证H^res_{t←s}的稳定，最后实验也验证了这些改动的有效性。不过，也有一些新实验如《你的deepseek mHC可能不需要"m"》显示H^res_t直接设为单位阵就足够好了。"
- On the static query: the first AttnRes used a_{t+1,s} ∝ exp(w_{t+1}·y_s), "即直接以一个数据无关的静态向量为Q、而K、V都是y_s去做Softmax Attention，这便是第一版AttnRes。令人惊喜的是，就这么个简单的设计，它相比Residuals的提升已经非常显著！" And: "我们还尝试了一些更复杂的设计，发现它们大多不如这个简单的版本，只有给K多加个RMSNorm的操作，能取得比较稳定的收益". The static Q was chosen for convenience and was then valued for inference: "'点睛之笔'正是开始图方便的静态Q设计！"
- On the residual baseline and SWA: "Residuals本身已经是一个非常强的Baseline，它对应于所有状态向量的等权求和，任何新设计想要超越它，那么至少在形式上要能够覆盖它". SWA "实际效果很糟糕，甚至还不如Residuals".
- He writes nothing about fully input-independent weights (no key) for AttnRes. The paper's Table 4 reports that arm (item 4).
- Versus our claim: it partly agrees. He repeats, without his own numbers, that identity H_res is "足够好" for mHC, which is our mHC result. He also reports that simple designs beat complex ones. But his AttnRes keeps input dependence through the keys, and the paper reports that removing it hurts. Setting difference: blog, no numbers of its own; Kimi MoE runs with Muon.

### 9889: "注意力机制真的可以'集中注意力'吗？" (2023-12-12)

- Topic: whether linear attention can be as sparse as softmax attention. Summary: "二次型Attention机制有可能实现任意稀疏的Attention矩阵，线性Attention则并不容易实现这种稀疏，或者只能实现绝对位置相关的稀疏，这可能是线性Attention能力有所限制的原因之一。"
- It does not mention HC, mHC, AttnRes, static queries or identity mixing. It predates all of them. Its only link to our question is indirect: AttnRes (item 4, Sec. 6) frames (m)HC as depth-wise linear attention and AttnRes as depth-wise softmax attention, and this post argues that linear attention cannot concentrate. That argument predicts that learned routing should matter. Our result says it does not, at 48 layers.
- Versus our claim: no direct relation. It is a theory post about sequence attention.

### Other kexue posts found by search (FTS for mHC, Hyper-Connection, AttnRes)

- 11833 (2026-07-29, 解构Scaling Law): mentions mHC and AttnRes only as "残差改进我们也还没提及".
- 11848 (2026-08-04, 简单谈谈K3的MoE和Attention): "K3 = KDA + MLA + Stable LatentMoE + AttnRes，训练优化器依然是Moonlight版Muon". So AttnRes ships with Muon in K3.
- 11985 (2026-10-27, 从参数对称性到有效学习率): discusses Hyperball (arXiv 2606.16899) and notes that "不管你搭配的是常规Residual还是MHC、AttnRes，它的每一个子层都是f(Norm(X))". This bears on our MuonH setting (scale symmetry of the weights). None of the three posts reports a static or identity arm.

## 11. Search: who already reports that identity or static routing matches or beats learned routing in language models

Method: I searched the text and reference lists of items 1–9 for identity mixing, static weights, input-independent weights and the related names they cite. I also ran an FTS search of the kexue database.

- **Tian Xie, "Your DeepSeek mHC Might Not Need the 'm'", Zhihu blog post, 2026** (https://zhuanlan.zhihu.com/p/2010852389670908320). **NOT OPENED**: Zhihu returned HTTP 403 from della-vis1. It is known only through two citations:
  - AttnRes Sec. 6 (reference [58]): "[58] show that replacing A^× with the identity matrix still yields competitive performance, highlighting the role of state expansion."
  - 苏剑林, kexue 11664: "也有一些新实验如《你的deepseek mHC可能不需要"m"》显示H^res_t直接设为单位阵就足够好了。"
  - This is the closest prior statement of our mHC result. Its setting, numbers and depth are unknown.
- **DenseFormer (item 9a)**: input-independent learned weights over all earlier layers, at 48 blocks, beat the baseline by 0.77 perplexity. There is no dynamic arm, so it shows only that static routing is enough to gain, not that it matches dynamic routing.
- **HC (item 1, Table 2)**: "At an expansion rate of 2, the improvements of DHC and SHC are similar." At n = 4, dynamic is better by 0.010–0.013.
- **苏剑林, kexue 11664**: a static query (data-independent Q) was "非常显著" better than Residuals, and "更复杂的设计……大多不如这个简单的版本". The keys stay input-dependent.
- **AttnRes (item 4, Table 4)**: fully input-independent mixing (1.749) still beats the baseline (1.766), but it loses to Full AttnRes (1.737).
- **mHC-lite (item 3)**: two different ways to build a doubly stochastic H_res give the same loss, and unconstrained HC gives no gain at 24 layers. That fits "the exact routing does not matter", but there is no identity arm.
- **He et al. 2016 (item 6)**: outside language models. A learned gate on the shortcut is never better than the identity (6.91% vs 6.61% at best).
- Mentioned but not opened (not in the task list): ANCRe (Zhang, Li, He, Giannakis, arXiv 2602.09009; abstract only: it "parameterizes and learns residual connectivities from the data"; AttnRes Sec. 7 calls its coefficients "learned per-layer scalar coefficients fixed after training"). Also ReZero, LayerScale and DLCL (cited as "Static" in AttnRes Table 5 and MUDDFormer Sec. 4). I did not check any of them for a static-versus-dynamic comparison.

Published results that **contradict** our claim, each with a direct ablation:
- HC Table 3: freezing the width connections at their identity-like initialization costs +0.021 V2 loss (OLMo-1B, 16 layers, 500B tokens).
- mHC Table 1: identity H_res instead of learned H_res costs +0.022 loss (in HC, scale not stated).
- AttnRes Table 4: input-independent mixing costs +0.012 loss (16 blocks, 87.9B tokens).
- MUDDFormer Table 5: static dense instead of dynamic dense costs +0.35 perplexity (single-way) and +0.44 (multiway) (405M, 24 layers, 7B tokens).
- All four use at most 30 layers, and three of them (HC, mHC, MUDDFormer) use AdamW or the OLMo recipe. AttnRes does not name the optimizer for its ablation table. None uses 48 layers or MuonH. These differences are the obvious candidates for the disagreement. This report did not test them.

## Summary

Who already said a version of our claim:
1. Tian Xie (Zhihu, 2026; NOT OPENED, known through AttnRes [58] and 苏剑林 11664): identity H_res in mHC gives "competitive" or "足够好" results.
2. Pagliardini et al., DenseFormer (2024): input-independent depth weights give a large gain at 48 blocks. There is no learned-routing arm.
3. Zhu et al., HC (2024/25), Table 2: static ≈ dynamic at n = 2.
4. 苏剑林, kexue 11664 (2026): a static query works, and simpler designs beat more complex ones.
5. He et al. (2016): the identity shortcut beats every learned gate, though for a single path in CNNs.

What this post adds that none of them does:
1. One controlled setting (48 layers, MuonH, FineWeb-Edu) that tests the learned routing of all three families (HC, mHC, AttnRes) against an identity or static replacement.
2. A measured, written-up number for mHC with identity mixing. The only earlier report is a blog post that I could not open.
3. A result that goes against four published ablations (HC Table 3, mHC Table 1, AttnRes Table 4, MUDDFormer Table 5): static or identity routing is not only "competitive" but better for HC and AttnRes.
4. The separation of "several residual paths" from "learned routing between them": the gain comes from the paths. Earlier papers credit the routing (HC "rearrange layers", mHC "information exchange", AttnRes and MUDDFormer "input-dependent weighting", DepthBench "diversity of long-range routes"), but none of them removes the routing while keeping the paths at this depth.

Caveat for the post: because four published ablations point the other way, the post should say which setting difference (depth 48 vs ≤ 30, MuonH vs AdamW, token count) reverses the sign, or say that it is untested.
