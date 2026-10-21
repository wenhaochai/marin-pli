# kexue.fm (Su Jianlin) on residual design, depth, Muon/MuP and per-layer losses, mapped to the pls project

Source: local offline copy of kexue.fm (SQLite, plugin kexue-fm 0.1.4), searched on della-vis2 on 2026-10-21. About 25 query terms in Chinese and English: 残差, Pre/Post Norm, DeepNorm, 超连接/HC/mHC, AttnRes, 层间, Muon, MuP, 超球面, 权重衰减, 谱范数, 初始化, 多任务, 辅助损失, MTP, 预测未来, logit lens, 中间层, attention sink, massive activation, bf16, Scaling Law, 深度缩放, and others. About 40 articles were read in full or in their relevant sections. All links have the form https://kexue.fm/archives/<id>.

Article ids by relevance:
- Core: 9009, 8620, 8747, 8978, 8994, 11664, 11848, 8907, 8896, 8870, 11647, 11729, 11605, 11340.
- Supporting: 9889, 11371, 11126, 10739, 11416, 11241, 11196, 9698, 8998, 8990, 8027, 7818, 11833, 11549.

---

## Q1. Which articles explain why Pre-Norm wastes depth or makes late layers redundant, and what fixes do they propose?

### Takeaway
Su's main argument is the "watered depth" (深度有水分) of Pre-Norm. The residual stream grows as x_t = O(t), so for large t two consecutive layers act like one wider layer. A deep Pre-Norm stack therefore turns into a "shallow and wide" model. The fixes he discusses are Post-Norm and DeepNorm (identity weight α plus a smaller branch init λ), a branch scale ε = 1/√N, and ReZero/SkipInit-style gates. His preferred gate rises on a fixed schedule from 0 to 1 and is not trained. This mechanism directly predicts two of our findings: late layers become near-redundant, and many middle layers can be deleted.

### Cited Findings
- **9009, 为什么Pre Norm的效果不如Post Norm？ (2022-03-29)** — [link](https://kexue.fm/archives/9009)
  - Unrolling Pre-Norm gives x_{t+1} = x_0 + Σ_k F_k(Norm(x_k)). Every term has the same order, so x_{t+1} = O(t+1). For large t, x_t and x_{t+1} differ little, so F_t(Norm(x_t)) + F_{t+1}(Norm(x_{t+1})) ≈ (F_t + F_{t+1})(Norm(x_t)). Two layers then behave like one wider layer.
  - Quote: "Pre Norm结构无形地增加了模型的宽度而降低了模型的深度 … 层数越多，这个层就越'虚'" (Pre-Norm quietly adds width and removes depth; the more layers, the more "hollow" each one becomes).
  - Post-Norm shrinks the identity branch at every Norm, so its layers count fully.
  - Su reads DeepNet's remark ("gradients of Pre-LN at bottom layers tend to be larger than at top layers") as the same effect: Pre-LN leans on the identity path and degrades into a shallow, wide model.
  - Su's own experience: Pre-Norm and Post-Norm reach about the same pretraining result, but Post-Norm fine-tunes clearly better.
- **8620, 浅谈Transformer的初始化、参数化与标准化 (2021-08-17)** — [link](https://kexue.fm/archives/8620)
  - At init, Post-Norm unrolls to x_l = x_0/2^{l/2} + … + F_{l-1}/2^{1/2}, which weights early layers down exponentially.
  - Pre-Norm at init unrolls to x_l = x_0 + F_0(x_0) + F_1(x_1/√2) + … + F_{l-1}(x_{l-1}/√l). The input that layer k sees after its norm is effectively scaled by 1/√k, so all branches carry equal weight. A final Norm is needed before the prediction head because Var(x_l) keeps growing.
  - Su's preferred design is x_{t+1} = x_t + α_t F_t(x_t) with α_0 = 0 (SkipInit/ReZero/Fixup). He argues that α must not share the large learning rate of other parameters. In his experiments the best choice was to raise α_t by a small fixed step until it reaches 1 and then freeze it.
- **8747, 模型优化漫谈：BERT的初始标准差为什么是0.02？ (2021-11-08)** — [link](https://kexue.fm/archives/8747)
  - Post-Norm keeps every layer at the same scale, so you can attach a Dense classifier to layer 6 or to the last layer. With Pre-Norm you must add your own LN before reading an intermediate layer, because later layers have larger variance.
  - Post-Norm's vanishing gradient toward the input is useful for fine-tuning.
  - Without warmup, Post-Norm's top layers learn fastest, reach a bad optimum built on poor inputs, and then collapse.
- **8978, 训练1000层的Transformer究竟有什么困难？ (2022-03-09)** — [link](https://kexue.fm/archives/8978)
  - "Update explosion" (增量爆炸): with O(1) per-matrix gradients, ΔL = O(ηNK) grows with depth N.
  - Two fixes: warmup, or scale the gradients to O(1/√N).
  - Su re-derives DeepNorm, x_{l+1} = LN(α x_l + F(x_l)) with branch gain λ, from a scalar "magnitude probe" W = λU (U nearly orthogonal). The result is ∂L/∂λ = O(λ/α).
  - The required scaling depends on the optimizer:

    | Optimizer | Condition | α | λ |
    |---|---|---|---|
    | SGD | λ/α = (2N)^{-1/2} | (2N)^{1/4} | (2N)^{-1/4} |
    | Adam | λ/α = 1/(2N) | (2N)^{1/2} | (2N)^{-1/2} |
    | LAMB-type (update proportional to ‖θ‖) | λ²/α = 1/(2N) | 1 | (2N)^{-1/2} |

  - For Pre-Norm the same analysis holds with α = 1.
  - Two side cases: with α = 1 and λ = (2N)^{-1/2}, the small init "can leave that scale after a few steps", because the relative update size does not change. With λ = 1 and α = (2N)^{1/2}, learning becomes very slow.
  - DeepNet result: a 200-layer, 3.2B-parameter model beats a 48-layer, 12B-parameter model.
- **8994, 为什么需要残差？一个来自DeepNet的视角 (2022-03-19)** — [link](https://kexue.fm/archives/8994)
  - y = x + εf(x) with ε = 1/√N scales parameter gradients by 1/√N and keeps the forward and backward passes stable.
  - In high dimension, the per-layer expansion factor of the backward pass is about 1 + ε², not 1 + ε, because v and εv·∂f/∂x are nearly orthogonal. So ε = 1/√N is enough; the 1-D estimate would ask for 1/N.
- **8998, RoFormerV2 (2022-03-21)** — [link](https://kexue.fm/archives/8998)
  - Training Post-Norm from scratch used x_{t+1} = Norm(x_t + αF(x_t)), with α starting at 0 and rising linearly to 1.
  - Su reports this beat ReZero, "几乎是DeepNet之前的最优解" (nearly the best solution before DeepNet).
  - RoFormerV2 also removed all biases and the RMSNorm gamma.
- **8990, 门控注意力单元（GAU）还需要Warmup吗？ (2022-03-11)** — [link](https://kexue.fm/archives/8990)
  - GAU's branch output is about 0.01·x_l at init, two orders of magnitude below the stream. It behaves like a built-in DeepNorm with λ = 1 and α ≈ 100, so tens of Post-Norm GAU layers train without warmup.
- **8027, RealFormer (2020-12-24)** — [link](https://kexue.fm/archives/8027)
  - Puts the residual on the attention scores (A_n = Q_nK_n^T/√d + A_{n-1}) inside a Post-LN model. This keeps Post-LN quality and makes training easier.
- **11605, MuP之上：2 (2026-02-15)** — [link](https://kexue.fm/archives/11605)
  - Post-Norm is also an "In Norm" design. With y_{t+1} = x_t + F_t(x_t), Post-Norm becomes y_{t+1} = Norm(y_t) + F_t(Norm(y_t)).
  - So the only difference from Pre-Norm is whether the identity path carries x_t or Norm(x_t).

### Inferences
- **Deletable layers and the last-layer cost.** Separate heads force every x_l to be readable by its head, so each later block must make only a small correction to an already-good x_l. This strengthens the "hollow layer" effect in 9009. Test: for each design, plot the per-layer ratio ‖F_t(Norm(x_t))‖ / ‖x_t‖ and the cosine between x_t and x_{t+1}. Then check whether the layers that can be deleted are those where the ratio is smallest. A design whose identity path is shrunk (DeepNorm with α > 1, or Post-Norm-like) should show fewer deletable layers if 9009's account is right.
- **Gain init 1/√ℓ vs LNS.** At init, Pre-Norm already feeds layer k a 1/√k-scaled input (8620). LNS multiplies the norm output by a fixed 1/√ℓ. Gain init 1/√ℓ is the same at step 0 but is learnable. Su's view (8620) is that such a scale should not be trained at the main learning rate, and his best result used a fixed schedule. The planned follow-up is therefore a direct test of "fixed vs learnable depth scale".
  - Prediction from Su's reasoning: under Adam on gains with no weight decay, the update is O(η) per step regardless of the gain size. A gain of 1/√48 ≈ 0.14 therefore sees a relatively large update and can drift back toward O(1) within a few thousand steps.
  - Test: log the per-layer gains over training. If they converge toward 1 in deep layers, gain init only changes the early transient, and LNS keeps its effect.
- **MuonH vs Adam in the DeepNet table.** MuonH keeps every matrix at its initial Frobenius norm. This resembles the LAMB row more than the Adam row, because the update is a fixed fraction of the weight. That row says the identity weight α is unnecessary and the branch scale λ = (2N)^{-1/2} should be set at init.
  - Under MuonH the matrix part of λ can never drift. The branch scale is then set permanently by the init norm times the Adam-trained norm gains.
  - Consequence: in our recipe, DeepNorm's α > 1 may be redundant or harmful. The scale knobs that matter are the RMSNorm gains, which are trained by Adam with no weight decay.

### Gaps
- No kexue.fm article treats Sandwich-LN, LayerNorm Scaling (1/√ℓ, by that name), KEEL, Peri-LN or MoDA. All of these searches returned no hits.
- No article measures layer redundancy directly, for example by deleting layers.
- No article covers depth-μP (residual-branch 1/√L or 1/L scaling within the MuP framework). The searches "Depth-μP", "深度缩放", "层数缩放" and "1/L" returned nothing relevant. 10770 says outright that Normalization and residual effects were left out of its MuP analysis.

---

## Q2. What does Su say about Muon, norm-constrained or hyperspherical training, init, LR and depth (relevant to MuonH and to gain init 1/√ℓ)?

### Takeaway
Su derives Muon as steepest descent under a spectral-norm "update stability" condition. He also derives per-layer norms: the spectral norm for dense linear layers, the maximum row norm for the embedding, and the maximum column norm for the LM head. He argues against pinning a weight norm to a fixed value, which is MuonH's design. He prefers a bound ‖W‖ ≤ τ, enforced by minimal clipping or "pre-decay". He also states that the embedding and the LM head must not use Muon, and that the LM head's init and LR should scale as Θ(1/d). Every separate head in our project is such a loss-attached head.

### Cited Findings
- **11340, MuP之上：1. 好模型的三个特征 (2025-10-21)** — [link](https://kexue.fm/archives/11340)
  - A good model has three properties, each measured as a max of the RMS and required to be Θ(1): forward stability, dependency stability (Lipschitz in the input, which also covers backward stability through the Jacobian's spectral norm), and update stability.
  - The model f can be one layer, a block, or the whole network.
- **11605, MuP之上：2. 线性层与最速下降 (2026-02-15)** — [link](https://kexue.fm/archives/11605)
  - For a linear layer with In Norm, all three measures are √(d_in/d_out)·‖·‖₂.
  - The init std is σ = Θ(√(d_out/d_in)/(√d_in + √d_out)).
  - Steepest descent gives ΔW = −η√(d_out/d_in)·msign(G), which is the MuP version of Muon.
  - Muon still applies without an In Norm if ‖x‖_RMS is roughly constant.
- **11647, MuP之上：3. 特殊情况特殊处理 (2026-03-02)** — [link](https://kexue.fm/archives/11647)
  - Embedding: stability is measured by the maximum row RMS, so its steepest descent is row-wise Normalized SGD, with init variance Θ(1).
  - LM head: it is "对Loss负责" (answerable to the loss). Its input is (x, label t) and its output is the cross-entropy. Su bounds all three stability measures by 2d·max_i‖w_i‖_RMS, using |LSE(a) − LSE(b)| ≤ max|a_i − b_i|.
  - Hence the LM head's steepest descent is column-wise Normalized SGD, and its init std and LR both scale as Θ(1/d), unlike the embedding's Θ(1).
- **11416, Muon优化器指南 (2025-11-19)** — [link](https://kexue.fm/archives/11416)
  - Muon is only for "稠密输入的线性层的矩阵参数" (matrix parameters of linear layers with dense input). Using it for the embedding or the final classification layer, including GPT's LM head, makes results "明显差" (clearly worse).
  - The Moonlight version matches Adam's update RMS of 0.2. The MuP and Keller-Jordan versions need about a 0.2√d times larger LR, roughly 10×, and their LR transfers across widths.
- **11729, MuP之上：4. 坚守参数的稳定性 (2026-04-24)** — [link](https://kexue.fm/archives/11729)
  - Su mentions renormalizing to a fixed norm after every step (Hyperball, Nemotron-Flash) and reparametrizing f(x; τω/‖ω‖). He argues that fixing a parameter norm "仍然会破坏原本的训练动力学" (still distorts the original training dynamics), and that the norm should be left to training.
  - He proposes only ‖ω‖ ≤ τ, enforced in one of two ways:
    1. "Post Clip": project onto the norm ball after each update, using the minimal-change projection.
    2. "Pre Decay": shrink ω_{t−1} to (1−η)‖ω_{t−1}‖ with the same projection before the update. This guarantees ‖ω_t‖ ≤ max(‖ω_0‖, τ) for any norm and generalizes weight decay.
  - He prefers the minimal clip over plain multiplicative weight decay to avoid "过度干预" (over-intervention). Example: bounding the spectral norm at 5 with plain decay would need λ = 0.2, versus a typical 0.01.
- **10739, Muon续集 (2025-02-27)** — [link](https://kexue.fm/archives/10739)
  - Muon without weight decay converged faster early but was overtaken by Adam, and internal metrics showed "崩溃的苗头" (early signs of collapse).
  - Adding weight decay keeps ‖W_t‖₂ ≤ max(‖W_0‖₂, 1/λ), which matters "尤其…对于Attention Logits爆炸" (especially for attention-logit explosion).
- **11241, 流形上的最速下降：4. Muon + 谱球面 (2025-08-21)** — [link](https://kexue.fm/archives/11241)
  - Muon restricted to a spectral-norm sphere, followed by a spectral renormalization computed with power iteration.
- **11196, 流形上的最速下降：1. SGD + 超球面 (2025-08-01)** — [link](https://kexue.fm/archives/11196)
  - The same derivation for the Frobenius-norm hypersphere. This is the steepest descent closest to MuonH's constraint.
- **11549, 为什么我们偏爱各向同性？ (2026-01-20)** — [link](https://kexue.fm/archives/11549)
  - Normalization layers are cheap stand-ins for whitening. Per-sample RMSNorm is the "hypersphere" (超球面) choice.
- **11126, QK-Clip (2025-07-12)** — [link](https://kexue.fm/archives/11126)
  - A post-hoc rescaling of the Q and K weights against MaxLogit explosion in Muon + MLA at K2 scale. It leaves the forward computation unchanged.

### Inferences
- **MuonH is the constraint Su argues against, but it removes "update explosion".** MuonH fixes ‖W‖_F at its init value, so the relative update per step is constant. In the DeepNet table this is the LAMB row (α = 1, λ = (2N)^{-1/2}). A useful check: confirm that per-layer branch magnitudes ‖F_t‖ in our 48-layer runs stay at their init ratio and that only the Adam-trained gains move.
- **Separate heads are LM heads (11647).**
  - If they share the main head's weights, the logit scale per layer depends only on ‖Norm(x_l)·W‖, so differences between layers come from the final-norm gain or the hidden direction.
  - If they have their own weights, their init and LR should follow Θ(1/d) and they should be in the Adam or Normalized-SGD group, not Muon/MuonH.
  - Action: confirm that the pls separate heads are excluded from MuonH. If any are in the MuonH group, 11416 predicts clearly worse heads, which could confound "per-layer losses cost the last layer".
- **Gain init 1/√ℓ under MuonH.** The matrices cannot change norm, so a learned gain is the only per-layer scale. Unlike Adam weights, it has no counterweight in the weight norms. Combining 11729 and 8620, the clean comparison is three arms:
  1. LNS: fixed 1/√ℓ, no parameters.
  2. Gain init 1/√ℓ: learnable, Adam, no weight decay.
  3. Gain init 1/√ℓ with gains clipped to ≤ 1/√ℓ (Su's "Post Clip" applied to a scalar).

  Arm 3 has no extra parameters, which fits the owner's "parameter-free fixes" rule.

### Gaps
- Su has no article on how the Muon/MuP learning rate or init should scale with depth, and none on the per-layer LR of norm gains.
- No article analyses MuonH (fixed Frobenius norm) by name. 11729 only names Hyperball and Nemotron-Flash as fixed-norm methods.

---

## Q3. What is on Hyper-Connections, AttnRes, and attention over layers (static vs dynamic routing, near-binary reads, an outsized source)?

### Takeaway
11664 is Su's first-hand account of AttnRes. It answers several of our questions directly:
- The original and final AttnRes uses a **static, data-independent query** w_{t+1} with RMSNorm on the keys, and most richer designs were worse.
- HC/mHC are a special case of the same layer-attention form, with products of H^res matrices that can explode or collapse.
- Full AttnRes gives the **embedding a sizeable share of attention**, which is why Block AttnRes keeps the embedding as its own block.
- Sliding-window AttnRes lost to plain residuals because it cannot express the equal-weight sum. In Su's words, "compression beats sparsity".

### Cited Findings
- **11664, Attention Residuals 回忆录 (2026-03-19)** — [link](https://kexue.fm/archives/11664)
  - **Rewrite of residuals.** With y_t = f_t(x_{t−1}), residuals are y_{t+1} = f_{t+1}(y_0 + … + y_t), an equal-weight sum. AttnRes generalizes this to y_{t+1} = f_{t+1}(Σ_s a_{t+1,s} y_s) with a ≥ 0 and Σ a = 1.
    - Non-negativity keeps every source pushing in the same direction across layers.
    - The sum-to-1 constraint costs no expressiveness, because f has an input RMSNorm and RMSNorm(cx) = RMSNorm(x).
  - **HC as layer attention.**
    - HC: X_t = H^res_t X_{t−1} + H^post_t f_t(H^pre_t X_{t−1}), with k = 4 streams. Post-Norm and Highway are special cases.
    - Unrolled, a_{t+1,s} = H^pre_{t+1} H^res_{t←s+1} H^post_s, a scalar, so HC is "旋转90度的DeltaNet" (DeltaNet rotated by 90 degrees).
    - With tanh-computed H, the product H^res_{t←s} "有爆炸或坍缩的风险" (risks exploding or collapsing) and a can be negative.
    - mHC uses sigmoid (a ≥ 0) and doubly stochastic H^res, which is closed under products.
    - Su cites a Zhihu experiment ("你的deepseek mHC可能不需要'm'") showing that setting H^res = I is good enough.
  - **AttnRes form.**
    - First version: a_{t+1,s} ∝ exp(w_{t+1}·y_s), with a static learned query vector and keys = values = y_s. Su says it already gave a "非常显著" (very significant) gain over residuals.
    - "尝试了一些更复杂的设计，发现它们大多不如这个简单的版本" (more complex designs were mostly worse than this simple one). The only stable extra gain came from RMSNorm on K: a ∝ exp(w_{t+1}·RMSNorm(y_s)).
    - The static query makes inference practical, because a_{t,s} can be computed for all t > s as soon as y_s exists.
  - **Block AttnRes.**
    - SWA over layers was "很糟糕，甚至还不如Residuals" (very bad, even worse than residuals), because it cannot cover "所有状态向量等权求和" (the equal-weight sum of all states). So compression is better than sparsity.
    - Block design: the embedding is its own block, because the visualized Full AttnRes attention showed "模型偏向于给Embedding层可观的Attention" (the model tends to give the embedding layer substantial attention). The other layers are summed inside each block of m layers, and attention runs over block sums.
    - About 8 blocks recover most of the gain, at "5%以内的开销，换取25%的收益" (under 5% overhead for a 25% gain). Su does not define the 25% in the post.
    - Su gives one attention-matrix view that unifies Residuals (all ones, lower triangular), HC/mHC (products of H), Full AttnRes and Block AttnRes.
  - **Prior work** on depth or dense attention: DenseNet, DenseFormer, ANCRe, MUDDFormer, MRLA, Dreamer, ELMo, SKNets, LIMe, DCA.
- **11848, 简单谈谈K3的MoE和Attention (2026-08-04)** — [link](https://kexue.fm/archives/11848)
  - K3 = KDA + MLA + Stable LatentMoE + AttnRes, trained with Moonlight-style Muon. Attention weights are optimized per head, which Su says "没有效果上的优势" (has no quality advantage) and was chosen for correctness.
  - Su credits an RMSNorm before the LatentMoE up-projection with "无形增加了LatentMoE的等效深度" (quietly adding effective depth to LatentMoE), because the norm is a weak nonlinearity.
- **9889, 注意力机制真的可以"集中注意力"吗？ (2023-12-12)** — [link](https://kexue.fm/archives/9889)
  - For exp attention with keys k ~ N(μ, σ²I), the l1/l2 sparsity is S(a) = exp(−σ²‖q‖²/2).
  - S → 0 (one-hot) as ‖q‖ → ∞ or as the key spread σ → ∞. So attention becomes more concentrated by "增大q的模长，或者增大各个k之间的方差" (enlarging the norm of q or the variance across keys).
- **8027, RealFormer (2020-12-24)** — [link](https://kexue.fm/archives/8027)
  - A residual over the attention matrices across layers, an earlier "cross-layer" connection on scores rather than on states.

### Inferences
- **Our "outsized single source" in AttnRes is very likely the embedding y_0.** This is the same pattern that led Kimi to give the embedding its own block. Test: report the source index of the largest a_{t,s} per layer and how often it is s = 0. If it is the embedding, this is a known property of the design, not something our recipe causes. A new question for our project is whether separate heads change it.
- **Near-binary reads have a simple temperature reading (9889).** With K = RMSNorm(y_s), the key spread is fixed, so ‖w_{t+1}‖ alone sets the sharpness: S(a) ≈ exp(−c‖w‖²).
  - The pseudo-queries w are vectors, so in our recipe they are likely Adam-trained with no weight decay. Their norms can then grow without bound, and the softmax drifts toward one-hot.
  - Test: log ‖w_t‖ per layer over training and plot read entropy against ‖w_t‖.
  - Parameter-free intervention: renormalize w to a fixed norm, or clip it (Su's "Post Clip", 11729). If reads stop being near-binary and loss does not change, the binary behaviour is a temperature artefact and not a learned routing need.
- **Static vs dynamic routing.** Su's version is "static query, data-dependent keys". Note that it is already partly dynamic, because the keys are the token-specific y_s. That gives three arms for our follow-up:
  1. Fully static: learned scalar logits a_{t,s}, DenseFormer-like.
  2. Kimi's: static query w_{t+1} with RMSNorm(y_s) keys.
  3. Fully dynamic: query computed from x_t.

  Su found the richer designs mostly worse. That gives a prior: dynamic routing will not help much. If separate heads help a dynamic router more than a static one, that would be a surprising result.
- **HC vs AttnRes on one axis.** Using 11664's attention-matrix view, compute the effective A matrix a_{t+1,s} = H^pre_{t+1}·H^res_{t←s+1}·H^post_s for our trained HC and mHC models (per token, averaged), and plot it next to AttnRes's A.
  - This shows whether HC's effective routing is near uniform (residual-like) or concentrated like AttnRes.
  - It also tests whether ‖H^res product‖ explodes or collapses at 48 layers, which 11664 calls the HC failure mode.
- **Doubly stochastic constraint.** The Zhihu result that H^res = I suffices suggests one more arm for the static-vs-dynamic follow-up: mHC with H^res fixed to I, with only the pre and post maps learned.

### Gaps
- 11664 gives no numbers for how sharp AttnRes weights become. It only says the embedding gets "可观的" (substantial) attention. Whether Kimi saw near-binary reads is not stated.
- MoDA (attention over depth KV) is not discussed anywhere on kexue.fm. The search returned no hit.
- Su does not report a static-vs-dynamic ablation with numbers. He only says the simpler, static design beat "more complex designs".

---

## Q4. What is on deep supervision and auxiliary losses (why per-layer losses cost the last layer, and how to avoid it)?

### Takeaway
kexue.fm has no post on deep supervision of LLM layers, but the three-part multitask series gives the exact tools. Per-layer losses are a "main/auxiliary" (主次型) multitask problem: the last-layer loss is the main task and the per-layer losses are auxiliary. 8907 derives an update that maximizes progress on the main loss subject to each auxiliary loss not increasing. For one auxiliary loss it takes a closed form, g₀ + relu(−⟨g₀,g₁⟩)/‖g₁‖²·g₁. 8896 gives a cheap approximation that computes the conflicts on the shared hidden states rather than on all parameters.

### Cited Findings
- **8907, 多任务学习漫谈（三）：分主次之序 (2022-02-14)** — [link](https://kexue.fm/archives/8907)
  - When the goal is one main loss L₀ and auxiliary losses L₁…L_n are only regularizers, the "parallel" goal of treating every task equally "就可能会明显降低主任务的效果" (can clearly hurt the main task).
  - Formulation: max_u ⟨u, g₀⟩ − ½‖u‖² subject to ⟨u, g_i⟩ ≥ 0. By the minimax theorem this becomes min_{λ≥0} ‖g₀ + Σλ_i g_i‖².
  - For n = 1: λ = relu(−⟨g₀,g₁⟩)/‖g₁‖². In words, add the auxiliary gradient only as far as needed to stop the main step from increasing the auxiliary loss.
  - For general n, a Frank–Wolfe-style coordinate update. Examples given: L2 regularization without a tuned coefficient, and noisy-label training.
- **8896, 多任务学习漫谈（二）：行梯度之事 (2022-02-08)** — [link](https://kexue.fm/archives/8896)
  - Cache the Gram matrix ⟨g_i, g_j⟩ once per step.
  - For a shared encoder, replace parameter gradients with gradients with respect to the shared hidden states H, since ‖Σα_i g_i‖² ≤ ‖Σα_i ∇_H L_i‖²·‖∇_θ H‖². This minimizes an upper bound and needs only output-side gradients.
  - Su shows that the proof in "Multi-Task Learning as Multi-Objective Optimization" (Sener & Koltun) about this bound is wrong (counterexample with a positive definite M). The bound still works well in practice.
- **8870, 多任务学习漫谈（一）：以损失之名 (2022-01-18)** — [link](https://kexue.fm/archives/8870)
  - Loss weighting schemes:
    - Divide each loss by its initial value (log K for K-way cross-entropy) or by its prior value (label entropy).
    - Divide by stop_grad(L_i), which is equivalent to optimizing Σ log L_i, the geometric mean. Su says this is a strong baseline in many "民间实验" (informal experiments).
    - Use a generalized power mean L(γ), which interpolates from min to max.
    - Divide each loss gradient by its norm, Σ∇L_i/‖∇L_i‖. This is invariant to both scale and shift. Su relates it to GradNorm and calls GradNorm redundant.
- **7818, TeaForN (2020-10-27)** — [link](https://kexue.fm/archives/7818)
  - Auxiliary losses from repeated passes, loss = Σ_t Σ_i λ_i(−log p_t^{(i)}[w_t]), with shared weights across passes. Su: "这样处理后再不济应该也不会下降…稳赚不赔" (at worst it should not hurt; a no-lose strategy). This was his expectation, not a measured result.

### Inferences
- **The last-layer cost is a test of gradient conflict.** Experiment:
  1. In a separate-heads run, compute every K steps the cosine between g_final and g_l for each l, using the cheap version on hidden states from 8896, i.e. ∇_{x_l} of each loss at the shared stream positions.
  2. Plot the cosine against depth and training step.
  3. If the cost comes from conflict, the cosines for late or middle l should be negative where the last layer is hurt most.
- **A parameter-free fix to try.** Treat L_final as main and the per-layer losses as auxiliary, using 8907's projection. With n = 47 auxiliary losses, use the hidden-state bound from 8896 to keep it cheap, or group layers into a few blocks of auxiliary losses. This adds no parameters, which fits the owner's rule. It tests whether the last-layer cost is avoidable or a capacity trade-off.
  - A cheaper first arm: per-layer losses weighted by 1/stop_grad(L_l), Su's geometric-mean baseline, against equal weights. Early layers have larger L_l, so this down-weights them relative to the last layer.
- **Separate heads vs probes.** Probes are detached, so they are "auxiliary losses with λ = 0 on the trunk". Separate heads are the λ = 1 parallel version. 8907's projection is the natural middle point: auxiliary gradients enter only where they do not hurt the main loss.

### Gaps
- No kexue.fm article on deep supervision, early exit or layer-wise heads for LLMs. "深度监督", "Deep Supervision", "早退", "Early Exit" and "层剪枝" had no relevant hits. The only "Early Exit" hit (11494) is about optimizer early stopping, not early-exit inference.
- No article measures gradient conflicts between layers of one network.

---

## Q5. What bears on output heads, logit lens and last-layer behaviour (the last MLP pushes against the correct token)?

### Takeaway
Two mechanisms on kexue.fm may explain an odd final block. First, (8747) when the output logits are too small at init, gradient descent takes the cheapest route to scale them: the final norm's gain grows "明显偏大" (clearly too large). The last layer then becomes a logit-temperature knob rather than a feature layer, and BERT added Dense+LN before the MLM head to absorb this. Second, (11647) the LM head is "answerable to the loss", and its geometry and scaling differ from inner layers. No article discusses the logit lens itself.

### Cited Findings
- **8747 (2021-11-08)** — [link](https://kexue.fm/archives/8747)
  - With a small init (BERT std 0.02, about half of Xavier), the logits before softmax are near 0 and the predicted distribution is too uniform.
  - The model raises logit scale by the fastest path, which is the last LN's gamma. Embeddings are updated sparsely, so that path is slow.
  - Result: "最后一个LN层的gamma值会偏大…导致最后一层的方差会比其他层的明显大" (the last LN's gamma ends up large, so the last layer's variance is clearly larger than other layers'). Su reports that this is visible in released BERT weights.
  - BERT's extra Dense+LN before the MLM head moves the large gamma to the new LN and keeps the encoder layers consistent with each other.
  - With Pre-Norm, reading an intermediate layer requires inserting an LN (the logit-lens setting). With Post-Norm, any layer can be read directly.
- **9698, 语言模型输出端共享Embedding的重新探索 (2023-07-20)** — [link](https://kexue.fm/archives/9698)
  - With near-identity init (DeepNorm-like), the initial model is a 2-gram model. With tied embeddings the initial loss is about log(e^{dσ} + n − 1), far above log n, because the w_i·w_i term dominates.
  - Fixes: untie the weights, add an orthogonal projection after the final Norm (BERT's approach), shuffle dimensions, or use init σ = (log n)/d, which converged slowest. Su found the final results about the same for all fixes.
- **11647 (2026-03-02)** — [link](https://kexue.fm/archives/11647)
  - LM head stability bounds are 2d·max‖w_i‖_RMS. Its steepest descent is column-normalized SGD with Θ(1/d) init and LR.
- **11854, 除了交叉熵，LM Loss还有什么选择？ (2026-08-09)** — [link](https://kexue.fm/archives/11854)
  - Found by search. It discusses alternatives to cross-entropy for LM training. It was not read in detail and is not mapped to a finding here.

### Inferences
- **The "last MLP pushes against the correct token" may be calibration, not anti-learning.** If the final block's main job is to set logit scale or entropy, which 8747 attributes to the final norm gain, then its MLP may lower the correct token's logit on tokens where the earlier stream was over-confident.
- Test, with no new parameters:
  1. Split the last MLP's logit contribution into a component along the current logit vector (a pure temperature change) and an orthogonal component.
  2. Bucket tokens by the earlier layer's confidence and check whether the "push against" happens mostly on over-confident tokens.
  3. Also plot the final RMSNorm gain against the other layers' gains: 8747 predicts it is clearly larger in ordinary models.
  4. Check whether separate heads, which make every layer face the loss, remove this gap.
- **Separate heads at shallow layers.** Under 9698's analysis, a head reading layer 1 is close to a bigram model. If heads are tied to the input embedding, early-layer head losses start far above log n. This is worth checking when comparing early-layer loss curves.

### Gaps
- No kexue.fm article on the logit lens, tuned lens, or reading intermediate layers of decoder LLMs. "logit lens", "中间层" and "探针" returned only unrelated posts. 8747's note on reading BERT's layer 6 is the closest.

---

## Q6. Is there anything on multi-token prediction (after-next information appearing only past ~60% depth)?

### Takeaway
Almost nothing relevant. kexue.fm mentions MTP only as a decoding or speculative-decoding concern in K3's attention choice (11848). The nearest training idea is TeaForN (7818), which predicts t+2, t+3, … by re-running the shared decoder on its own outputs. Nothing on kexue.fm explains where in depth future-token information appears.

### Cited Findings
- **11848 (2026-08-04)** — [link](https://kexue.fm/archives/11848)
  - Under MTP or speculative decoding, MLA's decoding is already compute-bound (an MQA with head_dims 512+), so "MLA+MTP很容易吃亏" (MLA plus MTP easily loses out).
  - Su lists four conditions an attention better than MLA should meet. One is less decoding compute, so that it suits MTP.
- **7818, TeaForN (2020-10-27)** — [link](https://kexue.fm/archives/7818)
  - Teacher forcing makes h_t "没什么远见" (short-sighted), because it only has to predict y_t.
  - TeaForN feeds [e_0, h_1, …, h_{L−1}] back through the same decoder M for N passes and adds weighted losses for t+1 … t+N.
  - In the paper, sharing M across passes beat separate weights. Using top-k embedding mixtures in place of h was unstable and no better.

### Inferences
- TeaForN's assumption is that a deep-layer output h_t can be used as if it were an input embedding. This hints that late-layer states lie in an "input-like" space. That fits our finding that after-next information appears only late, where the stream starts to encode the next token as if it were already given. This is speculative and has no source support.

### Gaps
- No kexue.fm article on MTP for pretraining (DeepSeek-V3 style), on Medusa, or on where future-token information sits by depth. "多token", "Multi-Token", "预测未来", "Medusa" and "多步预测" returned no relevant hits.

---

## Q7. What is on massive activations, attention sinks, softmax saturation and low precision?

### Takeaway
There is no dedicated post on massive activations or attention sinks. Related material:
- SwiGLU outliers of order ‖x‖⁴, fixed in K3 with softcaps (11848).
- MaxLogit explosion and QK-Clip (11126).
- Biased BF16 rounding in attention when a softmax row has several exact maxima, which is a saturated, near-binary case (11371).
- Weight decay or norm bounds as protection against logit explosion (10739, 11729).

### Cited Findings
- **11848 (2026-08-04)** — [link](https://kexue.fm/archives/11848)
  - In SwiGLU, a row of W₁ can align with an input x. If the same happens in W₂ at the same position, the intermediate has outliers of order O(‖x‖⁴).
  - K3 replaces SiLU with SiTU(x; β) = β·tanh(x/β)·σ(x) (β₁ = 4) and also softcaps the linear branch (β₂ = 25). In the same bounds, softcap worked better than the hard clip used in GPT-OSS and DSV4.
- **11371, 低精度Attention可能存在有偏的舍入误差 (2025-10-27)** — [link](https://kexue.fm/archives/11371)
  - In BF16, P̄V with P̄ = exp(S − S_max) has biased rounding error when a row of S has several exact maxima, so several P̄ entries equal 1. The bias can build up into MaxLogit explosion or loss spikes.
  - The fix in the cited paper triggers when a row has two or more max values. Su thinks the detection condition "有很大调整空间" (leaves much room for improvement).
- **11126, QK-Clip (2025-07-12)** — [link](https://kexue.fm/archives/11126)
  - QK-Clip rescales the Q and K weights after the fact when the max logit exceeds a threshold. Su presents it as a "Muon + MLA" stability tool for K2.
- **9889 (2023-12-12)** — [link](https://kexue.fm/archives/9889)
  - The sharpness formula from Q3 (S(a) = exp(−σ²‖q‖²/2)) is the softmax-saturation law.

### Inferences
- **AttnRes near-binary reads in BF16.** If several sources reach the same max logit and the depth-softmax runs in BF16 with the max subtracted, 11371's bias applies to the depth mixture. That would pick a direction systematically. Cheap check: run the AttnRes depth softmax in FP32 and compare loss and read sharpness on one 48-layer run. Low prior; worth doing only if near-binary reads come with instability.
- **Under MuonH** the matrices cannot grow, so the K2-style MaxLogit explosion (from weight-norm growth) should be muted. Large logits would then have to come from the Adam-trained norm gains or the AttnRes queries. This links Q7 back to the gain-logging experiment in Q2.

### Gaps
- No kexue.fm post on "massive activations", "attention sink", "注意力沉没" or first-token sinks. These terms returned no hits; 9948 only mentions Attention Sink in passing as a length-extrapolation technique.

---

## Q8. Index of all relevant articles (id, title, date, link, idea, how it maps to pls)

### Takeaway
Fourteen core articles map directly to pls questions. The highest-yield ones are 11664 (AttnRes memoir), 9009 (Pre-Norm's watered depth), 8907/8896 (main/auxiliary gradient projection) and 11647/11729 (head geometry and norm constraints).

### Cited Findings
- 9009, 为什么Pre Norm的效果不如Post Norm？, 2022-03-29 — [link](https://kexue.fm/archives/9009)
  - Idea: Pre-Norm stream is O(t); adjacent layers collapse into a wider layer.
  - Maps to: deletable layers; last-layer redundancy.
- 8620, 浅谈Transformer的初始化、参数化与标准化, 2021-08-17 — [link](https://kexue.fm/archives/8620)
  - Idea: Pre-Norm at init gives F_k(x_k/√k); a ReZero-style gate is best raised on a fixed schedule.
  - Maps to: gain init 1/√ℓ vs LNS (fixed vs learnable).
- 8747, 模型优化漫谈：BERT的初始标准差为什么是0.02？, 2021-11-08 — [link](https://kexue.fm/archives/8747)
  - Idea: the final LN gain inflates to scale logits; Pre-Norm intermediates need an LN before a head.
  - Maps to: last MLP anomaly; logit lens.
- 8978, 训练1000层的Transformer究竟有什么困难？, 2022-03-09 — [link](https://kexue.fm/archives/8978)
  - Idea: update explosion; DeepNorm α and λ for SGD, Adam and LAMB.
  - Maps to: DeepNorm in a MuonH recipe; the LAMB row is closest to MuonH.
- 8994, 为什么需要残差？一个来自DeepNet的视角, 2022-03-19 — [link](https://kexue.fm/archives/8994)
  - Idea: branch scale ε = 1/√N suffices in high dimension (expansion 1 + ε²).
  - Maps to: LNS and 1/√ℓ arguments.
- 8998, RoFormerV2：自然语言理解的极限探索, 2022-03-21 — [link](https://kexue.fm/archives/8998)
  - Idea: Post-Norm with α ramped linearly from 0 to 1 beat ReZero.
  - Maps to: fixed schedule vs learnable depth scale.
- 8990, 门控注意力单元（GAU）还需要Warmup吗？, 2022-03-11 — [link](https://kexue.fm/archives/8990)
  - Idea: a branch two orders smaller than the stream acts as a built-in DeepNorm.
  - Maps to: branch-to-stream ratio as a diagnostic.
- 11664, Attention Residuals 回忆录, 2026-03-19 — [link](https://kexue.fm/archives/11664)
  - Idea: AttnRes with a static query and RMSNorm keys; HC as layer attention; embedding as its own block; sparse loses to compressed.
  - Maps to: outsized source = embedding; static vs dynamic routing; HC A-matrix analysis.
- 11848, 简单谈谈K3的MoE和Attention, 2026-08-04 — [link](https://kexue.fm/archives/11848)
  - Idea: K3 uses Block AttnRes; SiTU softcaps for SwiGLU outliers; MTP vs MLA.
  - Maps to: K3 design context; activation outliers.
- 8907, 多任务学习漫谈（三）：分主次之序, 2022-02-14 — [link](https://kexue.fm/archives/8907)
  - Idea: main/auxiliary projection, λ = relu(−⟨g₀,g₁⟩)/‖g₁‖².
  - Maps to: a parameter-free fix for the last-layer cost.
- 8896, 多任务学习漫谈（二）：行梯度之事, 2022-02-08 — [link](https://kexue.fm/archives/8896)
  - Idea: Pareto direction; cheap hidden-state gradient bound; error in Sener & Koltun's proof.
  - Maps to: cheap per-layer conflict measurement.
- 8870, 多任务学习漫谈（一）：以损失之名, 2022-01-18 — [link](https://kexue.fm/archives/8870)
  - Idea: weights 1/L_init, 1/L_prior, 1/sg(L), power mean, gradient-norm normalization.
  - Maps to: per-layer loss weighting arms.
- 11340, MuP之上：1. 好模型的三个特征, 2025-10-21 — [link](https://kexue.fm/archives/11340)
  - Idea: forward, dependency and update stability, each Θ(1).
  - Maps to: framework for per-layer diagnostics.
- 11605, MuP之上：2. 线性层与最速下降, 2026-02-15 — [link](https://kexue.fm/archives/11605)
  - Idea: Muon from spectral update stability; Post-Norm is also In Norm.
  - Maps to: MuonH background.
- 11647, MuP之上：3. 特殊情况特殊处理, 2026-03-02 — [link](https://kexue.fm/archives/11647)
  - Idea: embedding uses row-normalized SGD; LM head uses column-normalized SGD with Θ(1/d) init and LR.
  - Maps to: optimizer group and scaling for separate heads.
- 11729, MuP之上：4. 坚守参数的稳定性, 2026-04-24 — [link](https://kexue.fm/archives/11729)
  - Idea: bound norms (Post Clip / Pre Decay); fixing a norm "破坏原本的训练动力学" (distorts the training dynamics).
  - Maps to: a critique of MuonH; clip arms for gains and AttnRes queries.
- 11416, Muon优化器指南：快速上手与关键细节, 2025-11-19 — [link](https://kexue.fm/archives/11416)
  - Idea: no Muon on the embedding or LM head; LR conversion between versions.
  - Maps to: check the optimizer groups of the heads.
- 10739, Muon续集：为什么我们选择尝试Muon？, 2025-02-27 — [link](https://kexue.fm/archives/10739)
  - Idea: Muon needs weight decay to keep norms bounded.
  - Maps to: MuonH's no-weight-decay choice.
- 11241, 流形上的最速下降：4. Muon + 谱球面, 2025-08-21 — [link](https://kexue.fm/archives/11241)
  - Idea: Muon on a spectral sphere.
  - Maps to: alternative to MuonH.
- 11196, 流形上的最速下降：1. SGD + 超球面, 2025-08-01 — [link](https://kexue.fm/archives/11196)
  - Idea: steepest descent on a hypersphere.
  - Maps to: MuonH's constraint family.
- 9889, 注意力机制真的可以"集中注意力"吗？, 2023-12-12 — [link](https://kexue.fm/archives/9889)
  - Idea: sharpness S(a) = exp(−σ²‖q‖²/2).
  - Maps to: near-binary AttnRes reads as a ‖w‖ temperature effect.
- 11371, 低精度Attention可能存在有偏的舍入误差, 2025-10-27 — [link](https://kexue.fm/archives/11371)
  - Idea: biased BF16 rounding with tied row maxima.
  - Maps to: precision check for saturated depth-softmax.
- 11126, QK-Clip：让Muon在Scaleup之路上更进一步, 2025-07-12 — [link](https://kexue.fm/archives/11126)
  - Idea: MaxLogit explosion under Muon; post-hoc rescaling of Q and K.
  - Maps to: logit growth under norm constraints.
- 9698, 语言模型输出端共享Embedding的重新探索, 2023-07-20 — [link](https://kexue.fm/archives/9698)
  - Idea: near-identity init equals a bigram model; tied-embedding initial loss far above log n.
  - Maps to: early-layer head losses.
- 7818, TeaForN：让Teacher Forcing更有"远见"一些, 2020-10-27 — [link](https://kexue.fm/archives/7818)
  - Idea: predict t+1…t+N by re-running shared weights on outputs.
  - Maps to: MTP (weak link).
- 8027, RealFormer：把残差转移到Attention矩阵上面去, 2020-12-24 — [link](https://kexue.fm/archives/8027)
  - Idea: residuals on attention scores across layers.
  - Maps to: early cross-layer connection.
- 11833, 解构Scaling Law：优化、架构、数据的三重奏, 2026-07-29 — [link](https://kexue.fm/archives/11833)
  - Idea: F_arch ~ α_W W^{−γ_W} + α_H H^{−γ_H}; with γ_W = γ_H = 1 (citing arXiv 2606.25008), W* ~ N^{1/3} and H* ~ N^{1/3}. Su names mHC and AttnRes as not yet covered.
  - Maps to: depth vs width context only.
- 11549, 为什么我们偏爱各向同性？基于最速下降的理解, 2026-01-20 — [link](https://kexue.fm/archives/11549)
  - Idea: normalization layers are cheap whitening; RMSNorm is the hypersphere choice.
  - Maps to: background.

### Inferences
- The most decision-relevant new experiments, in order of cost:
  1. Log ‖w‖ and the read entropy of the AttnRes queries, plus the argmax source index, to test whether the outsized source is the embedding. This needs only existing checkpoints.
  2. Log per-layer RMSNorm gains, with the final gain against the rest, to test 8747's final-gain inflation and the drift of gain init 1/√ℓ.
  3. Measure the per-layer gradient cosine to the final loss on hidden states (8896's bound).
  4. Run the parameter-free 8907 projection or a 1/sg(L_l) weighting arm against the last-layer cost.
  5. Compute the effective HC/mHC A matrix next to AttnRes's (11664's unified view).
  6. Run a "clip, don't pin" arm for gains or queries (11729).

### Gaps
- Not covered on kexue.fm, despite searching:
  - Sandwich-LN, LayerNorm Scaling, KEEL, Peri-LN.
  - MoDA.
  - Depth-μP.
  - Logit lens and tuned lens.
  - Deep supervision and early exit for LLMs.
  - MTP for pretraining.
  - Massive activations and attention sinks.
- 11854 (LM-loss alternatives) and 10831 (DyT, a normalization substitute) were found but not mapped to a finding.
