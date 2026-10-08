# pls literature survey: per-layer readouts, nested depth, deep narrow transformers (2026-10-08)

**One sentence.** Published methods report "every exit as good as an independent model" mainly when regularization limits the
models (IWSLT, CIFAR, ASR, fine-tuning). The few results from compute-limited LM pretraining have either a few exits
with low weights (EE-LLM: 2 exits, weights 0.1–0.5) or a small final-layer cost (LayerSkip 1.5B). The one mechanism
with direct evidence against the final-layer cost is cross-layer connectivity: MSDNet shows that one intermediate
classifier costs a ResNet up to 7 points of final accuracy and costs a DenseNet almost nothing. Our codebase already has
the transformer analogues of this mechanism (HC, Full AttnRes) under `ARCH=`.

Scope and method. I searched arXiv, Semantic Scholar, OpenReview, ar5iv and blogs with WebSearch and WebFetch. Each
entry below gives the method, the setting, the numbers and the arXiv id. Entries tagged **[fetched]** were checked
against the paper text in this session. Entries tagged **[memory]** come from my knowledge of the paper and were not
re-read in this session; check them before citing them in the blog. I tag anything I could not find as **[not found]**.
The figures read from plots are approximate, and I mark them "≈".

Our numbers, for reference: separate heads at w=1 have per-layer gaps of 0.168 0.132 0.072 0.073 0.090 0.110 nats to the
depth controls (3.971 … 3.260), with a mean of 0.107. At w=0.05 the last-layer gap is 0.021 and the first-layer gap is 0.39.
The cosine between the aux gradient and the final gradient on the trunk is 0.8–0.9, and PCGrad is a no-op.

---

## 1. Deep supervision and auxiliary intermediate losses

**DSN, Deeply-Supervised Nets (Lee, Xie, Gallagher, Zhang, Tu; AISTATS 2015; arXiv 1409.5185) [fetched].** Each hidden
layer gets a companion SVM-style objective. The companion term has a hinge threshold γ: once a layer's companion loss
falls below γ, the term "vanishes and no longer plays role". The authors also suggest decaying the weight α_m toward 0
over training. Their stated rationale is that the companion objectives must not change the goal of the output layer.
They act as a regularizer early in training and switch off later. The paper has no ablation of the schedule. Relevance:
DSN switches the companion losses off, so it never targets readouts that stay good at every layer.

**GoogLeNet auxiliary classifiers (Szegedy et al.; CVPR 2015; arXiv 1409.4842) [memory].** Two auxiliary classifiers on
intermediate Inception modules are weighted by 0.3 during training and discarded at inference. The paper later reports
that they help little. The design goal is regularization of the final classifier.

**Al-Rfou et al., Character-Level Language Modeling with Deeper Self-Attention (AAAI 2019; arXiv 1808.04444) [fetched].**
A 64-layer character transformer gets a separate classifier on every intermediate layer, plus multi-position and
multi-target losses. Layer l stops contributing after a fraction l/(2n) of training. With 4M steps and 64 layers, one
more layer's loss is dropped every 62.5K steps. After half of training only the final loss remains. Ablation on enwik8:
the full T64 model scores 1.062 bpc, and removing the intermediate-layer losses gives 1.158 bpc (+0.096). Relevance:
(a) this is the strongest published evidence that per-layer LM losses help optimization at large depth (64 layers),
which matters for our 48-layer strand. (b) The schedule removes the losses bottom-first and keeps the deep ones longer,
so this model never has good readouts at every layer. (c) The architecture is a 2018 Post-LN-era transformer. The benefit
may come from fixing an optimization problem that Pre-LN and Sandwich-LN already avoid.

**MSDNet (Huang et al.; ICLR 2018; arXiv 1703.09844) [fetched].** This paper diagnoses why early exits hurt plain chains.
In its Figure 3, one intermediate classifier on a ResNet lowers the final classifier's accuracy by up to 7%. On a
DenseNet the drop is minimal. The authors' explanation is that the intermediate classifier makes early features
"optimized for the short-term", which "collapses information required to generate high quality features in later layers".
Dense connectivity lets later layers bypass the features that the early exit shaped. The second fix is multi-scale
features, so that coarse features exist early. MSDNet weights all exit losses equally (w_k = 1). Relevance: **this is the
mechanism behind our final-layer cost, and the paper proposes a cure.** Our trunk is a plain chain, because the residual
stream is a single sum that every readout must decode.

**Shallow-Deep Networks / overthinking (Kaya, Hong, Dumitras; ICML 2019; arXiv 1810.07052) [fetched].** The authors attach
internal classifiers (ICs) through a pooling and feature-reduction layer and one FC layer. In end-to-end SDN training,
each IC weight τ_i rises linearly from 0.01 to C_i, where C_i is the IC's relative inference cost, so deeper ICs get more
weight. In IC-only training the backbone is frozen, so the original accuracy is kept exactly. This is our "probes only".
End-to-end SDN training "often improves" final accuracy on CIFAR and Tiny ImageNet, for example VGG-16 on Tiny ImageNet
58.6 → 63.1 for the best IC. Those datasets are small, and regularization limits the models. The paper also defines
overthinking: an IC is correct and the final layer is wrong. Relevance: a depth-proportional weight with a warm-up
ramp. We tried weights that rise with depth, but not the ramp from 0.01.

**Improved Techniques for Training Adaptive Deep Networks, IMTA (Li, Zhang, Qi, Yang, Huang; ICCV 2019; arXiv
1908.06294) [fetched].** The paper has three parts:
1. **Gradient Equilibrium (GE).** An identity-forward, scaled-backward operator R(x;s) on the trunk and branches gives
   block i the gradient (1/(k−i+1))·Σ_{j≥i} ∇L_j. This is a mean over the exits at or after block i, not a sum. The
   authors prove that the gradient variance stays bounded.
2. **Inline Subnetwork Collaboration.** Exit i's logits feed exit i+1 with the gradient detached.
3. **One-for-all KD.** L_i = α·CE_i + (1−α)·KL(p_final ‖ p_i) at temperature T, with the deepest exit as the teacher.

On CIFAR-100 the full method improves MSDNet at every exit by 1.6–2.1 points, for example exit 5: 71.81 → 73.45.
Relevance: KD and the detached logit cascade are untried by us. GE is a per-block uniform rescale; section 6 explains why
it does nothing under MuonH.

**Boosted Dynamic Neural Networks, BoostNet (Yu, Li, Hua, Huang, Shi; AAAI 2023; arXiv 2211.16726) [fetched].** The exit
prediction is additive: F_n = t_n·sg(F_{n−1}) + f_n(x), with t_n = 0.5. The gradient into F_{n−1} is disabled even
where F_{n−1} shares parameters with f_n. The gradient is rescaled as ∂L/∂b_n = (1/(N−n+1))·Σ_{i≥n} ∂L_i/∂b_n. On
CIFAR-100 and ImageNet every exit improves over MSDNet, including the last (CIFAR-100: 75.98 → 76.30). Relevance:
additive exits mean that layer k encodes only a correction to exit k−1. This removes the requirement that every prefix
recompute the full prediction. We have not tried it.

**Zero Time Waste (Wołczyk et al.; NeurIPS 2021; arXiv 2106.05409) [fetched abstract].** Each internal classifier reuses
its predecessors' predictions through cascade connections and a geometric ensemble. It is the same family as BoostNet
and IMTA-ISC.

**Meta-GF (ECCV 2022) [fetched abstract]** and **Deep Feature Surgery (Gong et al.; ECCV 2024; arXiv 2407.13986) [fetched
abstract].** Both name gradient conflict between exits as "the major cause" of accuracy loss in multi-exit networks.
Meta-GF fuses the exits' gradients with meta-learned weights. DFS partitions the shared features along depth so that
different exits own different parts. Relevance: our measured cosine is 0.8–0.9 and PCGrad is a no-op, so in our setting
the cost is not in the gradient *direction*. The DFS idea (give exits private feature slices) still applies; it is a
capacity argument, not a conflict argument.

**Be Your Own Teacher (Zhang et al.; ICCV 2019; arXiv 1905.08094) [fetched abstract].** Self-distillation from the deepest
section into the shallow sections, using soft labels and feature hints. Accuracy gains average 2.65% across CNNs, and
the full network improves too.

**Depth-Adaptive Transformer (Elbayad, Gu, Grave, Auli; ICLR 2020; arXiv 1910.10073) [fetched].** Decoder exits on
IWSLT De-En. Aligned training puts an exit loss at every decoder block with uniform weights; the paper reports uniform
weights as better than ω_n = n, √n, 1/√n and 1/n. In Table 1 the aligned model is at least as good as the separately
trained n-block baselines at every depth: n=1 34.1 vs 34.2, n=2 35.5 vs 35.3, n=3 35.8 vs 35.6, n=4 36.1 vs 35.7,
n=5 36.1 vs 35.6, n=6 36.2 vs 35.9 BLEU. Sampled-exit ("mixed") training is worse than aligned training. **This is the
only published table I found with exactly our comparison: every exit against an independently trained model of that
depth.** The caveats are a small dataset (IWSLT, about 160K sentence pairs), an encoder-decoder model (the encoder is
full depth and unsupervised) and BLEU instead of CE. In that regime, extra losses act as regularization.

**CALM, Confident Adaptive Language Modeling (Schuster et al.; NeurIPS 2022; arXiv 2207.07061) [fetched].** T5
fine-tuning uses one shared output head for all decoder layers and the loss Σ_i ω_i L_i with ω_i = i/Σj, so deeper
layers get more weight. The full model's ROUGE-L on CNN/DM drops by 0.08 (38.32 → 38.24). Relevance: this is a
fine-tuning regime with a shared head and depth-proportional weights. It is close to our shared-head arm with weights
rising with depth.

**LayerSkip (Elhoushi et al.; ACL 2024; arXiv 2404.16710) [fetched].** The method combines layer dropout and an
early-exit loss through one shared LM head:
- **Layer dropout.** p_{l,t} = S(t)·D(l)·p_max. D(l) is exponential in depth, from 0 to 1; S(t) is exponential in time
  for pretraining from scratch; p_max = 0.1–0.2.
- **Exit weights.** Normalized as ẽ(t,l) = C(t,l)e(l)/Σ_i C(t,i)e(i), with a quadratic depth scale e(l) = e_scale·Σ_{i≤l} i.
  The last layer gets a larger fixed scale.
- **Exit curricula.** Rotational: only every R-th layer has an exit loss, and the set rotates. Gradual: exits switch on
  from the top layer down.

In pretraining from scratch (Llama 1.5B and 7B, 26B tokens), the final-layer perplexity is ≈2.0 → ≈2.3 at 1.5B (a
regression, read from the plot) and roughly unchanged at 7B. The middle-layer perplexity drops from ≈1e5 to ≈3–4. The
authors report a 3% final-layer regression in one continual-pretraining setting. Relevance: the rotational curriculum
cuts exit-head FLOPs by a factor R. This matters for 47 heads at d48. LayerSkip also gives a weighting recipe for many
layers whose total exit weight is normalized.

**DeeBERT (Xin et al.; ACL 2020; arXiv 2004.12993), FastBERT (Liu et al.; ACL 2020; arXiv 2004.02178), PABEE (Zhou et
al.; NeurIPS 2020; arXiv 2006.04152) [memory].**
- DeeBERT trains in two stages. First the backbone and final classifier train; then the backbone is frozen and the
  off-ramps train. This is our probes-only arm.
- FastBERT also freezes the backbone, and trains the student classifiers by KL to the final classifier
  (self-distillation).
- PABEE trains all layers jointly with a weighted average of losses, weights ∝ layer index.

All three are fine-tuning methods. None reports exits that match independent shallow models.

**LeeBERT (Zhu; ACL 2021) [fetched abstract].** Learns the exit-loss weights by bi-level optimization on a held-out split,
and adds mutual distillation between exits.

**EE-LLM (Chen, Pan, Li, Ding, Zhou; ICML 2024; arXiv 2312.04916) [fetched].** Early-exit pretraining at scale. The 1.3B
model has exits at 1/4 and 1/2 depth with weights 1/4 and 1/2; the 7B model uses weights 0.1 and 0.2. Each exit is the
output embedding with an optional LayerNorm, and tied or untied embeddings are both options. The finding: "the
final-exit loss curve of each early-exit model is close to (or even slightly below) that of the standard model". The
authors recommend exits in middle pipeline stages, not the first or last. Relevance: no final-layer cost in pretraining,
but with only 2 exits, low weights and deep models. This matches our w=0.05 point (last-layer gap 0.021).

**Training early-exit ASR from scratch (Wright et al.; ICASSP-W 2024; arXiv 2309.09546) [fetched abstract].** ASR
models trained from scratch on the joint loss of all exits beat single-exit models, including at the final layer. ASR
datasets are small, so these models are limited by regularization.

**LLAL (Jin et al. 2026, blog "Mitigate Silent Expert Death in Ultra-Sparse MoE") [fetched secondary summary].** A
temporary LM loss on one early MoE layer (4K of 135K steps at 180B) prevents the lower-layer experts from collapsing.
The project's own pls.md already covers this.

**Greedy local learning and InfoPro (Wang, Ni, Song, Yang, Huang; ICLR 2021; arXiv 2101.10832) [fetched].** When local
modules are trained greedily, the early modules "collapse some of the task-relevant information that may be useless for
short-term performance". On CIFAR-10 with ResNet-32 split into K=16 modules, the error is 24.59% for greedy training,
12.75% with InfoPro and 7.37% end-to-end. InfoPro adds an information-preservation term, I(h, x), to each local loss.
Relevance: this is the same diagnosis as MSDNet's, for the extreme case of our "local heads" arm.

## 2. Nested, anytime and matryoshka depth

**Matryoshka Representation Learning (Kusupati et al.; NeurIPS 2022; arXiv 2205.13147) [memory].** Nested embedding
prefixes are trained with a summed loss, and each prefix matches an independently trained embedding of that size. The
nesting is over width or dimensions, not depth.

**MatFormer (Devvrit et al.; NeurIPS 2024; arXiv 2310.07707) [fetched abstract].** Nested FFN widths inside each block,
trained jointly. Extracted sub-models are reported "as accurate as" independently trained ones. The nesting is over
width only, and all layers stay.

**Flextron (Cai et al.; ICML 2024; arXiv 2406.10260) [fetched abstract].** An elastic width/depth LLM obtained by
post-training an existing model with a router. It uses 7.63% of the pretraining tokens. Not pretraining from scratch.

**SortedNet (Valipour et al.; 2023; arXiv 2309.00255) [fetched abstract]** and **Sorted LLaMA / SoFT (Kavehzadeh et al.;
EACL Findings 2024; arXiv 2309.08968) [fetched abstract].** Nested sub-networks with shared parameters, sampled per step
"in a sorted and probabilistic manner". SortedNet keeps at least 96% of the original performance across 160 sub-models.
Sorted LLaMA applies this to LLaMA-2 13B *fine-tuning* with the shared head at every layer. The paper reports that
intermediate layers become usable and that the doubled speed comes "while maintaining or exceeding performance" on
PandaLM. These results are "≥96% of the full model" claims, not "equal to an independent k-layer model" claims.

**Slimmable networks (Yu et al.; ICLR 2019; arXiv 1812.08928) [memory]** and **US-Nets (Yu & Huang; ICCV 2019; arXiv
1903.05134) [fetched abstract].**
- **Sandwich rule.** At each step, train the smallest width, the largest width and n−2 random widths. The bounds
  "implicitly optimize all sub-networks".
- **Inplace distillation.** The sub-networks train on the full network's soft predictions from the same step,
  detached.
- The universally slimmable MobileNets beat individually trained models at many widths.
- Caveat: width nesting with BatchNorm recalibration differs from depth nesting with a shared residual stream.

**Matryoshka Language Model Suites (Godey & Artzi; Aug 2026; arXiv 2608.09703) [fetched].** The most relevant recent
pretraining result. Sub-models of 500M, 1.5B and 3B are nested, with θ_1 ⊂ θ_2 ⊂ θ_3 except for the LM heads, which are
separate per sub-model. A larger sub-model adds layers *and* width: the layer split is (24, 10, 5) blocks, with
concatenated fresh embedding dimensions. Each sub-model's loss is L_m = (1−α_d)·CE_m + α_d·CE(p_teacher → p_m), with
the largest sub-model as the teacher and α_d = 0.3. Ablations show the distillation is worth 0.15 PPL, most of it on the
small sub-models. Rescaling the norm at the junction between sub-models is worth 0.2 PPL. The result is "near-parity
(within 0.5 average points at every size)" with independently trained baselines, at 36% less training compute. I could
not extract per-size perplexity gaps for the largest sub-model. Relevance: KD from the largest sub-model, with
α = 0.3, is the component that brings nested pretraining to near parity. We have not tried it. Their nesting adds width
with depth, which gives the deep part fresh capacity that a pure depth prefix lacks.

**LayerDrop (Fan, Grave, Joulin; ICLR 2020; arXiv 1909.11556) [fetched].** Structured layer dropout with p = 0.2 and
"every other layer" pruning. On WikiText-103 the 16-layer model goes from 18.7 to 18.3 PPL with LayerDrop. The pruned
RoBERTa at 6 layers scores 82.9 MNLI, against 81.6 for DistilBERT and lower for training from scratch. This is
prunable depth, but every-other-layer, not a prefix.

**Stochastic depth (Huang et al.; ECCV 2016; arXiv 1603.09382) [memory].** Linear decay to p_L = 0.5. It regularizes and
shortens the expected depth.

**Don't Drop Dropout (Elhoushi et al.; ICML 2026; arXiv 2609.05275) [fetched].** Over 2400 runs at 271M–8.2B parameters
and up to 160B tokens, with 13–96 layers. Layer dropout with an increasing depth distribution and a *decreasing* time
schedule gives "lower loss at the same training FLOPs" and up to 25% FLOP savings at similar loss. Early-exit loss
(without exit heads) is always better than in a dense model. There is no explicit exit loss, and I found no table of
early-exit loss against an independent shallow model. Relevance: layer dropout is a cheap, published, at-scale way to
make prefixes decodable without paying at full depth. We have not tried it.

**Mixture-of-Depths (Raposo et al.; 2024; arXiv 2404.02258) [memory]** and **NAG / norm-agnostic residual MoD (Figliolia
& Millidge; Jun 2026; arXiv 2606.16112) [fetched abstract].** These route per token, not by exiting a prefix. NAG
reports that 20–25% skipping matches the full-depth baseline at equal training compute. Not directly about readouts.

**Looped and recurrent-depth models with per-iteration losses [mixed].**
- Universal Transformers (Dehghani et al.; 2018; arXiv 1807.03819) [memory] use ACT and no per-step LM loss.
- Ouro / LoopLM (arXiv 2510.25741) [fetched abstract] trains with per-loop-step LM losses weighted by a learned exit
  distribution plus an entropy regularizer, at 7.7T tokens.
- Huginn recurrent depth (Geiping et al.; 2025; arXiv 2502.05171) [memory] trains with random iteration counts and no
  per-iteration loss.

Relevance: a shared block makes every exit the "same" computation, which removes the specialization conflict by
construction. It does not transfer to our untied layers.

**2025–2026 work on early exits or per-layer LM losses in pretraining [searched].**
- The Diminishing Returns of Early-Exit Decoding in Modern LLMs (Wei et al.; Mar 2026; arXiv 2603.23701) [fetched
  abstract]: newer pretraining recipes reduce layer redundancy and therefore the early-exit opportunity.
- NITP (arXiv 2605.24956) [abstract via search] reports that "supervision on intermediate layers brings only negligible
  performance gains" for final quality; I did not read the details.
- **[not found]** I found no 2025–2026 paper that pretrains a decoder LM from scratch with an exit at *every* layer and
  reports each exit against independently trained k-layer models. Our Figure 3 comparison appears to be new.

## 3. Probes and readouts of intermediate layers

**Logit lens (nostalgebraist, LessWrong 2020) [memory].** The final norm and unembedding are applied to intermediate
residuals of GPT-2. The predictions converge gradually, and the mid-layer outputs are often unreadable in other models.

**Tuned lens (Belrose et al.; 2023; arXiv 2303.08112) [fetched abstract].** A per-layer affine translator (A_l, b_l) is
fitted on a frozen model by KL to the final logits. It is more predictive and less biased than the logit lens, up to
20B parameters, and the perplexity falls monotonically with depth. Our probes-only arm is a tuned lens with a free
lm_head per layer trained on CE to the labels instead of KL to the model. The linear-algebra content is the same, plus a
free per-layer RMSNorm.

**Future Lens (Pal et al.; CoNLL 2023; arXiv 2311.04897) [memory].** A single mid-layer hidden state of GPT-J-6B predicts
tokens several positions ahead well above chance. Hidden states carry more than the next token, and per-layer
next-token losses may compete with this "future" content.

**Where prediction forms.**
- Do Language Models Use Their Depth Efficiently? (Csordás, Manning, Potts; NeurIPS 2025; arXiv 2505.13898) [fetched
  abstract]. The second-half layers contribute much less than the first-half layers. In a linear-map study, the layers
  of a deeper model map best to the layers *at the same relative depth* in a shallower model: "the larger model simply
  spreads the same computations out over its many layers".
- What Affects the Effective Depth of LLMs? (Hu, Zhou, Zhang; Dec 2025; arXiv 2512.14064) [fetched abstract]. The
  effective-depth *ratio* is stable across Qwen-2.5 1.5B–32B and does not grow for harder tasks.
- The Unreasonable Ineffectiveness of the Deeper Layers (Gromov et al.; 2024; arXiv 2403.17887) [memory] and Lad et
  al., stages of inference (2024; arXiv 2406.19384) [memory] report the same redundancy.
- Relevance: the Csordás result explains our probes-only gap (1.10 nats at layer 2 against the 2-layer model). A 6-layer
  model's layer 2 is not a 2-layer model's last layer. It is one third of a stretched computation. Per-layer supervision
  forces every prefix to be a complete predictor, which removes the stretching. The deep model may need that stretching
  for its full-depth advantage. This is a hypothesis, not a published result.

**Training-time supervision against probes.** LayerSkip's middle-layer perplexity falls from ≈1e5 to ≈3 when the same
head is supervised at every layer (section 1). Training changes the representation, and probes only read it. DeeBERT,
FastBERT and SDN-IC-only (frozen backbone) keep the final quality exactly, with weaker exits. That is our probes-only
arm. I found no paper that measures how much of a supervised model's exit advantage over probes comes from the features
and how much from the heads. Our head-only refits answer this for our setting: the advantage lies in the features.

## 4. Methods that claim all-depth readouts at no full-depth cost

The relevant question is whether a method shows *every* exit at or below an independent model of that depth, with the
last exit not worse than the full model.

| Method | Citation | Setting | Claim | Evidence that every exit ≤ independent? | Relevance to our goal | Tried by us? |
|---|---|---|---|---|---|---|
| Equal-weight exits (aligned) | Elbayad 2020, 1910.10073 | IWSLT decoder exits, 6 blocks | every exit ≥ separately trained n-block model | **Yes**, Table 1, including n=6 (36.2 vs 35.9) | Exactly our comparison, but regularization-bound | yes (w=1) |
| Depth-weighted exits | CALM 2207.07061; PABEE 2006.04152 | T5/BERT fine-tuning | full model −0.08 ROUGE-L | No (exits not compared with independent models) | Same as our depth-rising weights | yes |
| Low-weight, few exits | EE-LLM 2312.04916 | 1.3B/7B pretraining, 2 exits, w 0.1–0.5 | final loss ≈ or < baseline | Final only | Matches our w=0.05 end of the trade-off | yes (w sweep) |
| Hinge or decay switch-off | DSN 1409.5185; Al-Rfou 1808.04444 | CNNs; 64-layer char LM | final improves | Final only; exits are dropped | Our switch-off at step 80 | partly (switch-off; not bottom-up l/2n) |
| Frozen backbone + exits | SDN IC-only 1810.07052; DeeBERT; FastBERT | CNN; BERT fine-tuning | final unchanged | Final only | Our probes-only arm | yes |
| Warm-up ramp of exit weights | SDN 1810.07052 (τ: 0.01 → C_i) | CNNs | final ≥ baseline (small data) | Final only | Untried variant of late start | partly (late start is a step, not a ramp) |
| Gradient rescaling (mean over later exits) | IMTA-GE 1908.06294; BoostNet 2211.16726 | MSDNet, CIFAR/ImageNet | all exits improve over MSDNet | vs MSDNet, not vs independent models | **No-op under MuonH** (section 6) | no (skip) |
| KD from final exit to all exits | BYOT 1905.08094; IMTA-OFA; FastBERT; US-Nets inplace KD 1903.05134; Matryoshka-LM 2608.09703 | CNNs, BERT, 3B nested LM pretraining | sub-models near independent models | US-Nets and Matryoshka-LM: near parity, all sizes | **High**: the component that closes nested-LM gaps | **no** |
| Additive (boosted) exits, detached previous logits | BoostNet 2211.16726; ZTW 2106.05409; IMTA-ISC | CIFAR/ImageNet | all exits improve, final too (75.98 → 76.30) | vs MSDNet | Layer k only has to encode a correction | **no** |
| Dense / cross-layer connectivity | MSDNet 1703.09844 (DenseNet vs ResNet); DenseFormer 2402.02622; HC 2409.19606; AttnRes 2603.15031 | CNNs; LM pretraining | intermediate classifier barely hurts the DenseNet final (vs −7% for ResNet) | Final only (the mechanism) | **Highest**: attacks the cause; code exists (`ARCH=hc`, `attnres`) | **no** (at 6 layers with sep heads) |
| Feature partitioning per exit | Deep Feature Surgery 2407.13986 | CNN multi-exit | +up to 6.94% accuracy | vs multi-exit baselines | Private slices of the residual for exits | no |
| Random exit sampling (sandwich rule) | US-Nets 1903.05134; SortedNet 2309.00255; LayerSkip rotational 2404.16710 | width nets; LLaMA fine-tuning; LLM pretraining | ≥96% of full; cheaper | Width only (US-Nets) | Cost cut (head = 58% FLOPs); DAT finds mixed sampling worse | no |
| Layer dropout (+ exits) | LayerSkip 2404.16710; LayerDrop 1909.11556; Don't Drop Dropout 2609.05275 | LLM pretraining 271M–8.2B | final loss ≈ or < baseline at equal FLOPs; exits much better than dense | Final + early-exit loss vs dense, not vs independent | Cheap, published at scale | no |
| Learned / DRO exit weights | LeeBERT (ACL 2021); GradNorm 1711.02257; GroupDRO 1911.08731 | BERT; multitask | balances objectives | No | Target the *max gap* with known control CEs | no |
| Nonlinear exit adapters (MLP or block) | EE-LLM option; SDN feature reduction | LLM; CNN | flexible exits | No | Relaxes the linear-decodability constraint; costs compute | no |
| Information preservation in local losses | InfoPro 2101.10832 | local learning, CNNs | closes most of the greedy-to-E2E gap | vs E2E final | Explains the cost of our local heads | no |
| Separate norm + head per exit | MSDNet, SDN, Matryoshka-LM, DAT (IWSLT) | various | standard | – | Already our setup | yes |
| Stop-gradient into trunk (probes) | DeeBERT, SDN IC-only, tuned lens | – | final unchanged | – | Our probes-only arm | yes |
| Gradient surgery | PCGrad 2001.06782; Meta-GF | – | conflict resolution | – | No-op for us (cos 0.8–0.9) | yes |

**Summary of section 4.**
- Only one paper shows every exit at or below an independent model of the same depth: the Depth-Adaptive Transformer,
  on a small translation dataset.
- Two nested-model families report near parity at every size, and both use KD from the largest model: US-Nets (width)
  and Matryoshka LM Suites (width + depth, pretraining).
- Only one mechanism has direct evidence of removing the cost at the final exit: dense connectivity (MSDNet). BoostNet
  and IMTA report a better final exit than MSDNet, but MSDNet already has dense connectivity.

**Methods we have NOT tried, ranked by expected payoff per cost at 130m.**
1. Cross-layer connectivity with separate heads (HC, Full AttnRes, DenseFormer-style DWA). The code exists.
2. KD from the final exit to every exit (inplace distillation, α ≈ 0.3).
3. Layer dropout (LayerSkip / Don't Drop Dropout schedule) combined with separate heads.
4. Additive or boosted exits with detached previous logits.
5. Gap-targeted (DRO) exit weights that use the known depth-control CEs.
6. Nonlinear exit adapters: first as a frozen-backbone diagnostic, then end-to-end.
7. A bottom-up handoff from separate heads to probes on Al-Rfou's l/(2n) schedule.
8. Random exit sampling, mainly for cost; this matters at d48.
9. Gradient Equilibrium is not worth a run under MuonH (section 6).

## 5. Very deep narrow transformers (48 layers at width 512)

**DepthBench (Wang, Huang, Kang, González-Martínez, W. Liu, S. Liu; Sep 2026; arXiv 2609.32534) [fetched].**

*Setup.* About 400M parameters. Seven shapes from (L=16, d=1216) to (L=70, d=640). 20 tokens per parameter (about 8B
tokens). AdamW (0.9, 0.95), lr 2e-3 (LNS 1e-2), sequence 2048, batch 512, FineWeb-Edu.

*Results at L=70, d=640 (the deepest shape).*
- Pre-LN: ≈2.782, worse than at L=16.
- Sandwich-LN, LNS and KEEL: weak or non-monotonic scaling with depth.
- DeepNorm: diverged at lr 2e-3.
- mHC: no clear gain.
- MoDA: weak.
- HC: 2.699. Full AttnRes: 2.718. Both improve steadily with depth.
- Block AttnRes: 2.730, and it degrades after L=20.

*Layer analyses.*
- Angular distance: Pre-LN representations become "broadly similar across depth".
- Late-layer causal scores above 0.45: about 1% for Pre-LN, 7% for HC, 16% for Full AttnRes.
- Permutation scores above 0.15: 8% for Pre-LN, 35% for HC, 46% for Full AttnRes.
- Logit lens: "markedly less monotonic" under Full AttnRes, which the authors read as late integration of stored
  sub-layer outputs.
- mHC: the composed residual matrices have effective rank 1.44–1.65 (HC: 2.53–2.81), and the stream directions become
  "nearly collinear" at L=32.
- Block AttnRes has a "dead segment": a quarter of the L=32 network rereads the embedding before the first block
  boundary.

*Instabilities.* HC is sensitive to hyperparameters; a 500M HC run at the largest aspect ratio had a gradient explosion.
DeepNorm exploded. LNS needed a 5× learning rate.

*Relevance and predictions for our Figure 6 (all hypotheses).*
1. Under Pre-LN, Sandwich-LN, LNS, KEEL and DeepNorm, the probes-only curve will saturate early, so the second half of
   the 48 layers adds little.
2. Separate heads will force each layer to contribute, so they may *lower* the final-layer cost relative to probes in
   these designs (Al-Rfou 2019 at 64 layers).
3. Under Full AttnRes and HC, the logit lens is already non-monotonic, so probes will look worse in mid-depth and the
   per-layer supervision will conflict more with "late integration".
4. Under HC and Full AttnRes, later layers can also bypass the supervised sum (the MSDNet argument), so the final-layer
   cost of separate heads should be smaller.

These two forces (3 and 4) point in opposite directions. That makes HC and Full AttnRes the informative cells of
Figure 6.

**The depth architectures (ids verified by search).**
- **Sandwich-LN.** CogView, Ding et al. 2021, arXiv 2105.13290 [memory]. Related: Peri-LN (Kim et al.; 2025; arXiv
  2502.02732) [fetched abstract] puts norms around the sublayers and reports balanced variance growth, avoiding the
  massive activations of Pre-LN, up to 3.2B parameters.
- **Pre-LN vs Post-LN.** Xiong et al. 2020, arXiv 2002.04745 [memory]; Admin, Liu et al. 2020, arXiv 2004.08249
  [memory]. Post-LN needs warm-up and careful initialization at depth; Pre-LN is stable but its deep layers are weak.
- **Curse of Depth / LayerNorm Scaling, LNS.** Sun, S. Liu et al.; 2025; arXiv 2502.05795 [fetched abstract]. Pre-LN
  output variance grows with depth, so deep layers approach identity. LNS scales the norm output by 1/√ℓ, and the paper
  reports gains at 130M–1B.
- **Mix-LN.** Li, Yin, S. Liu; ICLR 2025; arXiv 2412.13795 [fetched abstract]. Post-LN in early layers and Pre-LN in
  late layers, 70M–7B.
- **DeepNet / DeepNorm.** Wang et al. 2022, arXiv 2203.00555 [memory]. Trains 1000-layer Post-LN models with α = (2L)^{1/4}.
- **KEEL.** "Post-LayerNorm Is Back: Stable, ExpressivE, and Deep", Chen & Wei, ByteDance Seed; Jan 2026; arXiv
  2601.19895 [fetched abstract]. A Post-LN highway with skip weight α = L (the number of sublayers); stable beyond 1000
  layers.
- **Hyper-Connections.** Zhu et al.; ICLR 2025; arXiv 2409.19606 [memory for the id]. Multiple residual streams with
  learned mixing.
- **mHC.** Xie et al., DeepSeek; arXiv 2512.24880 [fetched abstract]. Sinkhorn projection onto doubly stochastic
  matrices restores the identity mapping and stability.
- **Attention Residuals.** Kimi; Mar 2026; arXiv 2603.15031 [fetched abstract]. Softmax attention over earlier layer
  outputs replaces the fixed residual sum; it fixes "PreNorm dilution". Block AttnRes was used in Kimi Linear 48B-A3B at
  1.4T tokens.
- **MoDA, mixture-of-depths attention.** Mar 2026; arXiv 2603.15619 [fetched abstract]. Heads attend to the KV of the
  current layer and of earlier layers; +2.11% downstream at 3.7% extra FLOPs.
- **DenseFormer.** Pagliardini et al.; NeurIPS 2024; arXiv 2402.02622 [fetched abstract]. A learned depth-weighted
  average after each block. It is not in DepthBench, and it is the cheapest dense-connectivity variant.

**Depth against width at small scale.**
- MobileLLM (Liu et al.; ICML 2024; arXiv 2402.14905) [fetched abstract]: at 125M/350M, deep-and-thin (about 30
  layers) beats wide-and-shallow.
- Levine et al. 2020 (arXiv 2006.12467) [memory]: depth-width interplay; depth helps up to a width-dependent limit.
- DepthBench: Pre-LN at fixed 400M degrades past L=16 when going to d=640.
- "Depth Delusion" (arXiv 2601.20994) claims a critical depth D_crit ∝ W^0.44 [abstract via search; low confidence,
  unreviewed].
- Tapered LMs (Bayat, Behrouz, Courville; Jun 2026; arXiv 2606.23670) [fetched abstract]: more MLP capacity in early
  layers improves perplexity.

At width 512, 48 layers is beyond every depth-optimal estimate I found for Pre-LN. The run tests whether the residual
design can make use of the depth.

**Per-layer losses at large depth.** I found three sources: Al-Rfou (64 layers; the losses are essential), LayerSkip
(up to 32+ layers; normalized weights and a rotational curriculum) and Ouro (shared loops). **[not found]**: I found no
study of per-layer losses under HC, mHC, AttnRes, MoDA, KEEL or LNS. The Figure 6 comparison would be new.

**A practical warning for the queued d48 runs.** With w=1 at every one of 47 layers, the total aux weight is 47 against
1 for the final loss. LayerSkip and CALM both normalize the exit weights so that they sum to 1, and both give the final
layer extra weight. Under MuonH only the *direction* of each matrix update matters, so the final loss steers the trunk
with about a 1/48 share. Expect a large final-layer cost at d48 with w=1, unless the run normalizes the weights. Check
which `PLS_W` jobs 14949463/4 use, and add a normalized-weight arm (w = 1/47 per exit, or LayerSkip's ẽ).

## 6. Optimizer pitfalls with many large-vocabulary heads

**What the literature says.**
- LM-head gradient bottleneck (Godey & Artzi; COLM 2026; arXiv 2603.10145) [fetched abstract]: "95–99% of the gradient
  norm is suppressed by the output layer" when V ≫ D. Each of our heads is a separate rank-512 bottleneck for its
  layer's gradient.
- Toward a First-Principles Update Geometry for the LM Head (arXiv 2608.22253) [fetched abstract]: the spectral norm
  is the wrong geometry for the LM head because softmax ignores shared logit shifts. The paper proposes RowNorm. This
  applies to any Muon-style treatment of exit heads.
- Muon practice (Keller Jordan's blog 2024; Moonlight, arXiv 2502.16982 [memory]): embeddings and LM heads go to
  AdamW, not Muon.
- Small-scale proxies for instabilities (Wortsman et al. 2023; arXiv 2309.14322) [memory]: output-logit divergence,
  z-loss, and Adam ε at small gradient scales. LLAL (blog) names AdamW ε as the cause of the collapse of tiny-gradient
  experts.
- **[not found]**: I found no paper on global-norm clipping shared between a main head and many auxiliary heads. AdaGC
  (arXiv 2502.11034) [fetched abstract] argues that global clipping handles parameter-specific scales poorly.

**What our stack does (checked in the code; not from the literature).**
- `MuonHConfig` clips by global norm *inside each optimizer group*, not over all parameters.
- The separate heads (`aux_lm_heads`) share the AdamH group with `lm_head`, and the `aux_norms` share the Adam group
  with the embeddings and all norm gains. The aux heads' gradients therefore set the clip factor for the main head and
  the embeddings.
- The `pls-optsplit` branch (`PLS_OPT_SPLIT=1`, tag `-os`) already gives the aux heads and aux norms their own groups.
- AdamH, like MuonH, is norm-preserving (`lib/levanter/src/levanter/optim/adamh.py`: `norm_preserving_update`). Every
  separate head therefore keeps its initial Frobenius norm. Its logit scale (temperature) can only change through its
  RMSNorm gain, which the Adam group trains at adam_lr.
- The depth controls have the same constraint, so the comparison is fair. Still, compare the aux RMSNorm gains with the
  final norm gain: a shallow exit may need a different temperature than the norm gain gives.

**Reasoned consequences (my inference, not published).**
1. A per-block uniform gradient rescale (IMTA Gradient Equilibrium, BoostNet rescaling, LayerSkip's normalization, and
   the overall scale of w) cannot change a MuonH update. MuonH orthogonalizes each matrix's momentum and fixes the norm
   of its update. Only the *relative* weights between the final loss and the aux losses act. Across steps, they act
   through the momentum.
2. Clip-factor coupling is mostly benign for scale-invariant optimizers (MuonH, AdamH, Adam). A time-varying clip
   factor re-weights the steps inside the momentum and second-moment averages; it does not change a single step's
   direction.
3. At d48 the 47 aux heads hold 47 × V × 512 parameters. With V = 128k this is about 3.1B parameters, plus two AdamH
   moments, which is about 37 GB in fp32 before sharding. The aux heads would cost about 47× the main head in matmul
   FLOPs. The rotational exits of LayerSkip (every R-th layer, rotating) or one sampled exit per step would cut this cost
   by a factor R. Use this for the d48 strand if memory or step time bind.

## 7. Ranked next experiments at 130m (6 layers; compare with the depth controls 3.971 … 3.260)

1. **Separate heads under HC and Full AttnRes, 6 layers (the MSDNet dense-connectivity test).**
   - *Run.* `ARCH=hc` and `ARCH=attnres_full` with `PLS_W=1` separate heads, plus each architecture's plain 6-layer
     run. The last-layer gap needs a matched baseline in the same architecture. The k<6 controls per architecture can
     wait until the last-layer gap moves.
   - *Expected.* Under HC, later layers can read stream combinations that the exits do not decode. Under AttnRes, later
     layers can reread earlier layer outputs. Either way the final-layer gap at w=1 should fall well below 0.110. MSDNet
     saw the final-classifier damage go from up to 7 points (ResNet) to near zero (DenseNet).
   - *Cost.* 4 runs.
   - *Risk.* HC is hyperparameter-sensitive (DepthBench).
2. **KD from the final exit (inplace distillation).**
   - *Loss.* L_k = (1−α)·CE_k + α·CE(sg(p_final), p_k), with α = 0.3 as in Matryoshka LM Suites; also try α = 0.5. The
     extra cost is negligible, because p_final is already computed.
   - *Expected.* The shallow exits improve, because soft targets have lower gradient variance. Matryoshka-LM measured
     0.15 PPL on small sub-models, and US-Nets measured better sub-networks.
   - *Hypothesis on the last layer.* The exits now chase the trunk's own final distribution instead of one-hot labels,
     so less of the trunk's capacity goes to the labels. The final-layer gap may also shrink.
   - *Cost.* 1–2 runs.
3. **Layer dropout plus separate heads (LayerSkip / Don't Drop Dropout recipe).**
   - *Run.* p_max = 0.1, exponential in depth (layer 0 never dropped, the last layer most), decreasing in time. Run it
     with separate heads at w=1 and at w=0.2.
   - *Expected.* Later layers become robust to which prefix they receive, so each prefix can specialize as a predictor
     with less damage to the final layer. Don't Drop Dropout reports equal or lower final loss at equal FLOPs with this
     schedule, up to 8.2B parameters.
   - *Cost.* 2 runs. At 6 layers each dropped layer is a large share of the model, so keep p_max small.
4. **Gap-targeted DRO weights.**
   - *Rule.* w_k ← w_k·exp(η·(CE_k − c_k)), renormalized to Σw fixed, where c_k is the depth-control CE.
   - *Expected.* The weights move along the existing trade-off curve to its minimax point, so the gaps become roughly
     equal. The w-sweep suggests a uniform gap of about 0.07–0.09 nats.
   - *Value.* It does not expand the curve, but it gives the best "every layer close" point and a clean baseline for
     experiments 1–3.
   - *Cost.* 1 run.
5. **Boosted (additive) exits.**
   - *Logits.* logits_k = head_k(h_k) + t·sg(logits_{k−1}), with t = 0.5 as in BoostNet. The final exit stays standalone,
     so the full-depth model is unchanged at inference.
   - *Expected.* The shallow exits (k ≥ 2) improve because each layer only adds a correction. Each layer's linear
     readout also needs to carry less of the full prediction, so the trunk may be less constrained.
   - *Cost.* 1 run.
   - *Caveat.* Running exit k at inference then needs heads 1..k. That costs k× the head FLOPs (58% of FLOPs at 130m),
     so it is a fair "anytime" model only if the head cost is counted.
6. **Nonlinear-adapter diagnostic on frozen backbones, then end-to-end.**
   - *Diagnostic.* Refit each exit with a small SwiGLU MLP (512 → 1024 → 512) before its RMSNorm + head, on (a) the
     separate-heads w=1 backbone and (b) the probes-only backbone. The heads-only refit code exists.
   - *Reading.* Linear refits barely moved. If MLP refits close a large part of the probes-only gap, the prediction
     information is present but not linearly decodable.
   - *Follow-up.* If so, train separate heads with adapters end-to-end. Count the adapter compute: compare exit k with a
     k-layer control plus the same MLP.
   - *Cost.* Refits are cheap; 1 end-to-end run.
7. **Bottom-up handoff from separate heads to probes (Al-Rfou l/(2n) schedule).**
   - *Run.* Layer l's readout gradient into the trunk stops at a fraction l/(2·6) of training. The head keeps training
     as a probe.
   - *Expected.* The final layer recovers toward 3.260, and the shallow exits lose part of their gain. Our switch-off at
     step 80 left no trace, so the payoff is uncertain at 6 layers.
   - *Why run it.* It is the published recipe at 64 layers, so it is more relevant to the d48 strand.
   - *Cost.* 1 run.
8. **Sampled exits per step (sandwich rule: final + exit 1 + one random exit).**
   - *Run.* The weights are unbiased: 5/3 per sampled exit, so that the expected total matches w=1.
   - *Expected.* About the same quality as w=1, at about 60% of the aux-head FLOPs. DAT's "mixed" sampling was slightly
     worse than aligned training, so do not expect a quality gain.
   - *Why run it.* It is a cost tool to validate before d48.
   - *Cost.* 1 run.

Do not run Gradient Equilibrium, gradient rescaling or a global rescale of w by a constant. Under MuonH a per-block
uniform rescale gives the same update (section 6).

## References (arXiv ids; tags as defined above)

**Deep supervision and early exits**
- Lee, Xie, Gallagher, Zhang, Tu. Deeply-Supervised Nets. AISTATS 2015. arXiv:1409.5185 [fetched]
- Szegedy et al. Going Deeper with Convolutions. CVPR 2015. arXiv:1409.4842 [memory]
- Al-Rfou, Choe, Constant, Guo, Jones. Character-Level Language Modeling with Deeper Self-Attention. AAAI 2019. arXiv:1808.04444 [fetched]
- Huang et al. Multi-Scale Dense Networks for Resource Efficient Image Classification (MSDNet). ICLR 2018. arXiv:1703.09844 [fetched]
- Kaya, Hong, Dumitras. Shallow-Deep Networks. ICML 2019. arXiv:1810.07052 [fetched]
- Li, Zhang, Qi, Yang, Huang. Improved Techniques for Training Adaptive Deep Networks. ICCV 2019. arXiv:1908.06294 [fetched]
- Yu, Li, Hua, Huang, Shi. Boosted Dynamic Neural Networks. AAAI 2023. arXiv:2211.16726 [fetched]
- Wołczyk et al. Zero Time Waste. NeurIPS 2021. arXiv:2106.05409 [fetched abstract]
- Meta-GF: Training Dynamic-Depth Neural Networks Harmoniously. ECCV 2022 (arXiv id not checked) [fetched abstract]
- Gong et al. Deep Feature Surgery. ECCV 2024. arXiv:2407.13986 [fetched abstract]
- Zhang et al. Be Your Own Teacher. ICCV 2019. arXiv:1905.08094 [fetched abstract]
- Elbayad, Gu, Grave, Auli. Depth-Adaptive Transformer. ICLR 2020. arXiv:1910.10073 [fetched]
- Schuster et al. Confident Adaptive Language Modeling. NeurIPS 2022. arXiv:2207.07061 [fetched]
- Elhoushi et al. LayerSkip. ACL 2024. arXiv:2404.16710 [fetched]
- Xin et al. DeeBERT. ACL 2020. arXiv:2004.12993 [memory]
- Liu et al. FastBERT. ACL 2020. arXiv:2004.02178 [memory]
- Zhou et al. BERT Loses Patience (PABEE). NeurIPS 2020. arXiv:2006.04152 [memory]
- Zhu. LeeBERT. ACL 2021 (aclanthology 2021.acl-long.231) [fetched abstract]
- Chen, Pan, Li, Ding, Zhou. EE-LLM. ICML 2024. arXiv:2312.04916 [fetched]
- Wright et al. Training dynamic models using early exits for ASR. arXiv:2309.09546 [fetched abstract]
- Wang, Ni, Song, Yang, Huang. Revisiting Locally Supervised Learning (InfoPro). ICLR 2021. arXiv:2101.10832 [fetched]
- Wei et al. The Diminishing Returns of Early-Exit Decoding in Modern LLMs. arXiv:2603.23701 [fetched abstract]
- Jin et al. LLAL (blog, 2026), https://mooler0410.github.io/puguJin/blog/llal/ [secondary summary]

**Nested and anytime models**
- Kusupati et al. Matryoshka Representation Learning. NeurIPS 2022. arXiv:2205.13147 [memory]
- Devvrit et al. MatFormer. NeurIPS 2024. arXiv:2310.07707 [fetched abstract]
- Cai et al. Flextron. ICML 2024. arXiv:2406.10260 [fetched abstract]
- Valipour et al. SortedNet. arXiv:2309.00255 [fetched abstract]
- Kavehzadeh et al. Sorted LLaMA. EACL Findings 2024. arXiv:2309.08968 [fetched abstract]
- Yu et al. Slimmable Neural Networks. ICLR 2019. arXiv:1812.08928 [memory]
- Yu & Huang. Universally Slimmable Networks and Improved Training Techniques. ICCV 2019. arXiv:1903.05134 [fetched abstract]
- Godey & Artzi. Matryoshka Language Model Suites. arXiv:2608.09703 [fetched]
- Fan, Grave, Joulin. Reducing Transformer Depth on Demand with Structured Dropout (LayerDrop). ICLR 2020. arXiv:1909.11556 [fetched]
- Huang et al. Deep Networks with Stochastic Depth. ECCV 2016. arXiv:1603.09382 [memory]
- Elhoushi et al. Don't Drop Dropout. ICML 2026. arXiv:2609.05275 [fetched]
- Raposo et al. Mixture-of-Depths. arXiv:2404.02258 [memory]
- Figliolia & Millidge. Scaling Adaptive Depth with Norm-Agnostic Residual Networks. arXiv:2606.16112 [fetched abstract]
- Dehghani et al. Universal Transformers. ICLR 2019. arXiv:1807.03819 [memory]
- Scaling Latent Reasoning via Looped Language Models (Ouro). arXiv:2510.25741 [fetched abstract]
- Geiping et al. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv:2502.05171 [memory]

**Probes and depth use**
- nostalgebraist. interpreting GPT: the logit lens. LessWrong, 2020 [memory]
- Belrose et al. Eliciting Latent Predictions from Transformers with the Tuned Lens. arXiv:2303.08112 [fetched abstract]
- Pal et al. Future Lens. CoNLL 2023. arXiv:2311.04897 [memory]
- Csordás, Manning, Potts. Do Language Models Use Their Depth Efficiently? NeurIPS 2025. arXiv:2505.13898 [fetched abstract]
- Hu, Zhou, Zhang. What Affects the Effective Depth of Large Language Models? arXiv:2512.14064 [fetched abstract]
- Gromov et al. The Unreasonable Ineffectiveness of the Deeper Layers. arXiv:2403.17887 [memory]
- Lad et al. The Remarkable Robustness of LLMs: Stages of Inference? arXiv:2406.19384 [memory]

**Depth architectures**
- Wang, Huang, Kang, González-Martínez, W. Liu, S. Liu. DepthBench. arXiv:2609.32534 [fetched]
- Ding et al. CogView (Sandwich-LN). arXiv:2105.13290 [memory]
- Kim et al. Peri-LN. arXiv:2502.02732 [fetched abstract]
- Xiong et al. On Layer Normalization in the Transformer Architecture. arXiv:2002.04745 [memory]
- Liu et al. Understanding the Difficulty of Training Transformers (Admin). arXiv:2004.08249 [memory]
- Sun et al. The Curse of Depth in Large Language Models (LNS). arXiv:2502.05795 [fetched abstract]
- Li, Yin, Liu. Mix-LN. ICLR 2025. arXiv:2412.13795 [fetched abstract]
- Wang et al. DeepNet. arXiv:2203.00555 [memory]
- Chen & Wei. Post-LayerNorm Is Back: Stable, ExpressivE, and Deep (KEEL). arXiv:2601.19895 [fetched abstract]
- Zhu et al. Hyper-Connections. ICLR 2025. arXiv:2409.19606 [memory]
- Xie et al. mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880 [fetched abstract]
- Kimi Team. Attention Residuals. arXiv:2603.15031 [fetched abstract]
- Mixture-of-Depths Attention (MoDA). arXiv:2603.15619 [fetched abstract]
- Pagliardini et al. DenseFormer. NeurIPS 2024. arXiv:2402.02622 [fetched abstract]
- Liu et al. MobileLLM. ICML 2024. arXiv:2402.14905 [fetched abstract]
- Levine et al. Limits to Depth Efficiencies of Self-Attention. NeurIPS 2020. arXiv:2006.12467 [memory]
- Bayat, Behrouz, Courville. Tapered Language Models. arXiv:2606.23670 [fetched abstract]
- The Depth Delusion. arXiv:2601.20994 [search snippet only; low confidence]

**Optimization**
- Godey & Artzi. Lost in Backpropagation: The LM Head is a Gradient Bottleneck. COLM 2026. arXiv:2603.10145 [fetched abstract]
- Somasundaram et al. Toward a First-Principles Update Geometry for the Language-Model Head. arXiv:2608.22253 [fetched abstract]
- Liu et al. Muon is Scalable for LLM Training (Moonlight). arXiv:2502.16982 [memory]
- Wortsman et al. Small-scale proxies for large-scale Transformer training instabilities. arXiv:2309.14322 [memory]
- Wang et al. AdaGC. arXiv:2502.11034 [fetched abstract]
- Yu et al. Gradient Surgery for Multi-Task Learning (PCGrad). arXiv:2001.06782 [memory]
- Du et al. Adapting Auxiliary Losses Using Gradient Similarity. arXiv:1812.02224 [memory]
- Chen et al. GradNorm. arXiv:1711.02257 [memory]
- Sagawa et al. Distributionally Robust Neural Networks (GroupDRO). arXiv:1911.08731 [memory]
