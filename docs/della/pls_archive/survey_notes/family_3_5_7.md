# Families 3 / 5 / 7 — verified notes (fork of survey agent, 2026-09-29)

Scope: Family 3 = early-exit / adaptive-depth encoders and multi-exit training analyses; Family 5 = MT / seq2seq; Family 7 = intermediate supervision by alignment / distillation.
Verification: metadata from the arXiv API (export.arxiv.org) unless stated; ACL Anthology / AAAI OJS / ECVA / PMLR / NeurIPS pages for venues. "details from" names the source actually read (ar5iv or arxiv.org/html full text; ACL PDFs read through r.jina.ai).
Relevance scale (target experiment: temporary, annealed next-token LM loss at EVERY layer of a dense decoder LM trained from scratch, through the shared final norm + LM head): 5 = decoder LM with per-layer LM loss in pretraining, with final-layer evidence or a schedule; 4 = same-label per-layer losses in transformers with final-layer evidence or weighting/schedule insight; 3 = deep supervision elsewhere with a transferable insight; 2 = inference-motivated exits or alignment/distillation variants with little training-dynamics insight; 1 = tangential.

Counts (verified): Family 3 = 21, Family 5 = 4 (+1 checked and excluded: 2005.08081 has no supervision), Family 7 = 16. Total = 41.

---------------------------------------------------------------------
## Family 3 — early-exit / adaptive-depth encoders, multi-exit training
---------------------------------------------------------------------

### DeeBERT — DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference
- cite: Ji Xin, Raphael Tang, Jaejun Lee et al., ACL 2020; arXiv:2004.12993; link: https://arxiv.org/abs/2004.12993
- verified: arXiv API (comment "Accepted at ACL 2020"); details from: full text (ar5iv, Sec. 3 and Table 1)
- family: 3
- layers supervised: one "off-ramp" classifier after every transformer layer of BERT/RoBERTa (the last off-ramp is the usual classifier)
- target: same task label (cross-entropy)
- head sharing: separate classifier per layer; final norm shared: n.a.
- weighting/schedule: two-stage. Stage 1 = ordinary fine-tuning of embeddings + all layers + last off-ramp with L_n only. Stage 2 = freeze everything from stage 1 and train the other off-ramps with sum_{i<n} L_i (backbone gets no gradient from intermediate losses).
- purpose: inference speed
- final-layer effect (numbers): by construction the final layer equals the ordinary fine-tuned model. Stated reason: "The reason for freezing parameters of transformer layers is to keep the optimal output quality for the last off-ramp; otherwise, transformer layers are no longer optimized solely for the last off-ramp, generally worsening its quality." Early-exit inference: SST-2 −0.2 acc at −21% runtime; QQP −0.0 at −24%; largest drop −2.1 (BERT, MNLI) at ~40% savings.
- relevance (1-5): 3 — a direct statement (not an ablation) that joint per-layer supervision degrades the last layer; the design protects it by freezing.

### FastBERT — FastBERT: a Self-distilling BERT with Adaptive Inference Time
- cite: Weijie Liu, Peng Zhou, Zhe Zhao et al., ACL 2020; arXiv:2004.02178; link: https://arxiv.org/abs/2004.02178
- verified: arXiv API (comment "accepted to appear at ACL 2020"); details from: full text (ar5iv, Sec. 3)
- family: 3
- layers supervised: a "student classifier" after every transformer block
- target: self-distillation. Each student is trained with KL divergence to the final "teacher" classifier's soft labels; no ground-truth labels are needed, so unlabeled data can be used.
- head sharing: separate, mutually independent student classifiers; final norm shared: n.a.
- weighting/schedule: pre-train, then fine-tune the backbone + teacher (students disabled), then self-distil the students with the sum of KLs, unweighted. "The parameters in one module is always frozen while the other module is being trained", so the backbone is frozen during self-distillation.
- purpose: inference speed
- final-layer effect (numbers): the teacher/final layer is untouched (frozen). 2–6x speed-up "without losing accuracy for most datasets" (speed=0.1); 7–11x speed-up with small degradation.
- relevance (1-5): 2 — frozen-backbone distillation into exits, so the backbone's training dynamics are unchanged.

### PABEE — BERT Loses Patience: Fast and Robust Inference with Early Exit
- cite: Wangchunshu Zhou, Canwen Xu, Tao Ge et al., NeurIPS 2020; arXiv:2006.04152; link: https://arxiv.org/abs/2006.04152
- verified: arXiv API (comment "NeurIPS 2020"); details from: full text (ar5iv, Eq. 6, Tables 1/4, Fig. 2)
- family: 3
- layers supervised: an internal classifier on every layer of ALBERT/BERT
- target: same task label
- head sharing: separate classifiers C_1..C_n; final norm shared: n.a.
- weighting/schedule: joint training with a layer-index-weighted average L = sum_j j*L_j / sum_j j ("weighted average following [Kaya et al. 2019]"; "can correspond to the relative inference cost"). Weights are constant.
- purpose: both (speed, plus a claimed accuracy gain and robustness via the "overthinking" argument)
- final-layer effect (numbers): the early-exit model beats the vanilla fine-tuned model. ALBERT-base GLUE dev macro 84.4 → 85.1 (+0.7); ALBERT-large MNLI 86.4→86.8, SST-2 94.9→95.2, STS-B 90.4→90.6. Overthinking: "the error rate instead increases after 10 layers" for some inputs. (This compares patience-exit inference with the vanilla final layer, not the jointly trained last layer alone.)
- relevance (1-5): 4 — same-label losses on every transformer layer, trained jointly, with linear-in-depth weights and a net quality gain.

### BERxiT — BERxiT: Early Exiting for BERT with Better Fine-Tuning and Extension to Regression
- cite: Ji Xin, Raphael Tang, Yaoliang Yu et al., EACL 2021 (pp. 91–104); arXiv: none; link: https://aclanthology.org/2021.eacl-main.8/
- verified: ACL Anthology page (authors Ji Xin, Raphael Tang, Yaoliang Yu, Jimmy Lin); details from: full text (ACL PDF read through r.jina.ai; Eqs. 3–7, Fig. 2, Table 2)
- family: 3
- layers supervised: an off-ramp on every transformer layer
- target: same task label; also a learning-to-exit module for regression
- head sharing: separate off-ramps; final norm shared: n.a.
- weighting/schedule: three regimes compared. Joint = min sum_i L_i over all parameters (equal weights). Two-stage = the DeeBERT recipe. Alternating (proposed) = odd iterations minimize L_n only (all parameters), even iterations minimize sum_i L_i (all parameters).
- purpose: inference speed
- final-layer effect (numbers): "Joint treats all classifiers equally, and therefore its final classifier is less effective than that of Two-stage"; "Two-stage produces final classifiers with optimal quality at the price of earlier layers". Alternating is proposed as the trade-off. Table 2 (test; ALT early-exit models relative to raw models): 95–101% of BERT-base quality and 95–100% of BERT-large. Per-strategy final-layer curves are in Fig. 2 only, with no tabulated final-layer delta.
- relevance (1-5): 4 — names the equal-weight joint regime as the cause of final-layer degradation and fixes it by interleaving final-only steps (a 50% duty-cycle schedule on the auxiliary losses).

### Right Tool — The Right Tool for the Job: Matching Model and Instance Complexities
- cite: Roy Schwartz, Gabriel Stanovsky, Swabha Swayamdipta et al., ACL 2020; arXiv:2004.07453; link: https://arxiv.org/abs/2004.07453
- verified: arXiv API (comment "ACL 2020"); details from: full text (ar5iv, Sec. 2–4)
- family: 3
- layers supervised: classifiers at layers 0, 4, 12, 23 of BERT-large. Each classifier reads a learned weighted sum of all layers up to k.
- target: same task label; temperature calibration of each classifier
- head sharing: separate classifiers (<0.005% extra parameters); final norm shared: n.a.
- weighting/schedule: loss = plain sum of all classifier losses, trained jointly with the backbone (fine-tuning). Weights are constant and uniform.
- purpose: inference speed
- final-layer effect (numbers): "For our rightmost point (always selecting the most accurate classifier), we observe a smaller drop, mostly in SST and MNLI, compared to the corresponding baseline". There is a small final-layer drop under the joint sum; the paper gives no separate number.
- relevance (1-5): 3 — the joint equal-weight sum costs a little at the top layer.

### CATs — Consistent Accelerated Inference via Confident Adaptive Transformers
- cite: Tal Schuster, Adam Fisch, Tommi Jaakkola et al., EMNLP 2021; arXiv:2104.08803; link: https://arxiv.org/abs/2104.08803
- verified: arXiv API (comment "EMNLP 2021"); details from: full text (ar5iv, Sec. 4, Table 3)
- family: 3
- layers supervised: an early classification head after each intermediate layer (smaller first projection than the final head)
- target: same task label; conformal meta-classifier for consistency with the full model
- head sharing: separate heads; final norm shared: n.a.
- weighting/schedule: backbone frozen; heads trained on 70% of the training data. "We fix F rather than train it jointly with the new components of G to avoid any reduction in F's performance."
- purpose: inference speed with a distribution-free guarantee P(G(X)=F(X)) >= 1−eps
- final-layer effect (numbers): none by construction. At eps=0.10, >94% consistency with the full model.
- relevance (1-5): 2 — another explicit "freeze to avoid hurting the full model" design choice.

### RomeBERT — RomeBERT: Robust Training of Multi-Exit BERT
- cite: Shijie Geng, Peng Gao, Zuohui Fu et al., arXiv 2021 (preprint; no venue found); arXiv:2101.09755; link: https://arxiv.org/abs/2101.09755
- verified: arXiv API; details from: full text (ar5iv)
- family: 3
- layers supervised: all intermediate exits of BERT-base
- target: CE plus self-distillation from the final exit, L_sd = sum_{i<k} [(1−γ) CE(y, f_i) + γ KL_i], with γ=0.9 and T=3
- head sharing: separate exits; final norm shared: n.a.
- weighting/schedule: gradient regularization. When the self-distillation gradient g_s and the fine-tuning gradient g_f conflict (angle > 90°), g_f is projected onto the normal plane of g_s (PCGrad-style): g* = Proj(g_f) + g_s.
- purpose: inference speed / early-exit accuracy
- final-layer effect (numbers): the motivation is that "the performances of early exits in multi-exit BERT are significantly worse than late exits". RTE 69.5 (0.1 above BERT-base; DeeBERT 67.9); QNLI 88.7 at 34.1% runtime (DeeBERT 87.1 at 49.3%); QQP F1 70.8 (0.5 below BERT-base).
- relevance (1-5): 2 — documents gradient conflict between per-exit objectives on shared weights.

### ElasticBERT — Towards Efficient NLP: A Standard Evaluation and A Strong Baseline
- cite: Xiangyang Liu, Tianxiang Sun, Junliang He et al., NAACL 2022; arXiv:2110.07038; link: https://arxiv.org/abs/2110.07038
- verified: arXiv API (comment "Accepted to the main conference of NAACL-2022"); details from: full text (ar5iv, ElasticBERT section, Table 6, Table 7)
- family: 3
- layers supervised: every layer, during PRE-TRAINING from scratch (MLM + SOP at each exit): L = sum_{l=1..L} (L_MLM^l + L_SOP^l)
- target: same self-supervised targets (MLM/SOP) at every layer
- head sharing: "multiple pre-training heads attached at the intermediate layers" (read as per-exit heads; sharing is not stated); final norm shared: unknown
- weighting/schedule: equal weights. Grouped training: the L exits are split into G groups and optimized cyclically across batches (a rotating subset of exits per step). Gradient equilibrium (Li et al. 2019): the gradient of L_j into layer i<j is rescaled to counter the imbalance from overlapping subnetworks. Constant through pre-training (~160 GB text, 125K steps, batch 4096).
- purpose: both (elastic-depth backbone; a strong shallow model)
- final-layer effect (numbers): GLUE dev average at full 12 layers: ElasticBERT-base 85.6 vs BERT-base 82.9 and RoBERTa-base 86.1. The authors attribute the RoBERTa gap to about 8x fewer training samples. Not controlled: there is no ablation that pre-trains the same model without per-layer losses. Truncated 6L: 83.3 vs DistilBERT 80.7 and TinyBERT-6L 81.9. The Table 7 ablation of grouped training + GE shows only small differences at 12L.
- relevance (1-5): 4 — the only from-scratch pre-training with self-supervised losses at every transformer layer in this slice; it uses a rotating exit-group schedule and gradient rescaling, and the full-depth model stays competitive.

### Ensemble-IC — Early Exiting with Ensemble Internal Classifiers
- cite: Tianxiang Sun, Yunhua Zhou, Xiangyang Liu et al., arXiv 2021 (preprint; no venue found); arXiv:2105.13792; link: https://arxiv.org/abs/2105.13792
- verified: arXiv API; details from: full text (arXiv PDF read through r.jina.ai)
- family: 3
- layers supervised: an internal classifier on every layer of the PLM
- target: same label plus a diversity term: L = sum_i CE(x_i, y) − λ sum_{i>=2} min_{j<i} CE(x_i, x_j)
- head sharing: separate; final norm shared: n.a.
- weighting/schedule: unweighted sum ("we neglect the weights for different internal classifiers"; the appendix mentions α_i = i for BERT); constant
- purpose: inference speed
- final-layer effect (numbers): SST-2 93.5 vs PABEE 93.0 under the same exit strategy. No isolated final-layer comparison.
- relevance (1-5): 2

### LeeBERT — LeeBERT: Learned Early Exit for BERT with cross-level optimization
- cite: Wei Zhu, ACL-IJCNLP 2021 (Long, pp. 2968–2980); arXiv: none; link: https://aclanthology.org/2021.acl-long.231/
- verified: ACL Anthology page; details from: full text (ACL PDF read through r.jina.ai)
- family: 3
- layers supervised: an exit at every layer (BERT/ALBERT)
- target: same label plus mutual distillation among exits ("each exit learns from each other"; variants LLE = learn from later exits, LAE = learn from all exits)
- head sharing: separate exits; final norm shared: n.a.
- weighting/schedule: learnable weights w_i (CE terms) and w_{m,t} (distillation terms). They are updated by bi-level / cross-level optimization (CLO) on two splits of the training set.
- purpose: inference speed without accuracy drop
- final-layer effect (numbers): at 1.96x speed-up LeeBERT beats the full ALBERT-base model (MNLI 85.4 vs 84.6, QNLI 89.7 vs 89.2, QQP 90.2 vs 89.6, RTE 76.8 vs 75.6). PABEE at 1.91x: MNLI 83.9. The paper does not analyze whether the learned weights favour deep or shallow exits.
- relevance (1-5): 3 — learned per-layer loss weights; joint training plus distillation can exceed the full model.

### Kubaty-2025 — How to Train Your Multi-Exit Model? Analyzing the Impact of Training Strategies
- cite: Piotr Kubaty, Bartosz Wójcik, Bartłomiej Krzepkowski et al., ICML 2025 (PMLR 267:31821–31840); arXiv:2407.14320; link: https://arxiv.org/abs/2407.14320
- verified: arXiv API + PMLR page (via search); details from: full text (arxiv.org/html, Table 1, Sec. 4–5)
- family: 3
- layers supervised: internal classifiers (ICs) along ResNet-34/50, MSDNet, ViT-T/S/B and BERT-B (20-Newsgroups, STS-B)
- target: same label
- head sharing: separate ICs; final norm shared: n.a.
- weighting/schedule: three regimes. Disjoint = train the backbone, then train the ICs with the backbone frozen. Joint = everything together from scratch. Mixed = backbone first, then joint training of everything.
- purpose: both (a training-dynamics study of early-exit models)
- final-layer effect (numbers): the 100%-budget column is the final exit. Disjoint equals the backbone-only model.
  - ResNet-34/C100: disjoint 73.79, joint 74.17, mixed 75.88
  - MSDNet/C100: 70.36 / 75.86 / 76.51
  - ViT-T/C100: 63.99 / 66.49 / 70.25
  - ResNet-50/Tiny-IN: 65.71 / 65.01 / 67.24
  - ViT-T/IN-1k: 71.61 / 68.39 / 71.20. Joint from scratch costs −3.2 points of final-exit accuracy on ImageNet, and mixed is −0.4 below backbone-only.
  - BERT-B 20NG: disjoint up to 85.75, joint 84.24–84.41, mixed 84.99–85.25.
  - Mechanism: "In the joint regime, optimization focuses on subnetworks in the middle of the architecture, where gradients from intermediate classifiers exert the strongest influence". Joint lands in a different loss basin (mode connectivity), with a lower numerical rank of activations. Recommendation: "the mixed regime is generally preferred".
- relevance (1-5): 4 — the most systematic help-vs-hurt study. Joint all-exit training from scratch can hurt the final exit on large data (ImageNet, BERT), while adding exits after a backbone phase often helps. The ordering is the opposite of LLAL's early-then-remove schedule, which is an important contrast.

### ZTW — Zero Time Waste: Recycling Predictions in Early Exit Neural Networks
- cite: Maciej Wołczyk, Bartosz Wójcik, Klaudia Bałazy et al., NeurIPS 2021; arXiv:2106.05409; link: https://arxiv.org/abs/2106.05409
- verified: arXiv API (comment "Accepted at NeurIPS 2021"); details from: full text (ar5iv)
- family: 3
- layers supervised: an IC after each block
- target: same label; cascade connections (each IC also sees the previous IC's output, as in boosting) plus geometric ensembling
- head sharing: separate ICs; final norm shared: n.a.
- weighting/schedule: backbone frozen ("The weights θ will not be modified"); only ICs are trained
- purpose: inference speed
- final-layer effect (numbers): none (frozen backbone); no joint-vs-frozen comparison is reported
- relevance (1-5): 2

### BoostNet — Boosted Dynamic Neural Networks
- cite: Haichao Yu, Haoxiang Li, Gang Hua et al., AAAI 2023 (37(9):10989–10997); arXiv:2211.16726; link: https://arxiv.org/abs/2211.16726
- verified: arXiv API + AAAI OJS (via search); details from: full text (ar5iv, Eq. 7, Tables 1–2)
- family: 3
- layers supervised: all exits of MSDNet/RANet-style multi-exit CNNs
- target: same label, in additive (boosting) form. The output of exit n is a linear combination of its own output and all previous exits' outputs, with gradients through previous exits' outputs disabled.
- head sharing: separate exits; final norm shared: n.a.
- weighting/schedule: gradient rescaling dL/db_n = 1/(N−n+1) * sum_{i>=n} dL_i/db_n. This averages rather than sums the gradients arriving at shallow blocks from all deeper exits, so shallow blocks do not get disproportionate gradient. Constant.
- purpose: both (fixes the train/test mismatch of multi-exit training)
- final-layer effect (numbers): highest-budget (last-exit) anytime accuracy: ImageNet 75.08 vs RANet 74.69; CIFAR-100 76.30 vs MSDNet 75.98
- relevance (1-5): 3 — a concrete per-block gradient normalization for many overlapping per-layer losses. The same issue arises when every decoder layer gets an LM loss (layer i receives gradients from L−i+1 losses).

### MEViT — Multi-Exit Vision Transformer for Dynamic Inference
- cite: Arian Bakhtiarnia, Qi Zhang, Alexandros Iosifidis, BMVC 2021; arXiv:2106.15183; link: https://arxiv.org/abs/2106.15183
- verified: arXiv API (comment BMVC 2021); details from: full text (ar5iv)
- family: 3
- layers supervised: exit branches after intermediate ViT/DeiT layers (tested at every layer); seven branch designs (MLP, CNN variants, ViT-layer, MLP-Mixer, ResMLP)
- target: same label
- head sharing: separate branches; final norm shared: n.a.
- weighting/schedule: classifier-wise (frozen backbone), end-to-end ("the contribution of the final exit to the loss is double the contribution of the early exits"), or layer-wise
- purpose: inference speed
- final-layer effect (numbers): no direct final-exit vs original-backbone comparison is reported
- relevance (1-5): 2 — the 2:1 final-vs-intermediate weighting is a data point for weighting.

### LGViT — LGViT: Dynamic Early Exiting for Accelerating Vision Transformer
- cite: Guanyu Xu, Jiawei Hao, Li Shen et al., ACM MM 2023; arXiv:2308.00255; link: https://arxiv.org/abs/2308.00255
- verified: arXiv API (comment "ACM MM 2023"); details from: full text (arxiv.org/html)
- family: 3
- layers supervised: exit points at roughly equal-MAC spacing; local-perception heads in the lower half, global-aggregation heads in the upper half
- target: stage 2 uses self-distillation (heterogeneous + homogeneous + prediction distillation), L = α L_hete + β L_homo + L_pred
- head sharing: separate heads; final norm shared: n.a.
- weighting/schedule: two stages. Stage 1 is end-to-end with an alternating strategy. In stage 2 the backbone and final classifier are frozen and only exit heads and internal classifiers are trained.
- purpose: inference speed
- final-layer effect (numbers): early-exit inference on CIFAR-100: 88.5 vs ViT-B 90.8 at ~1.87x speed-up (about 2% cost on average). This is not a final-exit-only number.
- relevance (1-5): 2

### Meta-GF — Meta-GF: Training Dynamic-Depth Neural Networks Harmoniously
- cite: Yi Sun, Jian Li, Xin Xu, ECCV 2022 (LNCS 13671); arXiv: none; link: https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6337_ECCV_2022_paper.php
- verified: ECVA page (title, authors, abstract); details from: abstract
- family: 3
- layers supervised: all exits of multi-exit CNNs (CIFAR, ImageNet)
- target: same label
- head sharing: separate exits; final norm shared: n.a.
- weighting/schedule: meta-learned weights that fuse the per-exit gradients on shared parameters, taking into account the importance of the shared parameters to each exit
- purpose: training dynamics of multi-exit nets ("exits usually interfere with each other ... reducing model performance and negatively affecting convergence speed")
- final-layer effect (numbers): not in the abstract. DFS (below) reports Meta-GF at 77.47 on the last exit of MSDNet/C100.
- relevance (1-5): 2

### HDKD — Harmonized Dense Knowledge Distillation Training for Multi-Exit Architectures
- cite: Xinglu Wang, Yingming Li, AAAI 2021 (35(11)); arXiv: none; link: https://ojs.aaai.org/index.php/AAAI/article/view/17225
- verified: AAAI OJS page (title, authors, abstract); details from: abstract
- family: 3
- layers supervised: all exits
- target: same label plus dense distillation, where each exit learns from all later exits
- head sharing: separate; final norm shared: n.a.
- weighting/schedule: loss weights optimized by bi-level gradient descent against validation performance
- purpose: both
- final-layer effect (numbers): not in the abstract ("harmoniously improves" state-of-the-art multi-exit nets on CIFAR-100 and ImageNet)
- relevance (1-5): 2

### DFS — Deep Feature Surgery: Towards Accurate and Efficient Multi-Exit Networks
- cite: Cheng Gong, Yao Chen, Qiuyang Luo et al., ECCV 2024; arXiv:2407.13986; link: https://arxiv.org/abs/2407.13986
- verified: arXiv API (comment "ECCV 2024") + ECVA poster page; details from: full text (arxiv.org/html)
- family: 3
- layers supervised: all exits (MSDNet, ResNet18)
- target: same label
- head sharing: separate exits; weights are partitioned into shared and exit-specific parts; final norm shared: n.a.
- weighting/schedule: feature partitioning plus feature referencing, to remove "inconsistent or diametrically opposing gradient directions" among exits on shared weights
- purpose: both (accuracy; up to 50% less training time)
- final-layer effect (numbers): last exit MSDNet/C100 79.56 vs 77.47 (Meta-GF baseline); ResNet18/C100 79.93 vs 78.64 (BYOT). "Up to 6.94%" gains, mostly at early exits.
- relevance (1-5): 2

### LEAP — LEAP: Layer-wise Exit-Aware Pretraining for Efficient Transformer Inference
- cite: Shashank Kapadia, Deep Naryan Mishra, Sujal Reddy Alugubelli et al., ACL 2026 Industry Track (pp. 761–774); arXiv:2605.01058; link: https://arxiv.org/abs/2605.01058
- verified: arXiv API (journal_ref ACL 2026 Industry); details from: full text (arxiv.org/html)
- family: 3 (also 7: alignment-type target)
- layers supervised: every intermediate layer 1..L_s−1 of a distilled MiniLM-type sentence encoder
- target: alignment. Cosine hinge pushing each student layer's embedding toward (a) the teacher's final-layer embedding and (b) the student's own stop-gradient final embedding (weight 0.7).
- head sharing: n.a. (embedding space); final norm shared: n.a.
- weighting/schedule: L = L_final + 0.3 L_inter + 0.4 L_exit + 0.3 L_contrast, constant; training threshold τ=0.98 (inference θ=0.95)
- purpose: inference speed. Layer-aligned distillation "suppress[es] the representational convergence that early-exit mechanisms exploit", and LEAP restores it.
- final-layer effect (numbers): the final layer gets worse. STS-B Spearman 0.777 (baseline distilled MiniLM) → 0.760 ± 0.006 (−2.2%). BEIR: better on 3 of 5 tasks (+3.3% average). 91.9% of samples exit by layer 7; 1.61x wall-clock speed-up.
- relevance (1-5): 3 — a recent, clean data point that forcing every layer toward the final representation costs final-layer quality in an encoder.

### Scardapane-2020 — Why should we add early exits to neural networks?
- cite: Simone Scardapane, Michele Scarpiniti, Enzo Baccarelli et al., Cognitive Computation 2020; arXiv:2004.12814; link: https://arxiv.org/abs/2004.12814
- verified: arXiv API (journal_ref Cognitive Computation 2020); details from: full text (ar5iv)
- family: 3 (survey/tutorial)
- layers supervised: n.a. (taxonomy)
- target: n.a.
- head sharing: n.a.; final norm shared: n.a.
- weighting/schedule: taxonomy of joint training L + sum_i α_i L_i, layer-wise training (freeze previous layers), and separate/classifier-wise training. "If the auxiliary classifiers are only used to improve performance, it is customary to weight earlier classifiers less" (Inception α=0.3). Early exits reduce "tendency to overfitting and vanishing gradients".
- purpose: survey
- final-layer effect (numbers): none
- relevance (1-5): 2 — useful framing of the regimes and of the common down-weighting of shallow exits.

### Han-2021 — Dynamic Neural Networks: A Survey
- cite: Yizeng Han, Gao Huang, Shiji Song et al., IEEE TPAMI 44(11):7436–7456, 2022; arXiv:2102.04906; link: https://arxiv.org/abs/2102.04906
- verified: arXiv API + TPAMI citation (via search; DOI 10.1109/TPAMI.2021.3117837); details from: full text (ar5iv)
- family: 3 (survey)
- layers supervised: n.a.
- target: n.a.
- head sharing: n.a.; final norm shared: n.a.
- weighting/schedule: notes that "the multiple classifiers may interfere with each other, which degrades the overall performance". Points to MSDNet's dense/multi-scale design, gradient equilibrium, and bi-directional knowledge transfer as fixes.
- purpose: survey
- final-layer effect (numbers): none
- relevance (1-5): 1

---------------------------------------------------------------------
## Family 5 — MT / seq2seq
---------------------------------------------------------------------

### MultiLayerSoftmax — Multi-Layer Softmaxing during Training Neural Machine Translation for Flexible Decoding with Fewer Layers
- cite: Raj Dabre, Atsushi Fujita, arXiv 2019 (preprint; DBLP lists only CoRR); arXiv:1908.10118; link: https://arxiv.org/abs/1908.10118
- verified: arXiv API + search (DBLP/CoRR record); details from: full text (ar5iv, Algorithm 1, Table 1)
- family: 5
- layers supervised: ALL (encoder depth n, decoder layer m) combinations of a 6-6 autoregressive Transformer: N×M = 36 losses, each from the output of decoder layer m computed on encoder layer n
- target: same next-token label (teacher-forced CE)
- head sharing: a single softmax/output layer shared by all combinations; final norm shared: unknown
- weighting/schedule: plain average of the N×M losses (footnote: weighted averaging possible). Constant; trained from scratch.
- purpose: inference flexibility (decode with fewer layers)
- final-layer effect (numbers): WMT18 En→De. Full-depth 6-6 BLEU is 34.87 for both the N×M model and the vanilla 6-6 model, so there is no loss at full depth. The largest gap is at 1-1: 27.07 for a dedicated vanilla 1-1 model vs 24.24 for the N×M model. "when the number of decoder layers are increased there is no statistically significant difference". Training time ~9.5x vanilla, because each decoder is re-run for every encoder depth.
- relevance (1-5): 4 — the closest seq2seq analogue: an autoregressive transformer decoder trained FROM SCRATCH with equal-weight next-token CE at every decoder layer through one shared softmax. The top layer is unchanged: no harm, and no gain either.

### DSLP — Non-Autoregressive Translation with Layer-Wise Prediction and Deep Supervision
- cite: Chenyang Huang, Hao Zhou, Osmar R. Zaïane et al., AAAI 2022; arXiv:2110.07515; link: https://arxiv.org/abs/2110.07515
- verified: arXiv API + AAAI OJS page (via search); details from: full text (ar5iv, Eqs. 5–8, Tables 1–3)
- family: 5
- layers supervised: every NAT decoder layer: L = sum_n sum_t log p(y_t^(n) | h_t^(n))
- target: same target tokens. Layer-wise predictions are also fed forward: h~ = W_c[h; emb(argmax)]. With "mixed training", the fed token is the gold token with probability λ=0.3.
- head sharing: shared output projection W / softmax across layers; final norm shared: unknown
- weighting/schedule: equal-weight sum over layers, constant, from scratch
- purpose: both (quality and efficiency of NAT)
- final-layer effect (numbers): WMT14 En–De vanilla NAT 21.18 → deep supervision alone 21.84 (+0.66) → layer-wise prediction alone 21.22 → DSLP 22.72 (+1.54) → +mixed training 24.17. CMLM 19.91→21.76; GLAT 25.02→25.69; CTC 25.72→26.85 (27.02 with mixed training).
- relevance (1-5): 4 — equal-weight per-layer CE through a shared softmax improves the final layer of a transformer decoder trained from scratch (+0.66 BLEU from deep supervision alone), though it is NAT, not AR.

### MV-Transformer — Layer-Wise Multi-View Learning for Neural Machine Translation
- cite: Qiang Wang, Changliang Li, Yue Zhang et al., COLING 2020; arXiv:2011.01482; link: https://arxiv.org/abs/2011.01482
- verified: arXiv API (comment "COLING 2020"); details from: full text (ar5iv)
- family: 5
- layers supervised: one intermediate ENCODER layer (layer 1/3/6 for encoder depth 3/6/12) used as an auxiliary view, decoded by a partially shared decoder
- target: same target tokens (NLL for both views) plus KL consistency between the two views' predictions
- head sharing: the decoder shares self-attention and FFNs but has independent cross-attention per view ("a fully shared decoder has no sufficient capacity"); final norm shared: unknown
- weighting/schedule: L = (1−α)·½(NLL_pri + NLL_aux) + α·KL, with α ∈ [0.3, 0.5] "robust"; constant. The auxiliary modules are discarded at inference.
- purpose: training quality (no inference cost)
- final-layer effect (numbers): IWSLT'14 De→En 34.77 → 35.49 (+0.72); WMT'16 En→De 33.06 → 33.75 (+0.69); Ko→En +0.8 / +1.1
- relevance (1-5): 3 — supervising an intermediate encoder layer with the task loss plus consistency helps the final output modestly.

### Hint-NAT — Hint-Based Training for Non-Autoregressive Machine Translation
- cite: Zhuohan Li, Zi Lin, Di He et al., EMNLP-IJCNLP 2019; arXiv:1909.06708; link: https://arxiv.org/abs/1909.06708
- verified: arXiv API (comment "EMNLP-IJCNLP 2019") + ACL Anthology D19-1573 (via search); details from: full text (ar5iv)
- family: 5 (also 7: distillation-type target)
- layers supervised: all NAT decoder layers (hidden-state hints) and all layers/heads (attention hints)
- target: hints from an AR teacher. Hidden-state hint = penalize student hidden-state pairs whose cosine similarity is above γ_st when the teacher's is below γ_tr. Alignment hint = KL(teacher attention || student attention).
- head sharing: n.a.; final norm shared: n.a.
- weighting/schedule: L = L_nll + λ L_hid + μ L_align, λ=5.0, μ=1.0 (chosen so terms have similar scale at init); constant, no decay
- purpose: training quality
- final-layer effect (numbers): IWSLT14 De-En: NLL only 23.08 → +align 24.76 → +align+hid 25.55
- relevance (1-5): 2

### (excluded) LW-MV-Decoding — Rethinking and Improving Natural Language Generation with Layer-Wise Multi-View Decoding
- cite: Fenglin Liu, Xuancheng Ren, Guangxiang Zhao et al., arXiv 2020 (v7); arXiv:2005.08081; link: https://arxiv.org/abs/2005.08081
- verified: arXiv API; details from: full text (ar5iv)
- family: 5 — EXCLUDED from counts: it is purely architectural (each decoder layer attends to multiple encoder layers) and adds no auxiliary loss. "The proposal only relates to the injection of different mix of source representations".
- relevance (1-5): 1

---------------------------------------------------------------------
## Family 7 — intermediate supervision by alignment / distillation
---------------------------------------------------------------------

### REPA — Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think
- cite: Sihyun Yu, Sangkyung Kwak, Huiwon Jang et al., ICLR 2025 (oral); arXiv:2410.06940; link: https://arxiv.org/abs/2410.06940
- verified: arXiv API (comment "ICLR 2025 (Oral)"); details from: full text (arxiv.org/html, Tables 2–5)
- family: 7
- layers supervised: ONE early-middle block (layer 8 of 28 in SiT-XL/2). Depth ablation FID: layer 6 10.3, 8 10.0, 10 10.5, 12 11.2.
- target: alignment. Patch-wise cosine similarity between an MLP projection of the hidden state and frozen DINOv2 (-L/-B, etc.) features of the clean image.
- head sharing: separate MLP projector (discarded after training); final norm shared: n.a.
- weighting/schedule: L = L_velocity + λ L_REPA with λ=0.5 (robust over 0.25–1.0), CONSTANT for all of training
- purpose: training dynamics / speed
- final-layer effect (numbers): >17.5x faster. SiT-XL/2+REPA reaches FID 7.9 at 400K vs vanilla 8.3 at 7M (no CFG). With CFG: 1.80 at 800 epochs (vanilla 2.06); 1.42 with guidance interval. "limiting regularization to the first few layers further enhances generation performance ... this enables the remaining layers to concentrate on capturing high-frequency details".
- relevance (1-5): 3 — supervising an early layer accelerates the whole network, and supervising only early layers is better than deeper ones. A constant weight turns out to be suboptimal late in training (see HASTE).

### HASTE — REPA Works Until It Doesn't: Early-Stopped, Holistic Alignment Supercharges Diffusion Training
- cite: Ziqiao Wang, Wangbo Zhao, Yuhao Zhou et al., NeurIPS 2025; arXiv:2505.16792; link: https://arxiv.org/abs/2505.16792
- verified: arXiv API + NeurIPS 2025 virtual page / OpenReview (via search); details from: full text (arxiv.org/html, Fig. 1–2, Tables 1, 3, 4, 9)
- family: 7
- layers supervised: feature alignment at depth 8 (REPA) plus attention alignment (ATTA) into intermediate blocks 4–7 of SiT-XL/2
- target: alignment to DINOv2 features and attention maps (token-wise CE between attention maps)
- head sharing: separate projector; final norm shared: n.a.
- weighting/schedule: λ_r = λ_a = 0.5, then a HARD STOP at τ, after which training uses only the denoising loss. τ = 250K iterations for SiT-XL/2 and SiT-L/2, 100K for SiT-B/2; alternatively triggered by the gradient angle. No gradual annealing was tested ("not reported").
- purpose: training dynamics / speed
- final-layer effect (numbers):
  - Cosine between alignment and denoising gradients: relatively high in "ignition" (0–200K), near-orthogonal in "plateau" (200–400K), negative in "conflict" (>400K).
  - Table 4, SiT-XL/2 at 500K: no termination FID 8.1; τ=250K 5.3; τ=400K 7.4. At 400K: no termination 5.5 vs τ=250K 7.3, a transient dip right after the stop that is overtaken by 500K.
  - SiT-B/2 at 400K: 21.3 → 19.6 with τ=100K. SiT-L/2 at 400K: 8.9 → 7.9 with τ=250K.
  - Table 3 at 100 epochs: REPA-only 7.5; holistic without termination 8.1; holistic + termination 5.3.
  - Fig. 1 text: stopping REPA at 400K helps vs always-on, while stopping at 100K hurts.
  - HASTE matches vanilla SiT-XL/2 (7M iterations, FID 8.6) at 250K / 50 epochs, a "28x" reduction in steps.
- relevance (1-5): 4 — alignment rather than the same label, but it is the strongest quantitative evidence that intermediate-layer supervision helps early, then conflicts with the main loss, and that switching it off at the right time beats keeping it on. It is the direct analogue of LLAL's removal. The stop time matters: too early loses the benefit; there is a temporary dip after stopping.

### DeepFlow — Deeply Supervised Flow-Based Generative Models
- cite: Inkyu Shin, Chenglin Yang, Liang-Chieh Chen, ICCV 2025; arXiv:2503.14494; link: https://arxiv.org/abs/2503.14494
- verified: arXiv API (comment "Accepted to ICCV 2025"); details from: full text (arxiv.org/html, Eq. 4, Table 7)
- family: 7 (same-target deep supervision in a generative transformer, plus alignment options)
- layers supervised: k equal branches, each with a velocity output (e.g., layers {6,12} in B/2-2T; {8,16,24} in XL/2-3T), plus a VeRA refiner between branches
- target: the SAME flow-matching velocity target at every branch, plus a second-order acceleration loss; optional DINOv2 alignment
- head sharing: a velocity layer per branch (sharing not explicitly stated); final norm shared: unknown
- weighting/schedule: per-branch weights β_i, best with a LOW weight on intermediate predictions (β=0.2) and 1.0 on the final; λ_acc = 1.0; constant
- purpose: training dynamics / speed
- final-layer effect (numbers): SiT-B/2 FID 34.4 → +deep supervision 33.0 → +VeRA 28.1; with DINOv2 alignment 17.2. DeepFlow-XL/2-3T 10.3 at 80 epochs vs SiT-XL/2 13.8 ("8x faster").
- relevance (1-5): 4 — same-target intermediate losses in a transformer trained from scratch improve the final output when intermediate weights are about 0.2 of the final.

### SRA — No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves
- cite: Dengyang Jiang, Mengmeng Wang, Liuzhuozheng Li et al., ICLR 2026; arXiv:2505.02831; link: https://arxiv.org/abs/2505.02831
- verified: arXiv API (comment "ICLR 2026"); details from: full text (arxiv.org/html, Table 1)
- family: 7
- layers supervised: one early student block aligned to a later teacher block: 3→8 (B), 6→16 (L), 8→20 (XL) for SiT
- target: self-alignment. The teacher is an EMA copy (α=0.9999) fed a lower-noise input; patch-wise distance after a light MLP projector.
- head sharing: separate projector; final norm shared: n.a.
- weighting/schedule: λ=0.2, time gap sampled from 0–0.2; constant
- purpose: training dynamics / speed
- final-layer effect (numbers): SiT-B/2 at 400K FID 33.02 → 29.10 (3→8). Aligning 3→3 (same depth) gives 37.08, WORSE than baseline. SiT-XL: 1.58 at 800 epochs vs 2.06 vanilla at 1400.
- relevance (1-5): 2 — an early layer supervised by a deeper layer of the same model helps; a same-depth target hurts.

### iREPA — What matters for Representation Alignment: Global Information or Spatial Structure?
- cite: Jaskirat Singh, Xingjian Leng, Zongze Wu et al., arXiv 2025; arXiv:2512.10794; link: https://arxiv.org/abs/2512.10794
- verified: arXiv API; details from: abstract
- family: 7
- layers supervised: same as REPA (intermediate)
- target: alignment. Spatial structure of the target (pairwise patch cosine) drives gains, more than global semantic accuracy (27 encoders studied).
- head sharing: a conv projector replaces the MLP, plus spatial normalization of the target; final norm shared: n.a.
- weighting/schedule: as REPA
- purpose: training speed
- final-layer effect (numbers): "consistently improves convergence speed of REPA" (no numbers in the abstract)
- relevance (1-5): 1

### REPR-ALIGN — Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment
- cite: Fred Zhangzhi Peng, Alexis Fox, Anru R. Zhang et al., arXiv 2026; arXiv:2605.06885; link: https://arxiv.org/abs/2605.06885
- verified: arXiv API; details from: full text (arxiv.org/html, Eq. 3, Table 2)
- family: 7
- layers supervised: EVERY layer of a bidirectional masked-diffusion LM
- target: alignment. Mean over layers of (1 − cos) to stop-gradient hidden states of a frozen AR LM with the same architecture, at the same layer.
- head sharing: n.a.; final norm shared: n.a.
- weighting/schedule: L = L_diff + λ L_align, λ=10 (swept 1/5/10/20), constant
- purpose: training speed (AR→DLM conversion; "up to 4x")
- final-layer effect (numbers): layer-subset ablation, pass@1 / pass@10: lower third 11.28/25.00; middle third 15.73/31.10; upper third 8.96/31.71; ALL layers 18.00/31.00 ("only all-layer alignment gives the strongest pass@1"). HumanEval pass@10 24.9→31.0 (0.6B), 31.1→40.5 (1.7B).
- relevance (1-5): 2 — in LMs, aligning all layers beat any single third, but this is conversion from an existing AR model, not from-scratch LM pretraining.

### TinyBERT — TinyBERT: Distilling BERT for Natural Language Understanding
- cite: Xiaoqi Jiao, Yichun Yin, Lifeng Shang et al., Findings of EMNLP 2020; arXiv:1909.10351; link: https://arxiv.org/abs/1909.10351
- verified: arXiv API (comment "Findings of EMNLP 2020"); details from: full text (ar5iv)
- family: 7
- layers supervised: every student layer m, mapped to teacher layer g(m)=3m (4-layer student from a 12-layer teacher); embedding layer and prediction layer too
- target: distillation. Attention-matrix MSE plus hidden-state MSE (learnable projection W_h); embedding MSE; soft-label CE at the prediction layer.
- head sharing: n.a.; final norm shared: n.a.
- weighting/schedule: STAGED. General distillation uses no prediction-layer distillation. Task-specific distillation does "intermediate layer distillation ... for 20 epochs ... and then prediction layer distillation ... for 3 epochs".
- purpose: compression
- final-layer effect (numbers): ablation (GLUE-subset average 75.6): without Transformer-layer distillation 56.3 (−19.3); without attention loss 71.0; without hidden-state loss 72.9
- relevance (1-5): 2 — intermediate supervision first, output supervision last.

### MiniLM — MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers
- cite: Wenhui Wang, Furu Wei, Li Dong et al., NeurIPS 2020; arXiv:2002.10957; link: https://arxiv.org/abs/2002.10957
- verified: arXiv API + NeurIPS 2020 proceedings (via search); details from: full text (ar5iv, Table 7)
- family: 7
- layers supervised: ONLY the student's last layer, distilled from the teacher's last layer (self-attention distributions + value relations)
- target: distillation (KL)
- head sharing: n.a.; final norm shared: n.a.
- weighting/schedule: L = L_AT + L_VR, constant
- purpose: compression
- final-layer effect (numbers): last-layer-only beats layer-to-layer distillation: 6x384 student 80.0 vs 79.0; 4x384 student 78.1 vs 77.0 (Table 7 average)
- relevance (1-5): 2 — a counterpoint: dense layer-to-layer constraints were worse than constraining one top layer.

### MiniLMv2 — MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers
- cite: Wenhui Wang, Hangbo Bao, Shaohan Huang et al., Findings of ACL-IJCNLP 2021 (pp. 2140–2151); arXiv:2012.15828; link: https://arxiv.org/abs/2012.15828
- verified: arXiv API + ACL Anthology 2021.findings-acl.188 (via search); details from: full text (ar5iv)
- family: 7
- layers supervised: the student's last layer, supervised by ONE teacher layer, an upper-middle one for large teachers (layer 21 of 24 for BERT-large; 19 for RoBERTa-large / XLM-R-large)
- target: distillation of multi-head Q-Q, K-K, V-V relations (KL)
- head sharing: n.a.; final norm shared: n.a.
- weighting/schedule: constant sum of relation losses
- purpose: compression
- final-layer effect (numbers): an upper-middle teacher layer beats the last teacher layer for large teachers (the paper states it is better than MiniLM for large teachers)
- relevance (1-5): 2

### FitNets — FitNets: Hints for Thin Deep Nets
- cite: Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou et al., ICLR 2015; arXiv:1412.6550; link: https://arxiv.org/abs/1412.6550
- verified: arXiv API + search (ICLR 2015); details from: full text (ar5iv)
- family: 7
- layers supervised: ONE middle "guided" student layer, supervised by the teacher's middle "hint" layer
- target: distillation. L2 between the teacher hint and a regressor applied to the guided layer.
- head sharing: separate regressor; final norm shared: n.a.
- weighting/schedule: TEMPORARY and staged. Stage 1 pre-trains the student up to the guided layer on the hint loss. Stage 2 runs KD on the whole network, and the hint loss is gone.
- purpose: training dynamics (makes deep thin nets trainable) + compression
- final-layer effect (numbers): CIFAR-10 FitNet 91.61% (~2.5M params) vs teacher 90.18% (~9M). Standard backprop "could not train" the >5-layer thin nets in their budget, while hint training trained 13-layer ones.
- relevance (1-5): 3 — the classic "supervise an intermediate layer first, then remove it" recipe; it enabled optimization of otherwise untrainable depth.

### PKD — Patient Knowledge Distillation for BERT Model Compression
- cite: Siqi Sun, Yu Cheng, Zhe Gan et al., EMNLP 2019; arXiv:1908.09355; link: https://arxiv.org/abs/1908.09355
- verified: arXiv API (comment "Accepted to EMNLP 2019"); details from: full text (ar5iv)
- family: 7
- layers supervised: the student's intermediate layers. PKD-Skip learns from every k-th teacher layer ({2,4,6,8,10} for 12→6); PKD-Last learns from the last k teacher layers.
- target: distillation. MSE between normalized [CLS] hidden states.
- head sharing: n.a.; final norm shared: n.a.
- weighting/schedule: (1−α) CE + α KD + β PT, constant
- purpose: compression
- final-layer effect (numbers): BERT6 MNLI-m 81.5 (PKD) vs 80.2 (KD), test; PKD-Skip slightly better than PKD-Last
- relevance (1-5): 2

### MobileBERT — MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices
- cite: Zhiqing Sun, Hongkun Yu, Xiaodan Song et al., ACL 2020; arXiv:2004.02984; link: https://arxiv.org/abs/2004.02984
- verified: arXiv API (comment "Accepted to ACL 2020"); details from: full text (ar5iv, Tables 8–9)
- family: 7
- layers supervised: every layer (feature-map MSE + per-head attention KL to an IB-BERT teacher)
- target: distillation
- head sharing: n.a.; final norm shared: n.a.
- weighting/schedule: three strategies. Auxiliary (all layer-wise losses added to the pre-training-distillation loss): MNLI-m 83.0. Joint (layer-wise losses jointly first, then pre-training distillation): 83.5. Progressive (train layer by layer, freezing lower layers, then pre-training distillation): 83.9, the best.
- purpose: compression
- final-layer effect (numbers): ablation MNLI-m: bare 80.8 → +PD 81.1 → +PD+FMT 83.8 → +PD+FMT+AT 84.4
- relevance (1-5): 3 — using per-layer losses as a concurrent auxiliary during output-level training was the worst of the three; staging them before the output objective was better.

### DistillLens — DistillLens: Symmetric Knowledge Distillation Through Logit Lens
- cite: Manish Dhakal, Uthman Jinadu, Anjila Budathoki et al., arXiv 2026 (preprint; no venue found); arXiv:2602.13567; link: https://arxiv.org/abs/2602.13567
- verified: arXiv API; details from: full text (arxiv.org/html)
- family: 7
- layers supervised: a subset of decoder layers (GPT-2-120M: {2,4,6,8,10}; GPT-2-340M: {4,8,...,20}; TinyLlama-1.1B: {4,7,11,15,18}), mapped to teacher layers by depth ratio
- target: distillation in vocabulary space. Logit-lens distributions softmax(W_U h^(l)) of student and teacher, matched with Jensen–Shannon divergence.
- head sharing: each model's OWN unembedding (logit lens; no extra heads); final norm shared: not stated
- weighting/schedule: L = L_task + λ·mean_l JSD, λ=1.0, constant
- purpose: distillation quality (SFT stage, databricks-dolly-15k)
- final-layer effect (numbers): ROUGE-L vs standard KD: GPT-2-120M 17.20 → 21.12; GPT-2-340M 20.42 → 23.72
- relevance (1-5): 3 — supervising intermediate decoder layers through the shared unembedding helps the final output, but with teacher distributions (not labels), in fine-tuning, and on a subset of layers.

### NITP — NITP: Next Implicit Token Prediction for LLM Pre-training
- cite: Xiangdong Zhang, Debing Zhang, Shaofeng Zhang et al., ICML 2026; arXiv:2605.24956; link: https://arxiv.org/abs/2605.24956
- verified: arXiv API (comment "Accepted at ICML 2026"); details from: full text (arxiv.org/html, Eqs. 2–3, Tables 1, 3, 4)
- family: 7 (auxiliary target taken from an intermediate layer; the SUPERVISED state is the FINAL layer)
- layers supervised: the FINAL hidden state h_t, through an MLP projection head P. No intermediate layer receives a direct loss; the shallow layer is only the target.
- target: z_{t+1} = sg[shallow-layer representation of the next token], from a FIXED layer at ~20% depth (dense 0.5B: L5/24; 2B/3B: L6/28; MoE 1.9B: L3/16, 3B: L4/17, 9B: L5/24). Online model with stop-gradient (not EMA). Loss = 1 − cos(P(h_t), z_{t+1}).
- head sharing: separate projector P; final norm shared: n.a.
- weighting/schedule: L = L_NTP + λ L_NITP, λ=1.0 (0.8 for 9B MoE and 3B dense), CONSTANT for all of pretraining
- purpose: pretraining quality (representation regularization)
- final-layer effect (numbers): downstream average: dense 0.5B 24.42→25.44, 2B 31.91→33.70, 3B 37.18→38.53. 9B MoE MMLU-Pro 15.29→21.00, C3 56.65→63.01, CSQA 45.70→49.96, at ~2% extra FLOPs. Target-depth ablation (3B MoE avg): shallow L4 23.58 > deep L14 22.16 > middle L8 21.22. Motivation: "Transformer training typically exhibits a bottom-up convergence pattern, where layers closer to the input stabilize much faster than deeper layers". No validation-loss numbers for the dense models in the main tables.
- relevance (1-5): 3 — a decoder-LM pretraining auxiliary that uses shallow-layer states. It does not put a loss on intermediate layers, but it is useful evidence that ~20%-depth features are stable, useful targets.

### OISD — OISD: On-Policy Internal Self-Distillation of Language Models
- cite: Xinyu Liu, Darryl Cherian Jacob, Yang Zhou et al., Findings of EMNLP 2026; arXiv:2605.29089; link: https://arxiv.org/abs/2605.29089
- verified: arXiv API (journal_ref Findings of EMNLP 2026); details from: full text (arxiv.org/html)
- family: 7
- layers supervised: ONE intermediate layer (layer 6 of Qwen3-4B/8B; 27 of OctoThinker-3B); results are stable for layers 6/26/35
- target: self-distillation toward the detached FINAL layer's token distribution on the same on-policy rollouts. JSD, weighted by the GRPO advantage, plus an attention-distribution JSD.
- head sharing: logit lens through the model's own LN + unembedding, p^l = softmax(LN(h^l) E_u^T), i.e. the SHARED final norm and LM head, no new heads; final norm shared: yes
- weighting/schedule: L = L_GRPO + 1.0·L_think + 0.1·L_attn, constant (no schedule)
- purpose: RL post-training quality
- final-layer effect (numbers): Qwen3-4B Avg@K 55.08 (GRPO) → 64.75. AIME24 32.19→47.29; MATH500 82.41→90.99. Logit-only 61.67; attention-only 63.08.
- relevance (1-5): 3 — the same mechanism as the planned experiment (shared final norm + unembedding on an intermediate decoder layer), reported to help the final layer. However it is post-training, with a self-distillation target, on one layer.

### HLD-LLM — A Study on Hidden Layer Distillation for Large Language Model Pre-Training
- cite: Maxime Guigon, Lucas Dixon, Michaël E. Sander, arXiv 2026 (preprint); arXiv:2605.11513; link: https://arxiv.org/abs/2605.11513
- verified: arXiv API; details from: full text (arxiv.org/html)
- family: 7
- layers supervised: the student's middle layer (D_S/2), aligned to the teacher's middle layer (D_T/2). Teacher Gemma3 3.4B; students 123M and 735M, trained FROM SCRATCH on C4 (up to 168B tokens).
- target: alignment/distillation. Normalized MSE through a learned linear regressor.
- head sharing: separate regressor; final norm shared: n.a.
- weighting/schedule: two variants. HLDF (FitNets-style, TEMPORARY): Phase 1 = hint training of the student up to layer D_S/2 for the first P1 = 1–5% of the budget (1%/5% at 123M; 1%/4% at 735M); Phase 2 = standard KD, with no hint loss. HLDC (joint, constant): β L_data + α L_logits + γ L_emb, γ ∈ {0.05, 0.10}.
- purpose: pretraining quality
- final-layer effect (numbers): C4 eval metric (reported as perplexity; values look like log-perplexity), 123M / 735M: KD 3.005 / 2.609; HLDF 3.000 / 2.607; HLDC 3.005 / 2.608. "HLDF yields modest improvements in C4 perplexity across all shared-hyperparameter configurations", mostly "in regimes where KD itself performs poorly". Downstream: "no method dominates KD".
- relevance (1-5): 4 — the closest decoder-LM-from-scratch analogue of a short, early, TEMPORARY intermediate-layer objective that is then removed. The temporary version gives a small but systematic final-loss gain; the always-on joint version gives none.

---------------------------------------------------------------------
## Family-level takeaways
---------------------------------------------------------------------
- **Always-on, equal-weight joint supervision of intermediate layers tends to cost the final output.**
  - Stated as design motivation: DeeBERT ("generally worsening its quality"), CATs, FastBERT, ZTW, and BERxiT ("its final classifier is less effective than that of Two-stage").
  - Measured by Kubaty et al. (ICML 2025, 2407.14320): joint training from scratch lowers final-exit accuracy vs backbone-only on ViT-T/ImageNet-1k (68.39 vs 71.61, −3.2), ResNet-50/Tiny-IN (65.01 vs 65.71) and BERT-B/20NG (~84.3 vs 85.75). The mechanism is that intermediate-classifier gradients dominate the middle of the network.
  - LEAP (2605.01058): forcing every encoder layer toward the final representation costs −2.2% STS-B at the top layer.
- **When intermediate losses help, it is with low or depth-increasing weights, or when they are staged.**
  - PABEE's weights ∝ layer index (ALBERT-base GLUE 84.4→85.1); LeeBERT's learned weights; DeepFlow's β=0.2 on intermediate vs 1.0 on the final (SiT-B/2 FID 34.4→33.0 from deep supervision alone); Multi-Exit ViT's 2x weight on the final exit.
  - Kubaty's mixed regime (backbone first, then joint) beats backbone-only in 4 of 5 settings: ResNet-34/C100 +2.1, ViT-T/C100 +6.3, ResNet-50/Tiny-IN +1.5, MSDNet +6.2. It is −0.4 on ImageNet.
- **Decoders trained from scratch with next-token CE on every layer through one shared softmax are neutral to positive at full depth.**
  - Dabre & Fujita (1908.10118): AR NMT full-depth BLEU unchanged (34.87 vs 34.87 on WMT18 En→De), with a 3.24 BLEU deficit only at 1-1 depth.
  - DSLP (2110.07515): NAT decoder, deep supervision alone +0.66 BLEU (21.18→21.84); +1.54 with layer-wise prediction feeding.
  - Neither used a schedule; both used equal weights. These are the closest no-harm data points for "LM loss at every decoder layer".
- **Time-limited intermediate supervision is a recurring winning pattern.**
  - HASTE (NeurIPS 2025, 2505.16792): REPA-style alignment helps during "ignition" (0–200K iters), becomes gradient-orthogonal at 200–400K and conflicts after 400K. A hard stop at 250K turns SiT-XL/2 FID@500K from 8.1 into 5.3; stopping at 400K gives 7.4, and stopping at 100K hurts. SiT-B/2 at 400K: 21.3→19.6 (stop at 100K); SiT-L/2: 8.9→7.9 (stop at 250K). There is a transient dip right after stopping (at 400K the stopped run is 7.3 vs 5.5), so judge well after the switch-off.
  - Earlier staged recipes: FitNets (hint stage, then hint removed); TinyBERT (intermediate distillation 20 epochs, then prediction layer 3 epochs); MobileBERT (progressive 83.9 > joint 83.5 > concurrent auxiliary 83.0 MNLI-m).
  - HLD-LLM (2605.11513): a 1–5%-of-budget hint phase then removal gives a small consistent C4 gain (123M: 3.005→3.000; 735M: 2.609→2.607). The always-on joint variant gives none.
  - No paper in this slice compares a linear anneal (LLAL) against a hard stop (HASTE). That is an open question.
- **Where to supervise.**
  - REPA: one early block is best (layer 8 of 28: FID 10.0 vs 11.2 at layer 12), "enabl[ing] the remaining layers to concentrate on ... high-frequency details".
  - NITP: shallow ~20% depth is the best target layer (3B MoE avg 23.58 at L4 vs 22.16 at L14).
  - REPR-ALIGN (LM conversion): all-layer alignment beats any third (pass@1 18.00 vs 15.73 for the middle third).
  - MiniLM: last-layer-only beats layer-to-layer distillation (80.0 vs 79.0).
  - The evidence is split: early/partial supervision wins for representation targets in generative vision, while all-layer supervision wins when a same-architecture teacher is available.
- **Gradient bookkeeping matters when many overlapping per-layer losses share a trunk.**
  - Layer i of an L-layer decoder receives gradients from L−i+1 losses.
  - Remedies in the literature: BoostNet's 1/(N−n+1) rescaling, gradient equilibrium (ElasticBERT), RomeBERT's gradient projection when angles exceed 90°, ElasticBERT's cyclic exit groups (a rotating subset of exits per batch, similar to LayerSkip's rotation), and BERxiT's alternation of final-only and all-exit steps.
- **The logit-lens mechanism (shared final norm + unembedding on intermediate decoder states) is reported to help the final layer**, but so far only in post-training or distillation, not in from-scratch pretraining with labels.
  - OISD: RL; one layer; JSD to the detached final layer; Qwen3-4B Avg@K 55.08→64.75.
  - DistillLens: KD SFT; JSD to teacher layers; ROUGE-L +3.9.
- **Pretraining with self-supervised losses at every layer is viable.** ElasticBERT (MLM+SOP at all 12 exits, grouped training + gradient equilibrium) reaches GLUE dev 85.6 at full depth (BERT-base 82.9, RoBERTa-base 86.1). There is no controlled no-multi-exit ablation, so the effect on the final layer is unmeasured.

---------------------------------------------------------------------
## Could not verify
---------------------------------------------------------------------
- None of the entries above is unverified. BERxiT, LeeBERT, Meta-GF and HDKD have no arXiv id and were verified on ACL Anthology / ECVA / AAAI OJS instead.
- Venue unknown (arXiv-only preprints as far as checked): RomeBERT (2101.09755), Ensemble-IC (2105.13792), Multi-Layer Softmaxing (1908.10118; DBLP lists CoRR only), iREPA (2512.10794), REPR-ALIGN (2605.06885), DistillLens (2602.13567), HLD-LLM (2605.11513).
- Detail caveats:
  - BERxiT: per-strategy final-layer numbers exist only in figures; only the ALT relative scores are tabulated.
  - Right Tool: the final-layer "smaller drop" is not quantified in the text read.
  - ElasticBERT: whether exit heads are shared is not stated.
  - DeepFlow: whether branch velocity heads are shared is not stated.
  - HLD-LLM: the C4 metric is labelled perplexity, but the values (~3.0) look like log-perplexity.
