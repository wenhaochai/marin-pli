# Survey notes — Family 1 (deep supervision in CNNs / vision), 1b (iterative-refinement per-stage losses), 8 (local / greedy layer-wise learning)

Fork notes, 2026-09-29. Metadata verified with the arXiv API (export.arxiv.org), plus venue pages (PMLR, CVF open access, Springer/ECVA, AAAI OJS, NeurIPS proceedings, ICML virtual site) where noted. Details come from full text (ar5iv / arxiv.org/html) unless marked "abstract". Relevance scale: see the parent's brief (5 = decoder-LM per-layer LM loss in pretraining ... 1 = tangential).

---------------------------------------------------------------------
## Family 1 — classic deep supervision (CNN, ViT, MIM)
---------------------------------------------------------------------

### DSN — Deeply-Supervised Nets
- cite: Chen-Yu Lee, Saining Xie, Patrick Gallagher et al., AISTATS 2015 (PMLR v38, pp. 562-570); arXiv:1409.5185; link: https://arxiv.org/abs/1409.5185
- verified: arXiv API + proceedings.mlr.press/v38/lee15a.html; details from: full text (ar5iv), Eq. 3 and results tables
- family: 1
- layers supervised: every hidden (conv) layer gets a "companion objective"
- target: same class label; squared-hinge SVM (softmax variant also reported)
- head sharing: no — a separate classifier w^(m) per hidden layer ; final norm shared: n.a.
- weighting/schedule: total = ||w_out||^2 + L_out + sum_m alpha_m [ ||w^(m)||^2 + l_m - gamma ]_+ . The hinge on gamma switches a layer's companion loss off (zero gradient) once it falls below gamma, so each layer's aux loss removes itself. Optional decay "alpha_m x 0.1 x (1 - t/N) -> alpha_m to enforce the second term to vanish after certain number of iterations" (t = epoch, N = total epochs); the paper says the decay "might vary in different experiments".
- purpose: training dynamics (transparency of hidden layers, discriminative early features, vanishing/exploding gradients; acts as feature regularization)
- final-layer effect (numbers): test error, DSN vs same CNN: MNIST 0.39 vs 0.53; CIFAR-10 9.78 vs 10.41 (no aug), 8.22 vs 8.81 (aug); CIFAR-100 34.57 vs 35.68; SVHN 1.92 vs 2.35. Reports larger early-layer gradients with DSN.
- relevance (1-5): 3 — origin of the idea. The gamma-threshold self-deactivation and "decay to vanish" schedule are direct precedents for a temporary, annealed per-layer loss.

### GoogLeNet aux classifiers — Going Deeper with Convolutions
- cite: Christian Szegedy, Wei Liu, Yangqing Jia et al., CVPR 2015; arXiv:1409.4842; link: https://arxiv.org/abs/1409.4842
- verified: arXiv API + openaccess.thecvf.com (CVPR 2015); details from: full text (ar5iv), Sec. 5
- family: 1
- layers supervised: two auxiliary classifiers, on the outputs of Inception (4a) and (4d)
- target: same ImageNet label (softmax)
- head sharing: no. Each aux head is its own small net (5x5 avg-pool s3, 1x1 conv 128, FC 1024, 70% dropout, linear+softmax) ; final norm shared: n.a.
- weighting/schedule: constant, "the losses of the auxiliary classifiers were weighted by 0.3"; "At inference time, these auxiliary networks are discarded"
- purpose: training dynamics: "encourage discrimination in the lower stages ..., increase the gradient signal that gets propagated back, and provide additional regularization"
- final-layer effect (numbers): the paper gives no isolated number for the aux heads
- relevance (1-5): 3 — canonical constant-weight (0.3), train-only aux heads.

### Inception-v3 aux-head finding — Rethinking the Inception Architecture for Computer Vision
- cite: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe et al., CVPR 2016; arXiv:1512.00567; link: https://arxiv.org/abs/1512.00567
- verified: arXiv API + openaccess.thecvf.com (CVPR 2016); details from: full text (ar5iv), Sec. 4 "Utility of Auxiliary Classifiers"
- family: 1
- layers supervised: originally two side heads. The lower one was removed ("did not have any adverse effect on the final quality of the network") and only the upper one was kept.
- target: same label
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: constant (the 0.3 legacy); head kept for all of training
- purpose: re-examined as regularizer
- final-layer effect (numbers): aux classifiers "did not result in improved convergence early in the training: the training progression of network with and without side head looks virtually identical before both models reach high accuracy. Near the end of training, the network with the auxiliary branches starts to overtake ... and reaches a slightly higher plateau." They "act as regularizer": batch-normalizing the side head gives "+0.4% absolute" top-1 for the main classifier.
- relevance (1-5): 3 — important counter-evidence to an "early-only helps" story. In this CNN the aux loss did nothing early, and its benefit showed up late as regularization.

### CNDS — Training Deeper Convolutional Networks with Deep Supervision
- cite: Liwei Wang, Chen-Yu Lee, Zhuowen Tu et al. (with S. Lazebnik), arXiv 2015 (no archival venue found); arXiv:1505.02496; link: https://arxiv.org/abs/1505.02496
- verified: arXiv API; details from: full text (ar5iv), Sec. 3 and results tables
- family: 1
- layers supervised: only where gradients vanish. They run 10-50 backprop iterations with top-only supervision and add a branch after the layer whose mean gradient drops below 1e-7 (8-layer net: after conv4; 13-layer net: after layers 4, 7, 10).
- target: same label
- head sharing: no. Each branch is its own conv + FC(s) + softmax, because "feature maps at the lower convolutional layers are very noisy" ; final norm shared: n.a.
- weighting/schedule: L = L0 + alpha_t * L_s. alpha_t "starts with 0.3" and decays by alpha_t <- alpha_t*(1 - t/N) (t = epoch, N = total epochs), so the aux term is annealed toward 0.
- purpose: training dynamics ("regularization that gives better local minima"), enabling deeper nets
- final-layer effect (numbers): ImageNet val top-1/top-5 error CNN-8 34.7/14.0 -> CNDS-8 33.8/13.2; CNDS-13 31.8/11.8. MIT Places accuracy (val top-1/top-5) CNN-8 54.0/83.7 -> CNDS-8 54.7/84.1. CNDS-8 took about 5 days vs 6 days for the baseline on two K40s.
- relevance (1-5): 3 — the closest CNN precedent for an annealed aux weight (0.3 decaying to 0) that improved the final output.

### HED — Holistically-Nested Edge Detection
- cite: Saining Xie, Zhuowen Tu, ICCV 2015; arXiv:1504.06375; link: https://arxiv.org/abs/1504.06375
- verified: arXiv API + openaccess.thecvf.com (ICCV 2015); details from: full text (ar5iv), Sec. 2.3 and Table 2
- family: 1
- layers supervised: side outputs at conv1_2, conv2_2, conv3_3, conv4_3, conv5_3 (the last conv of each VGG stage)
- target: same dense edge map (class-balanced cross-entropy)
- head sharing: no (a 1x1 side-output layer per stage), plus a learned weighted-fusion layer ; final norm shared: n.a.
- weighting/schedule: alpha_m = 1 for all side outputs; kept for the whole run
- purpose: training dynamics + multi-scale output (the final output is a fusion of side outputs)
- final-layer effect (numbers): Table 2, fused output without vs with deep supervision: ODS .771 -> .782, OIS .785 -> .802, AP .738 -> .787
- relevance (1-5): 2 — its output is itself a fusion of the side outputs, so this is not a clean "final layer" test.

### PSPNet aux loss — Pyramid Scene Parsing Network
- cite: Hengshuang Zhao, Jianping Shi, Xiaojuan Qi et al., CVPR 2017; arXiv:1612.01105; link: https://arxiv.org/abs/1612.01105
- verified: arXiv API (comment: CVPR 2017); details from: full text (ar5iv), Sec. 3.3 and Table 2
- family: 1
- layers supervised: one auxiliary loss after res4b22 of ResNet-101
- target: same per-pixel label
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: constant alpha = 0.4; "In the testing phase, we abandon this auxiliary branch"
- purpose: training dynamics ("deeply supervised" optimization of a deep ResNet)
- final-layer effect (numbers): Table 2, mIoU / pixel accuracy: no aux 35.82/77.07; alpha 0.3 37.01/77.87; 0.4 37.23/78.01 (best); 0.6 37.09/77.84; 0.9 36.99/77.87. Every weight from 0.3 to 0.9 helps, and the result is insensitive to the exact value.
- relevance (1-5): 2 — a clean weight sweep. The aux loss helps the final output across a 3x range of weights.

### UNet++ deep supervision — UNet++: A Nested U-Net Architecture for Medical Image Segmentation
- cite: Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh et al., DLMIA 2018 (MICCAI workshop); arXiv:1807.10165; link: https://arxiv.org/abs/1807.10165
- verified: arXiv API (comment: DLMIA 2018); details from: full text (ar5iv), Sec. 2 and Table 3
- family: 1
- layers supervised: the four full-resolution nested decoder nodes X^{0,1..4}
- target: same mask (BCE + Dice)
- head sharing: no (a 1x1 conv per node) ; final norm shared: n.a.
- weighting/schedule: equal, constant. Inference has an "accurate" mode (average all branches) and a "fast" mode (pick one branch, i.e. prune).
- purpose: both (accuracy + prunability)
- final-layer effect (numbers): IoU without -> with deep supervision: cell nuclei 92.63 -> 92.52, colon polyp 33.45 -> 32.12, liver 79.70 -> 82.90, lung nodule 76.44 -> 77.21 (mixed)
- relevance (1-5): 1 — mixed sign, and far from LMs.

### MSDNet — Multi-Scale Dense Networks for Resource Efficient Image Classification
- cite: Gao Huang, Danlu Chen, Tianhong Li et al., ICLR 2018; arXiv:1703.09844; link: https://arxiv.org/abs/1703.09844
- verified: arXiv API + ICLR 2018 virtual page (iclr.cc/virtual/2018/poster/278); details from: full text (ar5iv), Sec. 3 (Fig. 3 right) and Sec. 4
- family: 1
- layers supervised: MSDNet puts classifiers at many depths. Motivating experiment: one intermediate classifier attached to ResNet / DenseNet at varying relative depth (CIFAR-100).
- target: same label
- head sharing: no (independent classifiers; the features are shared through dense connectivity) ; final norm shared: n.a.
- weighting/schedule: uniform w_k = 1 ("works well in practice"), constant
- purpose: inference (anytime prediction / budgeted batch classification)
- final-layer effect (numbers): KEY NEGATIVE: "the introduction of an intermediate classifier harms the final ResNet classifier ..., reducing its accuracy by up to 7%" (CIFAR-100). "The DenseNet ... suffers much less from this effect." Mechanism: "early classifiers influenc[e] the early features to be optimized for the short-term and not for the final layers. This improves the accuracy of the immediate classifier but collapses information required to generate high quality features in later layers." Dense connectivity "allows later layers to bypass features optimized for the short-term."
- relevance (1-5): 3 (CNN, but the key harm mechanism) — the residual stream of a decoder LM is additive like ResNet. This is the main warning for a per-layer shared-head LM loss.

### BranchyNet — BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks
- cite: Surat Teerapittayanon, Bradley McDanel, H. T. Kung, ICPR 2016; arXiv:1709.01686; link: https://arxiv.org/abs/1709.01686
- verified: arXiv API + ICPR 2016 (IEEE) listing via search (Semantic Scholar / Harvard DASH); details from: full text (ar5iv), Sec. 3-4
- family: 1
- layers supervised: 1-2 side branches (B-LeNet: after conv1; B-AlexNet: after conv1 and conv2; B-ResNet-110: after conv2 and conv37)
- target: same label
- head sharing: no (each branch has its own conv + FC layers) ; final norm shared: n.a.
- weighting/schedule: joint loss sum_n w_n L_n, constant. "First branch with 1.0 and the last branch with 0.3 provides a 1% increase in classification accuracy over weighting each branch equally."
- purpose: inference speed (claims each exit "provides regularization on the others")
- final-layer effect (numbers): overall accuracy at the chosen exit thresholds ("knee point"): B-AlexNet 79.19 vs AlexNet 78.38; B-ResNet 79.17 vs ResNet 80.70. The paper does not isolate the final exit.
- relevance (1-5): 2

### SDN — Shallow-Deep Networks: Understanding and Mitigating Network Overthinking
- cite: Yigitcan Kaya, Sanghyun Hong, Tudor Dumitras, ICML 2019; arXiv:1810.07052; link: https://arxiv.org/abs/1810.07052
- verified: arXiv API (comment: ICML 2019); details from: full text (ar5iv), Sec. 4 and Table 3
- family: 1
- layers supervised: internal classifiers (ICs) at the layers closest to 15/30/45/60/75/90% of inference cost
- target: same label
- head sharing: no (each IC = mixed max/avg pooling to 4x4 + one FC) ; final norm shared: n.a.
- weighting/schedule: in SDN training from scratch, "we start the training with tau_i = 0.01 and linearly increase it until tau_i = C_i", where C_i is the IC's relative cost (0.15 ... 0.9) and the final classifier has weight 1. So the weights are proportional to depth and warmed up from about 0 (the opposite direction to LLAL's decay). IC-only mode freezes the backbone.
- purpose: inference (early exit) + diagnosis ("overthinking")
- final-layer effect (numbers): Table 3 ("Max" early-exit accuracy, IC-only || SDN-training, original CNN in parentheses): CIFAR-100 VGG16 (70.9) 72.6 || 74.4; ResNet56 (68.8) 69.7 || 70.9; WRN (75.1) 75.5 || 77.3; MobileNet (64.9) 67.6 || 68.9; Tiny ImageNet VGG16 (58.6) 60.4 || 63.4. The paper does not isolate the final classifier's accuracy. Overthinking: 95% / 81% / 69% of CIFAR-10 / CIFAR-100 / Tiny ImageNet samples are correct at some IC before the end, and "up to ~50%" of misclassifications had a correct IC earlier.
- relevance (1-5): 3 — sensible depth-proportional weighting with a linear warm-up. Joint training does not destroy the final output.

### AdaLoss — Learning Anytime Predictions in Neural Networks via Adaptive Loss Balancing
- cite: Hanzhang Hu, Debadeepta Dey, Martial Hebert et al. (J. Andrew Bagnell), AAAI 2019; arXiv:1708.06832; link: https://arxiv.org/abs/1708.06832
- verified: arXiv API + ojs.aaai.org (AAAI 33(01) 3812-3821); details from: full text (ar5iv), Sec. 3-5
- family: 1
- layers supervised: anytime outputs at many depths (ResNet / DenseNet-style ANNs)
- target: same label
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: adaptive weights B_i proportional to 1 / (running average of loss_i), i.e. minimizing sum_i log(loss_i) (geometric mean of the losses). Compared against CONST (all 1), LINEAR (0.25 -> 1) and "half-end" (50% weight on the final output).
- purpose: inference (anytime)
- final-layer effect (numbers): with CONST weights, anytime nets make "15~18% more errors than the OPT [single-output nets], and the relative gap widens at later layers" (about 18.9% relative at full depth). AdaLoss shrinks the gap at the later layers.
- relevance (1-5): 3 — the clearest numbers on how equal per-layer weights tax the final output, plus a principled loss-normalized weighting (useful because shallow-layer LM losses are much larger).

### DKS — Deeply-supervised Knowledge Synergy
- cite: Dawei Sun, Anbang Yao, Aojun Zhou et al., CVPR 2019; arXiv:1906.00675; link: https://arxiv.org/abs/1906.00675
- verified: arXiv API + openaccess.thecvf.com (CVPR 2019); details from: full text (ar5iv), Sec. 3-4
- family: 1
- layers supervised: aux classifiers after conv3_x and conv4_x (ImageNet ResNets); 3 aux heads on CIFAR-100
- target: same label, plus pairwise knowledge matching among all heads including the final one (soft targets, both directions)
- head sharing: no (aux branches built from backbone-type blocks); discarded at inference ; final norm shared: n.a.
- weighting/schedule: all weights 1, constant
- purpose: training (final accuracy)
- final-layer effect (numbers): top-1 error baseline / plain DS / DKS: ImageNet ResNet-18 31.06 / 30.46 / 28.68; ResNet-50 25.47 / 25.09 / 23.53; ResNet-152 22.45 / 21.99 / 20.98; CIFAR-100 ResNet-32 29.97 / 29.89 / 26.81; ResNet-110 27.66 / 26.95 / 24.98. Plain same-label DS gains are small (0.08-0.92 points). Adding cross-head distillation gives most of the gain.
- relevance (1-5): 2

### BYOT — Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation
- cite: Linfeng Zhang, Jiebo Song, Anni Gao et al., ICCV 2019; arXiv:1905.08094; link: https://arxiv.org/abs/1905.08094
- verified: arXiv API + openaccess.thecvf.com (ICCV 2019); details from: full text (ar5iv), Sec. 3-4
- family: 1
- layers supervised: the net is split into sections (e.g. the 4 ResBlock stages of ResNet-50), with a shallow classifier (bottleneck + FC) after each
- target: (1-alpha)*CE(label) + alpha*KL(to the deepest classifier's softmax) + lambda*L2 feature hint to the deepest feature map
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: constant alpha, lambda (untuned, per the authors); temperature about 1
- purpose: training (final accuracy) + optional early exit
- final-layer effect (numbers): deepest classifier, CIFAR-100: ResNet-18 77.09 -> 78.64; ResNet-50 77.68 -> 80.56; ImageNet ResNet-50 73.56 -> 75.24. Beats plain deep supervision (DSN) at every depth (e.g. ResNet-50 ensemble 81.04 vs DSN 80.67).
- relevance (1-5): 3 — suggests that distilling the final output into shallow heads helps the final layer more than hard-label deep supervision does.

### IMTA — Improved Techniques for Training Adaptive Deep Networks
- cite: Hao Li, Hong Zhang, Xiaojuan Qi et al. (Gao Huang), ICCV 2019; arXiv:1908.06294; link: https://arxiv.org/abs/1908.06294
- verified: arXiv API + openaccess.thecvf.com (ICCV 2019); details from: full text (ar5iv), Sec. 3-4, Tables 2-3
- family: 1
- layers supervised: all exits of a multi-exit net (MSDNet-style)
- target: same label + one-for-all KD (L_i = alpha*CE_i + (1-alpha)*KL(exit i || final))
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: Gradient Equilibrium (GE) rescales the gradients flowing into the shared backbone (branch i: 1/(k-i+1) from its own exit and (k-i)/(k-i+1) from later ones), keeping gradient variance bounded instead of growing with the number of exits. Also inline subnetwork collaboration.
- purpose: both (anytime inference; explicitly targets gradient conflict among exits)
- final-layer effect (numbers): 5-exit CIFAR-100 final exit 71.81 -> 73.45 (+1.64), exit 2 63.73 -> 65.54; ImageNet final exit 71.34 -> 72.43 (+1.09), exit 1 56.64 -> 57.28
- relevance (1-5): 3 — the gradient-variance argument applies directly to summing L per-layer LM losses into one residual stream (L terms): each layer's gradient then has up to L-l extra contributions.

### DISCO — Deep Supervision with Intermediate Concepts
- cite: Chi Li, M. Zeeshan Zia, Quoc-Huy Tran et al., IEEE TPAMI 41(8) 2019; arXiv:1801.03399; link: https://arxiv.org/abs/1801.03399
- verified: arXiv API + PubMed / IEEE (DOI 10.1109/TPAMI.2018.2863285); details from: full text (ar5iv), Sec. 3-5
- family: 1
- layers supervised: a hierarchy of concepts at increasing depths (viewpoint -> keypoint visibility -> 3D structure -> 2D keypoints)
- target: different (coarse-to-fine) targets per depth, not the same label
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: constant; principle "supervision depths ... should be monotonically increasing" with concept complexity
- purpose: training (generalization)
- final-layer effect (numbers): KITTI-3D: DISCO beats "plain-all" (all supervision at the last layer) by +4.4% on 2D-All and +2.4% on 3D-Full
- relevance (1-5): 2 — suggests that shallow layers should get easier or coarser targets than the final one.

### Contrastive Deep Supervision (CDS)
- cite: Linfeng Zhang, Xin Chen, Junbo Zhang et al., ECCV 2022; arXiv:2207.05306; link: https://arxiv.org/abs/2207.05306
- verified: arXiv API (comment: ECCV 2022); details from: full text (ar5iv), Sec. 1, 3, Tables 1 and 3
- family: 1
- layers supervised: the K-1 intermediate stages
- target: contrastive (SimCLR / SupCon over augmentations) instead of the task CE at shallow layers
- head sharing: no (non-linear projection heads with extra conv layers) ; final norm shared: n.a.
- weighting/schedule: constant lambda
- purpose: training (final accuracy)
- final-layer effect (numbers): argument: task-loss deep supervision "conflicts with the well-known observation that the shallow layers learn low-level features instead of task-biased high-level semantic features" and "this conflict sometimes leads to accuracy degradation in the final classifier". Baseline / DSN / CDS: CIFAR-100 ResNet-18 77.45 / 78.30 / 80.84; ResNet-50 77.81 / 78.96 / 81.31; ImageNet top-1 ResNet-18 69.21 / 69.54 / 72.85; ResNet-34 73.17 / 73.29 / 76.19; ResNet-50 75.30 / 75.37 / 78.25 (DSN adds only +0.07 to +0.33 on ImageNet).
- relevance (1-5): 3 — the main "task loss at shallow layers is the wrong target" paper. On ImageNet, same-label DS gave almost nothing.

### DeepMIM — DeepMIM: Deep Supervision for Masked Image Modeling
- cite: Sucheng Ren, Fangyun Wei, Samuel Albanie et al., WACV 2025; arXiv:2303.08817; link: https://arxiv.org/abs/2303.08817
- verified: arXiv API + WACV 2025 (IEEE Xplore / ML Anthology); details from: full text (arxiv html), Sec. 3-4
- family: 1
- layers supervised: ViT-B blocks 6, 8, 10 (of 12), plus the usual final decoder
- target: same MAE pixel reconstruction. Optional "hybrid" targets t = alpha*x + (1-alpha)*x_hat (x = raw image, x_hat = a pretrained MAE's blurrier reconstruction) with alpha = 0, 1/3, 2/3 for the block-6, 8 and 10 decoders. The shallowest decoder gets the easiest target: "It may be beyond the capacity of these intermediate features to reconstruct the targets that are too complicated, i.e., raw pixels."
- head sharing: no (each extra decoder is an independent 4-layer Transformer); decoders dropped after pretraining ; final norm shared: no/n.a.
- weighting/schedule: plain sum of the M+1 reconstruction losses, constant for all of pretraining
- purpose: training (pretraining quality / convergence)
- final-layer effect (numbers): ImageNet fine-tune top-1, ViT-B/16: MAE 82.6 -> DeepMIM 83.4 (+0.8) -> hybrid-target 83.6 (+1.0) at 300 epochs; 83.6 -> 84.2 (+0.6) at 1600 epochs. Reports faster convergence and more discriminative intermediate blocks (CKA).
- relevance (1-5): 4 — transformer self-supervised pretraining with same-objective deep supervision. It improves the final model, and the gain shrinks with longer training (+0.8 at 300 epochs vs +0.6 at 1600).

### SDViT — Self-Distilled Vision Transformer for Domain Generalization
- cite: Maryam Sultana, Muzammal Naseer, Muhammad Haris Khan et al., ACCV 2022; arXiv:2207.12392; link: https://arxiv.org/abs/2207.12392
- verified: arXiv API (journal_ref: ACCV 2022); details from: full text (ar5iv), Sec. 3-4
- family: 1
- layers supervised: one randomly sampled intermediate block per training step ("distilling the knowledge to all of the sub-models at once poses optimization difficulties")
- target: KL to the final classifier's temperature-softened output (self-distillation), plus CE on the final output
- head sharing: YES. The intermediate class token goes through the same final classifier head h ; final norm shared: unknown
- weighting/schedule: constant lambda in {0.1, 0.2, 0.5}, temperature 3 or 5
- purpose: training (OOD generalization / less overfitting)
- final-layer effect (numbers): PACS DeiT-Small ERM 84.9 -> 86.3; CvT-21 88.3; OfficeHome CvT-21 75.6 (above baseline on all three backbones)
- relevance (1-5): 3 — a close structural analog (intermediate blocks through the shared final head). Its random-single-layer-per-step sampling is a cheap alternative to supervising every layer every step.

### LION-DG — LION-DG: Layer-Informed Initialization with Deep Gradient Protocols for Accelerated Neural Network Training
- cite: Hyunjun Kim, arXiv 2026; arXiv:2601.02105; link: https://arxiv.org/abs/2601.02105
- verified: arXiv API; details from: full text (arxiv html)
- family: 1
- layers supervised: auxiliary classifiers of deeply-supervised DenseNet / ResNet (CIFAR)
- target: same label
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: aux heads are zero-initialized, so at init "the gradient of the auxiliary loss with respect to backbone parameters is exactly zero". Aux gradients then "phase in" as head weights grow ("gradient awakening"), an implicit warm-up of the aux signal.
- purpose: training speed
- final-layer effect (numbers): DenseNet-DS CIFAR-10 81.11 (He init) vs 80.59 (LION-DG, +8.3% faster convergence) vs 81.92 (hybrid LSUV + LION-DG); ResNet-DS CIFAR-100 64.87 -> 64.96 (+11.3% speed-up). Small effects.
- relevance (1-5): 2 — a design note: with a SHARED, already-trained head (the LLAL / planned setup) there is no such implicit warm-up, and the aux gradient is full-strength from step 0.

### DTS — Deep Trajectory Supervision: Deep Supervision Strikes Back
- cite: Han Wang, Weijie Wang, Jiaqi Liu et al. (Hilde Kuehne, Nicu Sebe), ICML 2026; no arXiv id found; link: https://openreview.net/forum?id=N9kBjRH0Cx (ICML page https://icml.cc/virtual/2026/poster/64479)
- verified: ICML 2026 virtual poster page (title, authors, abstract); arXiv title search returns 0 hits; OpenReview page not machine-readable (bot challenge); details from: abstract only
- family: 1
- layers supervised: intermediate layers (Transformers and CNNs)
- target: SOFT targets "whose confidence increases with depth" instead of the hard label at every layer. The schedule follows an exponential evidence-accumulation curve they measured with the Tuned Lens (residual nets viewed as ODE discretizations).
- head sharing: unknown ; final norm shared: unknown
- weighting/schedule: per-depth target sharpness (exponential in depth); train-only ("zero overhead at inference")
- purpose: training (accuracy and convergence speed)
- final-layer effect (numbers): claims improved accuracy and convergence on ImageNet-1K and other benchmarks; numbers not retrieved
- relevance (1-5): 3 — directly addresses the "early layers forced to be as confident as the last" failure. The LM analog is per-layer temperature / label smoothing (or distilling from the final layer with depth-dependent temperature) instead of hard next-token CE at every layer.

### Deep-supervision review — A Comprehensive Review on Deep Supervision: Theories and Applications
- cite: Renjie Li, Xinyi Wang, Guan Huang et al., arXiv 2022; arXiv:2207.02376; link: https://arxiv.org/abs/2207.02376
- verified: arXiv abs page; details from: abstract
- family: 1 (survey)
- layers supervised / target / heads / schedule: n.a. (taxonomy of deep-supervision networks in CV)
- purpose: survey
- final-layer effect (numbers): n.a.
- relevance (1-5): 1

---------------------------------------------------------------------
## Family 1b — iterative-refinement architectures with per-stage / per-layer losses
---------------------------------------------------------------------

### CPM — Convolutional Pose Machines
- cite: Shih-En Wei, Varun Ramakrishna, Takeo Kanade et al. (Yaser Sheikh), CVPR 2016; arXiv:1602.00134; link: https://arxiv.org/abs/1602.00134
- verified: arXiv API + openaccess.thecvf.com (CVPR 2016); details from: full text (ar5iv), Sec. 3.3-3.4, Figs. 5-6
- family: 1b
- layers supervised: the output of every stage (belief maps)
- target: the same ground-truth belief maps at every stage (L2)
- head sharing: no (stages t >= 2 share structure but not weights) ; final norm shared: n.a.
- weighting/schedule: F = sum_t f_t, equal weights, constant
- purpose: training dynamics ("addresses the problem of vanishing gradients"; "replenish" gradients)
- final-layer effect (numbers): gradient histograms without intermediate supervision are "tightly peaked around zero" at early layers, and with it have "moderately large variance throughout the network". Joint training with intermediate supervision beats stage-wise training (which "saturate[s] at sub-optimal") and joint training without it (Fig. 6b; values only in the figure).
- relevance (1-5): 3 — the classic equal-weight, same-target-at-every-stage recipe, justified by gradient flow.

### Stacked Hourglass — Stacked Hourglass Networks for Human Pose Estimation
- cite: Alejandro Newell, Kaiyu Yang, Jia Deng, ECCV 2016 (LNCS 9912); arXiv:1603.06937; link: https://arxiv.org/abs/1603.06937
- verified: arXiv API + Springer (doi 10.1007/978-3-319-46484-8_29); details from: full text (ar5iv), Sec. 3.4 and ablations
- family: 1b
- layers supervised: the output of every hourglass. Predictions are mapped back by a 1x1 conv and added into the feature stream.
- target: "a loss is applied to the predictions of all hourglasses using the same ground truth" (MSE on Gaussian heatmaps)
- head sharing: no ("weights are not shared across hourglass modules") ; final norm shared: n.a.
- weighting/schedule: equal, constant
- purpose: training + iterative refinement
- final-layer effect (numbers): 2 / 4 / 8 stacks (all with intermediate supervision) reach 87.4 / 87.8 / 88.1 PCKh (MPII val). In the single-hourglass ablation, intermediate supervision "does offer an improvement ... but not enough to surpass" stacking + supervision.
- relevance (1-5): 2

### DETR aux decoding losses — End-to-End Object Detection with Transformers
- cite: Nicolas Carion, Francisco Massa, Gabriel Synnaeve et al., ECCV 2020; arXiv:2005.12872; link: https://arxiv.org/abs/2005.12872
- verified: arXiv API + ecva.net (ECCV 2020 supplementary); details from: full text (ar5iv) Sec. 3.2 "Auxiliary decoding losses" and Fig. 4; official code (facebookresearch/detr models/detr.py, models/transformer.py)
- family: 1b
- layers supervised: every decoder layer (6)
- target: the same Hungarian set-prediction loss at every layer
- head sharing: YES. "All predictions FFNs share their parameters. We use an additional shared layer-norm to normalize the input to the prediction FFNs from different decoder layers." In code, the decoder's final LayerNorm is applied to every intermediate output (intermediate.append(self.norm(output))). ; final norm shared: YES
- weighting/schedule: aux weights are identical to the final-layer weights (code copies weight_dict for each of dec_layers-1 layers), constant throughout training, can be turned off with --no_aux_loss
- purpose: training ("helpful ..., especially to help the model output the correct number of objects of each class")
- final-layer effect (numbers): no isolated with/without-aux ablation in the paper. Per-layer evaluation shows AP / AP50 rising by +8.2 / +9.5 from the first to the last decoder layer.
- relevance (1-5): 4 — the exact architectural analog of "shared final norm + shared head at every layer, equal weight, constant", in a transformer, as a default that the whole DETR family kept.

### Deformable DETR (iterative box refinement) — Deformable DETR: Deformable Transformers for End-to-End Object Detection
- cite: Xizhou Zhu, Weijie Su, Lewei Lu et al., ICLR 2021; arXiv:2010.04159; link: https://arxiv.org/abs/2010.04159
- verified: arXiv API (comment: ICLR 2021 Oral); details from: full text (ar5iv) Appendix A.4 and Table 1; official code main.py (--no_aux_loss: aux losses on by default)
- family: 1b
- layers supervised: every decoder layer (aux losses on by default in code)
- target: same detection loss
- head sharing: in the iterative-refinement variant "prediction heads for different decoder layers do not share parameters" ; final norm shared: unknown
- weighting/schedule: constant. For stability, gradients through the refined boxes are "blocked at sigma^-1 of previous layer predictions", following RAFT.
- purpose: training + refinement
- final-layer effect (numbers): 43.8 AP -> 45.4 with iterative box refinement -> 46.2 two-stage (50 epochs)
- relevance (1-5): 3 — the counterpoint to DETR. When each layer refines the previous layer's prediction, they unshare the heads and detach gradients.

### Mask2Former deep supervision — Masked-attention Mask Transformer for Universal Image Segmentation
- cite: Bowen Cheng, Ishan Misra, Alexander G. Schwing et al., CVPR 2022; arXiv:2112.01527; link: https://arxiv.org/abs/2112.01527
- verified: arXiv API (comment: CVPR 2022); details from: full text (ar5iv) Sec. 3-4
- family: 1b
- layers supervised: "An auxiliary loss is added to every intermediate Transformer decoder layer and to the learnable query features before the Transformer decoder"
- target: same mask + class loss
- head sharing: not stated explicitly in the text ; final norm shared: unknown
- weighting/schedule: constant
- purpose: training
- final-layer effect (numbers): no isolated ablation. AR@100 "consistently improves with more decoder layers."
- relevance (1-5): 2

### RAFT sequence loss — RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
- cite: Zachary Teed, Jia Deng, ECCV 2020; arXiv:2003.12039; link: https://arxiv.org/abs/2003.12039
- verified: arXiv API + ecva.net (ECCV 2020 paper PDF); details from: full text (ar5iv) Sec. 3.4 and ablation table
- family: 1b
- layers supervised: every recurrent refinement iterate (12 unrolled updates in training)
- target: the same ground-truth flow (L1)
- head sharing: YES (the update operator, a ConvGRU, is weight-tied across iterations) ; final norm shared: n.a.
- weighting/schedule: L = sum_i gamma^(N-i) ||f_gt - f_i||_1 with gamma = 0.8, so weights grow exponentially toward the last iterate (the first of 12 gets 0.8^11 ~ 0.086). Constant over training.
- purpose: training (iterative refinement)
- final-layer effect (numbers): tied vs untied update weights: Sintel clean / final EPE 1.63 / 2.83 (4.8M params) vs 1.96 / 3.20 (32.5M)
- relevance (1-5): 3 — a well-tested exponential depth weighting (later outputs weigh more). It is the natural opposite of LLAL's "early layer only" and a candidate per-layer weight profile.

---------------------------------------------------------------------
## Family 8 — local / greedy layer-wise learning
---------------------------------------------------------------------

### Greedy layer-wise (2006) — Greedy Layer-Wise Training of Deep Networks
- cite: Yoshua Bengio, Pascal Lamblin, Dan Popovici et al. (Hugo Larochelle), NIPS 2006 (pp. 153-160); no arXiv; link: https://proceedings.neurips.cc/paper/2006/hash/5da713a690c067105aeb2fae32403405-Abstract.html
- verified: NeurIPS proceedings page; details from: abstract
- family: 8
- layers supervised: one layer at a time (greedy), then global fine-tuning
- target: unsupervised (RBM / auto-encoder) per layer. The abstract contrasts this with earlier constructive methods that used "a supervised criterion at each stage".
- head sharing: n.a. ; final norm shared: n.a.
- weighting/schedule: stage-wise, then fine-tune the whole network
- purpose: training (optimization of deep nets)
- final-layer effect (numbers): not retrieved (PDF not machine-readable). Key hypothesis quoted: "using unsupervised learning at each layer in order to preserve information from the input".
- relevance (1-5): 1 — historical root of "per-layer targets should preserve information".

### Greedy ImageNet — Greedy Layerwise Learning Can Scale to ImageNet
- cite: Eugene Belilovsky, Michael Eickenberg, Edouard Oyallon, ICML 2019 (PMLR v97); arXiv:1812.11446; link: https://arxiv.org/abs/1812.11446
- verified: arXiv API + proceedings.mlr.press/v97; details from: full text (ar5iv)
- family: 8
- layers supervised: each new layer is trained (then frozen) with its own auxiliary classifier of depth k = 1, 2, 3
- target: same label
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: sequential. Each layer sees only its own loss.
- purpose: training (understanding / alternative to end-to-end)
- final-layer effect (numbers): ImageNet top-1 / top-5: k=1 58.1 / 79.7 (AlexNet 56.5 / 79.1); k=2 65.7 / 86.3; k=3 ensemble 71.6 / 89.8 (VGG-11 end-to-end 67.9 / 88.0). Linear separability rises monotonically with depth.
- relevance (1-5): 2

### DGL — Decoupled Greedy Learning of CNNs
- cite: Eugene Belilovsky, Michael Eickenberg, Edouard Oyallon, ICML 2020 (PMLR v119); arXiv:1901.08164; link: https://arxiv.org/abs/1901.08164
- verified: arXiv API + proceedings.mlr.press/v119; details from: full text (ar5iv)
- family: 8
- layers supervised: each of K modules has its own auxiliary loss, with no backprop between modules (update-unlocked, parallel)
- target: same label
- head sharing: no (small aux nets, about 5% of FLOPs) ; final norm shared: n.a.
- weighting/schedule: local losses only, constant
- purpose: training efficiency / parallelism
- final-layer effect (numbers): ImageNet top-1 / top-5 DGL vs backprop: VGG-13 (K=4) 67.8 / 88.0 vs 66.6 / 87.5; VGG-19 (K=4) 69.2 / 89.0 vs 69.7 / 89.7; ResNet-152 (K=2) 74.5 / 92.0 vs 74.4 / 92.1
- relevance (1-5): 2

### Greedy InfoMax — Putting An End to End-to-End: Gradient-Isolated Learning of Representations
- cite: Sindy Löwe, Peter O'Connor, Bastiaan S. Veeling, NeurIPS 2019; arXiv:1905.11786; link: https://arxiv.org/abs/1905.11786
- verified: arXiv API (comment: NeurIPS 2019 honorable mention); details from: full text (ar5iv)
- family: 8
- layers supervised: 3 gradient-isolated modules (ResNet-50 v2 on vision)
- target: self-supervised InfoNCE per module (not labels)
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: local only
- purpose: training (biologically plausible, memory)
- final-layer effect (numbers): STL-10 linear probe GIM 81.9 +- 0.3 vs end-to-end CPC 80.5 +- 3.1 (supervised 71.4). LibriSpeech speaker 99.4 vs 99.6; phone 62.5 vs 64.9.
- relevance (1-5): 1

### Local error signals — Training Neural Networks with Local Error Signals
- cite: Arild Nøkland, Lars Hiller Eidnes, ICML 2019; arXiv:1901.06656; link: https://arxiv.org/abs/1901.06656
- verified: arXiv API (comment: ICML 2019); details from: full text (ar5iv)
- family: 8
- layers supervised: every hidden layer, with no global backprop
- target: "pred" = CE through a local linear classifier (same label); "sim" = similarity matching; predsim = (1-beta) pred + beta sim, with beta = 0.99
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: constant, local only
- purpose: training (no global backprop)
- final-layer effect (numbers): test error global BP vs predsim: CIFAR-10 VGG11B(3x) 5.02 vs 3.97; CIFAR-100 23.7 vs 20.1; SVHN VGG8B 2.29 vs 1.74; STL-10 33.08 vs 20.51. "Sim" gives much lower training error than "pred" alone.
- relevance (1-5): 2 — a pure same-label local CE ("pred") is worse than adding a similarity target, another hint that hard labels at every layer are not the best local target.

### InfoPro — Revisiting Locally Supervised Learning: an Alternative to End-to-end Training
- cite: Yulin Wang, Zanlin Ni, Shiji Song et al. (Gao Huang), ICLR 2021; arXiv:2101.10832; link: https://arxiv.org/abs/2101.10832
- verified: arXiv API (comment: ICLR 2021); details from: full text (ar5iv), Sec. 2-3 (Table 1, Fig. 2), Sec. 5
- family: 8
- layers supervised: the outputs of K gradient-isolated modules
- target: greedy SL = same-label CE (baseline analyzed); InfoPro = reconstruction (keep I(h,x)) + task / contrastive term
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: constant lambda_1, lambda_2
- purpose: training memory (and analysis)
- final-layer effect (numbers): KEY NEGATIVE for same-label local losses: ResNet-32 CIFAR-10 test error end-to-end 7.37 -> greedy SL K=2 10.30 -> K=16 24.59 ("severe degradation"). Greedy SL "collapse[s] both I(h,x) and I(h,y) in their first few modules". InfoPro: ResNet-110 CIFAR-10 6.50 (E2E) vs 7.30 (K=4) vs 9.90 (K=16); ImageNet ResNet-101 top-1 error 22.03 (E2E) vs 21.85 (K=2) with 38.8% less memory.
- relevance (1-5): 3 — the information-collapse mechanism behind MSDNet's harm, measured. (Here without a global gradient. With end-to-end deep supervision the collapse pressure is weaker but has the same sign.)

### PGL — Locally Supervised Learning with Periodic Global Guidance
- cite: Hasnain Irshad Bhatti, Jaekyun Moon, ICML 2022 HAET workshop; arXiv:2208.00821; link: https://arxiv.org/abs/2208.00821
- verified: arXiv API; details from: abstract
- family: 8
- layers supervised: decoupled modules with local greedy losses
- target: same label (local), with the global objective reinstated periodically
- head sharing: no ; final norm shared: n.a.
- weighting/schedule: alternates local and global phases periodically
- purpose: memory / parallelism
- final-layer effect (numbers): abstract: local-only training "severely degrades the generalization", and periodic global guidance gives "significant performance gains" (no numbers in abstract)
- relevance (1-5): 2

### AugLocal — Scaling Supervised Local Learning with Augmented Auxiliary Networks
- cite: Chenxiang Ma, Jibin Wu, Chenyang Si et al. (Kay Chen Tan), ICLR 2024; arXiv:2402.17318; link: https://arxiv.org/abs/2402.17318
- verified: arXiv API (comment: ICLR 2024); details from: full text (arxiv html)
- family: 8
- layers supervised: every hidden layer (local), each with an auxiliary net built by uniformly sampling the primary network's subsequent layers. Aux depth decays pyramidally with layer index (tau = 0.5).
- target: same-label CE
- head sharing: partial (aux nets copy the architecture of later layers, not their weights) ; final norm shared: n.a.
- weighting/schedule: constant, local only
- purpose: memory
- final-layer effect (numbers): CIFAR-10 ResNet-110: BP 94.61 vs AugLocal (d=6) 93.96 vs InfoPro 86.95 vs PredSim 74.95; ImageNet ResNet-101 top-1 77.34 (BP) vs 76.70; about 40% memory saving. CKA: local layers resemble BP layers more as aux depth grows.
- relevance (1-5): 2 — local losses work when the aux head "looks like the rest of the network". Analog: the shared final norm + LM head is only the last piece of "the rest of the network".

### SOLO — SOLO: Pretraining Billion-Parameter Language Models with Shared-Output Local Learning
- cite: Bojian Yin, Shurong Wang, Yuqi Pan et al. (Guoqi Li), arXiv 2026 (submitted 2026-09-28, preprint); arXiv:2609.35440; link: https://arxiv.org/abs/2609.35440
- verified: arXiv abs page; details from: full text (arxiv html), method section, Tables 1, 4, 9, 10
- family: 8 (but directly relevant to family 2: decoder-LM pretraining with per-module next-token losses through a shared unembedding)
- layers supervised: K = 2 or 4 gradient-isolated modules of a 24-layer Transformer (every module's output gets a next-token loss)
- target: same next-token label (CE)
- head sharing: YES for the unembedding. Every aux head predicts through "a shared, read-only copy of the terminal readout" W from the previous step (W_{t-1}); "only the final loss updates W". Each aux head also has its own H = 2 Transformer++ blocks, "a final RMSNorm", a learned scalar temperature tau_k and a bias. ; final norm shared: NO (each head has its own RMSNorm)
- weighting/schedule: equal weights, constant, no annealing. Gradient is stopped between modules (pure local learning, no global backprop).
- purpose: training efficiency (no update locking, pipeline memory; up to 1.44x throughput)
- final-layer effect (numbers): WikiText ppl SOLO vs BP: 340M K=2 29.05 vs 28.04, K=4 31.19 vs 28.04; 1.3B K=2 22.29 vs 21.61, K=4 23.59; 2B K=2 21.57 vs 20.93, K=4 22.21. Zero-shot average within 0.5 (K=2) / 0.9 (K=4) points of BP. Readout ablation at 40M (WikiText ppl): rand 98.88, private readouts 75.34, SOLO shared 70.62, BP 67.08. A shared random readout beats per-module random readouts by about 12 ppl, and rotating the shared basis per module costs 13.3 ppl. So sharing the readout basis is what matters. Every module becomes a usable early exit (the 6-layer exit of the 1.3B model: 28.20 ppl, about a 340M BP model). SOLO exits agree with the terminal prediction 71-75% of the time vs 16-51% for BP's logit lens. Gradient cosine with BP: rand 0.32, priv 0.52, SOLO 0.70.
- relevance (1-5): 4 — the only decoder-LM pretraining result here with per-module next-token losses through the shared unembedding. It shows the shared readout is the right design. It also shows that in a BP-trained LM the intermediate states are far from logit-lens-decodable (16-51% agreement), so a shared-head per-layer loss is a strong constraint. It is not end-to-end deep supervision (no global gradient, no schedule), and its heads needed 2 extra blocks, so the numbers bound rather than predict the planned experiment.

### LoPT — Rethinking Local Learning: A Cheaper and Faster Recipe for LLM Post-Training
- cite: Hengyu Shi, Tianyang Han, Peizhe Wang et al., arXiv 2026; arXiv:2605.04913; link: https://arxiv.org/abs/2605.04913
- verified: arXiv API + abs page; details from: abstract
- family: 8
- layers supervised: one gradient boundary at the transformer midpoint. The second half learns from the task objective, and the first half from "a lightweight feature-reconstruction objective" (not an LM loss).
- target: task loss (top half) / feature reconstruction (bottom half)
- head sharing: not stated ; final norm shared: unknown
- weighting/schedule: n.a. (post-training)
- purpose: training efficiency + retention of pretrained capabilities
- final-layer effect (numbers): abstract: "competitive performance with lower memory cost ... and better retention of pretrained capabilities" (no numbers retrieved)
- relevance (1-5): 2 — an LLM-scale instance of "don't push narrow task gradients into early layers".

---------------------------------------------------------------------
## Family-level takeaways
---------------------------------------------------------------------
- Same-label deep supervision gives small but mostly positive final-output gains in CNNs when the aux weight is modest and/or annealed: DSN (CIFAR-100 35.68 -> 34.57 err), CNDS (alpha 0.3 decayed; ImageNet top-1 err 34.7 -> 33.8, about 1 day faster), PSPNet (any alpha 0.3-0.9 helps; 0.4 best, +1.4 mIoU), DKS's plain-DS baseline (+0.4 to +0.6 on ImageNet ResNets). On ImageNet it can be close to zero: Contrastive DS reports plain DSN at +0.07 to +0.33 top-1 for ResNet-18/34/50.
- It can clearly HURT the final output when shallow features get pushed toward the final task: MSDNet (one intermediate classifier costs ResNet's final classifier "up to 7%" on CIFAR-100; DenseNet far less, because dense connectivity lets later layers bypass short-term features); Contrastive DS ("sometimes leads to accuracy degradation in the final classifier"); InfoPro (greedy same-label local losses: ResNet-32 CIFAR-10 error 7.37 -> 10.30 at K=2 -> 24.59 at K=16, via collapse of I(h,x)); AdaLoss (constant equal weights: 15-18% more relative error than single-output nets, widest at the deepest outputs). Decoder LMs have a purely additive residual stream (ResNet-like, not DenseNet-like), so the MSDNet risk is the relevant one.
- The early-vs-late question has mixed CNN evidence. Inception-v3 found aux heads did NOT speed early convergence and helped only near the end (a regularizer). DSN / CNDS / CPM frame the benefit as fixing early gradient flow, and CNDS and DSN anneal the aux weight to about 0 (alpha_t <- alpha_t (1 - t/N); DSN's gamma-hinge turns each layer's companion loss off once it is low enough). No vision paper here tests "early window only, then removed" as LLAL does.
- Weighting profiles that tend to protect the final output grow with depth: SDN weights ICs by their depth fraction (0.15 ... 0.9, final 1), ramped linearly from 0.01; RAFT uses gamma^(N-i), gamma = 0.8 (first of 12 iterates about 0.09); BranchyNet found unequal weights worth about 1%; AdaLoss normalizes each loss by its running mean (geometric-mean objective); IMTA's gradient equilibrium rescales the extra backbone gradients so their variance stays bounded as exits are added.
- Softer or easier targets for shallow layers beat hard labels everywhere: DTS (ICML 2026: confidence increasing with depth, exponential schedule from the Tuned Lens), DeepMIM's hybrid (easier) targets for the shallow decoders (+0.2 more), DISCO's coarse-to-fine concepts (+4.4%), distillation from the final head (BYOT, DKS, IMTA, SDViT beat plain DS), and Contrastive DS (non-task target at shallow stages). LM analogs: temperature or label smoothing by depth, or KL to the (detached) final-layer distribution.
- Shared head + shared final norm across depths has a strong transformer precedent. DETR sends every decoder layer through one shared LayerNorm and shared FFN heads with loss weights equal to the final layer (on for the whole run, kept in Deformable DETR / Mask2Former). SDViT routes intermediate ViT blocks through the shared final classifier. SOLO shows, for decoder LMs, that sharing the unembedding basis is what makes per-module next-token losses work (40M: private readouts 75.34 vs shared 70.62 vs BP 67.08 ppl). When layers are meant to iteratively refine rather than all predict (Deformable DETR box refinement), heads are unshared and gradients detached.
- Transformer pretraining with deep supervision improved the final model in MIM: DeepMIM +0.8 top-1 at 300 epochs, +0.6 at 1600. The gain shrinks with longer training (a hint that the benefit is partly acceleration).
- Cost-saving variant: SDViT supervises ONE random intermediate block per step (they report that distilling into all blocks at once "poses optimization difficulties"). This is a cheap alternative to all-layer supervision for the dense experiment.
- Local-learning results (Greedy ImageNet, DGL, AugLocal, SOLO) show that per-layer losses can carry most of the learning signal. The gap to backprop grows with the number of isolated modules (SOLO 340M: +1.01 ppl at K=2, +3.15 at K=4; InfoPro K=16). This argues for keeping the global gradient (deep supervision, not local learning) and for keeping aux weights small or decaying.

## Could not verify
- "A comprehensive review on deep supervision in computer vision" (Neurocomputing 2025, ScienceDirect pii S0925231225028656): seen only as a search hit; not opened.
- DTS (ICML 2026) numeric results and exact target formula: OpenReview (N9kBjRH0Cx) blocked by a bot challenge and no arXiv version found. Only the ICML page's title, authors and abstract were verified.
- Bengio et al. 2006 results table (supervised vs unsupervised greedy variants): proceedings PDF not machine-readable. Only the abstract was verified.
- CPM ablation numbers (with vs without intermediate supervision; stage-wise vs joint): given only as a figure (Fig. 6b), no values in the text.
- GoogLeNet: no isolated quantitative effect of the aux classifiers in the paper text (a "~0.5%" figure sometimes quoted secondhand could not be found in the paper).
- SDN: the paper does not isolate the final classifier's accuracy after SDN training. Only early-exit accuracies (Table 3) are verified.
