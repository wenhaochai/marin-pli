# Family 4: Speech (intermediate-layer losses in ASR / ST / speech SSL)

Notes by the family-4 fork, 2026-09-29. Every citation below was checked against the arXiv API (export.arxiv.org), plus Crossref, the ISCA archive or the arXiv comment field for the venue. "details from" names the source of the method details and numbers. WER is test-clean/test-other unless a split is named.

Relevance scale (for the planned experiment: a temporary, annealed next-token LM loss at EVERY layer of a dense decoder LM trained from scratch, through the shared final norm + LM head): 5 = decoder LM with per-layer LM loss in pretraining; 4 = same-label per-layer losses in transformers, with final-layer evidence or a weighting/schedule insight; 3 = deep supervision elsewhere with a transferable insight; 2 = inference-motivated exits or alignment/distillation variants; 1 = tangential.

---

### InterCTC — Intermediate Loss Regularization for CTC-based Speech Recognition
- cite: Jaesong Lee, Shinji Watanabe, ICASSP 2021; arXiv:2102.03216; link: https://arxiv.org/abs/2102.03216
- verified: arXiv API (comment "Accepted at ICASSP 2021") + Crossref DOI 10.1109/ICASSP39728.2021.9414594. Details from the full text (ar5iv), Sec. 3–4 and Tables 1–3, copied verbatim. Head and norm sharing come from the ESPnet reference implementation (`espnet2/asr/encoder/conformer_encoder.py`, `espnet2/asr/espnet_model.py`, master branch, fetched 2026-09-29).
- family: 4
- layers supervised: one intermediate layer, ⌊L/2⌋ by default (layer 6 of 12, 12 of 24, 24 of 48; 12-layer Conformer). Variants: "lower" (layer 6 of 24), "multiple" (K=3 positions for 24 layers, K=7 for 48 layers), "random" position.
- target: the same CTC transcript as the final layer (same vocabulary).
- head sharing: the paper defines L_InterCTC := −log P_CTC(y | x_⌊L/2⌋), the same CTC formulation applied to the intermediate representation, "with a very small overhead". It mentions no extra parameters. In ESPnet, intermediate outputs go through the shared `self.ctc` module, and when the model is pre-LN they are first passed through the encoder's final LayerNorm (code comment: "# intermediate outputs are also normalized": `encoder_out = self.after_norm(encoder_out)`). final norm shared: yes (implementation; the paper is silent).
- weighting/schedule: L = (1−w)·L_CTC + w·L_InterCTC with w = 0.3, held constant for all of training. With several intermediate layers, ESPnet averages the intermediate losses (`loss_interctc / len(intermediate_outs)`) before applying w.
- purpose: training dynamics (regularizes the lower layers); no inference cost.
- final-layer effect (numbers): WSJ eval92, Transformer: 12 layers 16.5→13.6, 24 layers 13.9→12.4, 48 layers 13.8→12.6. 12-layer Conformer: 12.4→10.8. TED-LIUM2 test: 14.0→12.3 (12 layers), 12.2→10.6 (24), 10.9→10.3 (48). AISHELL-1 test CER: 6.3→6.2, 5.9→5.6, 5.7→5.5; Conformer 6.0→5.6. It stacks with stochastic depth (24 layers, both: eval92 11.8). Positions on WSJ eval92: lower (6/24) 12.9 vs default 12.4, still better than the 13.9 baseline; multiple 12.0 (24 layers, K=3) and 12.1 (48 layers, K=7) vs 12.4/12.6 default; random 12.0 (48 layers). Stated rationale: stochastic depth regularizes the higher layers, but the lower layers "may rely on the remaining higher layers rather than learn regularized representation by themselves".
- relevance (1-5): 4. This is the cleanest same-label intermediate loss through a shared head and shared final norm, on transformers trained from scratch, with final-layer gains at 12–48 layers. It uses a constant weight, and the model is an encoder (CTC), not a causal LM.

### SC-CTC — Relaxing the Conditional Independence Assumption of CTC-based ASR by Conditioning on Intermediate Predictions
- cite: Jumon Nozaki, Tatsuya Komatsu, Interspeech 2021; arXiv:2104.02724; link: https://arxiv.org/abs/2104.02724
- verified: arXiv API (comment "Accepted to INTERSPEECH2021"). Details from the full text (ar5iv), Sec. 3, Tables 1–3 (verbatim) and Fig. 3.
- family: 4
- layers supervised: K=5 intermediate layers of an 18-layer encoder at ⌊k·L/(K+1)⌋ (about every 3 layers), plus the final layer.
- target: the same transcript CTC. Intermediate posteriors are also fed back: X_{l+1}^in = LayerNorm(X_l^out) + Linear_{|V|→D}(Z_l).
- head sharing: intermediate predictions are computed "using Eq. (6) and Eq. (7)", i.e. the same LayerNorm (Eq. 6) and the same Linear_{D→|V|}+softmax (Eq. 7) as the final output. The back-projection Linear_{|V|→D} is "shared in the model". final norm shared: yes.
- weighting/schedule: L = (1−λ)·L_CTC + λ·(1/K)·Σ_k L_inter,k, with λ = 0.5, held constant.
- purpose: training dynamics plus accuracy of non-autoregressive decoding.
- final-layer effect (numbers), all 18-layer, CTC → InterCTC → SC-CTC: TEDLIUM2 test 12.2 → 10.1 → 9.4; WSJ eval92 14.9 → 12.7 → 11.9; AISHELL-1 test CER 6.2 → 5.7 → 5.3. Fig. 3: for InterCTC, "increasing the number of intermediate CTC loss gives no improvement for 12-layer and a little improvement for 18-layer". SC-CTC benefits from larger K in both.
- relevance (1-5): 4. This is the exact design of "a loss every few layers through the shared final norm + output head". It also shows that plain multi-layer same-label losses saturate unless the intermediate predictions are used.

### Deja-vu (iterated loss) — Deja-vu: Double Feature Presentation and Iterated Loss in Deep Transformer Networks
- cite: Andros Tjandra, Chunxi Liu, Frank Zhang et al., ICASSP 2020; arXiv:1910.10324; link: https://arxiv.org/abs/1910.10324
- verified: arXiv API (comment "Accepted in IEEE ICASSP 2020"). Details from the full text (ar5iv), Tables 1–2 (verbatim).
- family: 4
- layers supervised: 12-24, 8-16-24 or 6-12-18-24 of a 24-layer transformer; 12-24-36 of a 36-layer one.
- target: the same target as the final layer (CTC over 5k subwords; CE for hybrid).
- head sharing: a separate auxiliary MLP per intermediate loss ("two linear layers with 256 hidden units, LeakyReLU activation and softmax"). final norm shared: no / n.a.
- weighting/schedule: Loss(P_M, Y) + λ·Σ_k Loss(P_k, Y), λ = 0.3 ("based on our preliminary experiments"), constant.
- purpose: training dynamics in deep transformers.
- final-layer effect (numbers): LibriSpeech CTC, 24 layers, no SpecAugment: 5.0/13.1 → 4.5/12.2 (12-24), 4.6/12.3 (8-16-24), 4.4/12.0 (6-12-18-24). With SpecAugment: 24 layers 4.0/9.4 → 3.5/8.4 (8-16-24); 36 layers 4.0/9.4 → 3.4/8.1 (12-24-36). Without the iterated loss, the 36-layer baseline is no better than the 24-layer one; with it, depth pays off.
- relevance (1-5): 4. A constant same-label loss at every 6–12 layers improves the final layer and makes extra depth useful.

### Transformer hybrid AM (iterated loss) — Transformer-based Acoustic Modeling for Hybrid Speech Recognition
- cite: Yongqiang Wang, Abdelrahman Mohamed, Duc Le et al., ICASSP 2020; arXiv:1910.09799; link: https://arxiv.org/abs/1910.09799
- verified: arXiv API (DOI 10.1109/ICASSP40776.2020.9054345). Details from the full text (ar5iv), iterated-loss section and Table 3.
- family: 4
- layers supervised: outputs of layers 6/12/18 of a 24-layer model.
- target: the same senone/chenone cross-entropy as the final layer.
- head sharing: separate branches (a linear map to 256-d with ReLU, then a layer-specific softmax projection). "Intermediate-layer-specific parameters (e.g., the linear transformation before the softmax operation) are discarded after training." final norm shared: no.
- weighting/schedule: the auxiliary CE losses are "interpolated with the original CE loss with a 0.3 weight", constant.
- purpose: training dynamics (convergence of deep models).
- final-layer effect (numbers): LibriSpeech, vggTrf(768, 12 layers): 2.87/6.46 → 2.77/6.10. vggTrf(512, 24 layers): "not converged" without the loss → 2.66/5.64 with it. Quote: "Deep transformer models (deeper than 20 layers) often got stuck in training and made little progress for a long time."
- relevance (1-5): 4. This is the most direct speech evidence that a same-label per-layer loss rescues optimization with depth, the analog of LLAL's "weak learning signal to lower layers".

### RNN-T auxiliary tasks — Improving RNN Transducer Based ASR with Auxiliary Tasks
- cite: Chunxi Liu, Frank Zhang, Duc Le et al., IEEE SLT 2021; arXiv:2011.03109; link: https://arxiv.org/abs/2011.03109
- verified: arXiv API (comment "Accepted for publication at IEEE SLT 2021"). Details from the full text (ar5iv), method section and LibriSpeech table.
- family: 4
- layers supervised: auxiliary RNN-T branches on intermediate encoder layers (6, 12, 18 of 24). Chenone CE at layers {12, 24} (24 layers) or {18, 36} (36 layers).
- target: the same RNN-T loss on the auxiliary branch, plus a symmetric KL between the primary and auxiliary posteriors; chenone CE uses finer, context-dependent units.
- head sharing: the auxiliary branch reuses the primary predictor and joiner in the forward pass, but gradients stop there: "we do not update the decoder parameters if the gradients are back propagated from the auxiliary RNN-T loss". The auxiliary loss therefore trains only the lower trunk. final norm shared: unknown.
- weighting/schedule: fixed interpolation weights (λ for aux RNN-T+KL tuned; λ_ce = 0.6 reported); no schedule.
- purpose: training dynamics (encoder underfitting under teacher forcing; optimizing deep networks).
- final-layer effect (numbers): LibriSpeech, 24 layers: baseline 2.77/6.60; +aux+KL 2.48/5.62; +CE 2.42/5.75; +both 2.31/5.26. 36 layers with both: 2.2/4.7 without LM, 2.0/4.2 with LM. Quote: "without the auxiliary tasks, neither 24-layer transformer of FFN size 3072 nor 36-layer transformer of FFN 2048 is able to converge."
- relevance (1-5): 4. It confirms the depth-rescue effect, and the stop-gradient-into-shared-head design is a concrete option: the per-layer loss trains the trunk without distorting the shared LM head.

### Early-exit ASR from scratch (Wright et al.) — Training dynamic models using early exits for automatic speech recognition on resource-constrained devices
- cite: George August Wright, Umberto Cappellazzo, Salah Zaiem et al., ICASSP 2024 Workshop on Self-supervision in Audio, Speech and Beyond (SASB); arXiv:2309.09546; link: https://arxiv.org/abs/2309.09546
- verified: arXiv API (comment). Details from the full text (arXiv HTML), loss definition and Tables 2–3 (verbatim). The no-EE rows are "individual single-exit models trained independently" at each depth.
- family: 4
- layers supervised: exits after every other layer (2, 4, …, 12) of a 12-layer Conformer (CTC or AED), or of wav2vec2 / WavLM.
- target: the same transcript loss at every exit.
- head sharing: separate exit decoders (linear+softmax for CTC; a 4-layer transformer decoder for AED). final norm shared: no / unknown.
- weighting/schedule: an unweighted sum, L_EE = Σ_m L(ŷ^m, y), constant.
- purpose: inference (dynamic depth), but the paper measures the training effect.
- final-layer effect (numbers): from scratch, LibriSpeech layer-12 exit vs the single-exit 12-layer model: Conformer-CTC 5.1/15.1 vs 6.5/17.7; Conformer-AED 2.3/6.0 vs 2.5/6.1. TED-LIUM Conformer-CTC: 14.6 vs 16.4. The shallowest exit is worse than a dedicated shallow model (CTC layer 2: 23.9/43.8 vs 17.6/36.1). Adding exits while fine-tuning pretrained SSL models hurts the final layer: wav2vec2-CTC 4.3/12.2 vs 3.4/8.6; WavLM-CTC 3.6/8.8 vs 3.0/6.5. The authors call the compound loss "a regulariser".
- relevance (1-5): 4. This is the clearest from-scratch vs fine-tune contrast. The same all-layer, equal-weight losses improved the final exit from scratch and hurt it when bolted onto an already-trained model.

### DeCRED — DeCRED: Decoder-Centric Regularization for Encoder-Decoder Based Speech Recognition (earlier version: "Improving Automatic Speech Recognition with Decoder-Centric Regularisation in Encoder-Decoder Models")
- cite: Alexander Polok, Santosh Kesiraju, Karel Beneš et al., IEEE ASRU 2025; arXiv:2508.08938 (earlier arXiv:2410.17437); link: https://arxiv.org/abs/2508.08938
- verified: arXiv API for both ids; Crossref DOI 10.1109/ASRU65441.2025.11434661. Details from the full text (arXiv HTML) of both versions, including the layer/weight ablation (v2 Table VII; v1 Appendix Table 5).
- family: 4
- layers supervised: intermediate layers of the autoregressive decoder, i.e. the internal conditional LM. By default a single auxiliary classifier sits at decoder layer D−2. Ablation over layers 1–5 with β ∈ {0.1…0.5}.
- target: the same teacher-forced next-token cross-entropy as the final decoder layer.
- head sharing: separate, untied classifiers ("The weights of the additional classifiers are not tied to the one attached to the D-th layer"). final norm shared: unknown.
- weighting/schedule: L = α·L_CTC + (1−α)·Σ_d β_d·L_d^Attn with Σβ_d = 1. The default β_{D−2} = 0.4, so the final layer keeps 0.6. Constant; the auxiliary heads are dropped at inference (optional multi-layer logit fusion).
- purpose: training dynamics (regularizes the decoder's internal LM).
- final-layer effect (numbers): macro WER in-domain 6.4 → 6.3; out-of-domain 18.2 → 16.2. Internal-LM BPE perplexity falls 36.6% relative (mean over 11 test sets). TEDLIUM3 (37M model), greedy/beam: DeCRED 7.0/6.8 vs encoder-side InterCTC 7.5/7.1. v1: AMI 24.8 → 22.1, GigaSpeech 19.8 → 16.9. Ablation: middle-to-late decoder layers work best (β3=0.5 → 6.7; β4=0.4 → 6.8 WER on TEDLIUM3). Early layers give minimal gains, and several auxiliary classifiers gave no significant extra gain.
- relevance (1-5): 4. It is the only speech paper found with next-token losses on intermediate layers of an autoregressive decoder. Its evidence favors one late-ish layer with a separate head over many layers. The setting is encoder-decoder with a constant weight.

### ILO shared-decoder regularization — Intermediate-layer output Regularization for Attention-based Speech Recognition with Shared Decoder
- cite: Jicheng Zhang, Yizhou Peng, Haihua Xu et al., arXiv 2022 (submitted to Interspeech 2022; venue not confirmed); arXiv:2207.04177; link: https://arxiv.org/abs/2207.04177
- verified: arXiv API. Details from the full text (ar5iv).
- family: 4
- layers supervised: one intermediate encoder layer, whose output also goes to the decoder; the best position is about layer 9 of 12 (Fig. 3 scan).
- target: the same transcript. The intermediate encoder output feeds the SAME attention decoder (teacher-forced CE), in addition to / compared with intermediate CTC.
- head sharing: the decoder and its output layer are fully shared; the extra connection is removed at inference. final norm shared: unknown.
- weighting/schedule: L = α·L_ctc + β·L_att + γ·L_att^inter with α+β+γ = 1, α = 0.3, γ = 0.2 in most experiments; constant.
- purpose: training dynamics / regularization.
- final-layer effect (numbers): accented English about 8.2% relative WER reduction vs the baseline (7.7 vs 8.1 against the SpecAugment baseline, 4.9% relative). LibriSpeech test-other 6.9 → 6.8 with SpecAugment.
- relevance (1-5): 3. Intermediate supervision through a shared head helps, and the best position is late (9/12), not early.

### Layer pruning on demand — Layer Pruning on Demand with Intermediate CTC
- cite: Jaesong Lee, Jingu Kang, Shinji Watanabe, Interspeech 2021, pp. 3745–3749; arXiv:2106.09216; link: https://arxiv.org/abs/2106.09216
- verified: arXiv API + ISCA archive (DOI 10.21437/Interspeech.2021-1171). Details from the abstract only: ar5iv conversion failed and the PDF could not be parsed.
- family: 4
- layers supervised: intermediate CTC combined with stochastic depth, so that every truncated depth is usable (exact layers unknown).
- target: the same CTC transcript.
- head sharing: unknown; final norm shared: unknown.
- weighting/schedule: unknown.
- purpose: inference (depth reduction at run time without fine-tuning).
- final-layer effect (numbers): abstract only. Each pruned sub-model "maintains the accuracy of individually trained model of the same depth"; RTF 0.005 → 0.002 on GPU. Full-depth effect not verified.
- relevance (1-5): 2. It shows that InterCTC plus stochastic depth makes every depth usable (the LayerSkip recipe in speech), but the numbers are not verified.

### HuBERT-ILS (ILS-SSL) — Self-Supervised Learning for speech recognition with Intermediate layer supervision
- cite: Chengyi Wang, Yu Wu, Sanyuan Chen et al., arXiv 2021 (submitted to ICASSP 2022; acceptance not confirmed here); arXiv:2112.08778; link: https://arxiv.org/abs/2112.08778
- verified: arXiv API. Details from the full text (ar5iv).
- family: 4
- layers supervised: masked-prediction loss during PRETRAINING at layers {4, 12} for BASE (12 layers) and {9, 24} for LARGE. Layers were chosen by dev WER.
- target: the same k-means unit targets at every supervised layer.
- head sharing: separate prediction weights W^l and codeword embeddings e^l per layer. Sharing them across layers was "slightly worse" (<3% gap). final norm shared: n.a.
- weighting/schedule: unweighted sum over the supervised layers, constant for all of pretraining.
- purpose: training dynamics (push content information into the lower layers).
- final-layer effect (numbers): downstream ASR, BASE (100h fine-tuning): 6.3/13.2 (HuBERT) → 4.7/10.1. LARGE (960h fine-tuning): 2.1/4.3 → 1.9/3.8. SUPERB: ASR 6.42 → 5.45 WER, but speaker ID 81.42 → 79.29 accuracy. Intermediate supervision moves content into the lower layers at the cost of non-target (speaker) information.
- relevance (1-5): 3. This is from-scratch pretraining with a same-target intermediate loss and a clear downstream gain, and it documents a representational cost.

### OWSM-CTC — OWSM-CTC: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification
- cite: Yifan Peng, Yui Sudo, Muhammad Shakeel, Shinji Watanabe, ACL 2024; arXiv:2402.12654; link: https://arxiv.org/abs/2402.12654
- verified: arXiv API (comment "Accepted at ACL 2024 main conference"). Details from the full text (arXiv HTML v1).
- family: 4
- layers supervised: a 27-layer E-Branchformer (1.01B parameters, 180k hours, 151 languages). Self-conditioned intermediate CTC at layers 6, 12, 15 (ASR targets) and 21 (task-dependent), plus the final layer 27.
- target: transcript (+ language token) in the lower half; task-dependent (ASR transcript or ST translation) in the upper layers.
- head sharing: one CTC projection W1 shared across layers; W2 back-projects for self-conditioning. final norm shared: unknown.
- weighting/schedule: a plain average of the 5 CTC losses, 1/(1+|S|)·(L^(N) + Σ L^(s)), constant.
- purpose: training dynamics plus multitask quality at scale.
- final-layer effect (numbers): no with/without-intermediate ablation is reported. In the appendix ablation, making every intermediate CTC task-dependent diverged; ASR-only targets in the first half plus multitask in the second half worked best.
- relevance (1-5): 3. Equal-weight, shared-head, 5-layer supervision is stable at ~1B parameters and 180k hours. Target choice per depth mattered (the model diverged when all layers had hard task-dependent targets).

### HC-CTC — Hierarchical Conditional End-to-End ASR with CTC and Multi-Granular Subword Units
- cite: Yosuke Higuchi, Keita Karube, Tetsuji Ogawa et al., ICASSP 2022; arXiv:2110.04109; link: https://arxiv.org/abs/2110.04109
- verified: arXiv API (comment "Accepted to ICASSP2022"). Details from the full text (ar5iv).
- family: 4
- layers supervised: an 18-layer Transformer with CTC at layers 6, 12 and 18.
- target: granularity grows with depth: 256 → 2048 → 16384 subwords (LS-100/TED2), 512 → 4096 → 32768 (LS-960). Each level is self-conditioned on the lower-level predictions.
- head sharing: separate projections per level (different vocabularies). final norm shared: unknown.
- weighting/schedule: equal weights, 1/K over K = 3 losses; constant.
- purpose: training dynamics / accuracy.
- final-layer effect (numbers): LibriSpeech-100 test-clean: CTC 11.8 → SC-CTC 9.1 → HC-CTC 8.4. With the same 16k vocabulary at all layers, SC-CTC gets 8.9/21.0 vs HC-CTC 8.2/19.9 (LS-100 clean/other). The paper: "hierarchically increasing the subword units resulted in better performance than using the same vocabulary size for intermediate losses". Gains on LS-960 and TEDLIUM2 were marginal.
- relevance (1-5): 3. Easier (coarser) targets for shallow layers beat identical targets at every layer. That is an alternative to plain next-token loss at every layer.

### HMTL-CTC (Sanabria & Metze) — Hierarchical Multi Task Learning With CTC
- cite: Ramon Sanabria, Florian Metze, IEEE SLT 2018; arXiv:1807.07104; link: https://arxiv.org/abs/1807.07104
- verified: arXiv API (comment "In Proceedings at SLT 2018"). Details from the full text (ar5iv).
- family: 4
- layers supervised: CTC after successive BiLSTM encoder layers: characters (lowest) → 300 → 1k → 10k subwords (top).
- target: units get coarser with depth.
- head sharing: separate task-specific modules (BiLSTM + linear) per layer. final norm shared: n.a.
- weighting/schedule: not specified in the paper.
- purpose: accuracy.
- final-layer effect (numbers): Eval2000 SWB/CH with shallow fusion: HMTL (s10k) 12.5/23.7 vs single-task 15.6/26.7. Hierarchical placement (14–20% relative) beat putting all tasks at the top layer (block MTL, 9–18% relative).
- relevance (1-5): 2. It is an RNN with different targets, but it adds evidence that per-depth targets beat top-layer multitask.

### Hierarchical MTL for CTC (Krishna et al.) — Hierarchical Multitask Learning for CTC-based Speech Recognition
- cite: Kalpesh Krishna, Shubham Toshniwal, Karen Livescu, arXiv 2018 (technical report); arXiv:1807.06234; link: https://arxiv.org/abs/1807.06234
- verified: arXiv API (comment "Technical Report"). Details from the full text (ar5iv).
- family: 4
- layers supervised: phone CTC at intermediate layer i ∈ {1..5} of a 5-layer BiLSTM encoder; subword CTC at the top.
- target: phones (auxiliary) vs subwords (main).
- head sharing: separate softmax "added in parallel". final norm shared: n.a.
- weighting/schedule: L = λ·L_subword + (1−λ)·L_phone, best λ = 0.7; constant.
- purpose: accuracy.
- final-layer effect (numbers): Switchboard 21.5 → 18.6 (best multitask), 17.9 with pretraining plus multitask. i = 3, 4 beat i = 5 (top); "all models match or outperform the baseline".
- relevance (1-5): 2.

### Low-level auxiliary tasks (Toshniwal et al.) — Multitask Learning with Low-Level Auxiliary Tasks for Encoder-Decoder Based Speech Recognition
- cite: Shubham Toshniwal, Hao Tang, Liang Lu, Karen Livescu, Interspeech 2017, pp. 3532–3536; arXiv:1704.01631; link: https://arxiv.org/abs/1704.01631
- verified: arXiv API + ISCA archive (DOI 10.21437/Interspeech.2017-1118). Details from the full text (ar5iv).
- family: 4
- layers supervised: a phoneme decoder or CTC at encoder layer 3 (of 4); HMM-state CE at layer 2.
- target: lower-level units.
- head sharing: separate decoders/classifiers. final norm shared: n.a.
- weighting/schedule: equal average, L = 1/3·(L_c + L_pDec + L_s); constant.
- purpose: accuracy.
- final-layer effect (numbers): Eval2000 SWB/CHE/all 25.0/42.4/33.7 → 23.1/40.8/32.0. "The top-layer application of the phoneme loss produces worse performance than having the supervision at the lower (third) layer."
- relevance (1-5): 2.

### GIC — Improving CTC-based ASR Models with Gated Interlayer Collaboration
- cite: Yuting Yang, Yuke Li, Binbin Du, ICASSP 2023; arXiv:2205.12462; link: https://arxiv.org/abs/2205.12462
- verified: arXiv API (comment "Accepted by ICASSP 2023"). Details from the full text (ar5iv).
- family: 4
- layers supervised: intermediate CTC at layers {3, 6, 9, 12, 15} of an 18-layer encoder, plus the final layer.
- target: the same transcript CTC. Intermediate posteriors weight a sum of token embeddings, which a gate fuses back into the stream.
- head sharing: q^l = Softmax(Linear(LN(h^l))). Whether Linear/LN are shared with the final layer is not stated explicitly. final norm shared: unclear.
- weighting/schedule: (1−λ)·L_ctc + λ·(1/K)·Σ L_inter with K = 5, λ = 0.5; constant.
- purpose: accuracy (relax CTC conditional independence).
- final-layer effect (numbers), CTC / InterCTC / SC-CTC / GIC: AISHELL-1 test CER 5.2 / 4.7 / 4.6 / 4.4; TEDLIUM2 test WER 8.6 / 8.3 / 7.8 / 7.3. AIDATATANG: 5.5 → 4.4.
- relevance (1-5): 3. It is another every-3-layers design, and again the extra gains come from reusing the intermediate predictions.

### InterDecoder — Interdecoder: using Attention Decoders as Intermediate Regularization for CTC-Based Speech Recognition
- cite: Tatsuya Komatsu, Yusuke Fujita, IEEE SLT 2022 (proceedings published 2023), pp. 46–51; DOI 10.1109/SLT54892.2023.10022760. No arXiv version found.
- verified: Crossref record (title, authors, venue, pages, DOI). Details from the IEEE abstract as quoted in search results; no full text.
- family: 4
- layers supervised: intermediate encoder outputs are fed into an attention decoder.
- target: token-level CE given the previous ground-truth tokens (teacher forcing).
- head sharing: unknown; final norm shared: unknown.
- weighting/schedule: unknown.
- purpose: training dynamics (non-autoregressive CTC inference is kept).
- final-layer effect (numbers): up to 6% relative WER improvement over conventional non-autoregressive ASR on LibriSpeech/TEDLIUM2, and further gains combined with SC-CTC.
- relevance (1-5): 2. Details are unverified beyond the abstract.

### Multilingual auxiliary CTC — Improving Massively Multilingual ASR With Auxiliary CTC Objectives
- cite: William Chen, Brian Yan, Jiatong Shi et al., ICASSP 2023; arXiv:2302.12829; link: https://arxiv.org/abs/2302.12829
- verified: arXiv API (comment "accepted at ICASSP 2023"). Details from the full text (arXiv HTML v2).
- family: 4
- layers supervised: intermediate CTC at layer 3 (language-ID target) and layers 6, 9, 12, 15 (transcript), with self-conditioning.
- target: language ID at the lowest supervised layer, then the transcript.
- head sharing: not stated. final norm shared: unknown.
- weighting/schedule: intermediate weight w = 0.5 (Transformer); CTC/attention λ = 0.3; constant.
- purpose: accuracy.
- final-layer effect (numbers): best FLEURS CER 10.1 vs 14.1 for prior work (28.4% relative). The paper also reports LID-token conditioning beating plain SC-CTC by 3.0 MER absolute.
- relevance (1-5): 2.

### LAIL — Boosting CTC-Based ASR Using LLM-Based Intermediate Loss Regularization
- cite: Duygu Altinok, TSD 2025 (Springer LNAI); arXiv:2506.22846; link: https://arxiv.org/abs/2506.22846
- verified: arXiv API (comment: accepted to TSD 2025, Springer LNAI). Details from the full text (arXiv HTML).
- family: 4
- layers supervised: Conformer blocks 6, 12, 18, 24 (4 connector heads; 1–5 heads tested).
- target: a causal LM loss on the transcript, computed by a FROZEN LLaMA-3 (1B/3B/8B) from connector-projected intermediate acoustic features.
- head sharing: separate connectors (×32 downsampling + linear to 4096-d); the frozen LLM is shared across layers. final norm shared: n.a.
- weighting/schedule: L_CTC + α·L_LAIL with α = 0.3 and per-layer λ_l; constant.
- purpose: training (inject linguistic knowledge); no inference cost.
- final-layer effect (numbers): LibriSpeech 1.96/3.98 → 1.74/2.96; TEDLIUM2 7.7 → 6.0; WSJ 5.1 → 3.6. No InterCTC control.
- relevance (1-5): 2. It is an LM loss at several intermediate layers, but through an external frozen LLM, closer to distillation.

### PMS-SSL — Progressive Multi-Scale Self-Supervised Learning for Speech Recognition
- cite: Genshun Wan, Tan Liu, Hang Chen et al., arXiv 2022 (submitted to ICASSP 2023; venue not confirmed); arXiv:2212.03480; link: https://arxiv.org/abs/2212.03480
- verified: arXiv API. Details from the full text (ar5iv).
- family: 4
- layers supervised: masked-prediction losses at layers 6 and 12 during pretraining.
- target: coarse k-means targets (fewer clusters) at the intermediate layer, fine ones at the top ({300, 500}).
- head sharing: layer-specific codeword embeddings. final norm shared: n.a.
- weighting/schedule: unweighted sum; constant.
- purpose: training dynamics.
- final-layer effect (numbers): vs HuBERT, test-other with 10h fine-tuning 9.4 → 8.27 (13.7% relative); 100h 8.1 → 7.19 (12.7% relative); 5.8% relative better than ILS-SSL (10h).
- relevance (1-5): 2. It echoes HC-CTC: coarser targets at shallower layers.

### Inter-layer attention CTC (Hojo et al.) — Boosting CTC-based ASR using inter-layer attention-based CTC loss
- cite: Keigo Hojo, Yukoh Wakabayashi, Kengo Ohta et al., Interspeech 2024, pp. 2860–2864; DOI 10.21437/Interspeech.2024-1776. No arXiv version found.
- verified: ISCA archive page. Details from the abstract.
- family: 4
- layers supervised: every encoder layer. Each layer computes two CTC losses on attention-weighted mixtures of the lower-half and upper-half layer outputs.
- target: the same transcript CTC.
- head sharing: unknown; final norm shared: unknown.
- weighting/schedule: unknown.
- purpose: accuracy.
- final-layer effect (numbers): TEDLIUM2 dev/test 9.9/11.8 WER, better than the conventional intermediate-CTC methods (per the abstract).
- relevance (1-5): 2.

### CTC alignments for translation — CTC Alignments Improve Autoregressive Translation
- cite: Brian Yan, Siddharth Dalmia, Yosuke Higuchi et al., EACL 2023; arXiv:2210.05200; link: https://arxiv.org/abs/2210.05200
- verified: arXiv API + Crossref (DOI 10.18653/v1/2023.eacl-main.119). Details from the full text (ar5iv).
- family: 4 (speech/text translation)
- layers supervised: a hierarchical encoder. Source-language CTC on the intermediate encoder output (after SrcEnc); target-language CTC on the final encoder output; plus the attention decoder CE.
- target: the source transcript at the intermediate layer, the target translation at the final layer.
- head sharing: separate CTC heads. final norm shared: unknown.
- weighting/schedule: fixed weights on the three losses; constant.
- purpose: accuracy.
- final-layer effect (numbers): ablation (Table 3, En-De): no CTC 27.7 BLEU; +SrcCTC 27.8; +TgtCTC 28.1; both 28.3.
- relevance (1-5): 2.

### InterMPL — InterMPL: Momentum Pseudo-Labeling with Intermediate CTC Loss
- cite: Yosuke Higuchi, Tetsuji Ogawa, Tetsunori Kobayashi, Shinji Watanabe, ICASSP 2023; arXiv:2211.00795; link: https://arxiv.org/abs/2211.00795
- verified: arXiv API (comment "Accepted to ICASSP2023"). Details from the abstract.
- family: 4
- layers supervised: intermediate CTC (self-conditioned / hierarchical conditional) inside semi-supervised momentum pseudo-labeling.
- target: pseudo-labels at the intermediate layers.
- head sharing: unknown; final norm shared: unknown.
- weighting/schedule: unknown.
- purpose: accuracy (semi-supervised).
- final-layer effect (numbers): "up to a 12.1% absolute performance gain" over MPL.
- relevance (1-5): 1.

### HuBERT-EE — HuBERT-EE: Early Exiting HuBERT for Efficient Speech Recognition
- cite: Ji Won Yoon, Beom Jun Woo, Nam Soo Kim, Interspeech 2024; arXiv:2204.06328; link: https://arxiv.org/abs/2204.06328
- verified: arXiv API (comment "Accepted by INTERSPEECH 2024"); the ISCA archive entry appears in search results. Details from the full text (ar5iv), Table 2.
- family: 4
- layers supervised: early-exit branches (self-attention-based) at layers 5, 8 and 11 of the 12-layer HuBERT-base during FINE-TUNING.
- target: the same CTC transcript.
- head sharing: separate branch modules. final norm shared: no.
- weighting/schedule: L_FT1 + λ·L_FT2 with λ = 1 for joint training; also a two-stage variant (backbone frozen first).
- purpose: inference speed.
- final-layer effect (numbers): final-layer test-clean WER: HuBERT-base 3.88, joint 3.90, two-stage 3.88. Branches, joint vs two-stage: L5 21.11 vs 37.36; L8 8.60 vs 11.99; L11 4.04 vs 4.21. About 15% latency reduction at an entropy threshold of 0.0035.
- relevance (1-5): 2. In fine-tuning, joint exits barely move the final layer (+0.02) while greatly improving the shallow exits.

### Dynamic Encoder Transducer — Dynamic Encoder Transducer: A Flexible Solution For Trading Off Accuracy For Latency
- cite: Yangyang Shi, Varun Nagaraja, Chunyang Wu et al., Interspeech 2021, pp. 2042–2046; arXiv:2104.02176; link: https://arxiv.org/abs/2104.02176
- verified: arXiv API + ISCA archive (DOI 10.21437/Interspeech.2021-1272). Details from the abstract only (ar5iv conversion failed).
- family: 4
- layers supervised: "collaborative learning jointly trains multiple encoders with different depths in one single model"; the alternative is layer dropout.
- target: the same RNN-T loss at each depth.
- head sharing: unknown; final norm shared: unknown.
- weighting/schedule: unknown.
- purpose: inference (latency/accuracy trade-off).
- final-layer effect (numbers): "the full-size encoder in DET relatively reduces the word error rate of the same size baseline by over 8%" on LibriSpeech. The abstract does not say whether this comes from layer dropout or collaborative learning. The lightweight encoder trained with collaborative learning is 25% smaller with WER similar to the full-size baseline.
- relevance (1-5): 2.

### Splitformer — Splitformer: An improved early-exit architecture for automatic speech recognition on edge devices
- cite: Maxence Lasbordes, Daniele Falavigna, Alessio Brutti, arXiv 2025; arXiv:2506.18035; link: https://arxiv.org/abs/2506.18035
- verified: arXiv API. Details from the full text (arXiv HTML).
- family: 4
- layers supervised: exits every two layers of a 14-layer Conformer, with parallel downsampled layers.
- target: the same CTC transcript.
- head sharing: separate linear+softmax exits. final norm shared: unknown.
- weighting/schedule: unweighted sum over exits, trained from scratch (LibriSpeech 960h, 70 epochs).
- purpose: inference.
- final-layer effect (numbers): at the reported layer-12 exit, 4.8/14.7 vs 5.1/14.8 for the early-exit baseline.
- relevance (1-5): 2.

### Inter-KD — Inter-KD: Intermediate Knowledge Distillation for CTC-Based Automatic Speech Recognition
- cite: Ji Won Yoon, Beom Jun Woo, Sunghwan Ahn et al., IEEE SLT 2022; arXiv:2211.15075; link: https://arxiv.org/abs/2211.15075
- verified: arXiv API (comment "Accepted by 2022 SLT Workshop"). Details from the full text (ar5iv).
- family: 4 (distillation variant)
- layers supervised: intermediate CTC heads at layers 18, 24, 30 of a 33-layer Jasper-Mini student.
- target: ground-truth CTC at every head, plus an L2 distance to the teacher's (Jasper DR) softmax.
- head sharing: separate ("not shared with the original CTC's fully-connected layer"). final norm shared: no.
- weighting/schedule: L_CTC + λ·L_KD with λ = 0.25; constant.
- purpose: training (compression).
- final-layer effect (numbers): test-clean 8.85 → 6.30, vs sequence-level KD 9.10, guided CTC 8.29, SKD 7.81.
- relevance (1-5): 2.

---

## Verified but not scored (tangential; arXiv API checked, details not extracted)
- 2405.17376 — Federating Dynamic Models using Early-Exit Architectures for Automatic Speech Recognition on Heterogeneous Clients (Ali, Brutti, Falavigna, 2024): federated training of early-exit ASR.
- 2106.08595 — Multi-Speaker ASR Combining Non-Autoregressive Conformer CTC and Conditional Speaker Chain (Guo et al., Interspeech 2021): uses intermediate CTC for multi-speaker ASR.
- 2308.08449 — Improving CTC-AED model with integrated-CTC and auxiliary loss regularization (Zhu, Su, Zhang, 2023): the abstract mentions auxiliary-loss regularization; its placement is unclear.
- 2309.12234 — Bridging the Gaps of Both Modality and Language: Synchronous Bilingual CTC for Speech Translation and Speech Recognition (Xu et al., 2023): dual CTC objectives for speech translation.

## Family-level takeaways
- **Help, from scratch.** A same-label loss on intermediate layers of speech transformers trained from scratch consistently improves the final layer.
  - InterCTC, one middle layer, w = 0.3: WSJ eval92 improves by 1.2–2.9 absolute across 12/24/48 layers (16.5→13.6, 13.9→12.4, 13.8→12.6), TED-LIUM2 by 0.6–1.7.
  - Wright et al., equal-weight exits on every other layer: final exit 6.5/17.7 → 5.1/15.1 (Conformer-CTC) and 2.5/6.1 → 2.3/6.0 (AED).
  - HuBERT-ILS, pretraining with an intermediate same-target loss: BASE 6.3/13.2 → 4.7/10.1.
- **Depth rescue, the closest analog of LLAL's "weak signal to lower layers".** Without per-layer losses, deep speech transformers stalled.
  - Wang et al. 2020: 24 layers "not converged" → 2.66/5.64 with iterated loss.
  - Liu et al. 2021: 24-layer and 36-layer RNN-T encoders were "not able to converge" without the auxiliary tasks.
  - Deja-vu: the 36-layer baseline equals the 24-layer one (4.0/9.4) until iterated loss (3.4/8.1).
- **Hurt cases.**
  - Adding per-layer losses while FINE-TUNING an already-trained model hurts the final layer: wav2vec2 3.4/8.6 → 4.3/12.2, WavLM 3.0/6.5 → 3.6/8.8 (2309.09546). HuBERT-EE joint fine-tuning was neutral (3.88 → 3.90).
  - The shallowest exits stay worse than a dedicated shallow model (layer 2: 23.9 vs 17.6).
  - Supervising intermediate layers can strip non-target information (HuBERT-ILS speaker ID 81.42 → 79.29).
  - Hard task-dependent targets at every intermediate layer diverged in OWSM-CTC (1B parameters, 180k h).
  - Implication for a temporary per-layer loss: its early-training use matches the "from scratch" regime, where speech evidence is positive.
- **Precedent for a shared head and shared final norm.**
  - The standard ESPnet InterCTC/SC-CTC implementation sends intermediate outputs through the encoder's final LayerNorm (`after_norm`) and the same CTC linear head, the exact analog of the planned "shared final norm + LM head". The SC-CTC paper says so explicitly (Eq. 6–7 reused).
  - Separate heads also work: Deja-vu MLPs, Wang et al., DeCRED untied, HuBERT-ILS (where sharing was "slightly worse", <3%).
  - No speech paper reports a shared head failing.
  - Liu et al. share the decoder forward but stop the auxiliary gradient into it: a ready-made option if the per-layer loss distorts the shared LM head.
- **Weighting.**
  - The total auxiliary weight is typically constant at 0.3 (InterCTC, Deja-vu, Wang et al., LAIL) to 0.5 (SC-CTC, GIC, multilingual).
  - The ESPnet convention averages over supervised layers, so the auxiliary mass does not grow with the number of supervised layers (L = (1−w)·L_final + w·mean_l L_l).
  - Wright et al. / Splitformer used an unweighted sum over 6–7 exits (the final exit is about 1/6 of the loss) and still improved the final layer from scratch.
  - DeCRED keeps 0.6 on the final decoder layer.
  - No speech paper found anneals or removes the intermediate loss: all use constant weights for the whole run. LLAL-style temporary schedules are untested in this family.
- **Position.** Middle-to-late positions work best.
  - InterCTC: lower (6/24) is worse than middle (12.9 vs 12.4) but still beats the baseline (13.9).
  - ILO shared decoder: best at layer 9/12.
  - DeCRED: best at middle-late decoder layers; early decoder layers give minimal gains, and several auxiliary heads gave no extra gain.
  - With lower-level targets, lower layers are the right place (Krishna i=3/4 of 5 beat the top layer; Toshniwal layer 3 beat the top).
- **One layer vs many layers.**
  - Plain InterCTC gains little from supervising more layers: K=3 at 24 layers gives 12.4 → 12.0, K=7 at 48 layers gives 12.6 → 12.1, and 12 layers show no gain (SC-CTC Fig. 3).
  - Many-layer supervision pays off when the intermediate predictions are fed back into the residual stream: SC-CTC TEDLIUM2 10.1 → 9.4; GIC 7.8 → 7.3.
  - It also pays off when shallow layers get coarser targets than deep ones: HC-CTC 9.1 → 8.4 on LS-100; HMTL/PMS-SSL show the same.
  - For the dense experiment: identical next-token targets at every layer may saturate like plain InterCTC. Lower weights or softer/coarser targets for the first few layers are the speech-supported variants to ablate.

## Could not verify
- "Layer Pruning on Demand with Intermediate CTC" (2106.09216): the citation is verified, but its method details (which layers, head sharing, weights, full-depth WER) could not be read. The ar5iv conversion failed and the PDF was not parseable. A search-engine summary claimed an equal weighting w ≈ 0.66 over two intermediate layers plus the final layer; that claim is unverified.
- InterDecoder (Komatsu & Fujita, SLT 2022): the citation is verified via Crossref, but I found no full text or arXiv version. Its method details come only from the abstract text in search results.
- Dynamic Encoder Transducer (2104.02176): the citation is verified, but its method details come from the abstract only (ar5iv conversion failed).
- Venues of 2207.04177 (submitted to Interspeech 2022), 2112.08778 (submitted to ICASSP 2022) and 2212.03480 (submitted to ICASSP 2023) are given as the arXiv comment states; acceptance is not confirmed.
