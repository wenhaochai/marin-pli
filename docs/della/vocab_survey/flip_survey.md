# Vocabulary size × repetition × regularisation: literature survey (2026-10-23)

Scope: questions (a)–(d) from the vocab-overfitting project (300m Qwen3-style, MuonH, ~28 GB fineweb-edu, 8 passes, Paloma bpb).

Verification key:
- [V] title, authors, date and the quoted claim checked on the arXiv abstract page or the full text (arxiv.org / ar5iv / arxiv html) on 2026-10-23.
- [V, venue mem] same, but the venue is from memory and was not shown on the page.
- [U] not verified; do not cite without checking.

Excluded: wenhaochai.com/exps/vocab-overfitting.html showed up in search. It is our own page, not prior work.

---

## (a) Regulariser effect depends on vocabulary / granularity / data regime, or flips sign

| # | Citation | Claim relevant to us | Status |
|---|---|---|---|
| a1 | Chai & Xiong, "What Is a Repeated Token Worth? The Scaling Geometry of Multi-Epoch Pretraining", arXiv 2610.05591 (Oct 2026) | **Sign flip of a tokenisation-noise regulariser with repetition count.** BPE-dropout re-tokenisation (rate 0.05, GPT-2 BPE, 45M models) "raises loss by 0.021 and 0.010 bpb at R=1 and 4" but "lowers it by 0.022 and 0.154 bpb at R=8 and 16". Also: "Lower-entropy sources tolerate less repetition" (59M, 7 sources, r = −0.89 between unigram entropy and loss rise at 16 epochs). Runs use no weight decay and no dropout. | [V] |
| a2 | Cosma, Ruseti, Radoi, Dascalu, "Training Language Models with homotokens Leads to Delayed Overfitting", arXiv 2601.02867 (Jan 2026) | **Tokenizer-dependent regulariser.** Alternative segmentations ("homotokens") delay overfitting only "when the data is repeated more than 16 times"; below that no effect. "Gains are strongest when canonical tokens are highly compressed"; with GPT-2 (~50K) no multilingual benefit, with aya-23 (~255K) "definite benefits". 88M and 244M models. | [V] |
| a3 | Merity, Keskar, Socher, "Regularizing and Optimizing LSTM Language Models" (AWD-LSTM), arXiv 1708.02182, ICLR 2018 | **Regulariser strength explicitly tied to vocabulary size.** For WikiText-2 vs PTB: "we increase the input dropout to 0.65 to account for the increased vocabulary size" (0.4 on PTB, 10K vocab; WT2 >30K). Embedding (row) dropout 0.1; removing it costs +5.4 test ppl on PTB, +2.3 on WT2. | [V, venue mem] |
| a4 | Sennrich & Zhang, "Revisiting Low-Resource Neural Machine Translation: A Case Study", ACL 2019, arXiv 1905.11901 | Smaller BPE vocabulary (14k→2k) gives +4.9 BLEU at 100k words; aggressive word dropout (p=0.3 for words) gives +3.5 BLEU at 100k words but only +1.6 at 3.2M words. Gain magnitude depends on data size; no sign flip shown. | [V] |
| a5 | Provilkov, Emelianenko, Voita, "BPE-Dropout: Simple and Effective Subword Regularization", ACL 2020, arXiv 1910.13267 | "The effect from using BPE-dropout vanishes when a corpora size gets bigger"; "models trained with subword regularization are less sensitive to the choice of vocabulary size". | [V, venue mem] |
| a6 | Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates", ACL 2018, arXiv 1804.10959 | Gains "especially on low resource and out-of-domain settings". | [V] |
| a7 | Liu, Bauer, Manning, "Drop Dropout on Single-Epoch Language Model Pretraining", ACL Findings 2025, arXiv 2505.24788 | Dropout **hurts** single-epoch pretraining (BERT, Pythia 160M/1.4B). With a1 and c1: dropout sign depends on the repetition regime. | [V] |
| a8 | Gowda & May, "Finding the Optimal Vocabulary Size for Neural Machine Translation", Findings of EMNLP 2020, arXiv 2004.02334 | Vocabulary trades sequence length against class imbalance; imbalance is "especially affecting the recall of rare classes"; heuristic: "use the largest possible BPE vocabulary such that at least 95% of classes have 100 or more examples"; optimal vocab grows with data; 8K "seems to find a good trade-off" for small/medium data. | [V] |
| a9 | Merity, Keskar, Socher, "An Analysis of Neural Language Modeling at Multiple Scales", arXiv 1803.08240 (2018) | Char-level enwik8 model uses near-zero input/hidden dropout (0.01); word-level WT-103 uses 0.1; char PTB uses 0.1–0.25. Confounded by data and model size; the paper makes no causal claim. Weak evidence only. | [V] |
| a10 | Zhang, Sheng, Zhang, Jiang, Han, Deng, Zheng, "Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Prediction Models", CIKM 2022, arXiv 2209.06053 | Learning rate: "With an extremely small learning rate, such a phenomenon is less obvious, but it is at the expense of model performance"; Adam/RMSprop are "more prone to the one-epoch phenomenon"; "the weight decay algorithm neither improves the model performance nor helps alleviate the one-epoch problem"; "dropout does not help solve the one-epoch problem". | [V] |

Not found: any paper that shows label smoothing, weight decay or learning-rate effects changing sign with vocabulary size in LM pretraining. Searched; treat as a gap.

## (b) Memorisation and rare-token / sparse-feature embedding rows

| # | Citation | Claim relevant to us | Status |
|---|---|---|---|
| b1 | Zhang et al. 2022 (a10) | One-epoch overfitting tied to sparse ID features: the trained-vs-untrained embedded distribution shift "is mainly dominated by the sparse feature fields". **But** "the parameter changes of the embedding layer are generally stable", "the variation of the MLP layers suddenly increases at the second epoch", and "freezing the MLP layers after the first epoch alleviates the one-epoch phenomenon". The harm sits in the dense layers that adapt to the sparse rows, not in the rows' movement. | [V] |
| b2 | Liu, Fan, Liang, Kong, Li, "Multi-Epoch Learning for Deep Click-Through Rate Prediction Models", arXiv 2305.19531 (May 2023) | Hypothesis: the phenomenon is due to "the susceptibility of the embedding layer to overfitting"; MEDA "reinitializ[es] the embedding layer in each epoch". Opposite localisation to b1. | [V] |
| b3 | Land & Bartolo, "Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models", EMNLP 2024, arXiv 2405.05417 | Under-trained ("glitch") tokens exist across many LLM tokenizers; detected from weight indicators. Rows of rare tokens can stay close to initialisation. | [V] |
| b4 | Gao, He, Tan, Qin, Wang, Liu, "Representation Degeneration Problem in Training Natural Language Generation Models", ICLR 2019, arXiv 1907.12009 | Tied/output embeddings: "the rare words behave similarly to a non-appeared word"; "the embedding of the word will be pushed towards the negative directions of most hidden states"; low-frequency words "are clustered together in a local region". Output-side mechanism. | [V] |
| b5 | Yu, Song, Kim, Lee, Ryu, Yoon, "Rare Tokens Degenerate All Tokens: Improving Neural Text Generation via Adaptive Gradient Gating for Rare Token Embeddings", ACL 2022, arXiv 2109.03127 | A specific part of the gradient on rare-token embeddings drives degeneration of all tokens; gating it (AGG) helps LM and MT. | [V] |
| b6 | Gong, He, Tan, Qin, Wang, Liu, "FRAGE: Frequency-Agnostic Word Representation", NeurIPS 2018, arXiv 1809.06858 | "The embeddings of high-frequency and low-frequency words lie in different subregions of the embedding space"; this hurts rare words most. | [V] |
| b7 | Xie, Wang, Li, Lévy, Nie, Jurafsky, Ng, "Data Noising as Smoothing in Neural Network Language Models", ICLR 2017, arXiv 1703.02573 | Unigram and blank noising of context tokens as a smoothing analogue; adaptive rate γ_AD noises contexts with fewer distinct continuations more ("commonly seen bigrams are less likely to be noised"); Kneser-Ney proposal favours tokens with diverse histories. Word-level only (10K–50K vocab); vocab size not isolated. | [V, venue mem] |
| b8 | Gal & Ghahramani, "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks", NeurIPS 2016, arXiv 1512.05287 | Word-type embedding dropout: "Since the embedding matrix is optimised it can lead to overfitting"; the embedding is "the largest layer in most language applications, yet it is often not regularised". | [V, venue mem] |
| b9 | Merity et al. 2017 (a3) | Embedding dropout = dropping whole rows ("all occurrences of a specific word will disappear within that pass"). | [V] |
| b10 | Provilkov et al. 2020 (a5) | With standard BPE "rare tokens are in general separated from the rest" in embedding space; BPE-dropout embeddings have more meaningful neighbours. | [V] |
| b11 | Wang, Pham, Dai, Neubig, "SwitchOut: an Efficient Data Augmentation Algorithm for Neural Machine Translation", EMNLP 2018, arXiv 1808.07512 | Replaces words with "other random words from their corresponding vocabularies" (our uniform-token noise is this family); comparable or better than word dropout. Replacement distribution not stated on the abstract page. | [V] |
| b12 | Stoehr, Gordon, Zhang, Lewis, "Localizing Paragraph Memorization in Language Models", arXiv 2403.19851 (Mar 2024) | Gradients of memorised paragraphs are larger in lower layers; a low-layer attention head "focuses mainly on rare tokens"; "a few distinctive tokens early in a prefix can corrupt the entire continuation". Rare tokens act as keys for memorised text. | [V]; venue not checked |
| b13 | Tirumala, Markosyan, Zettlemoyer, Aghajanyan, "Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models", arXiv 2205.10770 (NeurIPS 2022) | "Models memorize nouns and numbers first", hypothesised as unique identifiers of examples; larger models memorise faster. | [V, venue mem] |
| b14 | Chung & Kim, "Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training", NeurIPS 2025, arXiv 2508.15390 | Vocab 24K→196K at fixed data/compute: gains come from the ~2,500 most frequent words "even though loss on the rare tail rises"; above 24K a larger vocab mainly widens frequency imbalance. Single-epoch. | [V] |
| b15 | Kexue.fm 6869, 苏剑林, 《Keras实现两个优化器：Lookahead和LazyOptimizer》 2019-07-30 | Any momentum optimiser keeps updating rows of tokens absent from the batch, which "可能导致Embedding层过拟合"; Lazy update raises IMDB val acc 83.7% → 84.9%. Blog. | [V] (local DB) |
| b16 | Kexue.fm 9736, 苏剑林, 《Lion/Tiger优化器训练下的Embedding异常和对策》 2023-08-28 | Sign-based optimisers drive rows of low-frequency tokens to ±1/λ (±100) because sign(m) stays constant after the token disappears; Adam's update decays. Fixes: lazy row updates, tied embeddings, or Adam for embeddings. Blog. | [V] (local DB) |
| b17 | Kexue.fm 11647 / 11416, 苏剑林, 《MuP之上：3》2026-03-02; 《Muon优化器指南》2025-11-19 | Embedding steepest descent is row-wise normalised SGD, not Muon; embedding and LM head should not use Muon. Relevant to how MuonH treats embedding rows. Blog. | [V] (local DB) |
| b18 | Wang, Zhang, et al., "Muon Outperforms Adam in Tail-End Associative Memory Learning", ICLR 2026, arXiv 2509.26030 | Muon learns tail classes in heavy-tailed data more evenly than Adam, whose errors depend on the feature embeddings; gain comes from VO and FFN weights. | [V] (search result + ICLR page listing; abstract not fetched) |
| b19 | Rumbelow & Watkins, "SolidGoldMagikarp" (LessWrong, 2023) | Origin of glitch tokens. | [U] |

## (c) Multi-epoch pretraining, regularisation, and vocabulary

| # | Citation | Claim relevant to us | Status |
|---|---|---|---|
| c1 | Xue, Fu, Zhou, Zheng, You, "To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis", NeurIPS 2023, arXiv 2305.13230 | "Using dropout alone is highly effective in alleviating the multi-epoch degradation"; 0.2–0.3 best for XL; "gradually integrating dropout during training can achieve comparable performance to using dropout from the beginning"; weight decay "can even make the training process unstable"; label smoothing only "slight improvement"; "larger models are more susceptible to overfitting". T5 / C4; vocab not varied. | [V] |
| c2 | Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel, "Scaling Data-Constrained Language Models", NeurIPS 2023, arXiv 2305.16264 | "Up to 4 epochs of repeated data yields negligible changes to loss"; returns then decay to zero. Dropout and weight decay fixed at 0.1, not studied; "returns from additional epochs may heavily depend on hyperparameters such as learning rate, dropout, or the optimizer choice". GPT-2 tokenizer only. | [V, venue mem] |
| c3 | Hernandez et al., "Scaling Laws and Interpretability of Learning from Repeated Data", arXiv 2205.10487 (2022) | Repeating 0.1% of data 100× makes an 800M model perform like 400M; repetition harms copying and induction heads. | [V] |
| c4 | Kim, Kotha, Liang, Hashimoto, "Pre-training under infinite compute", arXiv 2509.14786 (Sep 2025) | With fixed data, "the optimal weight decay is 30× larger than standard practice"; ensembling reaches a lower asymptote than regularisation alone (relevant to our weight-EMA result). | [V] |
| c5 | Lovelace, Belardi, Kundurthy, Sudhakar, Weinberger, "Prescriptive Scaling Laws for Data Constrained Training", arXiv 2605.01640 (May 2026) | One overfitting coefficient for repetition; weight decay λ=1.0 cuts it by ~70%. | [V] |
| c6 | Xu, Wu, Cho, Hu, Wang, "Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws", arXiv 2606.06888 (Jun 2026) | Masked-input regularisation (extra NTP loss on randomly masked inputs) on top of strong weight decay lowers val loss at 72M–1.4B; worth ~1.3× unique data. | [V] |
| c7 | Chen, Zhang, Bai, Hu, Wang, "Demystifying Training-Time Augmentation for Data-Constrained Language Model Pretraining", arXiv 2606.16246 (Jun 2026) | Token-level noise, permutations, offset targets all delay overfitting; **uniform random token replacement** ("sampled uniformly from the non-special vocabulary"; best rate 15%) gives the lowest minimum loss of single methods. Only Qwen2 tokenizer (151,646 vocab), one ~150M model, 100 epochs, from the start; vocab and dropout not varied. Closest analogue to our 128K arm. | [V] |
| c8 | Ni et al., "Diffusion Language Models are Super Data Learners", arXiv 2511.03276 (Nov 2025) | "Input or parameter noise improves AR under data constraint but cannot close the gap" to diffusion LMs. | [V] |
| c9 | Prabhudesai, Wu, Zadeh, Fragkiadaki, Pathak, "Diffusion Beats Autoregressive in Data-Constrained Settings", arXiv 2507.15857 (Jul 2025) | Masking acts as "an implicit data augmentation that AR's fixed left-to-right factorization lacks". | [V] |
| c10 | Yan, Wen, Li, Luo, Chen, Lyu, "Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression", arXiv 2511.13421 (Nov 2025) | Effective reuse rate saturates at a value that grows with dataset size (Zipf data included); challenges the flat "4 epochs is free" reading. | [V] |
| c11 | Tao, Liu, Dou, Muennighoff, Wan, Luo, Lin, Wong, "Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies", NeurIPS 2024, arXiv 2407.13623 | Compute-optimal vocab grows with model size (Llama2-70B should have ≥216K). Single-epoch; says nothing on repetition or rare-token overfitting. | [V] |
| c12 | Huang, Zhu, Wu, Zeng, Wang, Min, Zhou, "Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling", ICML 2025, arXiv 2501.16975 | Decouples input and output vocabularies; log-linear gain from larger input (n-gram) vocab. Single-epoch. A tool to separate "many sparse input rows" from "short sequences". | [V] |
| c13 | Kexue.fm 11833, 苏剑林, 《解构Scaling Law：优化、架构、数据的三重奏》 2026-07-29 | Multi-epoch penalty term α(K/D)^γ; cites c5 and c10; notes the predicted optimal-epoch trend conflicts with c10. Blog. | [V] (local DB) |

No paper found that varies vocabulary size under multi-epoch repetition (as of 2026-10-23). Closest: a1 (re-tokenisation × epochs), a2 (tokenizer × repetition), b14 (vocab × rare-token loss, single epoch).

## (d) Byte / character models and memorisation

| # | Citation | Claim relevant to us | Status |
|---|---|---|---|
| d1 | Kharitonov, Baroni, Hupkes, "How BPE Affects Memorization in Transformers", arXiv 2110.02782 (2021) | Larger BPE vocabularies "greatly facilitat[e] memorization" (random-label fitting, membership inference, training-text reproduction); "the effect persists when we ensure that the number of trained parameters remains constant"; conjectured cause: "reduction of the sequence length is the primary factor". | [V]; arXiv only as far as found |
| d2 | Xue, Barua, Constant, Al-Rfou, Narang, Kale, Roberts, Raffel, "ByT5: Towards a token-free future with pre-trained byte-to-byte models", TACL 2022, arXiv 2105.13626 | "Byte-level models are significantly more robust to noise." Input-noise robustness, not memorisation. | [V] |
| d3 | Gigant, Peng, Quesnelle, "Decoupling the Benefits of Subword Tokenization for Language Model Training via Byte-level Simulation", arXiv 2604.27263 (2026) | Subword benefit traced to throughput and boundary priors; abstract says nothing on memorisation. | [V], not relevant on memorisation |

No study found that measures whether memorisation in byte or character models is less harmful to held-out loss, or where it is stored. Treat (d) as open beyond d1.

## Kexue.fm search log

Queries run on the local SQLite (`kexue.py search`): 词表 过拟合; 词表 大小; tokenizer 词表; dropout 过拟合; 重复 epoch; 低频词 embedding (0 hits); Embedding 稀疏; Embedding层; 多epoch (0 hits); Muon Embedding; 词表 Embedding 学习率; Dropout 预训练. Relevant posts: 6869, 9736, 11647, 11416, 11833 (above); 9762 and 9768 (large-vocab continuation problem; Subword Regularization as a fix, "甚至能增强泛化能力"); 7758 (word-level models have more embedding parameters and overfit more without pretraining).

---

## Synthesis: what prior work says about hypothesis 3

Hypothesis 3: with a large vocabulary, memorisation is stored in rows of rare tokens; these rows mislead on new text; input noise feeds those rows random contexts; a small vocabulary has no such rows and pays only the corruption cost.

Supported:
1. Rare tokens are memorisation keys: b12 (low-layer head on rare tokens; distinctive prefix tokens drive memorised continuations), b13 (nouns and numbers first, as unique identifiers).
2. Larger vocab raises rare-tail loss and frequency imbalance even in one epoch: b14, a8. Repetition multiplies exposure of the same few contexts for those rows.
3. Rare-token rows sit apart and train differently: b3, b4, b5, b6, b10. Noise on segmentation (a5) pulls them toward the rest.
4. Optimiser-driven over-update of rare rows: b15, b16. This depends on whether MuonH updates embedding rows with momentum (and with sign/normalised steps). Check the MuonH embedding rule.
5. Tokenizer-dependent regulariser value: a2 (homotokens help only with a highly compressing tokenizer and at R>16), a3 (input dropout raised "to account for the increased vocabulary size").
6. Uniform random replacement works at a 151K vocab under heavy repetition: c7. Matches our 128K result. No paper tests it at a small vocab.

Contradicted or complicated:
1. Alternative mechanism, sequence length (d1): larger vocab memorises more because sequences are shorter, with parameters held fixed. Our observation 1 (train-loss drop only 0.043 vs 0.023 bpb) says large vocabs do not memorise much more, so d1's "more memorisation" does not explain our 3.5× eval gap; but shorter sequences could still make memorised text more harmful. Test: an over-tokenized input (c12) or n-gram-hash input rows on the 8K model gives many sparse rows at 8K sequence length.
2. Localisation (b1 vs b2): in CTR models the embedding rows barely move in epoch 2; the MLP adapts to the trained rows and freezing the MLP helps. b2 claims the opposite and fixes it by re-initialising embeddings. Test both: freeze embeddings after pass 1 vs freeze the body after pass 1, at 128K.
3. Sign flips need no vocab-specific mechanism: a1 shows a tokenisation-noise regulariser hurts at 1–4 epochs and helps at 8–16 epochs; a7 vs c1 show the same for dropout. A simpler account of our observation 2: a regulariser helps only when the repetition cost exceeds its bias cost; at 8K the 8-pass cost is only 0.023 bpb, so the corruption cost wins. Hypothesis 3 then explains only why the repetition cost is larger at 128K. Test: give 8K a larger repetition cost (16 or 32 passes) and see whether the same noise turns helpful.
4. a1 finds lower unigram entropy predicts more damage from repetition (across sources, fixed tokenizer). Across tokenizers our result goes the other way: bytes (lowest unigram entropy) lose least. So per-token unigram entropy is not the predictor across tokenizers.
5. b1 and c1: weight decay did not help in CTR and destabilised T5 multi-epoch training; c4 and c5 find very large weight decay helps LMs. Weight-decay strength by vocab is untested by anyone, including us.
6. Learning rate: b1/a10 find fast-converging optimisers and larger LR worsen one-epoch overfitting. Our 2× LR from pass 2 helps 8K; the 128K arm is untested; prior work predicts it helps 128K less or hurts.

Gaps (no prior work found): vocab size × number of passes in LM pretraining; label smoothing or weight decay with sign depending on vocab; harm of memorisation in byte models.
