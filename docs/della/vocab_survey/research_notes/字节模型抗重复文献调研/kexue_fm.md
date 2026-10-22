# kexue.fm (苏剑林, 科学空间) on tokenizers, embeddings, rare tokens, optimizers and data repetition

Source: the local SQLite of all of kexue.fm (kexue-fm plugin 0.1.4), searched 2026-10-22 with Chinese and English keywords (分词, 词表, BytePiece, 字节, Byte, Embedding, 词向量, 低频, 稀疏, LazyOptimizer, Tied, 重复, Epoch, 过拟合, 记忆, 遗忘, 权重衰减, 滑动平均, Muon, 超球面, Hyperball, Scaling Law, Zipf, Over-Encoding, Engram). 23 articles were read in full or in their relevant sections. All quotes are verbatim Chinese from the article text; the English gloss follows.

Overall finding: kexue.fm has no article on byte-level language models, and none compares multi-epoch overfitting across vocabulary sizes. It does contain the pieces of a mechanism:
1. Embedding rows of rare tokens get sparse gradients, and optimizers with state "over-optimize" them (6869, 9736).
2. Larger vocabularies mean more embedding parameters and sparser per-token data, so they overfit more easily (7758).
3. The steepest-descent geometry of an embedding table is row-wise normalized SGD. Each row therefore takes a full-size step every time its token appears, whatever the token frequency (11647, 10592).
4. Weight decay and learning rate set an EMA memory window of about 1/(λη) steps. For multi-epoch training, Su recommends sizing it to one epoch (11459).
5. A data-side scaling term penalizes epochs, (K/D)^γ, and embedding or memory tables count as sparse "memory" parameters (11833).

---

## Q1. Tokenizers, vocabulary size, byte-level models, embedding tables, weight tying

### Takeaway
Su's own work treats bytes as the right base unit: 256 symbols, no OOV, and more uniform information per unit. He states that larger vocabularies, or coarser units such as words, add embedding parameters and make each token's data sparser. In his account this makes them overfit more easily, train rare units less well, and cut the character-level links between tokens. He has no article on pure byte-level LMs or on how granularity interacts with repeated data.

### Cited Findings
- **BytePiece uses bytes as the base unit because there are only 256 of them:** 「因为不同的单Byte只有256个，所以只要词表里包含了这256个单Byte，那么就可以杜绝OOV」 ("there are only 256 distinct bytes, so with all 256 in the vocabulary, OOV is eliminated"). — [kexue.fm/archives/9752](https://kexue.fm/archives/9752)
- **Bytes carry a more uniform amount of information per unit than characters:** 「每个Byte的信息熵则更加均匀【比如，大部分汉字的UTF-8编码对应3个Byte，而汉字的平均信息熵正好是英文字母（对应一个Byte）的2～3倍左右】，因此用Byte的统计结果会更加无偏」 ("each byte's entropy is more uniform... so byte-level statistics are less biased"). — [9752](https://kexue.fm/archives/9752)
- **Compression rates (bytes/token) Su measured for 50K–100K vocabularies:**
  - 50K vocab on WuDao: SP-BPE 4.80, SP-Unigram 4.73, BytePiece 5.05.
  - 100K vocab on a mixed Chinese–English corpus (same-source test): SP-BPE 4.52, BytePiece 5.39. Out-of-distribution: 4.46 and 4.51.
  - These ratios are the same order as the 4.74× step ratio in our bytes-vs-BPE setup. — [9752](https://kexue.fm/archives/9752)
- **Character vs word granularity:** Su lists the benefits of character units as 「1、参数更少，不容易过拟合；2、不依赖于分词算法…；3、没那么严重的稀疏性，基本上不会出现未登录词」 ("fewer parameters, so less prone to overfitting; ...; less severe sparsity"). — [kexue.fm/archives/7758](https://kexue.fm/archives/7758)
- **Su's reading of the "characters beat words" result (Shannon.AI, ACL 2019):** 「以词为单位的模型Embedding层参数更多，自然就更容易过拟合，效果容易变差，这不用做实验都能猜个大概」 ("word-level models have more embedding parameters, so naturally they overfit more easily — one can guess this without experiments"). He argues that more thorough pretraining mitigates this overfitting. — [7758](https://kexue.fm/archives/7758)
- **WoBERT (vocab 33,586, words added to the character vocab) lost about 3% on People's Daily NER. Su traced the loss to sparsity, not segmentation errors:** 「掉点的原因并不是因为切分错误，而是因为稀疏性（平均来说每个词的样本更少，所以训练得没那么充分）」 ("the drop is from sparsity — on average each word has fewer samples, so it is trained less thoroughly"). — [7758](https://kexue.fm/archives/7758)
- **A large vocabulary cuts character-level links between tokens and may hurt generalization:** 「增大词表的缺点也很明显，最直接的就是会割裂token与token之间在字符层面之间的联系，从而可能会影响泛化」 ("the most direct downside of a larger vocab is that it severs the character-level connection between tokens, which may hurt generalization"). Example: the model cannot know that 太阳能 is 太+阳+能. — [kexue.fm/archives/9762](https://kexue.fm/archives/9762)
- **Su on why large vocabularies are attractive:** 「增大词表只需要增大Embedding层和输出的Dense层，这部分增加的计算量几乎不可感知」 ("a bigger vocab only enlarges the embedding and output dense layers; the added compute is almost imperceptible"). He adds that 「也不能无节制地增加词表大小」 ("one cannot grow the vocabulary without limit"). — [9762](https://kexue.fm/archives/9762)
- **Stochastic segmentation (Subword Regularization / BPE-dropout / his Viterbi Sampling) "may even strengthen generalization":**
  - 「也可以考虑在训练阶段就使用带有随机性的tokenize算法…此时分词结果不至于严重影响后续效果，甚至能增强泛化能力」 — [9762](https://kexue.fm/archives/9762)
  - 「看起来这个技巧确实能缓解前述问题，甚至有助于增强语言模型的容错能力」 — [kexue.fm/archives/9768](https://kexue.fm/archives/9768); a corrected exact sampler is in [kexue.fm/archives/9811](https://kexue.fm/archives/9811)
- **Embedding updates are sparse:** 「Embedding层的更新是稀疏的，一个个放大太麻烦」 ("embedding-layer updates are sparse; scaling them up one by one is cumbersome"). — [kexue.fm/archives/8747](https://kexue.fm/archives/8747)
- **Weight tying:**
  - Naive tying makes the initial loss about dσ instead of log n.
  - Fixes are: untie; add an orthogonal projection; or swap the two halves of the feature channels before the output product, 𝒮[w]=w[d/2:]∘w[:d/2].
  - Su reports that all fixes except the reduced-init one converge at similar speed and give similar final results. — [kexue.fm/archives/9698](https://kexue.fm/archives/9698)
- **Tying makes the embedding gradient dense, which removes the rare-row pathology:**
  - 「这样由于输出端重用了整个Embedding矩阵，因此整个Embedding矩阵都有非零梯度，从而让m_t不至于长期为常量」 ("because the output reuses the whole embedding matrix, every row has a nonzero gradient").
  - Su's result: 「使用将模型特征的channels对半交换的Tied Embedding，能解决以上问题，并且效果似乎比Untied Embedding还要好一点」 ("half-channel-swapped tied embedding solved the problem and seemed slightly better than untied"). — [kexue.fm/archives/9736](https://kexue.fm/archives/9736)
- **Embedding and memory tables as sparse "memory" parameters:**
  - Su groups PKM, UltraMem, Over-Encoding (2501.16975) and Engram (2601.07372) as sparse memory layers: 「它们的“Expert”简化为没有任何计算的可训练向量」 ("their 'experts' reduce to trainable vectors with no compute").
  - He notes the open question of whether N should count embedding parameters: 「参数量N的统计要不要包含Embedding，主流做法是不计入，但这在小规模时可能会有较大偏差」 ("mainstream practice excludes them, but at small scale this can cause large deviations"). — [kexue.fm/archives/11833](https://kexue.fm/archives/11833)
- **Zipf frequency and learnability (quantization model):**
  - If learning a quantum needs at least τ samples, quanta with Dp_k < τ cannot be learned: 「我们假设学习一个量子至少需要τ个样本，那么Dp_k < τ的量子都无法学到」. This gives L ~ D^{-γ/(γ+1)}.
  - In a later article Su calls such derivations a swap of one assumption for another. — [kexue.fm/archives/9607](https://kexue.fm/archives/9607); critique in [11833](https://kexue.fm/archives/11833)

### Inferences
- 7758 states a direct, if informal, prior for our result: a coarser vocabulary means more embedding parameters and fewer samples per unit, so it overfits more easily, and character units have the opposite properties. It matches our ordering (128K 0.082 > 8K 0.026 > bytes ~0) and the finding that the cost sits in rare tokens' input rows. Su's evidence is anecdotal (one NER bad-case analysis, BERT-era), not a controlled multi-epoch study.
- Under the 9607 / Zipf view, a 128K vocabulary pushes many token types into the regime where Dp_k is small. Each such row is fit from a handful of contexts, and those same contexts recur on every pass. With bytes every row has a huge Dp_k, so no row is fit from a few contexts.
- Stochastic segmentation (9762, 9768, 9811) is a remedy that kexue.fm points to. Re-tokenizing each pass differently would change which rare-row contexts repeat. It could be an ablation: BPE-dropout on the 128K model under 8 passes.
- Tying (9736) is a second cheap ablation: it makes every row's gradient dense on every step. Su found the half-swap variant slightly better than untied in his own runs.

### Gaps
- No kexue.fm article on byte-level LMs (ByT5, MambaByte, BLT, etc.) or on character-level LMs for LLM pretraining. The search for "Byte Latent", "BLT", "字符级" returned nothing relevant.
- No kexue.fm article measures overfitting or repetition cost as a function of vocabulary size.
- WoBERT's sparsity claim comes from a bad-case analysis on one NER task. Su reports no numbers for it.

---

## Q2. Repeated data, multi-epoch training, memorization, and how long a weight "remembers"

### Takeaway
Su models multi-epoch cost as a separate power-law penalty α10(K/D)^γ10 on top of the optimization term. Training then has an optimal number of epochs. He also models the weights as an EMA of past updates whose memory window is about 1/(λη) steps, and he recommends sizing it to one epoch's steps when training is multi-epoch. The two most direct pieces on embeddings say the same thing:
- Momentum keeps updating a rare token's row on steps where the token is absent, which Su calls Embedding over-optimization or overfitting.
- Under Adam this decays exponentially, so the effect is mild. Under sign-based optimizers it persists.

### Cited Findings
- **Data-side scaling term with explicit multi-epoch penalty:** 「我们认为Multi-Epoch越严重，泛化性能越差，所以贡献一项α10 (K/D)^γ10」. This gives F_data ~ α9 D^{-γ9} + α10 (K/D)^{γ10}, where K = samples seen and D = unique data. — [11833](https://kexue.fm/archives/11833)
- **Optimal epochs:** combining the two terms gives K*/D ~ D^{-γ̂1/(γ̂1+γ10)}: 「数据越少，反而应该训更多轮；数据越多，最优Epoch数越小」 ("the less data, the more epochs one should train"). Su notes that this contradicts 2511.13421. He proposes generalizing the penalty to α10 K^{γ10} D^{-γ11} so the sign can flip. — [11833](https://kexue.fm/archives/11833)
- **Su's report on 2605.01640 (Prescriptive Scaling Laws for Data Constrained Training):** the paper makes the penalty grow with model size: 「模型参数量越大，越容易过拟合，所以越不应该Multi-Epoch训练，因此让α10跟N正相关」 ("the larger the model, the easier it overfits... so α10 is made positively correlated with N"). — [11833](https://kexue.fm/archives/11833)
- **Su prefers penalizing repetition directly to "effective data" discounts:** he discusses Muennighoff et al. 2305.16264 and writes 「我们已经区分了K和D，只需要直接对Multi-Epoch带来的过拟合风险进行惩罚」. — [11833](https://kexue.fm/archives/11833)
- **EMA view of weight decay:** θ_t = (1−λη)θ_{t−1} + λη(−u_t/λ). Unrolled, θ_t = β3^t θ_0 + (1−β3^t)(−ū_t/λ), with β3 = 1−λη. — [kexue.fm/archives/11459](https://kexue.fm/archives/11459)
- **Memory window:** 「模型顶多能记住O(1/λη)步的数据，这是它的记忆周期」 ("the model can remember at most O(1/λη) steps of data; this is its memory period"), from β3^s ≥ c ⇒ s ≤ −log c/(λη). — [11459](https://kexue.fm/archives/11459)
- **Multi-epoch prescription:** 「比较适合的设置是让1/λη正比于训练步数，如果是Multi-Epoch的训练场景，则考虑让1/λη正比于单个Epoch的训练步数」 ("make 1/λη proportional to the training steps; in multi-epoch training, proportional to the steps of a single epoch"). — [11459](https://kexue.fm/archives/11459)
- **Weight decay also erases the initialization and bounds weight RMS:** 「Weight Decay还有一个作用是帮助模型忘掉初始化」 ("another role of WD is to help the model forget its initialization"). Without WD, AdamW's weight RMS grows like η√t. — [11459](https://kexue.fm/archives/11459); [kexue.fm/archives/11307](https://kexue.fm/archives/11307)
- **Asymptotic weight RMS:** ‖θ‖_RMS ≈ √(η/2λ) for AdamW. Su says the same √(η/λ) scaling holds for the homogeneous adaptive family (RMSProp, Adam, SignSGDM, Muon). — [11307](https://kexue.fm/archives/11307); [11459](https://kexue.fm/archives/11459)
- **Optimal schedule for equal weighting of every batch:** with constant λ, η_s ≈ η_max/(λη_max s + 1). With λ_s ∝ η_s, η_s ≈ η_max/√(2λ_max η_max s + 1). Each batch's weight in the final weights is ∝ its learning rate: 「每一Batch的权重…正比于Learning Rate」. — [11459](https://kexue.fm/archives/11459)
- **Su's caveat on the "each batch brings equal information" assumption:** 「学得越多，覆盖面越广，后来Batch的独特信息就越少」 ("the more is learned, the less unique information later batches carry"). He suggests splitting knowledge into 规律 (regularities) and 事实 (facts) and applying the assumption to facts: 「事实型知识…只能靠记忆」 ("factual knowledge can only be memorized"). — [11459](https://kexue.fm/archives/11459)
- **LazyOptimizer and Embedding overfitting:** 「当前batch中没被采样到的词，依然会使用历史动量来更新，这可能导致Embedding层过拟合…这样一来就算没有被反复采样的词，对应的Embedding也被反复更新了，就导致了过拟合」 ("words not sampled in the current batch are still updated by their historical momentum; this can make the embedding layer overfit — even words that are not repeatedly sampled get repeatedly updated").
  - On IMDB, LazyOptimizer(Adam) raised best validation accuracy from about 83.7% to above 84.9%.
  - Su's advice: 「Embedding层很大的模型（尤其是以词为单位的模型）都可以试一下…因为Embedding层参数量太大了，减少更新频率，让模型重点去优化其余部分」 ("models with large embedding layers, especially word-level ones, can try it — reduce the update frequency and let the model focus on optimizing the rest"). — [kexue.fm/archives/6869](https://kexue.fm/archives/6869); restated in [kexue.fm/archives/7094](https://kexue.fm/archives/7094) as 「Embedding层的更新应当稀疏化一些，这有助于防止过拟合」
- **Rare-token over-update and why Adam is milder than sign optimizers:**
  - 「这个问题会出现在所有带动量的优化器中，包括Adam和Tiger，不过在Adam中，这可能不会有明显感知，因为Adam的更新量跟动量成正比，如果一个token长期不重复出现，那么动量就会指数下降」 ("it happens in all momentum optimizers including Adam, but in Adam it may go unnoticed because the update is proportional to momentum, which decays exponentially if the token does not recur").
  - Under Tiger/Lion, sign(m_t) stays ±1. Rare-token rows reach the fixed point θ* = −sign(m)/λ = ±100 for λ = 0.01.
  - Feedback loop: 「一个token的Embedding由于过度更新偏向了某个方向之后，它的梯度可能会适应并助长这种变化」 ("once a token's embedding has drifted in one direction from over-updating, its gradient may adapt to and reinforce that drift").
  - Observed facts: 「这些token看上去都是相当低频的token」; 「除Embedding外，其他权重没有这个问题」; 「模型的总体表现（比如训练Loss、生成测试）都正常」 ("these tokens all appear to be quite low-frequency; no weights other than the embedding have the problem; overall training loss and generation looked normal"). — [9736](https://kexue.fm/archives/9736)
- **Fixes Su lists for rare-token rows:** (1) Lazy updates, i.e. update only rows whose gradient is nonzero. (2) Tied embeddings, which make the gradient dense. (3) The Lion authors' fix: 「混合优化器，比如Embedding层就用Adam，其他层才用Lion/Tiger」 ("a mixed optimizer: Adam for the embedding, Lion/Tiger elsewhere"). — [9736](https://kexue.fm/archives/9736)
- **Why the loss can miss rare-row damage:** 「有些能力在Loss体现不出来…这种能力依赖于低频Token的Embedding质量」 ("some abilities don't show up in the loss... they depend on the quality of low-frequency token embeddings"). — [9736](https://kexue.fm/archives/9736)

### Inferences
- **Confound to control:** 11459 measures memory in optimizer steps (≈1/λη). Bytes need 4.74× more steps for the same text. At the same per-step λ and η, the byte model's memory window therefore covers about 1/4.74 as much text as the 128K model's. The byte model may forget earlier passes more, independent of any tokenizer effect. To separate the two, match 1/(λη) in units of text, or use 11459's rule of 1/λη ∝ steps per epoch for each tokenizer. This applies to every Adam/AdamW parameter, including the input embedding; MuonH's norm-preserving matrices are covered in Q3.
- **Our Adam input embedding is the 6869/9736 setting:** sparse-gradient rows with momentum. 9736 says Adam's over-update decays quickly, so momentum leakage alone probably does not explain the cost. A second Adam effect is not discussed on kexue.fm and is my own inference: a rare row's second moment v decays between appearances. When the token reappears, m/√v ≈ (1−β1)/√(1−β2) is larger than a frequent row's typical update (≈3.2 for β1=0.9, β2=0.999). Rare rows thus take large, near-normalized steps on exactly the few contexts that repeat each pass. This should be checked by logging per-row update RMS against token frequency.
- 11833 offers a clean way to write our result: α10 (the repetition penalty) depends on the tokenizer. In Su's framing the tokenizer changes the coefficient, not the exponent: 「改变的应当是幂律前面的系数，而不是指数」. Fitting α10 per vocab from our 1/2/4/8-pass points would test whether γ10 is shared across vocabularies.

### Gaps
- No kexue.fm article measures memorization of repeated text, or links multi-epoch overfitting to specific parameter groups.
- 11459 analyses AdamW-type EMA. It does not compute the memory window for parameters kept on a fixed-norm sphere (MuonH-like). See Q3 for a derivation from 11196's geometry, which is my inference.
- The 6869 IMDB result is the only number on kexue.fm for embedding overfitting under momentum, and it comes from a small LSTM classifier.

---

## Q3. Muon/MuonH and embeddings, learning rate and steps

### Takeaway
Su derives that the embedding table should not use Muon. Its natural norm is the maximum row RMS, so its steepest descent is row-wise normalized SGD, and every row whose gradient is nonzero takes an update of the same size. In practice Adam is used for the embedding and LM head. Norm-fixing schemes such as Hyperball (the family MuonH belongs to) appear only briefly. Su treats them as a way to bound weights that alters the training dynamics, and he prefers bounding the norm (Pre-Decay, i.e. WD, or clipping) to fixing it.

### Cited Findings
- **Embedding and LM head must not use Muon:** 「Muon是只管“稠密输入的线性层的矩阵参数”…只需要记住Embedding层和最后的分类层（包括GPT的LM Head）的矩阵参数都不能用Muon，否则效果会明显差」 ("Muon is only for matrix parameters of linear layers with dense inputs... embedding and LM-head matrices must not use Muon, otherwise results are clearly worse"). Default: 「直接用Adam就行」. — [kexue.fm/archives/11416](https://kexue.fm/archives/11416)
- **Reason, in the Muon framework:** 「输入和输出的Embedding虽然也是矩阵，但它们使用上是稀疏的，所以更合理的方式也是将它们当成多个向量独立处理」 ("input and output embeddings are matrices, but they are used sparsely, so it is more reasonable to treat them as many independent vectors"). Under msign, each (1,d) row gives Normalized SGDM. — [kexue.fm/archives/10592](https://kexue.fm/archives/10592); [11416](https://kexue.fm/archives/11416)
- **Formal derivation (MuP之上 3):**
  - For f(i;E) = E_i, the update-stability criterion is max_i ‖ΔE_i‖_RMS = Θ(1): 「虽然都是矩阵，但Embedding层度量“稳”的指标不应该用谱范数，而应该是最大行范数，这就导致它的最速下降并不是Muon」 ("though a matrix, the embedding's stability metric should be the max row norm, not the spectral norm, so its steepest descent is not Muon").
  - Solution: ΔE_i = −η G_i/‖G_i‖_RMS, which is 「对梯度逐行做RMS Norm（Normalized SGD）」 ("RMS-normalize the gradient row by row").
  - Embedding init std is Θ(1). The LM head is column-wise normalized with lr and init ∝ 1/d. — [kexue.fm/archives/11647](https://kexue.fm/archives/11647)
- **Adam is still the default for these layers:** 「即便有了Muon，Adam仍然是主流优化器的重要组成部分（对于Embedding、LM Head等）」 ("even with Muon, Adam remains essential for the embedding, LM head, etc."). — [kexue.fm/archives/11882](https://kexue.fm/archives/11882)
- **Muon needs weight decay:** without WD, 「Muon前期收敛是快，但很快就被Adam追上了，而且各种“内科”还有崩溃的苗头」 ("Muon converges fast early but is soon caught by Adam, and internal metrics show signs of collapse"). With ΔW = −η[msign(M)+λW], Muon stays ahead. — [kexue.fm/archives/10739](https://kexue.fm/archives/10739)
- **Learning rate:** with MuP/KellerJordan-scaled Muon, Su says to multiply Adam's lr by about 0.2√d (≈10×), or results look underfit: 「直接将Adam的学习率放大10倍来作为Muon的学习率」. — [11416](https://kexue.fm/archives/11416)
- **Norm-fixing (Hyperball / MuonH family):**
  - Su names 「每一步优化后通过归一化将范数重新缩放成τ（例如Hyperball和Nemotron-Flash）」 ("rescale the norm back to τ after every step, e.g. Hyperball and Nemotron-Flash").
  - His view: 「我们真的应该将一个参数的某种范数严格控制成某个值吗？…这仍然会破坏原本的训练动力学」 ("should we really pin a parameter's norm to a fixed value? ... it still disrupts the original training dynamics").
  - He proposes keeping ‖ω‖ ≤ τ instead, via Post-Clip or Pre-Decay. He shows that RMS-norm Pre-Decay is exactly ordinary weight decay: 「RMS范数下的Pre Decay就是我们常用的权重衰减」. — [kexue.fm/archives/11729](https://kexue.fm/archives/11729)
- **For the embedding, the matching clip operator is per-row:** 「Embedding层的范数是行RMS的最大者，裁剪算子就是将每行向量的RMS都裁剪到不超过τ」 ("the embedding's norm is the max row RMS; the clip operator caps each row's RMS at τ"). — [11729](https://kexue.fm/archives/11729)
- **Sphere geometry:**
  - On ‖w‖₂=1, steepest descent projects the gradient to the tangent space, g − ⟨g,w⟩w.
  - A tangent step of size η changes the norm to √(1+η²). A retraction (re-normalization) is then needed: 「w − ηφ的模长是并不是精确地等于1，而是√(1+η²)」. — [kexue.fm/archives/11196](https://kexue.fm/archives/11196); spectral-sphere Muon in [kexue.fm/archives/11241](https://kexue.fm/archives/11241)
- **Steps vs learning rate in the optimization term:** F_opt ~ α1(Tη)^{-γ1} + α2 B^{-γ2} + α3 η^{γ3}, where 「Tη才是模型走过的“路程”」 ("Tη is the distance the model travels"). With theory values γ1=γ3=1, γ2=1/2, the optimal lr is η* ~ T^{-1/2}. — [11833](https://kexue.fm/archives/11833)

### Inferences
- **Row-normalized steepest descent is frequency-blind per appearance (from 11647):** a token that appears in 3 contexts per pass gets the same size of row step on each appearance as a common token. Its row is shaped only by those 3 contexts, repeated on each pass. Plain Adam approximates this per-coordinate (9736 and 11459 group Adam with the homogeneous "gradient/√gradient²" family). This is a candidate mechanism for why the 8-pass cost concentrates on rare input rows, and why it vanishes for bytes, where every row appears in millions of contexts.
- **Memory on a fixed-norm sphere (my derivation from 11196's √(1+η²), not stated by Su):** with retraction w ← (w − ηφ)/√(1+η²) and φ ⟂ w, the component along any old direction shrinks by 1/√(1+η²) ≈ 1−η²/2 per step. This gives a memory window of order 2/η² steps, the analogue of 11459's 1/(λη). As in Q2, it is measured in steps, so the byte model (4.74× more steps per text) has a shorter window in units of text at the same η. MuonH's matrices may therefore forget repeated text faster under bytes by construction. This needs an equal-text-window control before claiming a tokenizer effect.
- **Remedies that kexue.fm suggests and that fit our setup:**
  1. Lazy or row-masked updates for the input embedding.
  2. Tied, half-swapped embeddings, which give dense gradients.
  3. Per-row RMS clip or decay on the embedding (11729), so rare rows cannot drift far from a small number of repeated contexts.
  4. WD/LR sized so that 1/λη ≈ steps per epoch (11459).

### Gaps
- kexue.fm has no article on MuonH specifically, and no analysis of the memory or forgetting window of norm-preserving (hyperspherical) updates. 11729 mentions Hyperball only in passing.
- No kexue.fm article on Adam's second-moment behaviour for rarely-seen embedding rows. Su's discussions of sparse-row dynamics are about momentum (6869, 9736).
- No kexue.fm discussion of tokenizer granularity relative to model capacity: for example, how many embedding rows a 300M model can support before rows become memorization slots. 11833's "记忆之层" section is the closest, and it is about MoE/Engram-style memory, not vocabularies.
