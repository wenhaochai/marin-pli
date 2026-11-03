# Task: write four short texts in English and Chinese for one running experiment page

The page is https://wenhaochai.com/exps/per-layer-contribution.html, on the owner's site. Readers are ML researchers who did not run the experiments. Every text exists in English and Chinese, and a reader picks one language. Do not commit, push or submit anything. You run headless: keep calling tools until the output file is written. Do not end your turn with a plan.

Output: write `/scratch/gpfs/GROUP/USER/tmp/pls_hcid/argon_out.json`, one JSON object with exactly the keys below, each holding plain text with no HTML:
fig38_short_en, fig38_short_zh, fig38_insert_en, fig38_insert_zh, fig50_short_en, fig50_short_zh, fig50_insert_en, fig50_insert_zh

## Rules

1. Short captions: at most 200 English characters, at most 90 Chinese characters. Words only, no numbers. They say what the figure shows.
2. Inserts go in a "Details" fold under the figure. They carry the numbers. Write full sentences that make a point. Keep every number in the draft, exactly as written, in both languages. Do not add any number.
3. Keep the meaning exactly. Do not add causes or links the draft does not state ("therefore", "confirms", "因此"). Do not drop a qualifier ("about", "after", "at the final checkpoint"). If a draft sentence reports what happens together with something else, do not turn it into cause and effect.
4. No random-seed talk and no run counts. No run names, job ids or code names.
5. Style: plain words, active voice, one idea per sentence, short sentences. Use "we" where the authors act. No em dashes. No "delve", "crucially", "notably", "moreover". No "not X but Y". No colon reveals, no rhetorical questions, no closing aphorisms. A consequence is its own sentence, not a ", so" tail. Same word for the same thing. Do not use "pays" more than the draft does.
6. Chinese: write fluent Chinese as a strong Chinese technical blog reads (苏剑林's kexue.fm). Do not translate word by word. Short to medium sentences, explicit connectives where the logic needs them. No long attributive chains, no semicolon chains, no 将/进行/被 habits. Do not write "付". Say "代价是 0.169 nats". Keep in English: token, MLP, nats, Hyper-Connections, mHC, AttnRes, Sinkhorn.
7. Glossary (one name per thing): separate heads 独立头; the separate-heads model 独立头模型; probes 探针; ordinary model 普通模型; design 结构; stream 流; the stream sum 各流之和; read 读取; write 写入; mixing 混合; identity mixing 单位混合; cost 代价; final training loss 最终训练损失; layer k 第 k 层; Figure 50 图 50; evaluation loss 评估损失; state 状态; sublayer 子层; token 不译.

## The figures

Figure 38 is a bar chart. It plots, per design variant at 48 layers, the last-layer training loss of the separate-heads model minus that of the ordinary model with probes. That difference is the cost.
Figure 50 is a dot chart. Each row is one design. The light dot is the last-layer final training loss of the ordinary model. The dark dot is the same loss for the separate-heads model. The segment between them is the cost.
Both figures now show a new variant: Hyper-Connections with its mixing fixed to the identity.

## fig38_short: replaces this caption

Now: "Identity mixing and bounded read-write maps do not explain mHC's full cost, and signed maps lower it only slightly. HC with one shared write pays about the same, and clipped AttnRes queries still spike."
Now (Chinese): "单位混合和有界读写解释不了 mHC 的全部代价，有符号读写也只让代价略降。共用一个写入时 Hyper-Connections 代价差不多，裁剪 query 的 AttnRes 仍有尖峰。"
Draft of the new caption. Keep these four points in this order. Shorten as needed: identity mixing and bounded read-write maps do not explain mHC's full cost; in Hyper-Connections, identity mixing raises the cost more than any other change; one shared write leaves it about the same; clipped AttnRes queries still spike.

## fig38_insert: new sentences in the Details fold

They come right after this sentence: "Hyper-Connections with one write shared across streams pays 0.042 nats, close to Hyper-Connections."
And right before this sentence: "The two AttnRes seeds differ because the ordinary model of seed 0 has more loss spikes."
Draft:
"Hyper-Connections with its mixing fixed to the identity pays 0.169 nats, the largest cost of any 48-layer design we trained. Figure 50 splits this cost into its two models. The learned mixing of the ordinary Hyper-Connections model shrinks the stream sum in the early layers, to about half per layer over layers 2-11, whereas identity and Sinkhorn mixing leave the sum unchanged. Three more Hyper-Connections variants are still training, so their bars are empty. In the first, the dynamic parts of read and write come from one joint projection of all streams, as in mHC. In the second, the mixing keeps the stream sum, because each stream's outgoing weights add up to one. In the third, the mixing is one learned number per sublayer, which scales the whole state and moves nothing between streams. The last two test whether the effect of the mixing on the stream sum is what matters."

## fig50_short: replaces this caption

Now: "With four or more streams, the separate-heads models of Hyper-Connections and mHC are about equally good. Every mHC ordinary model beats every Hyper-Connections one, raising mHC's cost."
Now (Chinese): "4 条及以上流时，Hyper-Connections 与 mHC 的独立头模型相近。所有 mHC 普通模型都优于所有 Hyper-Connections 普通模型，因此 mHC 的代价更高。"
Draft of the new caption. Keep these points in this order: every mHC ordinary model beats every Hyper-Connections one; with its mixing fixed to the identity, the ordinary model of Hyper-Connections comes close to mHC's; its separate-heads model then has the highest loss in this figure.

## fig50_insert: new sentences at the end of the Details fold

They come after this sentence: "The mHC design has a slightly better separate-heads model, at 3.003 against 3.021 nats, and a far better ordinary model, at 2.918 against 2.979."
Draft:
"Fixing the mixing of Hyper-Connections to the identity tests whether the learned mixing separates its ordinary model from mHC's. All numbers here are on the newer code. The ordinary model falls from 2.956 to 2.931 nats, below every Hyper-Connections run with learned mixing and above every mHC run. Its evaluation loss reaches mHC's level, at 3.579 nats against 3.576 for mHC with identity mixing and linear read and write, and 3.638 for Hyper-Connections. The probes change far more than the last layer. At layers 10-32 they end 1.03-1.23 nats above Hyper-Connections, and at layers 12-48 they stay within 0.23 nats of that mHC design. The separate-heads model moves the other way and rises from 3.003 to 3.100 nats, the highest in this figure. The cost rises from 0.047 to 0.169 nats. The separate-heads gap to Hyper-Connections stays near 0.02 nats until about step 2,750 of 4,959, and then grows while the gradients of the first three layers grow. At the final checkpoint, about 3% of the tokens have a state more than 100 times the median. For most of them this starts with one MLP write at layer 5, about 10^5 times the median write, and identity mixing keeps it through every later layer. These tokens account for about a fifth of the rise on 4,096 tokens of held-out text. We have not yet found the source of the rest."

Before you write the file, check for each key that every number in the draft appears in your text, and check the length limits of the short captions. Write the file, then reply with one line.
