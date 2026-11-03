# Task: rewrite one passage, in English and in Chinese, for one running experiment page

The page is https://wenhaochai.com/exps/per-layer-contribution.html, on the owner's site. Readers are ML researchers who did not run the experiments. Every text exists in English and Chinese, and a reader picks one language. Do not commit, push or submit anything. You run headless: keep calling tools until the output file is written.

Output: write `/scratch/gpfs/GROUP/USER/tmp/pls_hcid/argon_out2.json`, one JSON object with exactly two keys, `en` and `zh`, each holding plain text with no HTML.

## Rules

1. The passage goes in a "Details" fold under a bar chart. Write full sentences that make a point.
2. Keep every number in the draft, exactly as written, in both languages. Do not add any number.
3. Keep the meaning exactly. Do not add causes or links the draft does not state ("therefore", "confirms", "因此"). Do not drop a qualifier ("only", "still", "fixed", "also"). If a draft sentence reports what happened together with something else, do not turn it into cause and effect. In particular, do not say why the variant diverged: we do not state a cause here.
4. No random-seed talk and no run counts. No run names, job ids or code names.
5. Style: plain words, active voice, one idea per sentence, short sentences. Use "we" where the authors act. No em dashes. No "delve", "crucially", "notably", "moreover". No "not X but Y". No colon reveals, no rhetorical questions, no closing aphorisms. A consequence is its own sentence, not a ", so" tail. Keep the first sentences close to the current text: only the changed parts need new words.
6. Chinese: write fluent Chinese as a strong Chinese technical blog reads (苏剑林's kexue.fm). Do not translate word by word. Short to medium sentences, explicit connectives where the logic needs them. No long attributive chains, no semicolon chains, no 将/进行/被 habits.
7. Glossary (one name per thing): stream 流; the stream sum 各流之和; read 读取; write 写入; mixing 混合; dynamic part 动态部分; state 状态; sublayer 子层; ordinary model 普通模型; bar 柱子; variant 变体; diverged 发散; layer k 第 k 层.

## The chart

The chart plots, per design variant at 48 layers, the last-layer training loss of the separate-heads model minus that of the ordinary model with probes. Hyper-Connections has four parallel streams, and every readout reads the sum of the streams. A variant with an empty bar has not finished training.

## Current text (English)

Three more Hyper-Connections variants are still training, and their bars are empty. In the first, the dynamic parts of read and write come from one joint projection of all streams, as in mHC. In the second, the mixing keeps the stream sum, because each stream's outgoing weights add up to one. In the third, the mixing is one learned number per sublayer, which scales the whole state and moves nothing between streams. The last two test whether the effect of the mixing on the stream sum is what matters.

## Current text (Chinese)

另外三个 Hyper-Connections 变体还在训练，它们的柱子是空的。第一个变体的读取和写入，动态部分像 mHC 一样来自所有流的同一个联合投影。第二个变体的混合保持各流之和不变，因为每条流流出的权重之和等于 1。第三个变体的混合是每个子层一个学到的数，它只缩放整个状态，不在流之间移动任何东西。后两个变体用来检验混合对各流之和的影响是不是关键。

## Draft of the new passage. Keep these six points in this order

1. Three more Hyper-Connections variants are still training, and their bars are empty.
2. In the first, the dynamic parts of read and write come from one joint projection of all streams, as in mHC.
3. In the second, the mixing is one learned number per sublayer, which scales the whole state and moves nothing between streams.
4. In the third, the mixing only shrinks the stream sum. Its gains are fixed, one per sublayer, and come from the ordinary Hyper-Connections model. Every difference between streams stays as it is.
5. The last two test whether the effect of the mixing on the stream sum is what matters.
6. We also tried a variant whose mixing keeps the stream sum and moves content between streams. Its ordinary model diverged at step 27, and we stopped it, so it has no bar.
