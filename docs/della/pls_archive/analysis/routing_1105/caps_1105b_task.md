# Rewrite caption passages of a research page (English and Chinese)

Page: an experiment page on per-layer supervision of language models. Each figure has a short caption (at most two lines) and a longer "Details" text. Below are my English drafts. Polish the English lightly and write fluent, natural Chinese that a Chinese researcher would write, not a translation. Write in plain, direct sentences, each with one fact.

Hard rules:
- Keep every number, every comparison and every claim. Add no new claim, no new number, no explanation that the draft does not have.
- Terms: "Hyper-Connections" stays in Latin letters in Chinese, as do mHC, Pre-LN, nats. 独立头 = separate heads. 普通模型 = ordinary model. 各流之和 = stream sum. 混合 = mixing. 单位混合 = identity mixing. 标量混合 = scalar mixing. 只移动混合 = moving-only mixing. 半剂量缩小 = half-dose shrinking. 固定增益 = fixed gains. 完整学到混合 = full learned mixing. 评估损失 = evaluation loss. 代价 = cost.
- No em-dashes. No "although X, Y" or "X can, but Y" constructions. No words like "crucially", "notably", "interestingly". No talk of random seeds.
- Short captions: no numbers. English at most 200 characters. Chinese at most 90 characters.
- In Chinese, write numbers with units as in the draft (0.157 nats), and use 第 N 层.
- Chinese style: short to medium sentences, explicit connectives (因为、所以、但是), no chains of 的, no 将/进行/被 habits. Do not use English hyphens or en-dashes in place of Chinese punctuation.

Write the result to /scratch/gpfs/GROUP/USER/tmp/pls_hcid/caps_1105b_out.json with the keys listed below, each with _en and _zh (plain text, no HTML). Check that the JSON loads. Reply "done".

## f38_scalar (one sentence added to the Details of Figure 38; the sentences before it say that in this variant the mixing is one learned number per sublayer, which scales the whole state and moves nothing between streams)
It pays 0.157 nats.

## f38_empty (a passage of the Details of Figure 38)
The bars of the second and fifth variants are still empty, because their separate-heads models are still training. So far, scalar mixing, half-dose shrinking and moving-only mixing all pay more than the 0.047 nats of full learned mixing.

## f50_cap (second sentence of the short caption of Figure 50; the first sentence says that mHC ordinary models beat Hyper-Connections with full learned mixing, and "it" below means Hyper-Connections)
Identity, scalar or moving-only mixing brings its ordinary model near mHC's and raises its separate-heads loss.

## f50_range (a clause inside a sentence of the Details of Figure 50; give only the replacement of this exact clause. In Chinese it is a parenthesis in the form （下文……的 Hyper-Connections 除外）)
apart from the Hyper-Connections rows with identity, scalar, half-dose or moving-only mixing described below.

## f50_gain (one sentence in the Details of Figure 50 about the product of the top singular values of three MLP weights; a sentence before it says the largest product is 1,606 with moving-only mixing, at layer 5. In Chinese write three short sentences in the form 半剂量缩小是 359，在第 2 层。)
It is 359 with half-dose shrinking, at layer 2, 612 with Pre-LN's fixed gains, at layer 12, and 97 with scalar mixing, at layer 10.

## f50_scalar (a passage of the Details of Figure 50; the sentence before it says that with the mixing reduced to one learned number per sublayer, the ordinary model reaches 2.918 nats, inside the range of the mHC runs)
Its evaluation loss is 3.565 nats, against 3.638 for Hyper-Connections. The separate-heads model reaches 3.075 nats, and the cost is 0.157 nats, close to the 0.169 of identity mixing.

## f50_new (a passage of the Details of Figure 50; it comes after sentences about half-dose shrinking and before a sentence about Pre-LN's fixed gains)
With mixing that only shrinks the stream sum, the ordinary model reaches 3.030 nats, above half-dose shrinking. With mixing that keeps the stream sum and allows negative weights, the ordinary model reaches 2.928 nats, close to identity mixing. The separate-heads models of these two variants are still training.
