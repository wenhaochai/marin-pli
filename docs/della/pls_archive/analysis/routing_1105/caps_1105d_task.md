# Write short texts for a research page (English and Chinese)

Page: an experiment page on residual designs of language models. It is organized by questions. Each question has a short answer under its heading, and each figure has a short caption (at most two lines). Below are my English drafts. Polish the English lightly and write fluent, natural Chinese that a Chinese researcher would write, not a translation. Plain, direct sentences, one fact each.

Hard rules:
- Keep every claim. Add no new claim and no number. Captions and the answer have no numbers except the depths 12, 24 and 48 where I wrote them.
- Terms: Hyper-Connections, HC, mHC, AttnRes, Pre-LN stay in Latin letters in Chinese. 残差通路 = residual path. 路由 = routing. 流之和 = stream sum. 混合 = mixing. 学到的混合 = learned mixing. 没有混合 = no mixing. 单位混合 = identity mixing. 留出损失 = held-out loss. 普通模型 = ordinary model. 尺度 = scale. 增益 = gain of a sublayer. 收益 = gain from mixing or from paths.
- No em-dashes. No "although X, Y" or "X can, but Y". No "crucially", "notably", "interestingly".
- Captions: English at most 200 characters, Chinese at most 90 characters each.
- Chinese style: short to medium sentences, explicit connectives (因为、所以、但是), no chains of 的, no 将/进行/被 habits.

Write /scratch/gpfs/GROUP/USER/tmp/pls_hcid/caps_1105d_out.json with the keys below, each with _en and _zh (plain text). Check that it loads. Reply "done".

## q_title (a question heading)
Does learned routing between residual paths pay?

## q_answer (the answer under the heading; two or three sentences)
Not in deep models. Extra residual paths lower the held-out loss at 12, 24 and 48 layers, while learned mixing between them pays a little at 12 layers and costs at 48. Its cost follows how far it changes the scale of the stream sum, and that change grows with depth.

## q_intro (a short paragraph after the answer)
Every model in this question is an ordinary model, and every number is the held-out loss of the last layer on Paloma C4. At 48 layers mHC, which is Hyper-Connections with a constraint on its mixing, beats Hyper-Connections. We expected the constraint to be the reason, so we trained Hyper-Connections with its mixing fixed to the identity. It reached mHC's loss. The papers that introduced these designs report the opposite at 16 layers, so we trained the same pair at 12 and 24 layers.

## cap_removal (caption of a figure: held-out loss of HC, mHC and AttnRes, each with and without its learned routing, 48 layers)
To see whether learned routing pays in ordinary models, we remove it before training. Without it, Hyper-Connections and AttnRes reach a lower held-out loss, and mHC stays about the same.

## cap_scale (caption of a figure: the product of the gains that HC's learned mixing applies to the stream sum, layer by layer, at 12, 24 and 48 layers)
To see what depth changes in the learned mixing, we multiply the gains it applies to the stream sum, layer by layer. The deeper the model, the wider the product swings.

## cap_cost (caption of a scatter: gain over no mixing against the orders of magnitude the product of gains spans, for learned mixing at three depths and six kinds of mixing at 48 layers)
To find what the cost of mixing follows, we plot its gain over no mixing against how far it changes the scale of the stream sum. The larger the change, the lower the gain.

## cap_paths (caption of a figure: held-out loss of designs with one path and designs with several paths and no mixing, 48 layers)
To see where the gain comes from, we compare designs with one path against designs with several paths and no mixing. Every design with several paths reaches a lower held-out loss.
