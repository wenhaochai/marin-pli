# Rewrite parts of a blog post draft, English and Chinese

The post is `draft_v1.json` in this folder; its brief is `argon_task.md` (voice, hard rules, terms: all still apply). Numbers come only from `facts.md`. I audited the draft and changed the story in the middle. Write the passages below. Polish my English lightly and write fluent, natural Chinese for its own readers, not a translation. Keep every number and every claim in my drafts. Add no new claim or number. Keep <span class="chk"> marks exactly where I put them. Write minus signs as − in both languages.

Chinese terms: keep the names Hyper-Connections, HC, mHC, Attention Residuals, AttnRes, Pre-LN, MLP, RMSNorm, token, loss in Latin letters, never 超连接 or 注意力残差. 单位矩阵 for identity. 评估点 for evaluation point. 移动内容 for moving content, never 搬移. 尺度变化 for change of scale. 增益 for gain of a sublayer. 收益 for gain from mixing or from paths. 流之和 for stream sum.

Write `draft2.json` in this folder: one object with the keys below, each {"en": "...", "zh": "..."}. Check that it loads. Reply "done".

## motivation (one paragraph, footnote markers as given)
Manifold-Constrained Hyper-Connections (mHC)<sup class="fn">2</sup> is Hyper-Connections (HC)<sup class="fn">1</sup> with a constraint on its mixing matrix, and at 48 layers it reaches a lower held-out loss than HC. We expected the constraint to be the reason. So we replaced HC's learned mixing with the identity and trained again from scratch. In Figure 1, HC then reaches mHC's held-out loss, 0.03 to 0.08 nats lower than with learned mixing. Removing the mixing from mHC itself moves the held-out loss by 0.006 nats. In Attention Residuals (AttnRes)<sup class="fn">4</sup>, replacing the token-dependent weights with weights that ignore the token lowers it by 0.08 to 0.18 nats. Yet the papers that introduced these designs report the opposite. Removing the routing raises the loss by 0.021 in HC at 16 layers, by 0.012 in AttnRes at 16 blocks, and by 0.022 in mHC's table of HC at a size it does not state. What decides whether learned routing pays?

## setup (one paragraph)
We train ordinary models, which optimize only the last layer's next-token loss, at 12, 24 and 48 layers. Width is 512 and the MLP width is 1792 at every depth, so the deeper models have more parameters: 177M, 223M and 314M. Every model trains from scratch for the same 4,959 steps on 2.6B tokens of FineWeb-Edu, in the same data order, with the MuonH recipe of the Marin speedrun. The held-out loss is the last layer's next-token cross-entropy on Paloma C4, in nats. One quantity runs through the post: the gain from learned mixing, which is the held-out loss with no mixing minus the held-out loss with learned mixing at the same depth.

## depth (one paragraph)
In Figure 3, the gain from learned mixing is +0.016 nats at 12 layers, −0.020 at 24 layers and −0.056 at 48 layers. At 12 layers this has the size the papers report. So the papers and our 48-layer result need not contradict each other. Before the 12- and 24-layer trainings started, we wrote down at least +0.01 and about 0. At all five evaluation points the three depths keep the same order, and at every depth the gain ends lower than it was at step 1,000. <span class="chk">We read a difference under 0.03 nats as no effect, so the claim rests on the trend across depth, not on the 24-layer point alone.</span> Back at layer 12 of the 48-layer model, what is this mixing doing that costs more the deeper the model is?

## controls_1 (one paragraph, replaces the first paragraph of the controls section)
Four controls rule out simpler explanations. First, the 12- and 24-layer pairs share code, data order and steps, and differ only in the mixing. Second, at 48 layers the parameter count differs by at most 0.1 percent within each comparison and at most 1.5 percent across designs.

## scale_heading
The mixing's change of scale compounds with depth

## scale_1 (one paragraph)
Return to layer 12 of the 48-layer model. The four row sums average 0.33, so this sublayer shrinks the stream sum to a third. Every sublayer multiplies the stream sum by such a gain, so depth multiplies the effect. Figure 4 plots the product of these gains, layer by layer. It spans about 1.7 orders of magnitude at 12 layers, 1.9 at 24 and 4.6 at 48. In the 48-layer model the product falls below one ten-thousandth by layer 23, then climbs back to about 1 at the last layer.

## scale_2 (one paragraph)
The state shows the result. With learned mixing, the median norm of the stream sum at the last layer is 114, 455 and about 10 million at the three depths, against 311, 398 and 667 with no mixing. The last write is 0.30, 0.10 and about 0.00001 of the new stream sum, against 0.40, 0.29 and 0.29. RMSNorm reads only the direction, so nothing in the loss limits the size of the state.

## scale_3 (one paragraph)
<span class="chk">Once the growth did not stop.</span> From step 4,408 the state norm passes about 1.8e19, where a float32 sum of squares overflows. RMSNorm then returns zero, the readout gives a uniform prediction for 70 percent of the tokens, and the held-out loss ends at 9.885. The gradient norm stays flat, so gradient monitors see nothing.

## cost_heading
The cost follows the change of scale, not the moving of content

## cost_1 (one paragraph)
Learned mixing does two things: it moves content between streams, and it changes the scale of the stream sum. We trained each one alone from scratch at 48 layers. Mixing that only moves content and keeps the stream sum reaches a held-out loss of 3.578 with non-negative weights and 3.591 with signed weights, against 3.579 and 3.581 with no mixing. So moving content neither helps nor hurts. Mixing that moves nothing and applies a fixed gain per sublayer, taken from a trained HC model, reaches 3.687, and 3.676 at half the dose, above learned mixing at 3.615 to 3.656.

## cost_2 (one paragraph)
Figure 5 puts these beside the depth test. Its horizontal axis is the number of orders of magnitude that the product of gains spans. At 1.7 orders or fewer the gain from mixing is near zero or slightly positive: +0.015 with one learned number per sublayer and +0.016 with learned mixing at 12 layers. From 1.8 orders on it is negative: −0.029 with constant mixing, −0.020 with learned mixing at 24 layers, −0.056 at 48 layers, −0.096 at half dose and −0.107 with fixed gains. The points come from different depths and different kinds of mixing, and they fall in about one order. The order is not perfect: the half dose costs more than learned mixing at 48 layers, which changes the scale more.

## cost_3 (one paragraph)
<span class="chk">We think this is why depth hurts. A write reaches the stream sum only through the gains after it. When those gains multiply to thousands, some layers' writes count thousands of times more than others, so the mixing in effect closes paths. That would explain the late writes above, and the early writes in the model with fixed gains. We did not test it directly.</span>

## paths_1 (one paragraph, replaces the first paragraph of the paths section)
The gain from paths is the held-out loss of Pre-LN minus the held-out loss with no mixing at the same depth. In the left panel of Figure 3 it is +0.061 nats at 12 layers, +0.085 at 24 and +0.090 at 48. So paths pay at every depth, while learned mixing stops paying.

## limitations (one paragraph; every sentence keeps its mark)
<span class="chk">All trainings use one optimizer (MuonH), one held-out text (Paloma C4) and one budget of 2.6B tokens against up to 500B tokens in the papers, and at every depth the gain from learned mixing falls during training, so the depth where it crosses zero may move with the budget.</span> <span class="chk">Depth and parameter count move together here, from 177M to 314M.</span> <span class="chk">At 12 and 24 layers we removed only the mixing, and the link between the change of scale and the cost across depths is a correlation; only the 48-layer variants change the scale by construction.</span> <span class="chk">Larger models such as Kimi K3 are not tested and may differ.</span>

## takeaways_1 (one paragraph)
To sum up, extra residual paths lower the held-out loss at 12, 24 and 48 layers. Learned mixing between the paths helps slightly at 12 layers (+0.016 nats) and hurts at 48 (−0.056 nats). Its cost follows its change of scale, which compounds with depth, and moving content between streams does not pay at 48 layers.

## related_mhc (one sentence, in the Related work paragraph on the designs tested here; the note marker stays)
Xie et al.<sup class="fn">2</sup> projected HC's mixing matrix onto doubly stochastic matrices via Sinkhorn iterations in mHC, reported that this lowers the composite gain from nearly 3000 in HC to about 1.6, and reported a 0.022 loss rise from identity mixing in a table of HC whose size is not stated, whereas our 48-layer trainings show identity mixing matches Sinkhorn mixing.

## related_depthbench (one sentence, same paragraph)
Wang et al.<sup class="fn">5</sup> compared Pre-LN, HC, mHC and AttnRes across depths up to 70 layers in DepthBench with AdamW, found that HC keeps gaining from depth up to 32 layers, from 2.729 at 16 layers to 2.699, and observed that mHC's products converge toward a rank-1 average, without testing identity or static routing.

## related_su (one sentence)
Jianlin Su<sup class="fn">10</sup> noted in his AttnRes retrospective that a data-independent query beats more complex query designs, and that multiplying HC's matrices risks explosion or collapse, and he cited a concurrent report that identity mixing is good enough for mHC; we measure identity mixing at three depths.

## dek_removal
At 48 layers, removing learned mixing or token dependence lowers the held-out loss in HC and AttnRes and leaves mHC about the same.
## cap_removal (what is plotted and how; no numbers; at most 200 English characters, 90 Chinese characters)
Held-out loss on Paloma C4 at the end of training, for each residual design with its learned routing and with that routing removed from the start.
## dek_depth
Extra paths pay at every depth, and the gain from learned mixing turns negative as the model gets deeper.
## cap_depth (same limits)
Left: final held-out loss on Paloma C4 at each depth. Right: no mixing minus learned mixing at each evaluation during training.
## dek_scale
The deeper the model, the wider the product of the learned mixing's gains swings.
## cap_scale (same limits)
Product of the gains that the learned mixing applies to the stream sum, up to each layer: median over training tokens at the final checkpoint.
## dek_cost
The larger the change of scale, the lower the gain over no mixing, across depths and kinds of mixing.
## cap_cost (same limits)
Each dot is one kind of mixing at one depth: its held-out gain over no mixing against the orders of magnitude that the product of its gains spans.
## dek_paths
At 48 layers, every design with several paths and no mixing beats every design with one path.
## cap_paths (same limits)
Held-out loss on Paloma C4 at the end of training, for designs with one path and designs with several paths and no mixing.
