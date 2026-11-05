# Write the caption of a new figure on a research page (English and Chinese)

Page: an experiment page on residual designs of language models. Each figure has a short caption (at most two lines) and a longer "Details" text. Below are my English drafts. Polish the English lightly and write fluent, natural Chinese that a Chinese researcher would write, not a translation. Write in plain, direct sentences, each with one fact.

Hard rules:
- Keep every number, every comparison and every claim. Add no new claim, no new number, no explanation that the draft does not have.
- Terms: "Hyper-Connections" stays in Latin letters in Chinese, as do mHC, AttnRes, Pre-LN, nats. 混合 = mixing. 单位混合 = identity mixing. 完整学到混合 = full learned mixing. 读写 = read and write. 留出损失 = held-out loss. 第 N 层 = layer N. 通路 = path. 溢出 = overflow. 种子 = seed. 流 = stream.
- No em-dashes. No "although X, Y" or "X can, but Y" constructions. No words like "crucially", "notably", "interestingly".
- Short caption: no numbers except the depths are not needed either; give no digits at all. English at most 200 characters. Chinese at most 90 characters.
- In Chinese, write numbers with units as in the draft (0.041 nats), and use 第 N 步, N 层.
- Chinese style: short to medium sentences, explicit connectives (因为、所以、但是), no chains of 的, no 将/进行/被 habits. Use a real minus sign − for negative numbers in both languages, as in the draft.

Write the result to /scratch/gpfs/GROUP/USER/tmp/pls_hcid/caps_1105c_out.json with the keys cap_en, cap_zh, more_en, more_zh (plain text, no HTML). Check that the JSON loads. Reply "done".

## cap (short caption of the figure)
To test whether learned mixing pays at every depth, we train Hyper-Connections with and without it at three depths. The deeper the model, the less it pays. Pre-LN, with one path, stays worst at every depth.

## more (Details of the figure)
Hyper-Connections, mHC and AttnRes report that removing learned routing raises the loss at 16 layers. At 48 layers we find that fixing the mixing to the identity lowers it. We ask whether depth explains the difference. We train Hyper-Connections at 12 and 24 layers on the same code and the same budget, with full learned mixing and with identity mixing, and add the 48-layer runs. Read and write stay learned in both models, so this removes only the mixing. The left panel plots the final held-out loss of the last layer. Dots are runs and lines join their means. Pre-LN is the model with one path. The right panel plots identity mixing minus full learned mixing at each evaluation step, so a positive value means learned mixing helps. At the end the value is +0.016 nats at 12 layers, −0.020 at 24 and −0.056 at 48. Before the 12- and 24-layer runs started, we predicted at least +0.01, about 0 and about −0.06. The two shallower values are below 0.03 nats, so we read them as no effect larger than that. The 12-layer value matches the 0.012–0.022 nats the papers report at 16 layers, with other optimizers and far more tokens. Two seeds of Hyper-Connections at 48 layers differ by 0.041 nats, and each shallower depth has one pair of runs. So only the change from 12 to 48 layers, 0.072 nats, exceeds the difference between seeds. The three depths keep the same order at all five evaluation steps, and each depth ends lower than it was at step 1,000. Pre-LN stays above identity mixing by 0.061, 0.085 and 0.090 nats at the three depths. Its 12- and 24-layer runs use the older code, which moves Pre-LN by 0.006 nats at 48 layers. An older-code run of Hyper-Connections reaches 3.764 nats at 12 layers and 3.704 at 24, which would make the values +0.020 and −0.044. One more 48-layer run with learned mixing overflowed from step 4,408, and we leave it out.
