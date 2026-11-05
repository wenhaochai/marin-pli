# Check the related work of a blog post (read-only; no jobs, no pushes)

The post's claim, on 48-layer language models trained with MuonH on FineWeb-Edu: several residual paths lower held-out loss, and removing the learned routing between those paths does not raise it. Concretely: Hyper-Connections with its learned mixing replaced by the identity, or by one learned number, is better; mHC with its Sinkhorn mixing replaced by the identity is the same; Attention Residuals with input-independent weights is better.

Open each source and report what it actually says. Fetch through a vis node, never from the login node, for example `ssh della-vis1 'curl -sL https://arxiv.org/abs/2409.19606'`, and the HTML version at `https://arxiv.org/html/<id>` or the PDF (`pdftotext` if present). The local copy of kexue.fm is the skill `kexue-fm:kexue`; find its SQLite under `/home/USER/.claude-kn/plugins/cache/` or `~/.claude/plugins/cache/` and query it with sqlite3 on a vis node.

Sources:
1. Hyper-Connections, arXiv 2409.19606. Static versus dynamic numbers (Table 2 or wherever they are), what happens with frozen mixing or no mixing, what they claim mixing does.
2. mHC: Manifold-Constrained Hyper-Connections, arXiv 2512.24880. What they claim the constraint does, what they compare with, numbers, any identity-mixing baseline.
3. mHC-lite, arXiv 2601.05732. Claim and numbers.
4. Attention Residuals, arXiv 2603.15031. The ablation table with input-independent mixing (Table 4 per my notes), model size and depth, numbers.
5. DepthBench, arXiv 2609.32534. Its results for HC, mHC, AttnRes at 48 layers or nearest, any static or identity variant; its recipe (optimizer, tokens).
6. He, Zhang, Ren, Sun, Identity Mappings in Deep Residual Networks, arXiv 1603.05027. Its comparison of gating and identity shortcuts, numbers.
7. Srivastava, Greff, Schmidhuber, Highway Networks, arXiv 1505.00387, and Training Very Deep Networks if it is the better citation.
8. Huang et al., DenseNet, arXiv 1608.06993: the one claim that relates to extra fixed paths.
9. DenseFormer, arXiv 2402.02622, and MUDDFormer, arXiv 2502.12170: static versus learned dense weights in language models, numbers.
10. kexue.fm archives 11664 and 9889 (苏剑林): what he says about HC, mHC, AttnRes, static queries and identity mixing. Quote the Chinese sentences.
11. Search those papers' reference lists and text for anyone who already reports that identity mixing or static routing matches or beats the learned version in language models. Name them.

Write /scratch/gpfs/GROUP/USER/tmp/pls_routing/related.md. For each source: exact title; authors in order; venue and year; the claim with its numbers, quoted where short, with the table or section it comes from; then one sentence on how it agrees or disagrees with our claim, and one clause on any setting difference (scale, optimizer, tokens). Mark anything you could not open as NOT OPENED. Do not fill a gap from memory. Plain English, short sentences. End with a list: who already said a version of our claim, and what this post adds that none of them does.
