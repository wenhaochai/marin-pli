# Task: polish the English and rewrite the Chinese of one part of the page "Quantifying the Per-layer Contribution of LLMs"

Owner, 2026-11-01: 「你让argon重新给你整体润色一下中英语写作」 (have Argon polish all the English and Chinese writing of the
page). The page is https://wenhaochai.com/exps/per-layer-contribution.html, a running experiment page on the owner's site.
Every text on it exists in English and in Chinese, and the reader picks one language. Five writers polish five parts of
the page at the same time; the glossary below keeps the parts consistent. The prompt tells you which part is yours.

Do not commit, push or submit jobs. You run headless: your process ends when your turn ends, so keep calling tools until
your part and your report are done, and never end your turn with a plan or a status message. You need no python; if you
run any, run it on della-vis2 (`ssh della-vis2 'CUDA_VISIBLE_DEVICES= nice -n 19 python3 -I ...'`), never on this node.

## Files

- Your part: `/scratch/gpfs/GROUP/USER/tmp/pls_polish/units/part_K.txt`. Edit it in place, unit by unit. Each unit
  is one `@@` line followed by one line of text: the English unit, then the Chinese unit of the same pair. The `@@` line
  names the unit (id | figure or section | kind | language). Lines that start with `##` give context: the question the
  units belong to and each figure's title. Change only the text lines. Keep every `@@` line and every `##` line exactly
  as it is, keep the units in their order, and keep each text on one line.
- Context, read only: the whole page as it is now, `/scratch/gpfs/GROUP/USER/tmp/pls_polish/page_before.html`
  (questions, figure titles and the other parts); the owner's rules for experiment pages, in the "Spec" blocks of
  `/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/exps/pretrain-practice-template.html`; the style rules,
  `/home/USER/.claude/plugins/marketplaces/wenhaochai/writing/skills/style/SKILL.md` (RULE-01 to RULE-18).
- Your report: `/scratch/gpfs/GROUP/USER/tmp/pls_polish/units/report_K.md`, in Chinese (see the end).

## Unit kinds

- short caption: the caption printed under a figure.
- long caption: the text of the figure's "Details" fold, which a reader opens under the figure.
- answer: the one or two sentences printed under an answered question.
- paragraph: a paragraph of Setup, a paragraph between figures, or a paragraph in a question's "Details" fold.
- setup panel: the sentence under one setup's picture in a row of setup schematics. The panel's name (in the `@@` line)
  is fixed.
- note: a one-line pointer to another page.
- related work item: one item of Related work.

## What the owner wants

1. Write for an outside reader: an ML researcher who did not run these experiments. Plain words, complete sentences,
   active voice, one idea per sentence, short paragraphs. No run names, job ids, flags or code names; say what a setup
   does.
2. A short caption fits in two lines at desktop width: at most about 200 English characters, and about 90 Chinese
   characters. It says in words what the figure shows, plus a short reading key when the chart needs one. It does not
   repeat the question above it (the motivation) or the figure's title (what is plotted), and it carries no supporting
   numbers, because the chart shows them.
3. A long caption says why this figure comes here, then what is plotted and how to read it, then what it shows. The
   numbers and caveats that do not belong in the short caption live here. Write them into sentences that make a point;
   a run of values in parentheses after a claim reads worse than a sentence that carries them.
4. An answer is one or two sentences in words, at the altitude the figures support, with no supporting numbers.
5. Short captions, answers and paragraphs between figures carry no defensive text: no seed counts, comparisons with the
   spread between seeds, failed attempts, or caveats that change no conclusion. Nothing is deleted, only moved: such a
   sentence moves from the short caption into the long caption of the same figure.
6. Style: no em dashes; no LLM tells ("delve", "crucially", "notably", "moreover" chains, "it is worth noting"); no "not
   X but Y" framing; no colon reveals; no rhetorical questions or closing aphorisms. A consequence is its own sentence,
   not a ", so" tail. Narrate as "we" where the authors act (we train, we test, we find). Prefer a sentence to a
   parenthesis. Use the same word for the same thing; do not hunt for synonyms.

English example. Now: "The separate-heads cost is the difference of two models, so a design can pay less because its
separate-heads model is better or because its ordinary model is worse." Better: "The separate-heads cost is the
difference between two models. A design can pay less because its separate-heads model is better, or because its
ordinary model is worse."

## The Chinese: write it as fluent Chinese

The owner is a native Chinese speaker and an ML researcher. On another post this week he could not read the Chinese
(「感觉这个中文我完全读不懂」): it was compressed translationese. This page's Chinese has the same problem. For every
Chinese unit, read the English unit, understand it, and write the Chinese that a fluent Chinese author would write with
the same content, as a strong Chinese technical blog reads (苏剑林's kexue.fm, a well-edited 知乎 column). Do not
translate sentence by sentence.

- Natural Chinese word order and rhythm: short to medium sentences, explicit connectives where the logic needs them
  (因为、所以、但是、也就是说、换句话说), one idea per sentence. No long attributive chains before a noun (……的……的……的),
  no stacked clauses joined by semicolons, no "将……" / "进行……" / "被……" habits of translated text, no abstract
  nominalisations where a verb works.
- "pays" is English idiom. In Chinese say what the cost is: "Hyper-Connections 的代价是 0.047 nats", "代价更小", not
  "Hyper-Connections 付 0.047 nats" or "付得少".
- Keep in English what Chinese ML researchers keep in English: token, embedding, MLP, softmax, sigmoid, Sinkhorn, logit
  lens, tuned lens, nats, and every model or method name (Pre-LN, DeepNorm, Hyper-Connections, mHC, AttnRes, Kimi K3,
  MuonH, Qwen3, Marin speedrun, LayerSkip, DeepSeek-V3).
- Examples from this page.
  Now: "多数 run 训练 130m 模型，即 6 层、宽 512、2.6B token，或者宽度和数据相同、换成十种残差结构的 48 层模型；少数到 1.2b。"
  Better: "大部分运行训练的是 130m 模型：6 层，宽 512，训练 2.6B token。另一部分运行用同样的宽度和数据训练 48 层的模型，
  比较十种残差结构。少数运行做到 1.2b。"
  Now: "浅色点是加探针的普通模型的最后一层最终训练损失，它的主干只用最后一层的损失训练；深色点是独立头模型的同一损失；两点之间的线段就是代价。"
  Better: "浅色点是普通模型最后一层的最终训练损失。这个模型挂着探针，但骨干只用最后一层的损失训练。深色点是独立头模型的
  同一个量，两点之间的线段就是代价。"
- Read each Chinese unit in your head before moving on. If a sentence makes you stop to parse it, rewrite it.

## Glossary: one name per thing, in every part

| English | Chinese |
|---|---|
| separate heads (each layer predicts through a head of its own, and its loss trains the backbone) | 独立头 |
| the separate-heads model | 独立头模型 |
| probes (a head on each layer that sends no gradient into the model) | 探针 |
| ordinary model | 普通模型 |
| shared head; output head | 共享头；输出头 |
| backbone | 骨干 (never 主干) |
| design (a residual design such as Pre-LN or Hyper-Connections) | 结构 (残差结构) |
| stream; the stream differences | 流；各流之差 |
| read, write, mixing (of the residual streams) | 读取、写入、混合 |
| readout (reading a layer through a head) | 读出 |
| cost (the separate-heads cost) | 代价 (独立头代价) |
| final training loss | 最终训练损失 |
| the last layer; layer k | 最后一层；第 k 层 |
| run | 运行 |
| seed | 种子 |
| next token; the token after next | 下一个词；下下个词 |
| multi-token prediction | 多 token 预测 |
| Figure 12 | 图 12 |

The figure titles, the legends inside the charts and the setup panel names do not change in this pass, so a name you use
for a setup or a series must match the name the chart prints (for example "HC, mixing = I", "Probes only", "Separate
heads, 48 layers"). Hyper-Connections may stay "HC" where the text already uses that short form.

## Hard constraints (a merge script checks them; a unit that fails keeps its old text)

1. Same facts, same numbers. Every number of a unit stays in that unit, written the same way: digits stay digits, with
   the same decimals, signs and units. The one exception: a number may move from a figure's short caption into the long
   caption of the same figure, in the same language. Never delete, add, round or change a number, and never change a
   number written as a word ("four", "half", "两") into digits or back. The same holds for claims and for hedges that carry
   meaning: you may rephrase, reorder, merge and split sentences and cut repeated words; you may not add a result or drop
   one.
2. The English and the Chinese of a pair say the same thing, with the same numbers.
3. Every HTML tag inside a unit stays, with its attributes. A link `<a href="...">...</a>` keeps its href and stays on
   words that describe its target. The link text of a related-work item (the work, its authors and year) stays as it is.
4. Each unit stays on one line. Add no HTML tags and no ⟦ ⟧ marks.
5. Length: short captions within the limits of point 2 above; long captions and paragraphs no longer than now. Shorter is
   welcome where a sentence repeats the short caption, the title or another sentence.

## Report (in Chinese, short; write it to report_K.md)

1. English words and Chinese characters of your part, before and after.
2. Every number you moved from a short caption into its long caption, by figure.
3. Any term you had to choose that is not in the glossary, and your choice.
4. Anything in this brief you could not follow, and why.
