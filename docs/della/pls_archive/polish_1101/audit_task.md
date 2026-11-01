# Task: audit a polish pass of the page "Quantifying the Per-layer Contribution of LLMs" for changes of meaning

The owner asked Argon (another model) to polish the English and rewrite the Chinese of
https://wenhaochai.com/exps/per-layer-contribution.html, a bilingual experiment page. Five Argon writers each polished
one part, and five auditors now check one part each; the prompt tells you which part K is yours. A merge script has
already checked numbers, links and tags mechanically. Your job is the check a script cannot do: did any unit change its
meaning? Do not commit, push or submit jobs, and do not edit the page or the part files. You
run headless: keep calling tools until audit_K.md and overrides_K.json are written, and never end your turn with a plan or a
status message. You need no python; if you run any, run it on della-vis2
(`ssh della-vis2 'CUDA_VISIBLE_DEVICES= nice -n 19 python3 -I ...'`), never on this login node.

## Files

- `/scratch/gpfs/GROUP/USER/tmp/pls_polish/review_K.md`: every changed unit of your part, OLD then NEW. A unit is one text of
  the page: a figure's short caption (printed under the figure), its long caption (the "Details" fold under the figure),
  an answer (printed under a question), a paragraph, a setup-panel sentence, a note or a related-work item. Units come in
  pairs: `uNNNen` is the English and `uNNNzh` the Chinese of the same text. A unit marked "accepted despite: figure
  numbers lost" had a number removed that the same figure's captions repeated; the number must still appear in that
  figure's other caption (check it), and nothing else may be lost.
- `/scratch/gpfs/GROUP/USER/tmp/pls_polish/page_before.html`: the page before the pass, for context (the questions,
  figure titles and neighbouring text).
- `/scratch/gpfs/GROUP/USER/tmp/pls_polish/argon_task.md`: the brief the writers followed, with its glossary. Its
  rules allowed this: rephrase, reorder, merge and split sentences; cut repeated words; move a number or a caveat from a
  figure's short caption into the long caption of the same figure.

## What to check, for every changed unit

1. Meaning. The NEW text keeps every claim, hedge, qualifier, comparison and condition of the OLD text, and adds none.
   Watch for: a claim made stronger or weaker ("about half" becoming "half", "can" becoming "does", a hedge dropped, a
   "within Pre-LN" or "at 130m" scope dropped); a comparison turned around or attached to the wrong design, layer or
   model; a number now attached to a different quantity; a cause stated where the old text stated a correlation; a
   sentence of the old text lost (for a figure, check the short and long caption together, since text may move between
   them).
2. Pairs. The English and the Chinese of a pair say the same thing.
3. Names. Every setup, design or series name still matches the name the figure prints (titles in page_before.html;
   legends in `/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/assets/data/plc-figures.js`), and the glossary of argon_task.md
   is followed (one name per thing).
4. Severe Chinese problems only: a sentence a Chinese ML researcher cannot parse, or one that reads as a wrong statement.
   Do not flag matters of taste.

## Output

1. `/scratch/gpfs/GROUP/USER/tmp/pls_polish/audit_K.md`, in Chinese: one entry per problem: the unit id, the kind of
   problem (meaning / pair / name / Chinese), the OLD and NEW wording that differ, and why it matters. End with a count
   of units checked and problems found.
2. `/scratch/gpfs/GROUP/USER/tmp/pls_polish/overrides_K.json`: a JSON object mapping each unit id with a problem to
   its corrected full text. Make the smallest edit to the NEW text that fixes the problem; keep the polished wording
   everywhere else. Keep every number, tag and link of the unit exactly as in NEW, and keep each text on one line. If a
   unit cannot be fixed by a small edit, map it to its OLD text instead and say so in audit.md. Write `{}` if there are
   no problems.
