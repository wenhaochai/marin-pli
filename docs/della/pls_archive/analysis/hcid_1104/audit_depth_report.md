# Audit: HC with and without mixing at 12 and 24 layers (read-only)

Scope: the uncommitted diff of `scripts/della/pls_arch3_fetchable.sbatch` (cases `arch3-hcid-d12`, `arch3-hcid-d24`), the launcher, the HC code, the W&B configs in `tmp/pls_paths/`, the old job logs, and `real_step_depth.{py,txt}`. Nothing was submitted, committed or pushed. The dry run was run again on della-vis1 (CPU only, `nice -n 19`).

## Blockers

There are two blockers. Both are about how the results will be read. Neither is a fault in the job wiring.

**B1. The approved jobs alone cannot satisfy your reading rule.** The rule asks for "same code, same depth". The only learned-mixing runs at 12 and 24 layers are `-sep-bbfrozen-ahc-vs-d12` and `-d24`. They ran on commits a3210d3641 and ac5e91ebd1 (2026-10-24), and they differ from the planned hcid jobs in four ways (details in item 2): attention (JAX flash, not FA4 THD), microbatch (1, not 4), probe norm folding (off, not on), and two weeks of code. So you need `arch3-hc-d12-probes` and `arch3-hc-d24-probes` to apply the rule as written. They are cheap: about 5 h and 10 h on 4 GPUs (item 4). Submit them with the hcid pair, or change the rule to say "against the old reference, with a code margin of 0.01".

**B2. At 12 layers the planned test cannot detect the effect it is meant to test.** The papers report +0.012 to +0.022. Your own prediction for 12 layers is +0.01 or more. Both values fall inside the ±0.03 "cannot tell" band. So the most likely outcome at 12 layers is "cannot tell, run a second seed", and that is predictable now. Decide on the seeds before you submit (item 3), not after the first pair.

## 1. Job names, run ids, knobs

- Names to knobs: `arch3-hcid-d12-probes` gives `BB=1`, `DESIGN=arch3-hcid-d12` (sbatch:104), then `LAYERS=12 PDP=4 ARCH=hc HC_RES=identity` (sbatch:152). `-d24` is the same at sbatch:153. `arch3-hc-d{12,24}` are at sbatch:111-112 and differ only by the missing `HC_RES`. OK.
- Run ids (dry run again on della-vis1, no errors): `...-pls1-sep-bbfrozen-ahc-hcid-fa4thd-vs-d12`, `-d24`, `...-sep-bbfrozen-ahc-fa4thd-vs-d12`, `-d24`. None of them exists under `marin_store_big/speedrun/`. The existing d12/d24 directories have no `-fa4thd` tag, so there is no collision and no accidental resume. There is no `logs/pair_failed/*.fail` and no `logs/smoke_pass/*.key` for any d12 or d24 id. OK.
- The launcher accepts `LAYERS=12` and `LAYERS=24` with `HC_RES=identity`. The check at launcher:308 needs only `ARCH=hc`. OK.
- Depth-dependent parts of HC. Both arms are the same on every one:
  - The 1/sqrt(m) init scale depends on the stream count m = 4, not on depth (depth_arch_qwen3.py:331).
  - The read constants use sublayer mod m (depth_arch_qwen3.py:257-261). 2L = 24 and 48 are both multiples of 4.
  - The residual constant is the identity in both arms. `HC_SUM_GAIN` is read only by `sumshrink` and `preln_leak` (depth_arch_qwen3.py:252-254, 389, 454). These jobs do not reach it.
  - PDP: `PDP=4` is first overridden to 1 (launcher:395-396), then `HCMB` sets `PARAM_MAP_PDP=4 LEVANTER_CE_BTILED_MIN_BATCH=1048576` for every `ARCH=hc` job (pls_h100x8_single.sbatch:90-92). Both arms get microbatch 4 and the streaming CE.
  - Gradient checkpointing uses the same `scan_via` and policy for both arms (depth_arch_qwen3.py:733-743).
  - The only difference is the identity branch at depth_arch_qwen3.py:444-446. In that branch `res` and `w_res` get no gradient.
- Minor: the smoke's train-path readout gate runs only for `LAYERS=48` (pls_h100x8_single.sbatch:215). The d12/d24 jobs get only the eval-gap gate (≤ 0.005). The old d12/d24 runs had the same gap. You can extend `GATE=1.0` to every probes run.

## 2. Same training? Old against current code

Same in the old d12, d24 and d48 W&B configs and the current launcher: 4959 steps, batch 128 × 4096 (2.6B tokens), linear schedule with decay 0.8 and no warmup, lr 0.02 / adam 0.008, `seed` 0, `data_seed` 42, `permutation_type` linear, eval every 1000 steps. The data order is identical across depths and arms. OK.

Differences between the old reference `-sep-bbfrozen-ahc-vs-d12` (`cfg-...-vs-d12.json`) and what the current code gives:

| field | old d12/d24 | current (planned) | moves the last layer? |
|---|---|---|---|
| `model.attn_backend` | JAX_FLASH | FA4_THD | bf16-level. Pre-LN 48 layers: +0.001 (w500/w501); reruns 3.660 vs 3.678 |
| `trainer.per_device_parallelism` | 1 | 4 (HCMB) | summation order only |
| CE path | streaming (microbatch 1) | streaming (threshold 1048576) | same path |
| `model.pls_fold_norm` | off | on | probe heads only. The probe gradients still enter the global clip (max_grad_norm 1.0), so the backbone sees a rounding-level effect |
| `model.depth_hc_res` | absent | `identity` / absent | the treatment |
| code | a3210d3641 / ac5e91ebd1 | HEAD | `depth_arch_qwen3.py` +240 lines, `loss.py` +141, `attention.py` +204. The learned-mixing path at depth_arch_qwen3.py:468 looks unchanged |

Size of a code change: Pre-LN moves +0.006 to +0.010 (3.668/3.669 → 3.674/3.678). HC moves +0.023 (3.615 → 3.638), but its seed spread on the old code is 0.041 (3.615 / 3.656), so this shift cannot be separated from seed noise. My estimate of the code effect is ≤ 0.01, and HC's seed noise is about 0.02-0.03 (one standard deviation). One comparison against the old reference is therefore usable only with a 0.01 code margin, and seed noise dominates it anyway. The two current-code runs remove the margin at about 15 GPU-hours × 4. Recommendation: run them (B1).

The 48-layer "same code" pair is not on exactly the same code either. It is the comparison the 12/24 result will be set against:
- The learned run `-ahc-fa4thd-vs-d48` (3.638) is wq w388, `arch3-hc-newstack`: microbatch 1, b-tiled CE on 4096-token shards with the lse kernel, `PLS_SERIAL=1` (sbatch:135).
- The mixing = I runs (w811 and its seed 1) used microbatch 4 with the streaming CE.
- The learned seed-1 run on the HCMB path, `-ahc-fa4thd-vs-d48-s1` (w890), overflowed: held-out 9.885, state norm 1.5e20 (facts.md:8, 61).

So no learned HC run at 48 layers has finished cleanly on the exact path the new jobs use. State this in the write-up. An overflow at 12/24 layers is less likely, but it is possible. Decide now what happens if one occurs: exclude the run and run the next seed.

## 3. Reading rule

The noise on held-out, last layer:
- mixing = I, current code: 3.579 / 3.581. The spread is 0.002, so this arm is quiet.
- learned, old code: 3.615 / 3.656. The spread is 0.041. With n = 2 the standard deviation is about 0.029.

The standard deviation of a one-pair difference is therefore about 0.03. A ±0.03 band means about a one-sigma call. That is too loose to declare "helps" or "hurts", yet too tight to catch a +0.01 to +0.02 effect. The 48-layer result passes the rule: 3.579 against the learned mean of 3.636 gives −0.057 (range −0.036 to −0.077).

What I would change:
1. Use the same code and the same eval key, `eval/L{L-1}/loss`: L11 for 12 layers, L23 for 24 layers, not L47.
2. Plan 2 seeds of the learned arm at each depth from the start. The learned arm carries the noise; mixing = I does not need a second seed unless it surprises. Compare against the mean of the learned seeds.
3. Thresholds: ≥ +0.03 against the learned mean (about 1.5 standard errors with 2 seeds) = helps; ≤ −0.03 = hurts. Report anything in between as "no effect larger than 0.03 at this depth". At 12 layers that answer is informative: it bounds the literature's effect at this scale. Do not report it as a failure.
4. Read the trend across depth (12 / 24 / 48), not three separate verdicts. The question is about a sign change with depth. A monotone fall from 12 to 48, with 48 clearly negative, answers it even when 12 sits inside the band.
5. Write down now what happens if a learned run overflows (exclude it, run the next seed).

## 4. Wall time, memory, idle GPUs

- Old speeds at microbatch 1: d12 6.3-6.4 s/step (w83), d24 12.6-14.7 s/step (w81). HCMB gave HC 48 layers 24.8 → 19.4 s/step. mixing = I at 48 layers and microbatch 4 runs 14.2 s/step (w811).
- Estimates: d12 about 3.5-5 s/step → 5-7 h. d24 about 7-10 s/step → 10-14 h. Add about 15 min for the smoke and 5 evals. The 20 h request is too long for d12 (about 3×) and somewhat too long for d24.
- A `--partial` job that finishes early exits 0. wq marks it done, and fetch_pending refills its 4 GPUs (fetch_pending.sh:8, 459). No GPU sits idle as long as 4-GPU jobs wait in wq. The cost of the oversized `--time` is in planning: fetch_pending balances planned ends and requires `--time` + 10 min of host time left (fetch_pending.sh:10-22), so a 20 h request lands later and pairs badly. Pass `--time 8:00:00` for d12 and `14:00:00` for d24, if wq takes a `--time` override. The shared sbatch fixes 20 h for every arch3 design.
- Memory: HC 48 layers at microbatch 4 needs 21.7 GiB of temp, under half of the about 64 GiB pool. 12 and 24 layers need less. No risk.
- Idle risk inside the job: a 12-layer model keeps the GPUs busy every step. There is no long phase at 0%. The watchdog first-step limit (40 min) is far above the expected compile time. No risk.

## 5. Production check (`real_step_depth.py` / `.txt`)

**It did not pass, and it does not test what it says.** All four cases end in `AssertionError` on the backbone-gradient check (difference 0.19 at 12 layers, 0.39 at 24). The `res` / `w_res` checks never ran.

The cause is a bug in the test, not a probe leak:
- `PerLayerQwen3LMHeadModel.config` is `self.transformer.config` (qwen.py:233). `eqx.tree_at` puts `model.transformer` into `model0` (real_step_depth.py:52), so `model0` takes back the full config: `pls_weight` 1, detach on. `cfg0` is lost.
- `model0` has no aux heads, so its 11 or 23 readouts run through the shared final norm and lm_head (per_layer_qwen3.py:477-494).
- This explains the "last layer alone" value of 145.996 (12 × 12.2), not about 12.
- `g.transformer` contains `transformer.norm`. Its gradient differs because those readouts reach it. The difference grows with the readout count: 0.19 → 0.39 is about ×2 for 11 → 23 readouts.

The loss check is also mislabeled. "Loss 146.98 near ln V 11.76" compares a sum of L readouts with ln V. Per readout it is about 12.2, which is fine, but the script does not assert it.

It also differs from production:
- attention VANILLA, not FA4_THD
- T = 1024, B = 1, not 4096 × microbatch 4
- config from the old run, so `pls_fold_norm` is off. The production probes path (folded norm) is not exercised.

Fix:
- Compare only `g.transformer.layers` and `g.embeddings`, or build the reference as the same model with `aux_lm_heads` / `aux_norms` stop-gradiented, not through a config swap.
- Assert the mean readout loss is within about 1 nat of ln V.
- Set `pls_fold_norm=True`.
- Run it again before citing it.

The 48-layer hcid runs on the same code trained cleanly. Only `num_layers` changes. So I do not expect a real defect, but the check as written is a fail.

## 6. Other things that could make the result unreadable

- The eval key changes with depth (L11 / L23 / L47). `eval/loss` equals the last readout (the smoke gate checks this to 0.005). Use either one, consistently.
- The 12/24/48 comparison mixes a learned 48-layer reference from another CE path (w388) with mixing = I on the HCMB path. The 48-layer gap of −0.06 carries this caveat. Optionally, a learned HC 48 seed 2 on the HCMB path would also tell whether w890's overflow is rare.
- The papers' baselines differ in scale, tokens and metric (they report other losses or accuracy). Compare signs and the trend with depth, not magnitudes.
- Nothing else found: no collision, no shared output directory, no fail markers, no depth-dependent knob that differs between the arms.
