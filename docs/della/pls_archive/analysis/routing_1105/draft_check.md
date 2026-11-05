# Number Check for `draft.json` against `facts.md` and `facts.json`

Every number appearing in `draft.json` (both English and Chinese) is listed below by section, along with its exact source in `facts.md` / `facts.json` (and `related.md` where `facts.md` points to `related.md` under `## Papers`). Figures `cap` fields contain zero numbers, as required.

## Standfirst
- `12`, `24`, `48` (layers): `facts.md` lines 3, 71–73 (`## Depth: 12 layers, 24 layers, 48 layers`).

## Section 2: `motivation`
- `48` (layers): `facts.md` line 3 (`Runs below: 48 layers`).
- `0.03 to 0.08` (nats, HC without learned mixing vs with learned mixing): `facts.md` line 30 (`hc_remove_mixing_heldout: [0.036, 0.077, 0.059, 0.034, 0.075, 0.057]`, which rounds to `0.03` to `0.08`).
- `0.006` (nats, mHC without mixing vs mHC): `facts.md` line 31 (`mhc_remove_mixing_heldout: 0.006`).
- `0.08 to 0.18` (nats, AttnRes with weights that ignore the token vs AttnRes): `facts.md` line 34 (`attnres_remove_token_dependence_heldout: [0.176, 0.08]`, i.e. `0.08` to `0.18`).
- `24` (up to 24 layers in papers): `facts.md` line 80 (`muddformer_table5: {'static_perplexity_rise': 0.35, 'layers': 24, 'tokens': '7B'}`).
- `0.012 to 0.022` (loss rise in papers when removing routing): `facts.md` lines 77–79 (`attnres_table4: 0.012`, `hc_table3: 0.021`, `mhc_table1: 0.022`).
- `0.021` and `16` (HC Table 3 loss rise at 16 layers): `facts.md` line 77 (`hc_table3: {'freeze_width_connections_loss_rise': 0.021, 'layers': 16, 'tokens': '500B', 'model': 'OLMo-1B'}`).
- `0.022` (mHC Table 1 identity mixing loss rise): `facts.md` line 78 (`mhc_table1: {'identity_mixing_loss_rise': 0.022, 'layers': 'not stated'}`).
- `0.012` and `16` (AttnRes Table 4 input-independent loss rise at 16 blocks): `facts.md` line 79 (`attnres_table4: {'input_independent_loss_rise': 0.012, 'layers': 16, 'tokens': '87.9B'}`).

## Section 3: `background`
- `4` ($m = 4$ streams): `argon_task.md` line 16 / `related.md` line 12 (`n = 4`).
- `1` (rows and columns of doubly stochastic matrix sum to 1): definition of doubly stochastic matrix (`argon_task.md` line 17).
- `12` and `48` (layer 12 of the 48-layer model): `facts.md` line 66 (`hc_layer12_mixing_row_sums_mlp`) and line 3 (`48 layers`).
- `0.81`, `0.10`, `-0.47`, `0.86` (four row sums in the MLP sublayer of layer 12): `facts.md` line 66 (`hc_layer12_mixing_row_sums_mlp: [0.81, 0.1, -0.47, 0.86]`).

## Section 4: `setup`
- `12`, `24`, `48` (layers): `facts.md` lines 71–73 (`## Depth`).
- `512` (width): `facts.md` line 88 (`width: 512`).
- `1792` (MLP width): `facts.md` line 89 (`mlp: 1792`).
- `177M`, `223M`, `314M` (parameters without probe heads): `facts.md` line 87 (`params_M_without_probe_heads: {'12': 177, '24': 223, '48': 314}`).
- `4,959` (steps): `facts.md` line 91 (`steps: 4959`).
- `2.6B` (tokens): `facts.md` line 90 (`tokens: 2.6B`).

## Section 5: `depth`
- `+0.016` (gain from learned mixing at 12 layers): `facts.md` line 71 (`12 layers: ... 'gain_from_learned_mixing': 0.016`).
- `-0.020` (gain from learned mixing at 24 layers): `facts.md` line 72 (`24 layers: ... 'gain_from_learned_mixing': -0.02`).
- `-0.056` (gain from learned mixing at 48 layers): `facts.md` line 73 (`48 layers: ... 'gain_from_learned_mixing': -0.056`).
- `0.021` at `16` layers (HC paper): `facts.md` line 77 (`hc_table3: {'freeze_width_connections_loss_rise': 0.021, 'layers': 16}`).
- `0.012` at `16` blocks (AttnRes paper): `facts.md` line 79 (`attnres_table4: {'input_independent_loss_rise': 0.012, 'layers': 16}`).
- `+0.01` and `0` (pre-registered predictions before 12- and 24-layer trainings): `argon_task.md` line 31 (`at least +0.01 and about 0`).
- `5` (five evaluation points): `facts.md` lines 71–73 (`gain_from_learned_mixing_by_step` has 5 keys: `1000`, `2000`, `3000`, `4000`, `4958`).
- `4,958` and `1,000` (steps): `facts.md` lines 71–73 (`gain_from_learned_mixing_by_step` keys `'1000'` and `'4958'`).
- `+0.067` to `+0.016` (12 layers step 1000 to step 4958): `facts.md` line 71 (`'1000': 0.067, ..., '4958': 0.016`).
- `+0.033` to `-0.020` (24 layers step 1000 to step 4958): `facts.md` line 72 (`'1000': 0.033, ..., '4958': -0.02`).
- `-0.032` to `-0.056` (48 layers step 1000 to step 4958): `facts.md` line 73 (`'1000': -0.032, ..., '4958': -0.056`).
- `0.03` (nats threshold for no effect): `argon_task.md` line 31 (`We read a difference under 0.03 nats as no effect`).

## Section 6: `controls`
- `12` and `24` (layer pairs): `facts.md` lines 71–72.
- `48` (layers): `facts.md` line 73.
- `0.2` (percent parameter count difference at 48 layers): `argon_task.md` line 32 and `facts.json` `params_M` (`(3404.9 - 3400.2) / 3400.2 = 0.138% < 0.2%`).
- `3,400.2M` to `3,404.9M` (`params_M` across Pre-LN, HC, mHC, AttnRes at 48 layers): `facts.json` lines 8 (`pre_ln`: `3400.2`), 60 (`hc_mixing`: `3400.5`), 196 (`mhc`: `3404.9`), 234 (`attnres`: `3400.3`), 256 (`attnres_static`: `3400.2`).
- `16,380` (FineWeb-Edu tokens): `facts.md` line 63 (`source: 'tmp/pls_hcid/mhc_mix_full.py, 16,380 FineWeb-Edu tokens'`).
- `3` percent / `0.029` (mean off-diagonal weight per row in mHC): `facts.md` line 63 (`'mean_offdiag_per_row': 0.029`).
- `2.873` to `3.480` (mHC loss on FineWeb-Edu tokens as trained vs with every $A$ replaced by $I$): `facts.md` line 63 (`'loss_as_trained': 2.873, 'loss_every_A_replaced_by_I': 3.48`).
- `10.44` (largest excess training loss jump in AttnRes, in nats): `facts.md` line 44 (`aattnres-vs-d48-s1-g4: {'events': 5, 'largest_excess': 10.44, 'step_of_largest': 626}`).
- `0.10` (largest excess with static weights in AttnRes, in nats): `facts.md` line 45 (`aattnres-rstatic-vs-d48: {'events': 0, 'largest_excess': 0.1, 'step_of_largest': 1391}`).
- `0` (spike events with unit RMS in AttnRes): `facts.md` line 46 (`aattnres-vnorm-vs-d48: {'events': 0, 'largest_excess': 0.1, 'step_of_largest': 1391}`).
- `3.611` (held-out loss of AttnRes with unit RMS): `facts.md` line 23 (`attnres_unit_rms: aattnres-vnorm-vs-d48: held-out 3.611`).
- `3.562` (held-out loss of AttnRes with static weights): `facts.md` line 24 (`attnres_static: aattnres-rstatic-vs-d48: held-out 3.562`).

## Section 7: `mechanism`
- `12` and `48` (layer 12 of the 48-layer model): `facts.md` line 85.
- `0.33` / a third (mean of the four row sums at layer 12): `facts.md` line 85 (`hc_layer12_mean_row_sum: 0.33`).
- `3.578` (held-out loss of mixing that moves content and keeps stream sum, non-negative weights): `facts.md` line 13 (`hc_moves_keeps_sum: ahc-hcmove-fa4thd-vs-d48: held-out 3.578`).
- `3.591` (held-out loss of mixing that keeps stream sum with signed weights): `facts.md` line 14 (`hc_keeps_sum_signed: ahc-hcsum-hcf32-fa4thd-vs-d48: held-out 3.591`).
- `3.579` and `3.581` (held-out loss of HC with no mixing): `facts.md` line 11 (`hc_no_mixing: ahc-hcid-fa4thd-vs-d48: held-out 3.579 ... ahc-hcid-fa4thd-vs-d48-s1: held-out 3.581`).
- `3.687` (held-out loss of mixing that only shrinks): `facts.md` line 15 (`hc_only_shrinks: ahc-hcshrink-fa4thd-vs-d48: held-out 3.687`).
- `3.676` (held-out loss of half-dose shrink): `facts.md` line 16 (`hc_half_dose_shrink: ahc-hcshrink-sp0.5-fa4thd-vs-d48: held-out 3.676`).
- `3.615` to `3.656` (held-out loss range of HC with learned mixing): `facts.md` line 7 (`hc_mixing: ahc-vs-d48-g4: held-out 3.615 ... ahc-vs-d48-s1: held-out 3.656; ahc-fa4thd-vs-d48: held-out 3.638`).
- `3.565` (held-out loss of HC with one learned scalar per sublayer): `facts.md` line 12 (`hc_scalar: ahc-hcscalar-fa4thd-vs-d48: held-out 3.565`).
- `363` (stream sum norm at last layer of 48-layer scalar HC): `facts.md` line 86 (`hc_scalar_stream_sum_norm_last_layer_48: 363.0`).
- `667` (stream sum norm at last layer of 48-layer HC with no mixing): `facts.md` line 65 and line 73 (`'no_mixing': 667.0`).

## Section 8: `scale`
- `1.7`, `1.9`, `4.6` (orders of magnitude of the mixing scale change at 12, 24, 48 layers): `facts.md` lines 71–73 (`'orders_of_magnitude_of_the_mixing_scale_change': 1.7` at 12, `1.9` at 24, `4.6` at 48).
- `114`, `455`, `10 million` (`10,289,532`) (median stream sum norm at last layer with learned mixing at 12, 24, 48 layers): `facts.md` lines 71–73 (`'median_stream_sum_norm_last_layer': {'learned_mixing': 114.0}` at 12, `455.0` at 24, `10289532.0` at 48).
- `311`, `398`, `667` (median stream sum norm at last layer with no mixing at 12, 24, 48 layers): `facts.md` lines 71–73 (`'no_mixing': 311.0` at 12, `398.0` at 24, `667.0` at 48).
- `0.30`, `0.10`, `0.00001` (`0.000012`) (last write over new sum with learned mixing at 12, 24, 48 layers): `facts.md` lines 71–73 (`'last_write_over_new_sum': {'learned_mixing': 0.3}` at 12, `0.1` at 24, `1.2e-05` at 48).
- `0.40`, `0.29`, `0.29` (last write over new sum with no mixing at 12, 24, 48 layers): `facts.md` lines 71–73 (`'no_mixing': 0.4` at 12, `0.29` at 24, `0.29` at 48).
- `4,408` (step of first rise in overflowed run): `facts.md` line 64 (`'step_first_rise': 4408`).
- `1.8e19` ($1.8 \times 10^{19}$, float32 sum of squares overflow threshold): `facts.md` line 64 (`'float32_sum_of_squares_overflow_at': 1.8e+19`).
- `70` percent (`70.1` percent, share of tokens at uniform loss): `facts.md` line 64 (`'share_tokens_at_uniform_loss': 0.701`).
- `9.885` (final held-out loss of overflowed HC run): `facts.md` lines 8 and 64 (`'final_heldout': 9.885`).
- `0.97` (median gradient norm steps 4400–4599): `facts.md` line 64 (`'grad_norm_median_steps_4400_4599': 0.97`).

## Section 9: `paths`
- `+0.061`, `+0.085`, `+0.090` (gain from paths at 12, 24, 48 layers, in nats): `facts.md` lines 71–73 (`'gain_from_paths': 0.061` at 12, `0.085` at 24, `0.09` at 48).
- `3.660` to `3.691` (one-path held-out loss range at 48 layers): `facts.md` line 36 (`one_path_range: [3.66, 3.691]`).
- `3.562` to `3.582` (several paths with no mixing held-out loss range at 48 layers): `facts.md` line 37 (`several_paths_no_mixing_range: [3.562, 3.582]`).
- `1`, `2`, `4`, `8` (stream counts in HC with mixing): `facts.md` line 38 (`hc_streams_with_mixing_heldout: {'1': 3.675, '2': 3.632, '4': [3.615, 3.656, 3.638], '8': 3.643}`).
- `3.675` (held-out loss of 1-stream HC): `facts.md` lines 17 and 38 (`'1': 3.675`).
- `3.632` (held-out loss of 2-stream HC): `facts.md` lines 17 and 38 (`'2': 3.632`).
- `3.615` to `3.656` (held-out loss range of 4-stream HC): `facts.md` lines 7 and 38 (`'4': [3.615, 3.656, 3.638]`).
- `3.643` (held-out loss of 8-stream HC): `facts.md` lines 17 and 38 (`'8': 3.643`).

## Section 10: `limitations`
- `2.6B` (our token budget): `facts.md` line 90 (`tokens: 2.6B`).
- `500B` (paper token budget): `facts.md` line 77 (`hc_table3: ... 'tokens': '500B'`).
- `177M` to `314M` (model parameters across 12 to 48 layers): `facts.md` line 87 (`params_M_without_probe_heads: {'12': 177, '24': 223, '48': 314}`).
- `12` and `24` (layers): `facts.md` lines 71–72.

## Section 11: `takeaways`
- `12`, `24`, `48` (layers): `facts.md` lines 71–73.
- `+0.016` (gain from learned mixing at 12 layers): `facts.md` line 71.
- `-0.056` (gain from learned mixing at 48 layers): `facts.md` line 73.

## Section 12: `open-questions`
- `16` (layers in HC and AttnRes papers): `facts.md` lines 77 and 79.
- `500B` (tokens in HC paper): `facts.md` line 77.
- `26B` (ten times our `2.6B` token budget): `argon_task.md` line 38 (`ten times the tokens`) and `facts.md` line 90 (`2.6B`).
- `48` (layers): `facts.md` line 73.

## Section 13: `related-work`
- `110` (ResNet-110 layers in He et al. 2016): `related.md` line 88 (`ResNet-110 on CIFAR-10`).
- `1x1` ($1 \times 1$ convolution shortcut in He et al. 2016): `related.md` line 94 (`1×1 conv shortcut`).
- `0.77`, `48`, `72` (DenseFormer perplexity drop of 0.77 at 48 blocks matching a 72-block baseline): `related.md` lines 125–129 (`18.61` to `17.84` = `0.77` perplexity at `48` blocks, matching `72` blocks).
- `0.021`, `16`, `500B` (HC Table 3 on 16-layer OLMo-1B at 500B tokens): `facts.md` line 77 and `related.md` lines 12, 18.
- `0.022` (mHC Table 1 identity mixing loss rise): `facts.md` line 78 and `related.md` line 32.
- `24` (mHC-lite 24-layer L model where HC loses its gain): `related.md` lines 42, 47.
- `0.012`, `16`, `87.9B` (AttnRes Table 4 input-independent loss rise at 16 blocks on 87.9B tokens): `facts.md` line 79 and `related.md` lines 58–62.
- `0.08 to 0.18` (our 48-layer AttnRes loss drop from removing token dependence): `facts.md` line 34 (`[0.176, 0.08]`).
- `0.35`, `24`, `7B` (MUDDFormer Table 5 static perplexity rise at 24 layers on 7B tokens): `facts.md` line 80 and `related.md` lines 136, 146.
- `70` and `1` (DepthBench up to 70 layers; rank-1 average $U = (1/m)\mathbf{1}\mathbf{1}^\top$): `related.md` lines 76, 79.
- `12`, `24`, `48` (our three depths): `facts.md` lines 71–73.
