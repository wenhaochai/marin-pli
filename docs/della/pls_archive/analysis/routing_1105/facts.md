# Fact sheet, Deeper Models Gain from Residual Paths, Not from Learned Routing

Held-out = eval/L47/loss, Paloma c4_en, last step. Train = mean of the last 50 steps of the last layer. Runs below: 48 layers, width 512, 2.6B tokens, ordinary models. Depth section: 12, 24, 48 layers at the same budget. Gain from learned mixing = no mixing minus learned mixing, held-out. Gain from paths = Pre-LN minus no mixing, held-out. A positive gain means the thing helps. Stream sum norm and write share: median over 8,192 FineWeb-Edu tokens, final checkpoint.

- pre_ln: apreln-vs-d48-g4: held-out 3.668, train 2.983; apreln-vs-d48-s1: held-out 3.669, train 2.983; apreln-fa4thd-vs-d48: held-out 3.674, train 2.988; apreln-vs-d48-rerun: held-out 3.66, train 2.987; apreln-fa4thd-vs-d48-rerun: held-out 3.678, train 2.988
- one_path_other: vs-d48-g4: held-out 3.691, train 2.991; alns-vs-d48-g4: held-out 3.682, train 2.996; akeel-vs-d48-g4: held-out 3.712, train 3.027
- hc_mixing: ahc-vs-d48-g4: held-out 3.615, train 2.944; ahc-vs-d48-s1: held-out 3.656, train 2.967; ahc-fa4thd-vs-d48: held-out 3.638, train 2.956
- hc_mixing_overflowed: ahc-fa4thd-vs-d48-s1: held-out 9.885, train 9.318
- hc_mixing_variants: ahc_pos-vs-d48-g4: held-out 3.643, train 2.954; ahc-hcrwsig-vs-d48: held-out 3.669, train 2.976; ahc-hcrwj-fa4thd-vs-d48: held-out 3.674, train 2.987; ahc-hcwsh-fa4thd-vs-d48: held-out 3.677, train 2.979
- hc_static_mixing: ahc-rstatic-vs-d48: held-out 3.609, train 2.936
- hc_no_mixing: ahc-hcid-fa4thd-vs-d48: held-out 3.579, train 2.931; ahc-hcid-fa4thd-vs-d48-s1: held-out 3.581, train 2.929
- hc_scalar: ahc-hcscalar-fa4thd-vs-d48: held-out 3.565, train 2.918
- hc_moves_keeps_sum: ahc-hcmove-fa4thd-vs-d48: held-out 3.578, train 2.925
- hc_keeps_sum_signed: ahc-hcsum-hcf32-fa4thd-vs-d48: held-out 3.591, train 2.928
- hc_only_shrinks: ahc-hcshrink-fa4thd-vs-d48: held-out 3.687, train 3.03
- hc_half_dose_shrink: ahc-hcshrink-sp0.5-fa4thd-vs-d48: held-out 3.676, train 3.013
- hc_streams_with_mixing: ahc-hcs1-vs-d48: held-out 3.675, train 2.972; ahc-hcs2-vs-d48: held-out 3.632, train 2.954; ahc-hcs8-vs-d48: held-out 3.643, train 2.95
- mhc: amhc-vs-d48-g4: held-out 3.576, train 2.922
- mhc_linear: amhc-mhcrwlin-vs-d48: held-out 3.564, train 2.913
- mhc_no_mixing: amhc-mhcid-vs-d48: held-out 3.582, train 2.926
- mhc_no_mixing_linear: amhc-mhcid-mhcrwlin-fa4thd-vs-d48: held-out 3.576, train 2.918; amhc-mhcid-mhcrwlin-fa4thd-vs-d48-s1: held-out 3.579, train 2.915
- attnres: aattnres-vs-d48-g4: held-out 3.738, train 3.045; aattnres-vs-d48-s1-g4: held-out 3.642, train 2.955
- attnres_unit_rms: aattnres-vnorm-vs-d48: held-out 3.611, train 2.92
- attnres_static: aattnres-rstatic-vs-d48: held-out 3.562, train 2.913
- attnres_block: aattnres_block-vs-d48-g4: held-out 3.626, train 2.953
- moda: amoda-vs-d48-g4: held-out 3.622, train 2.948

## Differences (held-out, positive = the design with routing is worse)

- hc_remove_mixing_heldout: [0.036, 0.077, 0.059, 0.034, 0.075, 0.057]
- mhc_remove_mixing_heldout: 0.006
- mhc_linear_remove_mixing_heldout: [0.012, 0.015]
- hc_remove_token_dependence_heldout: [0.006, 0.047, 0.029]
- attnres_remove_token_dependence_heldout: [0.176, 0.08]
- attnres_unit_rms_vs_static: 0.049
- one_path_range: [3.66, 3.691]
- several_paths_no_mixing_range: [3.562, 3.582]
- hc_streams_with_mixing_heldout: {'1': 3.675, '2': 3.632, '4': [3.615, 3.656, 3.638], '8': 3.643}
- repeat_spread: {'hc_no_mixing': 0.002, 'mhc_no_mixing_linear': 0.003, 'pre_ln': 0.018}

## Failures

- aattnres-vs-d48-g4: {'events': 10, 'largest_excess': 7.07, 'step_of_largest': 881}
- aattnres-vs-d48-s1-g4: {'events': 5, 'largest_excess': 10.44, 'step_of_largest': 626}
- aattnres-rstatic-vs-d48: {'events': 0, 'largest_excess': 0.1, 'step_of_largest': 1391}
- aattnres-vnorm-vs-d48: {'events': 0, 'largest_excess': 0.1, 'step_of_largest': 1391}
- ahc-vs-d48-g4: {'events': 0, 'largest_excess': 0.1, 'step_of_largest': 1391}
- ahc-vs-d48-s1: {'events': 0, 'largest_excess': 0.12, 'step_of_largest': 1391}
- ahc-fa4thd-vs-d48: {'events': 0, 'largest_excess': 0.1, 'step_of_largest': 1391}
- ahc-fa4thd-vs-d48-s1: {'events': 2, 'largest_excess': 1.5, 'step_of_largest': 4581}
- ahc-hcid-fa4thd-vs-d48: {'events': 0, 'largest_excess': 0.11, 'step_of_largest': 1391}
- ahc-hcid-fa4thd-vs-d48-s1: {'events': 0, 'largest_excess': 0.11, 'step_of_largest': 1391}
- amhc-vs-d48-g4: {'events': 0, 'largest_excess': 0.11, 'step_of_largest': 1391}
- amhc-mhcid-vs-d48: {'events': 0, 'largest_excess': 0.11, 'step_of_largest': 1391}
- apreln-vs-d48-g4: {'events': 0, 'largest_excess': 0.11, 'step_of_largest': 1391}
- apreln-vs-d48-s1: {'events': 0, 'largest_excess': 0.11, 'step_of_largest': 1391}
- apreln-fa4thd-vs-d48: {'events': 0, 'largest_excess': 0.1, 'step_of_largest': 1391}
- apreln-vs-d48-rerun: {'events': 0, 'largest_excess': 0.1, 'step_of_largest': 1391}
- apreln-fa4thd-vs-d48-rerun: {'events': 0, 'largest_excess': 0.1, 'step_of_largest': 1391}

## Measured elsewhere

- mhc_mixing_on_tokens: {'mean_offdiag_per_row': 0.029, 'p99': 0.333, 'max': 0.807, 'loss_as_trained': 2.873, 'loss_every_A_replaced_by_I': 3.48, 'source': 'tmp/pls_hcid/mhc_mix_full.py, 16,380 FineWeb-Edu tokens'}
- hc_overflow: {'step_first_rise': 4408, 'step_5_nats': 4420, 'final_train': 9.318, 'final_heldout': 9.885, 'median_state_norm_L48': 1.5e+20, 'float32_sum_of_squares_overflow_at': 1.8e+19, 'share_tokens_at_uniform_loss': 0.701, 'ln_V': 11.76, 'grad_norm_median_steps_4400_4599': 0.97, 'source': 'docs pls.md item 24, tmp/pls_hcid/state_margin.py, ce_by_target.py'}
- hc_state_norm_median_L48: {'hc_mixing_other_run': 12000000.0, 'hc_no_mixing': 667, 'source': 'res_gain tables tmp/pls_hcid/res_gain_all.txt, res_gain_b.txt (no mixing: 6.674e+02)'}
- hc_layer12_mixing_row_sums_mlp: [0.81, 0.1, -0.47, 0.86]
- source_row_sums: tmp/pls_hcid/row_sums.py, HC ordinary final checkpoint

## Depth

- 12 layers: {'heldout_one_path_preln': [3.845], 'heldout_no_mixing': [3.784], 'heldout_learned_mixing': [3.768], 'gain_from_learned_mixing': 0.016, 'gain_from_paths': 0.061, 'gain_from_learned_mixing_by_step': {'1000': 0.067, '2000': 0.034, '3000': 0.036, '4000': 0.022, '4958': 0.016}, 'median_stream_sum_norm_last_layer': {'learned_mixing': 114.0, 'no_mixing': 311.0}, 'orders_of_magnitude_of_the_mixing_scale_change': 1.7, 'last_write_over_new_sum': {'learned_mixing': 0.3, 'no_mixing': 0.4}}
- 24 layers: {'heldout_one_path_preln': [3.745], 'heldout_no_mixing': [3.66], 'heldout_learned_mixing': [3.68], 'gain_from_learned_mixing': -0.02, 'gain_from_paths': 0.085, 'gain_from_learned_mixing_by_step': {'1000': 0.033, '2000': 0.005, '3000': -0.003, '4000': -0.016, '4958': -0.02}, 'median_stream_sum_norm_last_layer': {'learned_mixing': 455.0, 'no_mixing': 398.0}, 'orders_of_magnitude_of_the_mixing_scale_change': 1.9, 'last_write_over_new_sum': {'learned_mixing': 0.1, 'no_mixing': 0.29}}
- 48 layers: {'heldout_one_path_preln': [3.668, 3.669, 3.674], 'heldout_no_mixing': [3.579, 3.581], 'heldout_learned_mixing': [3.638, 3.615, 3.656], 'gain_from_learned_mixing': -0.056, 'gain_from_paths': 0.09, 'gain_from_learned_mixing_by_step': {'1000': -0.032, '2000': -0.023, '3000': -0.03, '4000': -0.052, '4958': -0.056}, 'median_stream_sum_norm_last_layer': {'learned_mixing': 10289532.0, 'no_mixing': 667.0}, 'orders_of_magnitude_of_the_mixing_scale_change': 4.6, 'last_write_over_new_sum': {'learned_mixing': 1.2e-05, 'no_mixing': 0.29}}

## Four-stream arms at 48 layers: scale of the stream sum

- I/48: {'heldout': 3.579, 'orders_of_magnitude_of_the_mixing_scale_change': 0.0, 'median_stream_sum_norm_last_layer': 667.0, 'last_write_over_new_sum': 0.29}
- HC moves/48: {'heldout': 3.578, 'orders_of_magnitude_of_the_mixing_scale_change': 0.0, 'median_stream_sum_norm_last_layer': 562.0, 'last_write_over_new_sum': 0.23}
- HC keeps sum/48: {'heldout': 3.591, 'orders_of_magnitude_of_the_mixing_scale_change': 0.0, 'median_stream_sum_norm_last_layer': 725.0, 'last_write_over_new_sum': 0.28}
- HC scalar/48: {'heldout': 3.565, 'orders_of_magnitude_of_the_mixing_scale_change': 1.5, 'median_stream_sum_norm_last_layer': 363.0, 'last_write_over_new_sum': 0.13}
- HC static/48: {'heldout': 3.609, 'orders_of_magnitude_of_the_mixing_scale_change': 1.8, 'median_stream_sum_norm_last_layer': 1465.0, 'last_write_over_new_sum': 0.099}
- HC half dose/48: {'heldout': 3.676, 'orders_of_magnitude_of_the_mixing_scale_change': 3.3, 'median_stream_sum_norm_last_layer': 1612.0, 'last_write_over_new_sum': 0.32}
- HC only shrinks/48: {'heldout': 3.687, 'orders_of_magnitude_of_the_mixing_scale_change': 6.5, 'median_stream_sum_norm_last_layer': 1038.0, 'last_write_over_new_sum': 0.35}
- HC/48: {'heldout': 3.638, 'orders_of_magnitude_of_the_mixing_scale_change': 4.6, 'median_stream_sum_norm_last_layer': 10289532.0, 'last_write_over_new_sum': 1.2e-05}

## Papers (full detail in related.md)

- hc_table3: {'freeze_width_connections_loss_rise': 0.021, 'layers': 16, 'tokens': '500B', 'model': 'OLMo-1B'}
- mhc_table1: {'identity_mixing_loss_rise': 0.022, 'layers': 'not stated'}
- attnres_table4: {'input_independent_loss_rise': 0.012, 'layers': 16, 'tokens': '87.9B'}
- muddformer_table5: {'static_perplexity_rise': 0.35, 'layers': 24, 'tokens': '7B'}
- source: related.md

## Derived

- hc_layer12_mean_row_sum: 0.33
- hc_scalar_stream_sum_norm_last_layer_48: 363.0
- params_M_without_probe_heads: {'12': 177, '24': 223, '48': 314}
- width: 512
- mlp: 1792
- tokens: 2.6B
- steps: 4959
- parameter_spread_across_designs_48_layers_percent: 1.5
- parameter_spread_within_a_comparison_48_layers_percent: 0.1
- learned_mixing_48_product_of_gains: {'lowest_log10': -4.46, 'at_layer': 22.5, 'at_last_layer_log10': 0.14}
- gain_over_no_mixing_same_depth_48: {'one_learned_number': 0.015, 'constant_mixing': -0.029, 'half_dose_shrink': -0.096, 'only_shrinks': -0.107, 'moves_content_nonnegative': 0.002, 'moves_content_signed': -0.011}
