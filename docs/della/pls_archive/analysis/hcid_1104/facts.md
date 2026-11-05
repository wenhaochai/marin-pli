# Fact sheet, Residual Paths Help, Learned Routing Does Not

Held-out = eval/L47/loss, Paloma c4_en, last step. Train = mean of the last 50 steps of the last layer. All 48 layers, width 512, 2.6B tokens, ordinary models.

- pre_ln: apreln-vs-d48-g4: held-out 3.668, train 2.983; apreln-vs-d48-s1: held-out 3.669, train 2.983; apreln-fa4thd-vs-d48: held-out 3.674, train 2.988; apreln-vs-d48-rerun: held-out 3.66, train 2.987; apreln-fa4thd-vs-d48-rerun: held-out 3.678, train 2.988
- one_path_other: vs-d48-g4: held-out 3.691, train 2.991; alns-vs-d48-g4: held-out 3.682, train 2.996; akeel-vs-d48-g4: held-out 3.712, train 3.027
- hc_mixing: ahc-vs-d48-g4: held-out 3.615, train 2.944; ahc-vs-d48-s1: held-out 3.656, train 2.967; ahc-fa4thd-vs-d48: held-out 3.638, train 2.956
- hc_mixing_overflowed: ahc-fa4thd-vs-d48-s1: held-out 9.885, train 9.318
- hc_mixing_variants: ahc_pos-vs-d48-g4: held-out 3.643, train 2.954; ahc-hcrwsig-vs-d48: held-out 3.669, train 2.976; ahc-hcrwj-fa4thd-vs-d48: held-out 3.674, train 2.987; ahc-hcwsh-fa4thd-vs-d48: held-out 3.677, train 2.979
- hc_static_mixing: ahc-rstatic-vs-d48: held-out 3.609, train 2.936
- hc_no_mixing: ahc-hcid-fa4thd-vs-d48: held-out 3.579, train 2.931; ahc-hcid-fa4thd-vs-d48-s1: held-out 3.581, train 2.929
- hc_scalar: ahc-hcscalar-fa4thd-vs-d48: held-out 3.565, train 2.918
- hc_moves_keeps_sum: ahc-hcmove-fa4thd-vs-d48: held-out 3.578, train 2.925
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
