# Audit before submission: HC with and without mixing at 12 and 24 layers (read-only; no jobs, no pushes, no commits)

Repository: /scratch/gpfs/GROUP/USER/project/marin-pls, branch pls. scripts/della/pls_arch3_fetchable.sbatch has an uncommitted diff: cases arch3-hcid-d12, arch3-hcid-d24.

Question the runs answer: does learned residual mixing help shallow models and hurt deep ones? Reports: HC Table 3 (16 layers) +0.021 when frozen; mHC Table 1 +0.022; AttnRes Table 4 (16 blocks) +0.012. Here at 48 layers, on the current code, HC mixing 3.638, mixing = I 3.579 and 3.581 (held-out, last layer, W&B eval/L47/loss).

Planned jobs, all ordinary models (probes), through `wq submit --partial --project pls`:
- arch3-hcid-d12-probes, arch3-hcid-d24-probes: mixing = I. Approved by the owner.
- arch3-hc-d12-probes, arch3-hc-d24-probes: learned mixing on the current code. Not yet approved. The existing runs at these depths (-sep-bbfrozen-ahc-vs-d12, -d24) are on the earlier code.

Check, with file:line and evidence:
1. Each job name reaches exactly its knobs. Run ids (dry run tmp/pls_routing/dryrun_depth.sh): -sep-bbfrozen-ahc-hcid-fa4thd-vs-d12, -d24; -sep-bbfrozen-ahc-fa4thd-vs-d12, -d24. Do they collide with an existing run or a smoke marker? Does the launcher accept LAYERS=12 and 24 with HC_RES=identity? Is anything depth-dependent in hc (the 1 / sqrt(m) scaling, init, PDP, gradient checkpoint groups, HC_SUM_GAIN) that differs between the two arms?
2. Same steps, tokens, schedule, seed, data order as the 48-layer pair and as the old references? List every config difference between old and current code that could move the last layer: attention backend, cross-entropy path, HCMB, anything else, with the W&B config fields of -sep-bbfrozen-ahc-vs-d12 (tmp/pls_paths/cfg-sep-bbfrozen-ahc-vs-d12.json) against what a dry run gives. Is one comparison against the old references good enough, or are the two runs on the current code needed? Evidence on the size of a code change: HC 48 layers held-out 3.615 old, 3.638 current; Pre-LN 3.668, 3.674. Seeds on old code: HC 3.615, 3.656; mixing = I on current code 3.579, 3.581.
3. Reading rule proposed: difference = mixing = I minus learned mixing, held-out, last layer, same code, same depth. At or above +0.03: mixing helps at that depth. At or below -0.03: mixing hurts. Between: one pair cannot tell, propose a second seed. Prediction written now: +0.01 or more at 12 layers, near 0 at 24, -0.06 at 48. Is the rule defensible with the noise above? Say what you would change.
4. Wall time and memory: the job script asks 20 h and 4 GPUs. Is that right-sized for 12 and 24 layers? How does a --partial job behave if it ends well before its time? Any risk that a wq job at 12 layers leaves GPUs idle?
5. Production check: read tmp/pls_routing/real_step_depth.py and real_step_depth.txt. Does it test what it says?
6. Anything else that would make the result unreadable.

Do not run GPU code. CPU only on della-vis1 under `nice -n 19` with `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu`, never on the login node. Write /scratch/gpfs/GROUP/USER/tmp/pls_routing/audit_depth_report.md: blockers first, then the rest. Plain English, short sentences. Say clearly if there is no blocker.
