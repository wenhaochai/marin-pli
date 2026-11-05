# Audit before submission: one-stream decomposition (read-only review; do not submit, cancel or push anything)

Repository: /scratch/gpfs/GROUP/USER/project/marin-pls, branch pls, working tree (scripts/della/pls_arch3_fetchable.sbatch and scripts/della/depth_arch_s1_test.py are not committed yet; read `git diff` and the file).

Five ordinary-model runs (probes only, no gradient from the probes into the backbone) are about to be submitted through wq with --partial --prio 10, each stopped by hand at about step 700 of 4,959 by `wq cancel`:
- arch3-hcs1f-probes: ARCH=hc HC_STREAMS=1
- arch3-hcs1id-probes: ARCH=hc HC_STREAMS=1 HC_RES=identity
- arch3-hcs1scalar-probes: ARCH=hc HC_STREAMS=1 HC_RES=scalar
- arch3-hcs1static-probes: ARCH=hc HC_STREAMS=1 ROUTE=static
- arch3-plleak01-probes: ARCH=preln PRELN_LEAK=1 SHRINK_POWER=0.1

The question they answer: with one stream the update is h <- g h + w F(RN(r h)). Which of g and w gives the layer 12 probe its low loss, compared with Pre-LN (no g, no w)? Reference values, train cross-entropy at layer 12, steps 500-599: Pre-LN on the same backend 5.219 (run -sep-bbfrozen-apreln-fa4thd-vs-d48), HC one stream on the older backend 4.296 (run -sep-bbfrozen-ahc-hcs1-vs-d48). Fixed reading rule: within 0.3 of arch3-hcs1f = the knob is not needed; within 0.3 of Pre-LN = the knob is needed; between = partial.

Read: experiments/references/della_muonh_qwen3_scaling.py (env to config, run id, how HC_STREAMS, HC_RES, ROUTE, PRELN_LEAK, SHRINK_POWER map), experiments/references/depth_arch_qwen3.py (hyper_step, residual_step, _hc_constants, validation near line 700), experiments/references/per_layer_qwen3.py (config fields, _layer, probes), scripts/della/pls_h100x8_single.sbatch (what it unsets, the smoke, the training limit, how a resumed segment finds its run), scripts/della/pls_arch3_fetchable.sbatch, scripts/della/depth_arch_s1_test.py. Dry run: /scratch/gpfs/GROUP/USER/tmp/pls_hcid/dryrun_s1.sh. Real-size check: /scratch/gpfs/GROUP/USER/tmp/pls_hcid/real_step_s1.py. Wq rules: /scratch/gpfs/GROUP/USER/tools/wq/README.md. Do not run any GPU code. You may run the CPU test and the dry run on della-vis1 under `nice -n 19` with `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu` (never on the login node).

Check and report, with file:line and evidence:
1. Does each job name reach exactly the knobs listed? Is any env var left over from another design (the unset list in pls_h100x8_single.sbatch)? Do the run ids collide with any existing run?
2. Does each arm compute what its name says with m = 1: is g really 1 in identity; really one static number per sublayer in scalar; are r, g, w constants in static? With one stream read r only sets the sign and scale of the input, and RN removes the scale: is that right, and does this make any arm mean something other than stated? Does the 1 / sqrt(m) scaling of o_proj and down_proj at init (m = 1 versus Pre-LN) differ between arch3-hcs1id and Pre-LN in a way that would confound the comparison? Report every other difference between arch3-hcs1id and Pre-LN (init, norms, learning-rate groups, weight decay on the routing parameters).
3. Is PRELN_LEAK=1 SHRINK_POWER=0.1 correct: the gains used, their index by sublayer, the dtype? Is the comparison with Pre-LN clean: same seed, same data order, same optimizer groups?
4. Are the references comparable: same data, batch, schedule, seed, probes? HC one stream's reference is on the older attention backend; is the run arch3-hcs1f enough to correct that?
5. Are there failure modes a 30-step smoke and a check at step 100 would miss? How should the job be stopped so that its checkpoint stays usable for a later resume? Check that `wq cancel` of a running --partial job keeps the last checkpoint and that a later `wq requeue` resumes it.
6. Weak spots in the reading rule: is step 500-599 late enough, is 0.3 nats defensible given that two runs of HC with the same seed and different attention backends differ by 0.27 at layer 12 at steps 500-599?
7. Anything in the tests that does not test what it claims (a reference that calls the code under test, an assertion that cannot fail).

Write the report to /scratch/gpfs/GROUP/USER/tmp/pls_hcid/audit_report_s1.md: blockers first, then everything else, each with evidence. Say clearly if there is no blocker. Plain English, short sentences.
