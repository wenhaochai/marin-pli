"""CPU check: the optimizer schedule that della_muonh_qwen3_scaling builds for LR_SCHEDULE, against the formula."""
import os, sys, importlib, unittest.mock as mock
import numpy as np
for sched in (sys.argv[1] if len(sys.argv) > 1 else "",):
    os.environ.update(SIZE="130m", LAYERS="12", LR_SCHEDULE=sched)
    import experiments.references.della_muonh_qwen3_scaling as m
    step = m.muonh_qwen3_run("130m")
    cfg = step.build_config(mock.MagicMock()) if hasattr(step, "build_config") else None
    tc = cfg.train_config
    opt = tc.optimizer
    T = tc.trainer.num_train_steps
    f = opt.lr_scheduler(T)
    t = np.array([0, 1, 100, 991, 2480, 3967, 3968, 4500, 4958])
    lr = np.array([float(f(int(i))) for i in t])
    ref = 0.02 / np.sqrt(1 + 23 * 0.02 ** 2 * t)
    print(repr(sched), "run", tc.trainer.id, "T", T, "sched", type(opt.lr_schedule).__name__ if not isinstance(opt.lr_schedule, str) else opt.lr_schedule, "decay", opt.decay, "warmup", opt.warmup)
    print("   lr ", np.round(lr, 6)); print("   1/sqrt ref", np.round(ref, 6))
