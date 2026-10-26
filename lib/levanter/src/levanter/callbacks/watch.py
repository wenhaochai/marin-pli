# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
from dataclasses import dataclass
from typing import Literal, Sequence, TypeVar, Union, cast

import jax
from jaxtyping import PyTree

import levanter.tracker
from levanter.analysis.tree_stats import summary_statistics_for_tree
from levanter.callbacks import JitCallback
from levanter.tracker.histogram import SummaryStats
from levanter.trainer_state import InsideJitInfo, TrainerState
from levanter.utils.tree_utils import key_path_to_str


Target = Literal["grads", "params", "opt_state", "updates"]
M = TypeVar("M", bound=PyTree)
S = TypeVar("S", bound=TrainerState)
VALID_WATCH_TARGETS = {"grads", "params", "opt_state", "updates"}

# LEVANTER_NORM_GUARD=<relative> (2026-10-26, after the marin vocab run w158): stop the run when the norm of a weight that
# AdamH or MuonH keeps constant (the lm_head and the attention and MLP projections) moves by more than <relative> between
# two watch steps. In w158 one update read a wrong norm (that step's loss value): the lm_head norm went 353.33 -> 3.556 and
# the run collapsed to near-uniform predictions without a NaN. Between healthy watch steps these norms move by ~1e-6
# (f32 rounding; 375 runs scanned). 0 (the default) turns the check off.
_NORM_GUARD = float(os.environ.get("LEVANTER_NORM_GUARD", "0") or 0)
_CONSTANT_NORM_WEIGHTS = ("lm_head.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                          "gate_proj.weight", "up_proj.weight", "down_proj.weight")


def _validate_watch_targets(watch_targets: Sequence[str]) -> None:
    invalid_targets = set(watch_targets) - VALID_WATCH_TARGETS
    if invalid_targets:
        raise ValueError(f"Invalid watch targets: {invalid_targets}. Valid targets are: {VALID_WATCH_TARGETS}")


def compute_watch_stats(
    *,
    watch_targets: Sequence[Target],
    include_norms: bool,
    include_per_parameter_norms: bool,
    include_histogram: bool,
    split_scan_layers: bool,
    params: PyTree | None = None,
    grads: PyTree | None = None,
    updates: PyTree | None = None,
    opt_state: PyTree | None = None,
    model_tree_type: type | None = None,
) -> dict[str, jax.Array | SummaryStats]:
    """Compute watch metrics for selected training targets.

    Args:
        watch_targets: Targets to include, chosen from grads/params/updates/opt_state.
        include_norms: Whether to include norms.
        include_per_parameter_norms: Whether to include per-parameter norms.
        include_histogram: Whether to include histograms.
        split_scan_layers: Whether stacked scan layers are split for logging.
        params: Parameter tree for the ``params`` target.
        grads: Gradient tree for the ``grads`` target.
        updates: Update tree for the ``updates`` target.
        opt_state: Optimizer state tree for the ``opt_state`` target.
        model_tree_type: Optional type used to filter optimizer-state leaves.

    Returns:
        A dictionary of metric keys to scalar arrays or histograms.
    """
    _validate_watch_targets(watch_targets)

    to_log: dict[str, jax.Array | SummaryStats] = {}
    tree_targets: dict[Target, tuple[str, PyTree | None]] = {
        "grads": ("grad", grads),
        "params": ("params", params),
        "updates": ("updates", updates),
    }

    for target in watch_targets:
        if target in tree_targets:
            prefix, tree = tree_targets[target]
            if tree is None:
                raise ValueError(f"{target} must be provided when watch_targets includes '{target}'")
            stats = summary_statistics_for_tree(
                prefix,
                tree,
                split_scan_layers,
                include_histogram=include_histogram,
                include_norms=include_norms,
                include_per_parameter_norms=include_per_parameter_norms,
            )
            to_log.update(stats)
            continue

        if target == "opt_state":
            if opt_state is None:
                raise ValueError("opt_state must be provided when watch_targets includes 'opt_state'")

            if model_tree_type is None:
                leaves = jax.tree.leaves_with_path(opt_state)
            else:
                leaves = jax.tree.leaves_with_path(opt_state, is_leaf=lambda m: isinstance(m, model_tree_type))

            for path, value in leaves:
                if model_tree_type is not None and not isinstance(value, model_tree_type):
                    continue

                name = key_path_to_str(path)
                name_to_log = f"opt_state/{name}" if name else "opt_state"
                this_stats = summary_statistics_for_tree(
                    name_to_log,
                    value,
                    split_scan_layers,
                    include_histogram=include_histogram,
                    include_norms=include_norms,
                    include_per_parameter_norms=include_per_parameter_norms,
                )
                to_log.update(this_stats)

    return to_log


@dataclass(frozen=True)
class WatchConfig:
    watch_targets: Union[list[Target], Target] = dataclasses.field(
        default_factory=lambda: cast(list[Target], ["grads", "params"])
    )
    """
    What to watch during training. Can be a single target or a list of targets.
    Valid targets are: 'grads', 'params', 'opt_state', 'updates'.
    """
    include_norms: bool = True
    include_per_parameter_norms: bool = True
    include_histograms: bool = False
    split_scan_layers: bool = True

    interval: int = 10

    @property
    def is_enabled(self) -> bool:
        return len(self.watch_targets) > 0 and self.interval > 0 and (self.include_norms or self.include_histograms)

    def build(self) -> "WatchCallback":
        return WatchCallback(
            watch_targets=self.watch_targets,
            include_norms=self.include_norms,
            include_per_parameter_norms=self.include_per_parameter_norms,
            include_histogram=self.include_histograms,
            split_scan_layers=self.split_scan_layers,
        )


class WatchCallback(JitCallback[S, M, dict[str, jax.Array | SummaryStats]]):
    """
    A unified callback for watching various aspects of training (gradients, parameters, optimizer state, updates).
    This callback combines the functionality of GradWatchCallback, ParamWatchCallback, OptStateWatchCallback,
    and UpdatesWatchCallback into a single callback.

    Args:
        watch_targets (Union[Sequence[str], str]): What to watch. Can be a comma-separated string or list of strings.
            Valid targets are: 'grads', 'params', 'opt_state', 'updates'.

        include_norms (bool): Whether to include norms in the logging.
        include_histogram (bool): Whether to include histograms in the logging.
        split_scan_layers (bool): Whether to split the scan layers into separate histograms/norms.
    """

    def __init__(
        self,
        watch_targets: Union[Sequence[str], str] = ("grads", "params"),
        include_norms: bool = True,
        include_per_parameter_norms: bool = True,
        include_histogram: bool = False,
        split_scan_layers: bool = True,
    ):
        if isinstance(watch_targets, str):
            watch_targets = [t.strip() for t in watch_targets.split(",")]
        else:
            watch_targets = list(watch_targets)

        self.watch_targets = cast(Sequence[Target], watch_targets)
        self.include_norms = include_norms
        self.include_per_parameter_norms = include_per_parameter_norms
        self.include_histogram = include_histogram
        self.split_scan_layers = split_scan_layers

        # Validate watch targets
        _validate_watch_targets(watch_targets)

    def inside_step(
        self, state: TrainerState[M], inside_info: InsideJitInfo[M]
    ) -> dict[str, jax.Array | SummaryStats]:
        return compute_watch_stats(
            watch_targets=self.watch_targets,
            include_norms=self.include_norms,
            include_per_parameter_norms=self.include_per_parameter_norms,
            include_histogram=self.include_histogram,
            split_scan_layers=self.split_scan_layers,
            params=state.trainable_model,
            grads=inside_info.grads,
            updates=inside_info.updates,
            opt_state=state.opt_state,
            model_tree_type=type(state.model),
        )

    def on_step(self, step_info: S, cb_info: dict[str, jax.Array | SummaryStats]):
        if _NORM_GUARD > 0:
            self._check_constant_norms(int(step_info.step), cb_info)
        levanter.tracker.log(cb_info, step=int(step_info.step))

    def _check_constant_norms(self, step: int, cb_info: dict[str, jax.Array | SummaryStats]):
        """Raises when a weight that the optimizer keeps at a constant norm moved by more than LEVANTER_NORM_GUARD."""
        current = {
            k: float(v)
            for k, v in cb_info.items()
            if k.startswith("params/norm/") and k.endswith(_CONSTANT_NORM_WEIGHTS) and isinstance(v, jax.Array)
        }
        previous = getattr(self, "_constant_norms", {})
        for k, v in current.items():
            before = previous.get(k)
            if before is not None and before > 0 and abs(v - before) / before > _NORM_GUARD:
                raise RuntimeError(
                    f"norm guard: {k} moved from {before:.6g} to {v:.6g} by step {step} (relative "
                    f"{abs(v - before) / before:.2e} > {_NORM_GUARD:g}); its optimizer keeps it constant, so an update "
                    "read a wrong value. Stopping so that the run resumes from its last checkpoint."
                )
        self._constant_norms = current
