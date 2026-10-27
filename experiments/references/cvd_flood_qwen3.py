# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""The best 128K setup with flooding (owner popup 2026-10-26, "128K 最好设置 + flood 0.5"): VARIANT=cvd's input
rows (table_variants_qwen3.ComposedDropoutQwen3Config) and VARIANT=flood's batch loss |L - b| + b toward a per-step level
(flood_qwen3.FloodQwen3Config), in one model. The flood model computes its loss through super(), which here is the cvd
model's, so both act as they do alone. The level is the plain runs' one (logs/vocab_flood/), not one from cvd runs.
"""

from dataclasses import dataclass

from levanter.models.lm_model import LmConfig

from experiments.references.flood_qwen3 import FloodQwen3Config, FloodQwen3LMHeadModel
from experiments.references.table_variants_qwen3 import ComposedDropoutQwen3Config, ComposedDropoutQwen3LMHeadModel


@LmConfig.register_subclass("qwen3_cvd_flood")
@dataclass(frozen=True)
class CvdFloodQwen3Config(ComposedDropoutQwen3Config, FloodQwen3Config):
    @property  # type: ignore[override]
    def model_type(self):
        return CvdFloodQwen3LMHeadModel


class CvdFloodQwen3LMHeadModel(FloodQwen3LMHeadModel, ComposedDropoutQwen3LMHeadModel):
    """Method order: flood's loss, then cvd's training embedding, then the baseline."""

    @classmethod
    def init(cls, Vocab, config: CvdFloodQwen3Config, *, key):  # type: ignore[override]
        return ComposedDropoutQwen3LMHeadModel.init.__func__(cls, Vocab, config, key=key)
