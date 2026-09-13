# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a tutorial dataset on a Della login node (compute nodes cannot reach huggingface.co).

    MARIN_PREFIX=/scratch/gpfs/KARTHIKN/wc9403/marin_store uv run python -m scripts.della.prep_data wikitext
"""

import sys

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import run

from experiments.tutorials.train_tiny_model import dataset

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "wikitext"
    with build_context(BuildContext(versions=VersionCodex(default="dev", overrides={}))):
        handle = dataset(name)
    print(run(handle, max_concurrent=1))
