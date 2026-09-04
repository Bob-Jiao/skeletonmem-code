#!/usr/bin/env python3
"""Give every torchrun rank its own compile cache before importing torch."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


cache_root = Path(
    os.environ.get(
        "STAGE1_COMPILE_CACHE_ROOT",
        "/data/jiaoguanbo/skeletonmem/runtime/cache/train_compile",
    )
)
local_rank = os.environ.get("LOCAL_RANK", "0")
rank_root = cache_root / f"rank{local_rank}"
inductor = rank_root / "torchinductor"
triton = rank_root / "triton"
inductor.mkdir(parents=True, exist_ok=True)
triton.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)
os.environ["TRITON_CACHE_DIR"] = str(triton)

runpy.run_module("wan_va.train", run_name="__main__")
