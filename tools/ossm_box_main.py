#!/usr/bin/env python3
"""Launch STIP_main with wandb optional when OSSM logging is disabled."""

from pathlib import Path
import runpy
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import wandb  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("wandb")
    stub.init = lambda *args, **kwargs: None
    stub.watch = lambda *args, **kwargs: None
    stub.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = stub

runpy.run_path(str(ROOT / "STIP_main.py"), run_name="__main__")
