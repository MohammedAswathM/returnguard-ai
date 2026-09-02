#!/usr/bin/env python3
import shutil
from pathlib import Path

for target in (
    Path("artifacts/data"), Path("artifacts/baselines"), Path("artifacts/source"),
    Path("artifacts/data_uci"), Path("artifacts/data_uci_repeat"),
    Path("artifacts/baselines_uci"), Path("artifacts/data_locked_final"),
    Path("artifacts/data_uci_locked_final"), Path("artifacts/data_uci_repeat_locked_final"),
):
    if target.is_dir() and target.parent == Path("artifacts"):
        shutil.rmtree(target)
