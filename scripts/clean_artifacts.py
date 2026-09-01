#!/usr/bin/env python3
import shutil
from pathlib import Path

for target in (Path("artifacts/data"), Path("artifacts/baselines")):
    if target.is_dir() and target.parent == Path("artifacts"):
        shutil.rmtree(target)

