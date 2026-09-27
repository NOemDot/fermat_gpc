"""Small IO helpers: CSV append, JSON dump, seed control, timing."""
import csv
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np


def set_seeds(seed: int) -> None:
    """Set Python, NumPy, and (if available) PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def append_csv(path: str, row: dict) -> None:
    """Append a dict as a row to a CSV, writing the header if new."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def dump_json(path: str, obj) -> None:
    """Serialise obj to JSON, using str for non-serialisable values."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


@contextmanager
def timer(label: Optional[str] = None):
    """Context manager that yields a mutable dict and stores elapsed seconds."""
    box = {"elapsed": None}
    t0 = time.time()
    try:
        yield box
    finally:
        box["elapsed"] = time.time() - t0
        if label:
            print(f"[timer] {label}: {box['elapsed']:.3f}s")