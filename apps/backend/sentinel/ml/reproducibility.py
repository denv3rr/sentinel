from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
import random
from typing import Any

import numpy as np


def _resolve_git_head(repo_root: Path) -> str | None:
    git_dir = repo_root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None

    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()[:40]
        packed_refs = git_dir / "packed-refs"
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                sha, _, name = line.partition(" ")
                if name.strip() == ref:
                    return sha.strip()[:40]
        return None
    if len(head) >= 7:
        return head[:40]
    return None


def configure_reproducibility(seed: int, deterministic: bool = True) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch_deterministic = False
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch_deterministic = deterministic
    except Exception:
        torch_deterministic = False

    return {
        "seed": seed,
        "deterministic_requested": deterministic,
        "torch_deterministic": torch_deterministic,
    }


def default_run_metadata(config: dict[str, Any], dataset_hash: str | None = None) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": _resolve_git_head(repo_root),
        "config": config,
        "dataset_hash": dataset_hash,
    }

