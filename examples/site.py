"""Site-local paths for the examples, kept out of the repository.

Machine-specific locations -- data roots, scene directories -- live in a
``site.env`` file at the repository root (gitignored; see
``site.env.example``) or in the environment. Nothing in the repository
itself names a host, a mount point, or a storage layout.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def load_site():
    """Fold site.env into the environment (existing variables win)."""
    env = _ROOT / "site.env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def scene(day: str) -> Path:
    """The scene directory for a campaign day, from GPRI_SCENE_<day>."""
    load_site()
    v = os.environ.get(f"GPRI_SCENE_{day}", "")
    if not v:
        raise SystemExit(
            f"GPRI_SCENE_{day} is not set. Copy site.env.example to "
            f"site.env and point it at your data, or export the variable.")
    return Path(v)


def survey_roots() -> list[str]:
    load_site()
    v = os.environ.get("GPRI_SURVEY_ROOTS", "")
    return [p for p in v.split(":") if p]
