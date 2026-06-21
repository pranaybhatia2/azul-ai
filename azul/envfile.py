"""Minimal, dependency-free .env loader.

Reads KEY=VALUE lines from the repo-root .env (if present) into os.environ
without overriding variables already set in the real environment. Called from
the network-using CLI entry points (azul.play, scripts.eval_llm) so a local
ANTHROPIC_API_KEY in .env is picked up automatically. The .env file is
gitignored — never commit it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_env(path: Optional[str] = None) -> None:
    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Don't clobber a value already exported in the real environment.
        os.environ.setdefault(key, value)
