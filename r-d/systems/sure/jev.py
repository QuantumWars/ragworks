"""Minimal Jev (TypeSafe System One) client, routed via OpenRouter.

Local to this system on purpose -- see r-d/README.md. Jev returns typed answers
sampled from a supplied schema rather than generated as tokens, so an
out-of-schema verdict is structurally impossible. That is the property this
system is testing.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ENDPOINT = "https://openrouter.ai/api/v1/systemone"
DEFAULT_MODEL = "typesafe/jev-1.13"


def _key() -> str:
    if k := os.environ.get("OPENROUTER_API_KEY"):
        return k
    for parent in Path(__file__).resolve().parents:
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found in environment or any parent .env")


def noul(instructions: str, *, true: str | None = None, false: str | None = None) -> dict:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        q["criteria"] = {"true": true, "false": false}
    return q


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def system_one(
    state: Any, questions: dict[str, dict], *, model: str = DEFAULT_MODEL, timeout: float = 60.0
) -> dict:
    """One round trip. Every question is scored against the same state in parallel."""
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:300]}") from None
