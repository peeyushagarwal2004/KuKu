"""CLI:  python -m promofinder <show folder or .zip> [--out DIR] [--mode llm|rules] [--episodes 1-10]"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .llm import LLMConfig, PROVIDERS
from .pipeline import run


def parse_range(s: str | None) -> list[int] | None:
    if not s:
        return None
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="promofinder", description="Find promo clips in microdrama screenplays.")
    ap.add_argument("input", help="folder of episode .txt files, or a .zip of them")
    ap.add_argument("--out", help="output folder (default outputs/<show>-<mode>)")
    ap.add_argument("--mode", choices=["llm", "rules"], default="llm")
    ap.add_argument("--provider", choices=list(PROVIDERS), help="overrides LLM_PROVIDER")
    ap.add_argument("--model", help="overrides LLM_MODEL")
    ap.add_argument("--episodes", help="e.g. 1-10 or 1,3,5 (default: all)")
    ap.add_argument("--stitch", type=int, default=5, help="max stitched multi-episode promos (0 = off)")
    ap.add_argument("--show-name")
    a = ap.parse_args(argv)

    p = Path(a.input)
    name = a.show_name or (p.stem if p.suffix.lower() == ".zip" else p.name)
    out = a.out or str(Path("outputs") / f"{name}-{a.mode}")
    cfg = LLMConfig.from_env(provider=a.provider, model=a.model) if a.mode == "llm" else None
    try:
        run(a.input, out, mode=a.mode, cfg=cfg, episodes=parse_range(a.episodes), stitch_n=a.stitch,
            show_name=name)
    except Exception as e:  # surface a clean message on the CLI
        if os.getenv("PROMOFINDER_DEBUG"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
