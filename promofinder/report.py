"""Write raw outputs: JSON (full), CSV (one row per promo), Markdown (for an editor), QC report."""
from __future__ import annotations

import csv
import re
import json
import statistics as st
from collections import Counter
from pathlib import Path

from .clips import clip_script
from .parse import Episode


def promo_dict(c: dict) -> dict:
    d = {k: v for k, v in c.items() if k not in ("segments", "fx", "llm_raw")}
    d["segments"] = [s.to_dict() for s in c.get("segments", [])]
    if "fx" in c:
        d["fixups"] = c["fx"].notes
        d["flags"] = c["fx"].flags
        d["errors"] = c["fx"].errors
    d["episodes"] = sorted({s.episode for s in c.get("segments", [])})
    return d


def segments_text(c: dict) -> str:
    return " + ".join(s.label() for s in c["segments"])


def write_all(out: Path, ranked: list[dict], rejected: list[dict], moments: list[dict],
              eps: dict[int, Episode], meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "promos.json").write_text(json.dumps({"meta": meta, "promos": [promo_dict(c) for c in ranked]},
                                                indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "rejected.json").write_text(json.dumps(
        [promo_dict(c) if "fx" in c else c for c in rejected], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "moments.json").write_text(json.dumps(
        [{**{k: v for k, v in m.items() if k != "segment"}, **m["segment"].to_dict()} for m in moments],
        indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with open(out / "promos.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "type", "episodes", "segments (episode start-end, in order)", "duration_s",
                    "title", "hook", "cliffhanger", "why_it_works", "final_score", "hook_score",
                    "cliffhanger_score", "standalone_score", "critic_verdict", "spoiler", "flags", "post_note"])
        for c in ranked:
            cr = c.get("critic") or {}
            w.writerow([c["rank"], c["id"], c["type"], " ".join(f"E{e:02d}" for e in sorted({s.episode for s in c["segments"]})),
                        segments_text(c), c["duration_s"], c["title"], c["hook"], c["cliffhanger"], c["why"],
                        c["final"], c["scores"]["hook"], c["scores"]["cliffhanger"], c["scores"]["standalone"],
                        cr.get("verdict", ""), c["spoiler"], "; ".join(c["fx"].flags), c.get("post_note", "")])

    write_markdown(out / "promos.md", ranked, eps, meta)
    write_qc(out / "qc_report.md", ranked, rejected, meta)


def write_markdown(path: Path, ranked: list[dict], eps: dict[int, Episode], meta: dict) -> None:
    L = [f"# Promos: {meta['show']}", "",
         f"Model `{meta['model']}` · {len(meta['episodes_processed'])} episodes · "
         f"{meta['promos_kept']} promos kept, {meta['promos_rejected']} rejected · generated {meta['generated']}", ""]
    if meta.get("strategy"):
        L += [f"**Posting strategy:** {meta['strategy']}", ""]
    L += ["| # | id | cuts | len | title | score | critic |", "|---|---|---|---|---|---|---|"]
    for c in ranked:
        cr = c.get("critic") or {}
        L.append(f"| {c['rank']} | {c['id']} | {segments_text(c)} | {c['duration_s']:.0f}s | {c['title']} | "
                 f"{c['final']:.2f} | {cr.get('verdict', '')} |")
    L.append("")
    for c in ranked:
        cr = c.get("critic") or {}
        L += [f"## #{c['rank']} · {c['id']} · {c['title'] or '(untitled)'}", "",
              f"- **Cuts (in order):** {segments_text(c)} · **{c['duration_s']:.1f}s** · {c['type']}",
              f"- **Hook:** {c['hook']}",
              f"- **Cliffhanger:** {c['cliffhanger']}",
              f"- **Why it works:** {c['why']}"]
        if c.get("post_note"):
            L.append(f"- **Posting note:** {c['post_note']}" + (" *(would move)*" if c.get("would_move") else ""))
        s, p, r = c["scores"], c["proposer_scores"], c.get("rule_scores", {})
        L.append(f"- **Scores** (blend): hook {s['hook']}, cliffhanger {s['cliffhanger']}, standalone "
                 f"{s['standalone']} → quality {c['quality']}, final {c['final']} "
                 f"({'; '.join(c['adjustments']) or 'no adjustments'})")
        L.append(f"  - proposer {p} · rules {r}"
                 + (f" · blind critic stop {cr['stop_scroll']}, standalone {cr['standalone']}, "
                    f"cliff {cr['cliffhanger']}, verdict **{cr['verdict']}**" if cr else ""))
        if cr:
            L.append(f"- **Cold viewer understood:** {cr['understood']}")
            if cr.get("confusion"):
                L.append(f"- **Cold viewer confused by:** {cr['confusion']}")
            if cr.get("open_question"):
                L.append(f"- **Left wondering:** {cr['open_question']}")
        if c.get("context_needed"):
            L.append(f"- **Context a new viewer lacks:** {c['context_needed']}")
        if c.get("same_scene_as"):
            L.append(f"- Same scene as {c['same_scene_as']} (ranked lower so the posting plan stays varied)")
        if c["fx"].flags:
            L.append(f"- ⚠ **Flags:** {'; '.join(c['fx'].flags)}")
        if c["fx"].notes:
            L.append(f"- Auto-fixes: {'; '.join(c['fx'].notes)}")
        L += ["", "<details><summary>Script of the clip (episode timestamps)</summary>", "", "```",
              clip_script(c["segments"], eps, relative=False), "```", "</details>", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def write_qc(path: Path, ranked: list[dict], rejected: list[dict], meta: dict) -> None:
    L = [f"# Automatic quality checks: {meta['show']}", ""]
    singles = [c for c in ranked if c["type"] == "single"]
    covered = {s.episode for c in singles for s in c["segments"]}
    missing = [e for e in meta["episodes_processed"] if e not in covered]
    durs = [c["duration_s"] for c in ranked]
    L += ["## Volume", "",
          f"- Promos kept: {len(ranked)} ({len(singles)} single-episode, {len(ranked) - len(singles)} stitched)",
          f"- Rejected by rules: {len(rejected)}",
          f"- Episodes with at least one promo: {len(covered)}/{len(meta['episodes_processed'])}"
          + (f" (none for {', '.join(f'E{e:02d}' for e in missing)})" if missing else ""),
          f"- Length: min {min(durs, default=0):.0f}s, median {st.median(durs) if durs else 0:.0f}s, max {max(durs, default=0):.0f}s",
          f"- LLM calls: {meta['llm_calls']} (+{meta['llm_cache_hits']} cached), tokens {meta.get('tokens')}, "
          f"wall time {meta['seconds']}s", ""]

    reasons = Counter()
    for c in rejected:
        for e in (c.get("errors") or (c["fx"].errors if "fx" in c else [c.get("error", "?")])):
            reasons[e.split(" (")[0]] += 1
    L += ["## Why proposals were rejected", ""] + [f"- {n} × {r}" for r, n in reasons.most_common()] + [""]

    fix = Counter(re.sub(r"\d+(\.\d+)?s", "Ns", n) for c in ranked for n in c["fx"].notes)
    flags = Counter(f for c in ranked for f in c["fx"].flags)
    L += ["## Auto-fixes applied to kept promos (the model's cut points were off)", ""]
    L += [f"- {n} × {r}" for r, n in fix.most_common()] or ["- none"]
    L += ["", "## Soft flags on kept promos", ""] + ([f"- {n} × {r}" for r, n in flags.most_common()] or ["- none"])
    L.append("")

    crit = [c for c in ranked if c.get("critic")]
    if crit:
        verdicts = Counter(c["critic"]["verdict"] for c in crit)
        L += ["## Proposer vs blind critic", "",
              f"- Critic verdicts: {dict(verdicts)}",
              f"- Mean proposer standalone {st.mean(c['proposer_scores']['standalone'] for c in crit):.1f} vs "
              f"critic {st.mean(c['critic']['standalone'] for c in crit):.1f}",
              f"- Mean proposer hook {st.mean(c['proposer_scores']['hook'] for c in crit):.1f} vs "
              f"critic stop-scroll {st.mean(c['critic']['stop_scroll'] for c in crit):.1f}", "",
              "Biggest gaps (proposer thought it stood alone, the cold viewer was lost):", ""]
        gaps = sorted(crit, key=lambda c: c["proposer_scores"]["standalone"] - c["critic"]["standalone"], reverse=True)
        for c in gaps[:6]:
            L.append(f"- {c['id']} ({segments_text(c)}): proposer {c['proposer_scores']['standalone']:.0f} vs critic "
                     f"{c['critic']['standalone']:.0f}. Confused by: {c['critic']['confusion'] or '-'}")
        L.append("")
        skips = [c for c in crit if c["critic"]["verdict"] == "skip"]
        if skips:
            L += ["Clips the cold viewer would skip:", ""] + \
                 [f"- #{c['rank']} {c['id']}: {c['critic']['understood']}" for c in skips] + [""]

    if meta["mode"] == "llm" and any(c.get("rule_scores") for c in ranked):
        dis = sorted(ranked, key=lambda c: abs(c["scores"]["hook"] - c["rule_scores"]["hook"]), reverse=True)[:5]
        L += ["## Rules vs model on the hook (biggest disagreements; worth a human look)", ""]
        L += [f"- {c['id']}: model {c['scores']['hook']:.1f} vs rules {c['rule_scores']['hook']:.1f}. Hook: {c['hook'][:120]}"
              for c in dis] + [""]

    spoil = Counter(c["spoiler"] for c in ranked)
    L += ["## Spoiler risk", "", f"- {dict(spoil)}", ""]
    path.write_text("\n".join(L), encoding="utf-8")
