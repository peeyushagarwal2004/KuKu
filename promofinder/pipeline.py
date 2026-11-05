"""End-to-end: screenplays -> candidate promos -> blind critic -> stitched promos -> ranking."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from . import prompts
from .clips import (Fixups, Segment, check_promo, clip_script, make_segment, overlap_ratio)
from .heuristics import propose_rules, window_scores
from .llm import LLM, LLMConfig, LLMError
from .parse import Episode, fmt_ts, load_show

Log = Callable[[str], None]


def _items(data, key: str) -> list:
    """Models sometimes return a bare JSON list instead of {key: [...]}; accept both."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        v = data.get(key)
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []
    return []


# ---------------------------------------------------------------- proposals
def _clamp_score(x, default=5.0) -> float:
    try:
        return max(1.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return default


def _int(x):
    try:
        return int(str(x).lstrip("Bb"))
    except (TypeError, ValueError):
        return None


def propose_llm(llm: LLM, ep: Episode, show: str, total: int, raw_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    user = prompts.PROPOSER_USER.format(
        show=show, number=ep.number, total=total, duration=fmt_ts(ep.duration),
        logline=ep.logline or "(none)", notes=ep.notes or "(none)",
        beats="\n".join(b.line() for b in ep.beats))
    data = llm.chat_json(prompts.PROPOSER_SYSTEM, user)
    (raw_dir / f"episode-{ep.number:03d}.proposer.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    accepted, rejected = _build_promos(ep, data)

    # Repair loop: LLMs are bad at timestamp arithmetic. Tell the model the real lengths we
    # computed from the script and let it fix the clips that broke a hard rule. One round only.
    repairable = [c for c in rejected if any(e.startswith(("too long", "too short")) for e in c["fx"].errors)]
    repaired = False
    if repairable:
        problems = "\n".join(
            f"- promo {c['id'].split('-P')[1]} (B{c['llm_raw'].get('start_beat')}-B{c['llm_raw'].get('end_beat')}): "
            f"{c['fx'].errors[0]}. Measured from the script it runs "
            f"{c['segments'][0].label() if c['segments'] else '?'}." for c in repairable)
        fix_msg = (f"I measured your clips against the timestamps. These break the 20-60 second rule:\n{problems}\n"
                   "Pick tighter beat ranges for those promos (keep a hook in the first 3 seconds and an "
                   "unresolved ending), keep the other promos and moments unchanged, and reply with the full "
                   "JSON again in the same shape.")
        try:
            data2 = llm.chat_json(prompts.PROPOSER_SYSTEM, user, history=[
                {"role": "assistant", "content": json.dumps(data)}, {"role": "user", "content": fix_msg}])
            (raw_dir / f"episode-{ep.number:03d}.proposer.repair.json").write_text(
                json.dumps({"feedback": fix_msg, "reply": data2}, indent=2), encoding="utf-8")
            a2, r2 = _build_promos(ep, data2)
            if len(a2) >= len(accepted):   # only take the repair if it did not lose good clips
                accepted = a2
                rejected = r2 + [dict(c, id=c["id"] + "-before-repair") for c in repairable]
                data = {**data2, "moments": _items(data2, "moments") or _items(data, "moments")}
                repaired = True
        except (LLMError, ValueError):
            pass
    for c in accepted + rejected:
        c["repaired_episode"] = repaired
    return accepted, rejected, _build_moments(ep, data)


def _build_promos(ep: Episode, data) -> tuple[list[dict], list[dict]]:
    eps = {ep.number: ep}
    accepted, rejected = [], []
    for i, p in enumerate(_items(data, "promos"), start=1):
        fx = Fixups()
        sb, eb = _int(p.get("start_beat")), _int(p.get("end_beat"))
        hb, cb = _int(p.get("hook_beat")), _int(p.get("cliffhanger_beat"))
        seg = make_segment(ep, sb, eb, fx, hook_beat=hb) if sb is not None and eb is not None else None
        if sb is None or eb is None:
            fx.errors.append("missing start/end beat")
        total_s = check_promo([seg], fx, eps, hook_beat=hb, cliff_beat=cb) if seg else 0
        sc = p.get("scores") or {}
        cand = {
            "id": f"E{ep.number:02d}-P{i}", "type": "single", "segments": [seg] if seg else [],
            "duration_s": total_s, "title": p.get("title", ""), "hook": p.get("hook", ""),
            "cliffhanger": p.get("cliffhanger", ""), "why": p.get("why", ""),
            "context_needed": p.get("context_needed", ""), "spoiler": str(p.get("spoiler", "none")).lower(),
            "hook_beat": hb, "cliffhanger_beat": cb,
            "proposer_scores": {k: _clamp_score(sc.get(k)) for k in ("hook", "cliffhanger", "standalone")},
            "fx": fx, "llm_raw": p,
        }
        (rejected if fx.errors else accepted).append(cand)
    return accepted, rejected


def _build_moments(ep: Episode, data) -> list[dict]:
    moments = []
    for j, m in enumerate(_items(data, "moments"), start=1):
        fx = Fixups()
        sb, eb = _int(m.get("start_beat")), _int(m.get("end_beat"))
        if sb is None or eb is None:
            continue
        seg = make_segment(ep, sb, eb, fx)
        if not seg or not (2.0 <= seg.duration <= 20.0):
            continue
        moments.append({"id": f"m_E{ep.number:02d}_{j}", "segment": seg, "kind": m.get("kind", ""),
                        "description": m.get("description", ""), "intensity": _clamp_score(m.get("intensity"))})
    return moments


def dedupe(cands: list[dict], key) -> list[dict]:
    """Within one episode, drop a candidate that mostly overlaps a better one."""
    keep: list[dict] = []
    for c in sorted(cands, key=key, reverse=True):
        if all(overlap_ratio(c["segments"], k["segments"]) < 0.5 for k in keep):
            keep.append(c)
        else:
            c["fx"].errors.append("duplicate of a better clip in the same episode")
    return keep


# ---------------------------------------------------------------- critic
def run_critic(llm: LLM, cands: list[dict], eps: dict[int, Episode], batch: int, log: Log) -> None:
    groups = [cands[i:i + batch] for i in range(0, len(cands), batch)]
    with ThreadPoolExecutor(max(1, llm.cfg.workers)) as pool:
        list(pool.map(lambda g: _critic_batch(llm, g, eps, log), groups))
    log(f"  critic reviewed {sum(1 for c in cands if c.get('critic'))}/{len(cands)}")


def _critic_batch(llm: LLM, group: list[dict], eps: dict[int, Episode], log: Log) -> None:
    body = "\n\n".join(f"=== CLIP {c['id']} ({c['duration_s']:.0f}s) ===\n"
                       f"{clip_script(c['segments'], eps, relative=True)}" for c in group)
    try:
        data = llm.chat_json(prompts.CRITIC_SYSTEM, body)
    except (LLMError, ValueError) as e:
        log(f"  critic batch failed: {e}")
        return
    by_id = {str(r.get("id", "")).strip(): r for r in _items(data, "reviews")}
    for c in group:
        r = by_id.get(c["id"])
        if r:
            c["critic"] = {
                "stop_scroll": _clamp_score(r.get("stop_scroll")),
                "standalone": _clamp_score(r.get("standalone")),
                "cliffhanger": _clamp_score(r.get("cliffhanger")),
                "understood": r.get("understood", ""), "confusion": r.get("confusion", ""),
                "open_question": r.get("open_question", ""),
                "verdict": str(r.get("verdict", "maybe")).lower(),
            }


# ---------------------------------------------------------------- stitching
def run_stitch(llm: LLM, moments: list[dict], eps: dict[int, Episode], show: str, n: int,
               raw_dir: Path) -> tuple[list[dict], list[dict]]:
    if len(moments) < 2:
        return [], []
    lib = {m["id"]: m for m in moments}
    lines = [f"{m['id']} | E{m['segment'].episode:02d} | {m['segment'].duration:.1f}s | {m['kind']} | "
             f"intensity {m['intensity']:.0f} | {m['description']}" for m in moments]
    premise = "\n".join(f"E{e.number:02d}: {e.logline[:220]}" for e in eps.values())
    data = llm.chat_json(prompts.STITCH_SYSTEM.replace("{n}", str(n)),
                         prompts.STITCH_USER.format(show=show, premise=premise, moments="\n".join(lines)))
    (raw_dir / "stitch.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    accepted, rejected = [], []
    for i, p in enumerate(_items(data, "promos"), start=1):
        fx = Fixups()
        ids = [str(x) for x in p.get("moment_ids") or []]
        missing = [x for x in ids if x not in lib]
        if missing:
            fx.errors.append(f"unknown moment ids {missing}")
        segs = [lib[x]["segment"] for x in ids if x in lib]
        if len({s.episode for s in segs}) < 2:
            fx.errors.append("stitched promo uses fewer than 2 episodes")
        total = check_promo(segs, fx, eps) if segs else 0
        sc = p.get("scores") or {}
        cand = {"id": f"X{i}", "type": "stitched", "segments": segs, "moment_ids": ids, "duration_s": total,
                "title": p.get("title", ""), "hook": p.get("hook", ""), "cliffhanger": p.get("cliffhanger", ""),
                "why": p.get("why", ""), "context_needed": "", "spoiler": str(p.get("spoiler", "none")).lower(),
                "hook_beat": None, "cliffhanger_beat": None,
                "proposer_scores": {k: _clamp_score(sc.get(k)) for k in ("hook", "cliffhanger", "standalone")},
                "fx": fx, "llm_raw": p}
        (rejected if fx.errors else accepted).append(cand)
    return accepted, rejected


# ---------------------------------------------------------------- scoring
WEIGHTS = {"hook": 0.35, "cliffhanger": 0.35, "standalone": 0.30}
CRITIC_WEIGHT = 0.6   # the blind critic has no context, so it is the fairer judge of a cold viewer


def score(c: dict, eps: dict[int, Episode]) -> None:
    p, cr = c["proposer_scores"], c.get("critic")
    first, last = c["segments"][0], c["segments"][-1]
    hook_r = window_scores(eps[first.episode], first)["hook"]
    cliff_r = window_scores(eps[last.episode], last)["cliffhanger"]
    stand_r = sum(window_scores(eps[s.episode], s)["standalone"] for s in c["segments"]) / len(c["segments"])
    c["rule_scores"] = {"hook": hook_r, "cliffhanger": cliff_r, "standalone": round(stand_r, 1)}

    def blend(a, b):
        return a if b is None else (1 - CRITIC_WEIGHT) * a + CRITIC_WEIGHT * b
    parts = {
        "hook": blend(p["hook"], cr and cr["stop_scroll"]),
        "cliffhanger": blend(p["cliffhanger"], cr and cr["cliffhanger"]),
        "standalone": blend(p["standalone"], cr and cr["standalone"]),
    }
    quality = sum(WEIGHTS[k] * v for k, v in parts.items())

    adj, why = 0.0, []
    first_ep = min(s.episode for s in c["segments"])
    if first_ep <= 10:
        adj += 0.8; why.append("+0.8 episode 1-10 (viewer lands on the free episodes this clip shows)")
    elif first_ep <= 20:
        adj += 0.4; why.append("+0.4 episode 11-20")
    if c["spoiler"].startswith("high"):
        adj -= 1.0; why.append("-1.0 high spoiler risk")
    elif c["spoiler"].startswith("low"):
        adj -= 0.1; why.append("-0.1 low spoiler risk")
    if c["fx"].flags:
        adj -= 0.4 * len(c["fx"].flags); why.append(f"-{0.4 * len(c['fx'].flags):.1f} rule flags")
    if cr:
        v = cr["verdict"]
        if v == "post":
            adj += 0.3; why.append("+0.3 critic says post")
        elif v == "skip":
            adj -= 0.6; why.append("-0.6 critic says skip")
    c["scores"] = {k: round(v, 2) for k, v in parts.items()}
    c["quality"] = round(quality, 2)
    c["final"] = round(quality + adj, 2)
    c["adjustments"] = why


def _cast(c: dict, eps: dict[int, Episode]) -> set[str]:
    return {b.speaker.split(" (")[0].replace("LADY ", "").strip() for s in c["segments"]
            for b in eps[s.episode].beats if s.start <= b.start < s.end and b.kind == "dialogue"}


def same_scene(c: dict, o: dict, eps: dict[int, Episode]) -> bool:
    """Two clips from the same or adjacent episodes with nearly the same speakers are usually one
    scene that continues across an episode break (microdramas pick up exactly where they left off)."""
    if not any(abs(a.episode - b.episode) <= 1 for a in c["segments"] for b in o["segments"]):
        return False
    x, y = _cast(c, eps), _cast(o, eps)
    return bool(x and y) and len(x & y) / len(x | y) >= 0.6


def rank(cands: list[dict], eps: dict[int, Episode]) -> list[dict]:
    """Greedy ranking with diversity: penalise a second clip of the same scene, a third clip from
    the same episode, and overlapping cuts, so the top of the list is a varied posting plan."""
    pool, out, per_ep = list(cands), [], {}
    while pool:
        def eff(c):
            eps_used = max(per_ep.get(s.episode, 0) for s in c["segments"])
            ov = max((overlap_ratio(c["segments"], o["segments"]) for o in out), default=0)
            dup = next((o["id"] for o in out if same_scene(c, o, eps)), None)
            return c["final"] - 0.5 * max(0, eps_used - 1) - 1.5 * ov - (1.0 if dup else 0), dup
        scored = [(eff(c), c) for c in pool]
        (best_score, dup), best = max(scored, key=lambda t: t[0][0])
        best["rank_score"] = round(best_score, 2)
        if dup:
            best["same_scene_as"] = dup
        pool.remove(best)
        out.append(best)
        for s in best["segments"]:
            per_ep[s.episode] = per_ep.get(s.episode, 0) + 1
    for i, c in enumerate(out, start=1):
        c["rank"] = i
    return out


def run_rank_notes(llm: LLM, ranked: list[dict], top: int) -> dict:
    lines = []
    for c in ranked[:top]:
        segs = ", ".join(s.label() for s in c["segments"])
        lines.append(f"#{c['rank']} {c['id']} [{segs}] {c['duration_s']:.0f}s \"{c['title']}\" "
                     f"hook: {c['hook']} | cliffhanger: {c['cliffhanger']} | scores {c['scores']} "
                     f"| spoiler {c['spoiler']}")
    try:
        return llm.chat_json(prompts.RANK_SYSTEM, "\n".join(lines))
    except (LLMError, ValueError):
        return {}


# ---------------------------------------------------------------- main entry
def run(show_path, out_dir, mode: str = "llm", cfg: LLMConfig | None = None,
        episodes: list[int] | None = None, stitch_n: int = 5, critic_batch: int = 8,
        top_notes: int = 10, show_name: str | None = None, log: Log = print) -> dict:
    t_start = time.time()
    out = Path(out_dir)
    raw_dir = out / "raw_llm"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_eps = load_show(show_path)
    if not all_eps:
        raise ValueError(f"No episodes found in {show_path}")
    if show_name:
        show = show_name
    elif isinstance(show_path, (bytes, bytearray)):
        show = "show"
    else:
        sp = Path(show_path)
        show = sp.stem if sp.suffix.lower() == ".zip" else sp.name
    total_eps = len(all_eps)
    run_eps = [e for e in all_eps if not episodes or e.number in episodes]
    eps = {e.number: e for e in all_eps}
    log(f"Loaded {total_eps} episodes of '{show}'; processing {len(run_eps)} in {mode} mode")

    llm = None
    if mode == "llm":
        llm = LLM(cfg or LLMConfig.from_env(), log=log)
        log(f"Model: {llm.name}")

    accepted, rejected, moments = [], [], []
    proposals = {}
    if llm:
        def _one(ep):
            try:
                return propose_llm(llm, ep, show, total_eps, raw_dir)
            except (LLMError, ValueError) as e:
                return e
        with ThreadPoolExecutor(max(1, llm.cfg.workers)) as pool:
            futs = {pool.submit(_one, ep): ep for ep in run_eps}
            for f in as_completed(futs):
                proposals[futs[f].number] = f.result()
                log(f"  proposer done for E{futs[f].number:02d} ({len(proposals)}/{len(run_eps)})")
    for ep in run_eps:
        if llm:
            res = proposals[ep.number]
            if isinstance(res, Exception):
                log(f"E{ep.number:02d}: proposer failed: {res}")
                rejected.append({"id": f"E{ep.number:02d}-FAILED", "error": str(res)})
                continue
            a, r, m = res
            a = dedupe(a, key=lambda c: sum(c["proposer_scores"].values()))
            moments += m
        else:
            a = propose_rules(ep)
            for i, c in enumerate(a, start=1):
                c["id"] = f"E{ep.number:02d}-R{i}"
                c["type"] = "single"
                c["duration_s"] = round(sum(s.duration for s in c["segments"]), 2)
            r = []
        for c in list(a):
            if c["fx"].errors:
                a.remove(c); r.append(c)
        accepted += a
        rejected += [c for c in r if isinstance(c, dict) and "fx" in c]
        log(f"E{ep.number:02d}: {len(a)} promo(s) kept, {len(r)} rejected")

    if llm:
        log(f"Blind critic reviewing {len(accepted)} single-episode promos")
        run_critic(llm, accepted, eps, critic_batch, log)
        if stitch_n and len({m['segment'].episode for m in moments}) >= 2:
            log(f"Building stitched promos from {len(moments)} moments")
            try:
                st_ok, st_bad = run_stitch(llm, moments, eps, show, stitch_n, raw_dir)
            except (LLMError, ValueError) as e:
                log(f"  stitching failed: {e}")
                st_ok, st_bad = [], []
            run_critic(llm, st_ok, eps, critic_batch, log)
            accepted += st_ok
            rejected += st_bad

    for c in accepted:
        score(c, eps)
    ranked = rank(accepted, eps)
    notes = run_rank_notes(llm, ranked, top_notes) if llm and ranked else {}
    by_id = {n.get("id"): n for n in _items(notes, "notes") if isinstance(n, dict)}
    for c in ranked:
        if c["id"] in by_id:
            c["post_note"] = by_id[c["id"]].get("post_note", "")
            c["would_move"] = bool(by_id[c["id"]].get("would_move", False))

    meta = {
        "show": show, "mode": mode, "model": llm.name if llm else "rules",
        "episodes_total": total_eps, "episodes_processed": [e.number for e in run_eps],
        "promos_kept": len(ranked), "promos_rejected": len(rejected), "moments": len(moments),
        "llm_calls": llm.calls if llm else 0, "fallbacks": llm.fallbacks_used if llm else 0, "llm_cache_hits": llm.cache_hits if llm else 0,
        "tokens": llm.usage if llm else {}, "seconds": round(time.time() - t_start, 1),
        "strategy": notes.get("strategy", "") if isinstance(notes, dict) else "",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    from .report import write_all
    write_all(out, ranked, rejected, moments, eps, meta)
    log(f"Done: {len(ranked)} promos ranked, {len(rejected)} rejected -> {out}")
    return {"meta": meta, "ranked": ranked, "rejected": rejected, "moments": moments, "episodes": eps}
