"""Rule-based signals. Two uses:

1. A no-LLM fallback mode (free, offline, deterministic) that still produces promos.
2. An independent second opinion in LLM mode: where rules and the LLM disagree
   is a cheap place to look for bad picks.

The word lists are tuned by eye on genre conventions (period romance / revenge
drama), not learned. They will be weaker on other genres; that is a known limit.
"""
from __future__ import annotations

import re

from .clips import Fixups, Segment, make_segment, check_promo, MIN_LEN, MAX_LEN
from .parse import Beat, Episode

HOOK_WORDS = {
    # physical shocks
    r"\bslap": 3, r"\bstrik": 2, r"\bpunch": 3, r"\bshov": 2, r"\bgrab": 1.5, r"\bseiz": 2,
    r"\bthrow": 1.5, r"\bblood": 2, r"\bsword": 2, r"\bblade": 2, r"\bknife": 2, r"\bdagger": 2,
    r"\bcollaps": 2, r"\bfaint": 2, r"\bscream": 2, r"\bkneel": 2, r"\bone knee": 2, r"\bkiss": 2,
    r"\bburst": 1.5, r"\bslam": 1.5, r"\bdrag": 1.5, r"\bchok": 2, r"\bpoison": 3, r"\bdead\b|\bdeath": 2,
    r"\barrest": 2, r"\bguards?\b": 1, r"\bshatter": 1.5, r"\bmud\b": 1, r"\btear": 1, r"\bgasp": 1.5,
    # social shocks
    r"\bhow dare": 3, r"\bdisown": 3, r"\bbanish": 2, r"\bhumiliat": 2, r"\bmarry\b|\bmarri": 1.5,
    r"\bwedding|\baltar": 1, r"\bduke\b": 2, r"\bheir": 1.5, r"\bbastard|\bwhore|\bharlot|\bwitch": 2.5,
    r"\bliar\b|\blie\b|\blied\b": 1.5, r"\bbetray": 2, r"\bthief|\bstole": 2, r"\bkill": 2.5,
    r"\bon your knees": 3, r"\bget out": 2, r"\bworthless|\bplain|\bugly|\bbeggar|\bpeasant": 1.5,
}
CLIFF_WORDS = {
    r"cliffhanger": 4, r"cut to black|cuts to black|smash cut": 3, r"\bominous": 2, r"\bfreez": 1.5,
    r"\bswell": 1, r"\bsilence\b": 1, r"\bburst|\bbursts open|\bdoor": 1.5, r"\bwho are you": 3,
    r"\bwhat\b.*\?": 1, r"\?$": 1.5, r"\bunless\b|\bor else\b|\byou will pay|\bregret": 2.5,
    r"\breveal": 1.5, r"\bstunned|\bshock": 1.5, r"\btomorrow\b|\btonight\b": 1, r"\.\.\.$": 1,
}
DIRECTION_HEAT = r"furious|shout|scream|cruel|venom|contempt|disdain|outrag|shock|horrif|desperate|menac|cold|sneer|mock|trembl|enraged|seeth|defian"


def _score(text: str, table: dict[str, float]) -> float:
    t = text.lower()
    return sum(w for pat, w in table.items() if re.search(pat, t))


def hook_score(b: Beat) -> float:
    s = _score(b.text, HOOK_WORDS)
    if b.kind == "dialogue":
        s += 1.0
        if re.search(DIRECTION_HEAT, b.direction.lower()):
            s += 1.5
        if b.text.endswith("!"):
            s += 0.5
    if b.kind == "sfx" and re.search(r"crack|slap|thud|crash|scream|gasp", b.text.lower()):
        s += 1.5
    return s


def cliff_score(b: Beat, ep: Episode) -> float:
    s = _score(b.text, CLIFF_WORDS)
    if b.kind == "dialogue" and re.search(DIRECTION_HEAT, b.direction.lower()):
        s += 0.5
    if ep.duration - b.end < 4.0:      # the episode's own ending is written as a cliffhanger
        s += 3.0
    return s


def window_scores(ep: Episode, seg: Segment) -> dict:
    """Rule-based 0-10 scores for any segment; used as a cross-check on LLM picks."""
    inside = [b for b in ep.beats if seg.start <= b.start < seg.end]
    head = [b for b in inside if b.start - seg.start <= 3.0]
    tail = [b for b in inside if seg.end - b.end <= 5.0] or inside[-2:]
    hook = max((hook_score(b) for b in head), default=0)
    cliff = max((cliff_score(b, ep) for b in tail), default=0)
    lines = [b for b in inside if b.kind == "dialogue"]
    speakers = {b.speaker for b in lines}
    # standalone proxy: enough talk to explain the conflict, not a one-man monologue
    standalone = min(10.0, 3 + 0.5 * len(lines) + (1.5 if len(speakers) >= 2 else 0))
    return {"hook": round(min(10.0, 2 + hook), 1), "cliffhanger": round(min(10.0, 2 + cliff), 1),
            "standalone": round(standalone, 1)}


def propose_rules(ep: Episode, per_episode: int = 2) -> list[dict]:
    """Rules-only promos: start on the hottest beats, end on the strongest cliff 20-60 s later."""
    beats = ep.beats
    starts = sorted(beats, key=hook_score, reverse=True)[:8]
    cands = []
    for hb in starts:
        if hook_score(hb) < 2:
            continue
        ends = [b for b in beats if MIN_LEN + 2 <= b.end - hb.start <= MAX_LEN - 5]
        if not ends:
            continue
        cb = max(ends, key=lambda b: (cliff_score(b, ep), -abs(b.end - hb.start - 40)))
        fx = Fixups()
        seg = make_segment(ep, hb.id, cb.id, fx, hook_beat=hb.id)
        if not seg:
            continue
        check_promo([seg], fx, {ep.number: ep}, cliff_beat=cb.id)
        sc = window_scores(ep, seg)
        cands.append({"segments": [seg], "fx": fx, "hook_beat": hb.id, "cliffhanger_beat": cb.id,
                      "hook": f"{hb.label}: {hb.text}", "cliffhanger": f"{cb.label}: {cb.text}",
                      "title": "", "why": "Rule pick: strongest hook words near the cut-in and "
                                          "strongest cliffhanger cues near the cut-out.",
                      "context_needed": "", "spoiler": "low", "proposer_scores": sc,
                      "rank_key": sc["hook"] + sc["cliffhanger"]})
    cands.sort(key=lambda c: c["rank_key"], reverse=True)
    picked: list[dict] = []
    for c in cands:
        s = c["segments"][0]
        if all(min(s.end, p["segments"][0].end) - max(s.start, p["segments"][0].start) < 0.5 * s.duration
               for p in picked):
            picked.append(c)
        if len(picked) >= per_episode:
            break
    return picked
