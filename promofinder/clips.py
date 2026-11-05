"""Clip geometry: turn beat ranges into exact cut points and check them with plain rules.

The LLM only ever names beat ids. Every timestamp in the output is computed here
from the script, so the model cannot invent a timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .parse import Beat, Episode, fmt_ts

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0          # hook must start within this many seconds of the cut-in
MAX_EXTEND = 3.0           # how far we will stretch a cut to avoid chopping a spoken line
END_SNAP = 1.5             # snap to the episode's end (the stinger) if this close


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    start_beat: int
    end_beat: int

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def label(self) -> str:
        return f"E{self.episode:02d} {fmt_ts(self.start)}-{fmt_ts(self.end)}"

    def to_dict(self) -> dict:
        return {"episode": self.episode, "start": fmt_ts(self.start), "end": fmt_ts(self.end),
                "start_s": round(self.start, 2), "end_s": round(self.end, 2),
                "duration_s": self.duration, "start_beat": self.start_beat, "end_beat": self.end_beat}


@dataclass
class Fixups:
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)     # soft problems, kept but penalised
    errors: list[str] = field(default_factory=list)    # hard problems, clip rejected


def beats_in(ep: Episode, start: float, end: float) -> list[Beat]:
    return [b for b in ep.beats if b.start >= start - 1e-6 and b.start < end - 1e-6]


def make_segment(ep: Episode, start_beat: int, end_beat: int, fx: Fixups,
                 hook_beat: int | None = None) -> Segment | None:
    n = len(ep.beats)
    if not (0 <= start_beat < n and 0 <= end_beat < n):
        fx.errors.append(f"beat ids out of range (episode has {n} beats)")
        return None
    if end_beat < start_beat:
        start_beat, end_beat = end_beat, start_beat
        fx.notes.append("start/end beats were swapped")
    t0 = ep.beats[start_beat].start
    t1 = max(b.end for b in ep.beats[start_beat:end_beat + 1])

    # Hook must land within the first 3 s: if not, move the cut-in up to just before it.
    if hook_beat is not None and 0 <= hook_beat < n:
        hb = ep.beats[hook_beat]
        if hb.start < t0 - 1e-6 or hb.start > t1:
            fx.flags.append("named hook beat is outside the clip")
        elif hb.start - t0 > HOOK_WINDOW:
            # start on the shot that carries the hook if there is one just before it
            lead = [b for b in ep.beats if hb.start - 1.5 <= b.start <= hb.start and b.kind == "camera"]
            new_t0 = min((b.start for b in lead), default=hb.start)
            fx.notes.append(f"cut-in moved {new_t0 - t0:.1f}s later so the hook lands in the first 3s")
            t0 = new_t0

    # Do not start in the middle of a spoken line.
    for b in ep.beats:
        if b.kind == "dialogue" and b.start < t0 - 1e-6 < b.end - 0.3:
            if t0 - b.start <= 2.0:
                fx.notes.append("cut-in pulled back to the start of a line")
                t0 = b.start
            else:
                fx.notes.append("cut-in pushed past a half-heard line")
                t0 = b.end
    # Do not end in the middle of a spoken line.
    for b in sorted(ep.beats, key=lambda b: b.start):
        if b.kind == "dialogue" and b.start < t1 - 0.3 and b.end > t1 + 1e-6:
            if b.end - t1 <= MAX_EXTEND:
                fx.notes.append("cut-out extended to finish a line")
                t1 = b.end
            else:
                fx.flags.append("cut-out lands mid-line")
    if ep.duration - t1 <= END_SNAP:
        t1 = ep.duration

    inside = beats_in(ep, t0, t1)
    if not inside:
        fx.errors.append("empty clip after snapping")
        return None
    return Segment(ep.number, round(t0, 2), round(t1, 2), inside[0].id, inside[-1].id)


def check_promo(segments: list[Segment], fx: Fixups, eps: dict[int, Episode],
                hook_beat: int | None = None, cliff_beat: int | None = None) -> float:
    total = round(sum(s.duration for s in segments), 2)
    if total < MIN_LEN:
        fx.errors.append(f"too short ({total:.1f}s < {MIN_LEN:.0f}s)")
    elif total > MAX_LEN:
        fx.errors.append(f"too long ({total:.1f}s > {MAX_LEN:.0f}s)")
    if segments and cliff_beat is not None:
        last = segments[-1]
        ep = eps[last.episode]
        if 0 <= cliff_beat < len(ep.beats):
            cb = ep.beats[cliff_beat]
            if cb.end < last.end - 6.0:
                fx.flags.append("cliffhanger beat is not near the end of the clip")
            if cb.start > last.end:
                fx.flags.append("named cliffhanger beat is after the cut-out")
    for a, b in zip(segments, segments[1:]):
        if a.episode == b.episode and a.end > b.start and b.end > a.start:
            fx.flags.append("segments overlap")
    return total


def clip_script(segments: list[Segment], eps: dict[int, Episode], relative: bool = True,
                include_camera: bool = True) -> str:
    """The clip as a script. With relative=True timestamps restart at 00:00 (what a viewer sees)."""
    out, offset = [], 0.0
    for i, s in enumerate(segments):
        if len(segments) > 1:
            out.append(f"--- cut {i + 1}: episode {s.episode} ---")
        for b in beats_in(eps[s.episode], s.start, s.end):
            if not include_camera and b.kind == "camera":
                continue
            t = (b.start - s.start + offset) if relative else b.start
            out.append(f"[{fmt_ts(max(t, 0))}] {b.label}: {b.text}")
        offset += s.duration
    return "\n".join(out)


def overlap_ratio(a: list[Segment], b: list[Segment]) -> float:
    inter = 0.0
    for x in a:
        for y in b:
            if x.episode == y.episode:
                inter += max(0.0, min(x.end, y.end) - max(x.start, y.start))
    shorter = min(sum(s.duration for s in a), sum(s.duration for s in b)) or 1
    return inter / shorter
