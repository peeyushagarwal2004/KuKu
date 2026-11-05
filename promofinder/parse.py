"""Parse Kuku-style timestamped screenplays into a flat list of beats.

Each episode file looks like:

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. CATHEDRAL – DAY
    Scene start: 00:00.00
    ...
    **CAMERA:**
    Start: (00:00.00) | End: (00:04.80)
    Close-up on ...

    **LADY JULIANA (Confused, Breathless):**
    Start: (00:13.50) | End: (00:14.30)
    "Why?"

A "beat" is one of those blocks. Beats overlap in time (a CAMERA block runs
alongside the ACTION and dialogue it shows), so everything downstream works
with beat ids and [start, end] ranges, never with "the next line".
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = r"(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?"
BLOCK_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
TIME_RE = re.compile(rf"^Start:\s*\({TS}\)\s*\|\s*End:\s*\({TS}\)")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)")
HEADING_RE = re.compile(r"^Scene heading:\s*(.*)")
EP_RE = re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.I)
LOGLINE_RE = re.compile(r"^EPISODE LOGLINE:\s*(.*)", re.I)
NOTES_RE = re.compile(r"^CONTINUITY NOTES:\s*(.*)", re.I)
FILE_EP_RE = re.compile(r"episode[-_ ]?0*(\d+)", re.I)

NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx"}


def to_seconds(m: str, s: str, frac: str | None) -> float:
    return int(m) * 60 + int(s) + (float("0." + frac) if frac else 0.0)


def fmt_ts(t: float) -> str:
    """Seconds -> MM:SS.ss, the same format the scripts use."""
    m, s = divmod(round(t, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Beat:
    id: int                 # index within the episode (0-based)
    kind: str               # camera | action | sfx | dialogue
    start: float
    end: float
    text: str
    speaker: str = ""       # dialogue only
    direction: str = ""     # "(Cold, Commanding)" etc., dialogue only
    scene: int = 0
    heading: str = ""

    @property
    def label(self) -> str:
        if self.kind == "dialogue":
            return f"{self.speaker} ({self.direction})" if self.direction else self.speaker
        return {"camera": "CAMERA", "action": "ACTION", "sfx": "SOUND FX"}[self.kind]

    def line(self) -> str:
        """Compact one-line form used in prompts and reports."""
        return f"[B{self.id:03d} {fmt_ts(self.start)}-{fmt_ts(self.end)}] {self.label}: {self.text}"


@dataclass
class Episode:
    number: int
    logline: str
    notes: str
    beats: list[Beat] = field(default_factory=list)
    source: str = ""

    @property
    def duration(self) -> float:
        return max((b.end for b in self.beats), default=0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


def parse_episode(text: str, source: str = "", fallback_number: int = 0) -> Episode:
    lines = text.splitlines()
    number, logline, notes = fallback_number, "", ""
    beats: list[Beat] = []
    scene, heading = 0, ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if m := EP_RE.match(line):
            number = int(m.group(1))
        elif m := LOGLINE_RE.match(line):
            logline = m.group(1).strip()
        elif m := NOTES_RE.match(line):
            notes = m.group(1).strip()
        elif m := SCENE_RE.match(line):
            scene = int(m.group(1))
        elif m := HEADING_RE.match(line):
            heading = m.group(1).strip()
        elif m := BLOCK_RE.match(line):
            label = m.group(1).strip()
            # timing line follows (allow blank lines in between)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            tm = TIME_RE.match(lines[j].strip()) if j < len(lines) else None
            if not tm:
                i += 1
                continue
            start = to_seconds(*tm.group(1, 2, 3))
            end = to_seconds(*tm.group(4, 5, 6))
            body: list[str] = []
            k = j + 1
            while k < len(lines) and lines[k].strip() and not BLOCK_RE.match(lines[k].strip()) \
                    and not lines[k].startswith("---") and not SCENE_RE.match(lines[k].strip()):
                body.append(lines[k].strip())
                k += 1
            body_text = " ".join(body).strip()
            upper = label.upper()
            if upper in NON_DIALOGUE:
                beats.append(Beat(len(beats), NON_DIALOGUE[upper], start, end, body_text,
                                  scene=scene, heading=heading))
            else:
                pm = re.match(r"^(.*?)\s*\((.*)\)\s*$", label)
                speaker, direction = (pm.group(1), pm.group(2)) if pm else (label, "")
                beats.append(Beat(len(beats), "dialogue", start, end, body_text.strip('"“” '),
                                  speaker=speaker.strip(), direction=direction.strip(),
                                  scene=scene, heading=heading))
            i = k
            continue
        i += 1
    # beats are written roughly in time order; make it exact but stable
    beats.sort(key=lambda b: (b.start, b.id))
    for n, b in enumerate(beats):
        b.id = n
    return Episode(number, logline, notes, beats, source)


def _episode_sort_key(name: str) -> int:
    m = FILE_EP_RE.search(Path(name).name)
    return int(m.group(1)) if m else 10**6


def load_show(path: str | Path | bytes) -> list[Episode]:
    """Load every episode from a folder of .txt files, a .zip path, or zip bytes."""
    texts: list[tuple[str, str]] = []
    if isinstance(path, (bytes, bytearray)):
        zf = zipfile.ZipFile(io.BytesIO(path))
        texts = [(n, zf.read(n).decode("utf-8", "replace")) for n in zf.namelist()
                 if n.lower().endswith(".txt") and not n.startswith("__MACOSX")]
    else:
        p = Path(path)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                texts = [(n, zf.read(n).decode("utf-8", "replace")) for n in zf.namelist()
                         if n.lower().endswith(".txt") and not n.startswith("__MACOSX")]
        else:
            texts = [(str(f), f.read_text(encoding="utf-8", errors="replace"))
                     for f in p.rglob("*.txt")]
    texts.sort(key=lambda t: _episode_sort_key(t[0]))
    episodes = []
    for n, (name, txt) in enumerate(texts, start=1):
        ep = parse_episode(txt, source=Path(name).name, fallback_number=_episode_sort_key(name) or n)
        if ep.beats:
            episodes.append(ep)
    episodes.sort(key=lambda e: e.number)
    return episodes
