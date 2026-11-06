# Promo Finder

Reads a microdrama's timestamped screenplays and finds the best 20–60 second promo clips: clips that
open on a hook, end on a cliffhanger, and make sense to someone who has never seen the show.
For each promo you get the episode, exact start and end timestamps of every segment in order,
the hook, the cliffhanger, one line on why it works, and a posting rank.

**Live demo:** https://kuku-promo.streamlit.app (try the built-in sample show, or upload a .zip of episodes).

## Run it

```bash
pip install -r requirements.txt          # only Streamlit; the pipeline itself is standard library
cp .env.example .env                     # then paste a free Gemini key (aistudio.google.com/apikey)
```

Command line (writes to `outputs/<show>-llm/`):

```bash
python -m promofinder "path/to/show-folder-or.zip"
python -m promofinder "path/to/show.zip" --episodes 1-10          # quick partial run
python -m promofinder "path/to/show.zip" --mode rules             # no LLM: free, offline, weaker
python -m promofinder "path/to/show.zip" --provider openai --model gpt-4o-mini
```

Web app:

```bash
streamlit run app.py
```

## Swapping the model

Every provider is called through the OpenAI-compatible chat API, so a stronger model is a config change:

| Provider | Env vars |
|---|---|
| Gemini (default, free tier) | `GEMINI_API_KEY` |
| Groq (free tier) | `LLM_PROVIDER=groq`, `GROQ_API_KEY` |
| OpenAI | `LLM_PROVIDER=openai`, `OPENAI_API_KEY` |
| OpenRouter | `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY` |
| Local Ollama | `LLM_PROVIDER=ollama` |
| Any gateway (e.g. one Kuku provides) | `LLM_PROVIDER=custom`, `LLM_BASE_URL`, `LLM_API_KEY` |

`LLM_MODEL` overrides the model name. Responses are cached in `cache/`, so reruns are free and an
interrupted run resumes.

The Gemini default is `gemini-3.1-flash-lite` because, on the free tier, the Flash models allow only
about 20 requests a day and a full 50-episode run needs about 100. With a paid key, set
`LLM_MODEL=gemini-3.5-flash` (or anything stronger). When a model is overloaded (HTTP 503) or out of
daily quota, the client moves to the next model in `LLM_FALLBACK_MODELS`.

## How it works

1. **Parse** (`parse.py`): every CAMERA / ACTION / SOUND FX / dialogue block becomes a *beat* with its
   start and end time.
2. **Propose** (`pipeline.py`, one LLM call per episode): the model sees the episode as numbered beats
   and proposes up to 3 promos as *beat ranges* (never timestamps), naming the hook beat and the
   cliffhanger beat, plus 2–5 short standout "moments" for stitching.
3. **Cut and check with rules** (`clips.py`): timestamps are computed from the script, not the model.
   The cut-in moves so the hook lands within 3 s; cuts never chop a spoken line; length must be
   20–60 s; overlapping duplicates are dropped. Anything that fails is written to `rejected.json`
   with the reason.
4. **Blind critic** (batched LLM calls): a second prompt plays a TikTok viewer who has never seen the
   show. It only gets the clip itself, re-timed from 00:00, and says what it understood, what confused
   it, what it is left wondering, and whether it would stop scrolling.
5. **Stitch** (one LLM call): joins moments from 2+ episodes into multi-episode promos, checked by the
   same rules and critic.
6. **Rank** (code): blend of proposer and critic scores (critic weighted 60%), small bonus for promos
   from the free early episodes (+0.8 for episodes 1–10), penalties for spoilers and rule flags, and a
   diversity rule (a second clip of the same scene, or a third from one episode, drops down) so the top
   of the list isn't one episode. A final LLM call writes the posting note for the top 10.

## Outputs

| File | What |
|---|---|
| `promos.md` | Ranked, readable report with the script of every clip |
| `promos.csv` | One row per promo, for a spreadsheet |
| `promos.json` | Everything, including all scores and auto-fixes |
| `rejected.json` | Proposals the rules threw out, and why |
| `moments.json` | Short standout moments (the stitching library) |
| `qc_report.md` | Automatic quality checks |
| `raw_llm/` | The model's unedited JSON per episode |

## Input format

A folder or zip of `.txt` files named `episode-NNN-*.txt`, in the format of the Kuku input pack
(`**LABEL:**` then `Start: (MM:SS.ss) | End: (MM:SS.ss)` then text).

The input pack is confidential and is git-ignored (`data/`, `*.zip`). Don't commit it.
