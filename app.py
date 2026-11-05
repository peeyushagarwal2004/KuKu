"""Streamlit front end:  streamlit run app.py

Upload a .zip of episode screenplays, pick a model (or rules-only), get ranked promos
with exact cut points, the script of each clip, and downloadable raw outputs.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from promofinder.clips import clip_script
from promofinder.llm import LLMConfig, PROVIDERS, load_dotenv
from promofinder.pipeline import run

load_dotenv()
st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Promo Finder")
st.caption("Reads a microdrama's screenplays and finds 20-60 s promo clips that open on a hook, "
           "end on a cliffhanger and make sense to a stranger.")

# ------------------------------------------------------------ inputs
with st.sidebar:
    st.header("Model")
    mode = st.radio("Mode", ["llm", "rules"], format_func=lambda m: "LLM (recommended)" if m == "llm"
                    else "Rules only (free, offline, weaker)")
    provider = st.selectbox("Provider", list(PROVIDERS), index=0, disabled=mode != "llm")
    model = st.text_input("Model", value=PROVIDERS[provider][2], disabled=mode != "llm")
    key_env = PROVIDERS[provider][1]
    api_key = st.text_input(f"API key ({key_env or 'none needed'})", type="password",
                            value="", disabled=mode != "llm" or not key_env,
                            help="Leave blank to use the key from the environment / .env file.")
    base_url = st.text_input("Base URL", value=PROVIDERS[provider][0], disabled=mode != "llm",
                             help="Any OpenAI-compatible endpoint works.")
    st.header("Run")
    ep_range = st.text_input("Episodes (blank = all)", placeholder="e.g. 1-10")
    stitch_n = st.slider("Stitched multi-episode promos", 0, 8, 5, disabled=mode != "llm")

uploaded = st.file_uploader("Episode screenplays (.zip of .txt files)", type=["zip"])
local = st.text_input("…or a local folder / .zip path", value="")

if st.button("Find promos", type="primary", disabled=not (uploaded or local)):
    src: str | bytes = uploaded.getvalue() if uploaded else local
    show = Path(uploaded.name).stem if uploaded else (Path(local).stem if local.lower().endswith(".zip")
                                                       else Path(local).name)
    episodes = None
    if ep_range.strip():
        episodes = []
        for part in ep_range.split(","):
            a, _, b = part.strip().partition("-")
            episodes += list(range(int(a), int(b or a) + 1))
    cfg = None
    if mode == "llm":
        cfg = LLMConfig.from_env(provider=provider, model=model, base_url=base_url,
                                 api_key=api_key or None)
    out_dir = Path(tempfile.mkdtemp(prefix="promos-"))
    log_box = st.empty()
    lines: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        log_box.code("\n".join(lines[-14:]))

    try:
        with st.spinner("Reading episodes and picking promos…"):
            st.session_state["result"] = run(src, out_dir, mode=mode, cfg=cfg, episodes=episodes,
                                             stitch_n=stitch_n, show_name=show, log=log)
            st.session_state["out_dir"] = str(out_dir)
    except Exception as e:
        st.error(f"Run failed: {e}")

# ------------------------------------------------------------ results
res = st.session_state.get("result")
if res:
    meta, ranked, eps = res["meta"], res["ranked"], res["episodes"]
    out_dir = Path(st.session_state["out_dir"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Promos", meta["promos_kept"])
    c2.metric("Rejected by rules", meta["promos_rejected"])
    c3.metric("LLM calls", meta["llm_calls"])
    c4.metric("Time", f"{meta['seconds']:.0f}s")
    if meta.get("strategy"):
        st.info(f"**Posting strategy:** {meta['strategy']}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir))
    st.download_button("Download all raw outputs (.zip)", buf.getvalue(), file_name=f"{meta['show']}-promos.zip")

    st.subheader("Ranked promos")
    st.dataframe([{"#": c["rank"], "id": c["id"], "cuts": " + ".join(s.label() for s in c["segments"]),
                   "sec": round(c["duration_s"]), "title": c["title"], "score": c["final"],
                   "critic": (c.get("critic") or {}).get("verdict", "")} for c in ranked],
                 use_container_width=True, hide_index=True)

    for c in ranked[:30]:
        cr = c.get("critic") or {}
        with st.expander(f"#{c['rank']} · {c['id']} · {c['title'] or ''} · {c['duration_s']:.0f}s · "
                         f"score {c['final']:.2f}"):
            st.markdown(f"**Cuts (in order):** {' + '.join(s.label() for s in c['segments'])}")
            st.markdown(f"**Hook:** {c['hook']}  \n**Cliffhanger:** {c['cliffhanger']}  \n**Why:** {c['why']}")
            if c.get("post_note"):
                st.markdown(f"**Posting note:** {c['post_note']}")
            if cr:
                st.markdown(f"**Cold viewer:** {cr['understood']}  \n"
                            f"Confused by: {cr['confusion'] or '—'} · Left wondering: {cr['open_question'] or '—'} · "
                            f"verdict **{cr['verdict']}**")
            if c["fx"].flags:
                st.warning("; ".join(c["fx"].flags))
            st.code(clip_script(c["segments"], eps, relative=False), language=None)
