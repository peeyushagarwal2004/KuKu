"""Tiny, swappable LLM client.

Every provider here speaks the OpenAI chat-completions protocol, so swapping
models is a config change, not a code change:

    LLM_PROVIDER=gemini  GEMINI_API_KEY=...      (default model gemini-3.1-flash-lite: the free tier allows ~20 Flash requests a day)
    LLM_PROVIDER=groq    GROQ_API_KEY=...
    LLM_PROVIDER=openai  OPENAI_API_KEY=...
    LLM_PROVIDER=openrouter OPENROUTER_API_KEY=...
    LLM_PROVIDER=ollama                           (local, no key)
    LLM_PROVIDER=custom  LLM_BASE_URL=... LLM_API_KEY=...   (any compatible gateway)

LLM_MODEL overrides the model name for any provider. LLM_FALLBACK_MODELS (comma-separated) are tried
when the main model is overloaded. LLM_REASONING_EFFORT (none/low/medium/high) is passed through
for reasoning models; LLM_WORKERS sets how many episodes run in parallel.

Responses are cached on disk keyed by (model, messages), so a rerun is free
and an interrupted run resumes where it stopped. Only the standard library is
used, so there is nothing extra to install.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROVIDERS = {
    #            base_url                                                     key env              default model
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",     "gemini-3.1-flash-lite"),
    "groq":       ("https://api.groq.com/openai/v1",                          "GROQ_API_KEY",       "llama-3.3-70b-versatile"),
    "openai":     ("https://api.openai.com/v1",                               "OPENAI_API_KEY",     "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1",                            "OPENROUTER_API_KEY", "google/gemini-2.5-flash"),
    "ollama":     ("http://localhost:11434/v1",                               "",                   "llama3.1"),
    "custom":     ("",                                                        "LLM_API_KEY",        ""),
}


class LLMError(RuntimeError):
    pass


def load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=value lines from a local .env file into the environment (never overrides)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.3
    min_interval: float = 0.0     # seconds between calls (free-tier rate limits)
    max_retries: int = 6
    cache_dir: str = "cache"
    timeout: int = 180
    reasoning_effort: str = ""   # none | low | medium | high (reasoning models only)
    fallback_models: tuple = ()   # tried in order when the main model is overloaded
    workers: int = 1              # parallel requests

    @classmethod
    def from_env(cls, **overrides) -> "LLMConfig":
        load_dotenv()
        provider = (overrides.pop("provider", None) or os.getenv("LLM_PROVIDER") or "gemini").lower()
        if provider not in PROVIDERS:
            raise LLMError(f"Unknown provider {provider!r}. Choose one of {', '.join(PROVIDERS)}.")
        base, key_env, model = PROVIDERS[provider]
        cfg = cls(
            provider=provider,
            model=os.getenv("LLM_MODEL") or model,
            base_url=os.getenv("LLM_BASE_URL") or base,
            api_key=os.getenv("LLM_API_KEY") or (os.getenv(key_env, "") if key_env else ""),
            min_interval=float(os.getenv("LLM_MIN_INTERVAL", "2" if provider == "gemini" else "0")),
            reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low" if provider == "gemini" else ""),
            fallback_models=tuple(m.strip() for m in os.getenv(
                "LLM_FALLBACK_MODELS", "gemini-3.5-flash,gemini-3.8-flash" if provider == "gemini" else ""
            ).split(",") if m.strip()),
            workers=int(os.getenv("LLM_WORKERS", "3" if provider == "gemini" else "1")),
        )
        for k, v in overrides.items():
            if v not in (None, ""):
                setattr(cfg, k, v)
        return cfg


class LLM:
    def __init__(self, cfg: LLMConfig, log=print):
        if not cfg.base_url:
            raise LLMError("No base URL: set LLM_BASE_URL for provider 'custom'.")
        if not cfg.api_key and cfg.provider != "ollama":
            raise LLMError(f"No API key for provider '{cfg.provider}'. "
                           f"Set {PROVIDERS[cfg.provider][1]} (or LLM_API_KEY).")
        self.cfg = cfg
        self.log = log
        self._last_call = 0.0
        self._lock = threading.Lock()
        self.fallbacks_used = 0
        self.exhausted: set[str] = set()   # models whose daily quota ran out this run
        self.calls = 0
        self.cache_hits = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return f"{self.cfg.provider}:{self.cfg.model}"

    def _cache_path(self, messages: list[dict]) -> Path:
        h = hashlib.sha256(json.dumps([self.cfg.model, self.cfg.temperature, messages],
                                      sort_keys=True).encode()).hexdigest()[:24]
        return Path(self.cfg.cache_dir) / f"{h}.json"

    def chat(self, messages: list[dict], json_mode: bool = True) -> str:
        cp = self._cache_path(messages)
        if cp.exists():
            self.cache_hits += 1
            return json.loads(cp.read_text(encoding="utf-8"))["content"]

        models = [self.cfg.model, *[m for m in self.cfg.fallback_models if m != self.cfg.model]]
        models = [m for m in models if m not in self.exhausted] or models
        body = {"model": models[0], "messages": messages, "temperature": self.cfg.temperature}
        if self.cfg.reasoning_effort:
            body["reasoning_effort"] = self.cfg.reasoning_effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        delay = 10.0
        overloads = 0
        for attempt in range(self.cfg.max_retries + 1):
            with self._lock:   # space requests out across threads (free-tier rate limits)
                wait = self.cfg.min_interval - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.time()
            req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (500, 502, 503, 504):
                    overloads += 1
                    # after 2 overloads in a row, hop to the next model in the fallback list
                    if overloads >= 2 and models.index(body["model"]) + 1 < len(models):
                        body["model"] = models[models.index(body["model"]) + 1]
                        self.fallbacks_used += 1
                        self.log(f"  {self.cfg.provider} overloaded; falling back to {body['model']}")
                        overloads = 0
                        continue
                if e.code == 400 and "reasoning_effort" in body:
                    body.pop("reasoning_effort")   # model does not support it
                    continue
                m = re.search(r'"retryDelay":\s*"(\d+)', detail)
                retry_after = float(m.group(1)) if m else 0.0
                if e.code == 429 and ("per day" in detail.lower() or retry_after > 300):
                    # daily quota gone for this model: don't sleep for hours, move to the next model
                    self.exhausted.add(body["model"])
                    nxt = [x for x in models if x not in self.exhausted]
                    if not nxt:
                        raise LLMError(f"Daily free-tier quota used up for {', '.join(models)}. "
                                       "Wait for the reset, or set another provider/model.")
                    self.log(f"  {body['model']} daily quota used up; switching to {nxt[0]}")
                    body["model"] = nxt[0]
                    self.fallbacks_used += 1
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt < self.cfg.max_retries:
                    sleep_for = min(retry_after + 1, 120) if retry_after else delay
                    self.log(f"  LLM HTTP {e.code}; retrying in {sleep_for:.0f}s")
                    time.sleep(sleep_for)
                    delay = min(delay * 2, 120)
                    continue
                if e.code == 400 and json_mode and "response_format" in body:
                    body.pop("response_format")   # some models reject JSON mode; prompt still asks for JSON
                    continue
                raise LLMError(f"HTTP {e.code} from {self.name}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.cfg.max_retries:
                    self.log(f"  LLM network error ({e}); retrying in {delay:.0f}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                raise LLMError(f"Network error calling {self.name}: {e}") from None
        else:
            raise LLMError("LLM retries exhausted")

        with self._lock:
            self.calls += 1
            for k in self.usage:
                self.usage[k] += (data.get("usage") or {}).get(k, 0) or 0
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise LLMError(f"Unexpected response from {self.name}: {json.dumps(data)[:400]}") from None
        cp.write_text(json.dumps({"model": f"{self.cfg.provider}:{body['model']}", "content": content}), encoding="utf-8")
        return content

    def chat_json(self, system: str, user: str, history: list[dict] | None = None) -> dict:
        """Ask for JSON; on a parse failure, ask once more with the error shown.
        `history` continues an earlier conversation (assistant/user turns after the first user turn)."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}, *(history or [])]
        raw = self.chat(messages)
        try:
            return parse_json(raw)
        except ValueError as e:
            self._cache_path(messages).unlink(missing_ok=True)
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That was not valid JSON ({e}). "
                                                     "Reply with only the corrected JSON object."}]
            return parse_json(self.chat(messages))


def parse_json(raw: str) -> dict:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b > a:
            try:
                return json.loads(s[a:b + 1])
            except json.JSONDecodeError as e:
                raise ValueError(str(e)) from None
        raise ValueError("no JSON object found") from None
