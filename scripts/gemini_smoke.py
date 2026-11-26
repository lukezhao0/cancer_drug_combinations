"""Quick live check that the Gemini API (OpenAI-compatible endpoint) works.

Reads OPENAI_BASE_URL / OPENAI_API_KEY from the environment or ./.env, then runs:
  1. list models   - key is valid and the endpoint is reachable; is the model listed?
  2. completion    - one short non-streaming call, with latency and token usage
  3. streaming     - tokens printed to the console as they arrive
  4. json          - structured (JSON) output parses

    uv run python scripts/gemini_smoke.py
    uv run python scripts/gemini_smoke.py --model gemini-4-hydrogen
    uv run python scripts/gemini_smoke.py --only stream --prompt "Explain BLISS synergy in 3 sentences."

Exit code is 0 only if every selected check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

DEFAULT_MODEL = "gemini-3.5-flash-lite"
CHECKS = ("models", "complete", "stream", "json")


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env loader (no python-dotenv dependency); existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def usage_str(u) -> str:
    return f"tokens in={u.prompt_tokens} out={u.completion_tokens}" if u else "no usage reported"


def check_models(client: OpenAI, model: str) -> None:
    ids = sorted(m.id.removeprefix("models/") for m in client.models.list())
    gem = [i for i in ids if "gemini" in i]
    log(f"{len(ids)} models visible ({len(gem)} gemini)")
    if model not in ids:
        raise RuntimeError(f"{model!r} not in model list; gemini models: {', '.join(gem[:20])}")
    log(f"{model} is available")


def check_complete(client: OpenAI, model: str, prompt: str) -> None:
    t0 = time.perf_counter()
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    text = (r.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(f"empty response (finish_reason={r.choices[0].finish_reason})")
    log(f"{time.perf_counter() - t0:.2f}s, {usage_str(r.usage)}, finish={r.choices[0].finish_reason}")
    print(f"    > {text}", flush=True)


def check_stream(client: OpenAI, model: str, prompt: str) -> None:
    t0 = time.perf_counter()
    first = None
    n_chunks, parts = 0, []
    stream = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                            stream=True)
    print("    > ", end="", flush=True)
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            first = first or time.perf_counter()
            n_chunks += 1
            parts.append(delta)
            print(delta.replace("\n", "\n      "), end="", flush=True)
    print(flush=True)
    if not parts:
        raise RuntimeError("stream produced no content")
    log(f"first token {first - t0:.2f}s, total {time.perf_counter() - t0:.2f}s, {n_chunks} chunks")


def check_json(client: OpenAI, model: str) -> None:
    prompt = ('Return a JSON object with keys "drug" (string) and "target" (string) for imatinib. '
              "Output only the JSON.")
    t0 = time.perf_counter()
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                       response_format={"type": "json_object"})
    raw = r.choices[0].message.content or ""
    obj = json.loads(raw)
    if not {"drug", "target"} <= obj.keys():
        raise RuntimeError(f"missing keys in {obj}")
    log(f"{time.perf_counter() - t0:.2f}s, {usage_str(r.usage)}")
    print(f"    > {obj}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="In one sentence, what is drug synergy?")
    ap.add_argument("--only", choices=CHECKS, action="append", help="run only these checks (repeatable)")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    load_env()
    base, key = os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY")
    if not key:
        log("FAIL: OPENAI_API_KEY is not set (env or ./.env)")
        return 1
    log(f"endpoint {base or '(OpenAI default)'}  key ...{key[-4:]}  model {args.model}")
    client = OpenAI(base_url=base, api_key=key, timeout=args.timeout, max_retries=0)

    steps = {
        "models": lambda: check_models(client, args.model),
        "complete": lambda: check_complete(client, args.model, args.prompt),
        "stream": lambda: check_stream(client, args.model, args.prompt),
        "json": lambda: check_json(client, args.model),
    }
    results = {}
    for name in args.only or CHECKS:
        log(f"--- {name} ---")
        try:
            steps[name]()
            results[name] = True
            log(f"PASS {name}")
        except Exception as e:  # report every failure, keep going
            print(flush=True)
            results[name] = False
            cause = e.__cause__ or e.__context__
            detail = f" (cause: {type(cause).__name__}: {cause})" if cause else ""
            log(f"FAIL {name}: {type(e).__name__}: {e}{detail}")

    ok = all(results.values())
    log("summary: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
