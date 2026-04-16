#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from itertools import count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously send simple requests to an OpenAI-compatible vLLM endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default=None, help="Model name. If omitted, auto-detect from /models.")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent request loops.")
    parser.add_argument("--prompt", default="Write one short sentence about prime numbers.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between requests per worker.")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def http_json(url: str, payload: dict | None, api_key: str, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_model(base_url: str, model: str | None, api_key: str, timeout: float) -> str:
    if model:
        return model
    data = http_json(f"{base_url.rstrip('/')}/models", None, api_key, timeout)
    models = data.get("data") or []
    if not models:
        raise RuntimeError("No models returned from /models")
    return models[0]["id"]


def worker_loop(
    worker_id: int,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    sleep_s: float,
    stats: dict,
    lock: threading.Lock,
) -> None:
    request_counter = count(1)
    url = f"{base_url.rstrip('/')}/chat/completions"

    while True:
        req_id = next(request_counter)
        started = time.time()
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt} Request #{worker_id}-{req_id}.",
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            data = http_json(url, payload, api_key, timeout)
            elapsed = time.time() - started
            usage = data.get("usage") or {}
            output_tokens = usage.get("completion_tokens", "?")
            with lock:
                stats["ok"] += 1
                stats["last_ok"] = time.strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[worker {worker_id}] ok req={req_id} "
                f"elapsed={elapsed:.2f}s output_tokens={output_tokens}",
                flush=True,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            with lock:
                stats["errors"] += 1
                stats["last_error"] = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[worker {worker_id}] http_error req={req_id} code={exc.code} body={body}", flush=True)
            time.sleep(max(1.0, sleep_s))
        except Exception as exc:
            with lock:
                stats["errors"] += 1
                stats["last_error"] = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[worker {worker_id}] error req={req_id} err={type(exc).__name__}: {exc}", flush=True)
            time.sleep(max(1.0, sleep_s))

        if sleep_s > 0:
            time.sleep(sleep_s)


def main() -> None:
    args = parse_args()
    model = resolve_model(args.base_url, args.model, args.api_key, args.timeout)
    print(
        f"Starting keep_vllm_busy against {args.base_url} model={model} workers={args.workers}",
        flush=True,
    )

    stats = {"ok": 0, "errors": 0, "last_ok": None, "last_error": None}
    lock = threading.Lock()

    threads = []
    for worker_id in range(1, args.workers + 1):
        thread = threading.Thread(
            target=worker_loop,
            args=(
                worker_id,
                args.base_url,
                model,
                args.api_key,
                args.prompt,
                args.max_tokens,
                args.temperature,
                args.timeout,
                args.sleep,
                stats,
                lock,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    try:
        while True:
            time.sleep(30)
            with lock:
                ok = stats["ok"]
                errors = stats["errors"]
                last_ok = stats["last_ok"]
                last_error = stats["last_error"]
            print(
                f"[summary] ok={ok} errors={errors} last_ok={last_ok} last_error={last_error}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("Stopping keep_vllm_busy.", flush=True)


if __name__ == "__main__":
    main()
