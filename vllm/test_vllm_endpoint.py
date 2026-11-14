#!/usr/bin/env python3
"""
Exercise a running vLLM OpenAI endpoint: list models, do a plain chat (report
content + reasoning_content), and — if --tools — a tool-calling round-trip.
Pure stdlib (works with the cluster's system python3). Used by serve_vllm.sh's
smoke test and runnable standalone:

    python3 test_vllm_endpoint.py --url http://<node>:8000 --model <served-name> --tools \
        --api-key-file <repo>/output/vllm_<served-name>_<jobid>.apikey

API key: serve_vllm.sh servers require one by default. It is read from
--api-key-file, else from $VLLM_API_KEY — never taken as a command-line value
(that would show in `ps`). With neither, requests go out unauthenticated.
"""
import argparse, json, os, sys, urllib.request

API_KEY = None

def post(base, path, payload=None, timeout=180):
    url = base.rstrip("/") + path
    headers = {"Authorization": "Bearer " + API_KEY} if API_KEY else {}
    if payload is None:
        req = urllib.request.Request(url, headers=headers)
    else:
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def main():
    global API_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--api-key-file", help="file holding the server's API key (else $VLLM_API_KEY)")
    ap.add_argument("--auth-check", action="store_true",
                    help="only check that this key is accepted and --model is served, then exit")
    a = ap.parse_args()
    if a.api_key_file:
        with open(a.api_key_file) as f:
            API_KEY = f.read().strip()
    else:
        API_KEY = os.environ.get("VLLM_API_KEY") or None

    # identity check used by serve_vllm.sh: is the server on this port really ours?
    if a.auth_check:
        try:
            ids = [m["id"] for m in post(a.url, "/v1/models", timeout=30).get("data", [])]
        except Exception as e:
            print("[auth-check] FAIL:", e); sys.exit(1)
        if a.model not in ids:
            print("[auth-check] FAIL: %r not served here (got %s)" % (a.model, ids)); sys.exit(1)
        sys.exit(0)
    ok = True

    # 1) model list
    try:
        ids = [m["id"] for m in post(a.url, "/v1/models").get("data", [])]
        print("[models]", ids)
    except Exception as e:
        print("[models] FAIL:", e); ok = False

    # 2) plain chat (+ reasoning_content if the model emits it)
    try:
        r = post(a.url, "/v1/chat/completions", {
            "model": a.model,
            "messages": [{"role": "user", "content": "In one sentence, what is InfiniBand?"}],
            "max_tokens": 256,
        })
        msg = r["choices"][0]["message"]
        print("[chat] content:", (msg.get("content") or "").strip()[:220])
        rc = msg.get("reasoning_content")
        print("[chat] reasoning_content:", ("<%d chars>" % len(rc)) if rc else "none")
    except Exception as e:
        print("[chat] FAIL:", e); ok = False

    # 3) tool calling
    if a.tools:
        try:
            tools = [{"type": "function", "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"}},
                               "required": ["city"]}}}]
            r = post(a.url, "/v1/chat/completions", {
                "model": a.model,
                "messages": [{"role": "user", "content": "What's the weather in Paris? Call the get_weather tool."}],
                "tools": tools, "tool_choice": "auto", "max_tokens": 256,
            })
            msg = r["choices"][0]["message"]
            tc = msg.get("tool_calls")
            if tc:
                print("[tools] tool_calls:", [(t["function"]["name"], t["function"].get("arguments")) for t in tc])
            else:
                print("[tools] NO tool_calls returned; content:", (msg.get("content") or "").strip()[:150]); ok = False
        except Exception as e:
            print("[tools] FAIL:", e); ok = False

    print("[RESULT]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
