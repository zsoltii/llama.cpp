#!/usr/bin/env python3
# mtp-cancel-cache.py PORT: regression test for the MTP deferred catch-up (LLAMA_MTP_DEFER_CATCHUP) after a prompt-cache restore.
# P1 (long) -> P2 (different prompt, generation cancelled) -> P1 + tail: the server restores P1 from its prompt cache while the
# catch-up rows of P2 are still pending. Before the fix the last request failed with "inconsistent sequence positions".
# Start the server with an MTP model, --parallel 1 and LLAMA_MTP_DEFER_CATCHUP=1.
import json, sys, time, urllib.request, http.client
port = int(sys.argv[1])
R = "/home/zsoltii/IdeaProjects/llama.cpp/docs-dev/llama-server-perf/"
seed = open(R + "testdata/seed-hu.txt").read()
wiki = open(R + "bench/corpus/long-wiki.txt").read()
P1 = seed[:60000] + "\n\nSummarize the text above in two sentences:\n"
P2 = wiki[:20000] + "\n\nWrite a long essay about this:\n"

def complete(prompt, n):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps({"prompt": prompt, "n_predict": n, "cache_prompt": True, "seed": 1}).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=1800))
    except Exception as e:
        return {"error": str(e)}

def stream_cancel(prompt, keep):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    conn.request("POST", "/completion", json.dumps({"prompt": prompt, "n_predict": 3000, "stream": True, "cache_prompt": True, "seed": 1}).encode(), {"Content-Type": "application/json"})
    r = conn.getresponse(); n = 0
    for line in r:
        if line.decode().strip().startswith("data:"):
            n += 1
            if n >= keep: break
    conn.close()

r = complete(P1, 40); print("P1:", "ok" if "content" in r else r, flush=True)
stream_cancel(P2, 150); print("P2 cancelled", flush=True); time.sleep(0.5)
r = complete(P1 + seed[60000:63000], 40)
print("P1+tail:", "ok" if "content" in r and "error" not in r else "FAIL " + str(r)[:200], flush=True)
