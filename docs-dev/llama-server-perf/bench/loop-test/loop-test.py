#!/usr/bin/env python3
"""Start llama-server with a config, send story prompts with fixed seeds, detect repetition loops.
usage: loop-test.py NAME BIN "ENV=.. ENV=.." "extra args" seeds...   -> $BENCH_OUT/loop-test/loop-NAME.{txt,srv.log}"""
import json, os, subprocess, sys, time, urllib.request, zlib, re

name, binp, envs, extra = sys.argv[1:5]
seeds = [int(s) for s in sys.argv[5:]] or [1, 2, 3]
S = os.path.join(os.environ.get("BENCH_OUT", os.path.expanduser("~/llama-bench-results")), "loop-test")
os.makedirs(S, exist_ok=True)
MAXTOK = int(os.environ.get("MAXTOK", "5000"))
PROMPTS = [
    "tervezz meg egy 10000 szavas gyerekenek szóló mesét, amit később meg kell írnod",
    "csinálj 10000 szavas mesét avgy hosszabbat",
]

env = dict(os.environ)
for e in envs.split():
    k, v = e.split("=", 1); env[k] = v
args = [binp, "-hf", "unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL", "-c", "128000", "-fa", "on", "--parallel", "1",
        "-t", "16", "-tb", "16", "--port", "8091", "-ctk", "q8_0", "-ctv", "q8_0"] + extra.split()
log = open(f"{S}/loop-{name}.srv.log", "w")
p = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)

def loop_score(t):
    # longest run of a repeating unit (period 1..200 chars) at any position, in chars
    best = 0
    for per in range(1, 201):
        run = 0
        for i in range(per, len(t)):
            if t[i] == t[i - per]:
                run += 1
                if run > best and run >= per * 3: best = run
            else:
                run = 0
    return best

try:
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2)).get("status") == "ok": break
        except Exception: pass
        if p.poll() is not None: sys.exit(f"server died, see {S}/loop-{name}.srv.log")
        time.sleep(1)
    out = open(f"{S}/loop-{name}.txt", "w")
    for pi, pr in enumerate(PROMPTS):
        if os.environ.get("ONLY_P") and str(pi) != os.environ["ONLY_P"]: continue
        for s in seeds:
            body = {"messages": [{"role": "user", "content": pr}], "max_tokens": MAXTOK, "seed": s,
                    "stream": True, "timings_per_token": True}
            req = urllib.request.Request("http://127.0.0.1:8091/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=1200)
            rs, cs, t, fin, ls, chk = [], [], {}, None, 0, 0
            for raw in resp:
                raw = raw.decode("utf-8", "replace").strip()
                if not raw.startswith("data: ") or raw == "data: [DONE]": continue
                ev = json.loads(raw[6:])
                if ev.get("timings"): t = ev["timings"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("reasoning_content"): rs.append(d["reasoning_content"])
                    if d.get("content"): cs.append(d["content"])
                    fin = ch.get("finish_reason") or fin
                n_ch = sum(map(len, rs)) + sum(map(len, cs))
                if n_ch - chk > 400:
                    chk = n_ch
                    ls = loop_score(("".join(rs) + "".join(cs))[-1500:])
                    if ls >= 300: fin = "LOOP-ABORT"; break
            resp.close()
            txt = "".join(rs) + "\n=====\n" + "".join(cs)
            ls = max(ls, loop_score(txt[-1500:]))
            n = t.get("predicted_n"); acc = t.get("draft_n_accepted"), t.get("draft_n")
            line = f"{name} p{pi} seed={s} n={n} tg={t.get('predicted_per_second',0):.1f} draft={acc} finish={fin} LOOP={ls} rchars={sum(map(len,rs))} cchars={sum(map(len,cs))}"
            print(line, flush=True)
            out.write(f"##### {line}\n{txt}\n\n"); out.flush()
finally:
    p.terminate(); p.wait()
