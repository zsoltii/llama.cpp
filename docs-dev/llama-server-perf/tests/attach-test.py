#!/usr/bin/env python3
"""Attachment test: send a long Hungarian document (the 2011 Fundamental Law of Hungary, ~41k tokens) the way
the llama-server web UI sends a .txt attachment, ask for a summary, and record speed, draft acceptance and
repetition loops (separately for the thinking and the answer).

The web UI sends a text attachment as a content part "\\n\\n--- File: <name> ---\\n<content>", followed by the
prompt as a second text part (tools/ui/src/lib/services/chat.service.ts). The same seed is used for every
config, so different builds/GPUs can be compared run by run.

usage:
  attach-test.py --name b50-vk --bin build-vk/bin/llama-server --dev Vulkan2 [--sampling qwen] [--seeds 1,2,3]
                 [--env K=V ...] [-- extra server args]

sampling: default (server defaults), qwen (Qwen3.5 recommendation), qwen-dry (+ DRY)
output: <out>/attach-<name>.txt (texts), <out>/attach-<name>.jsonl (one record per run), <out>/attach-<name>.srv.log
"""
import argparse, json, os, subprocess, sys, time, urllib.request

HERE    = os.path.dirname(os.path.abspath(__file__))
DOC     = os.path.join(HERE, "..", "testdata", "alaptorveny-2011.txt")
DOCNAME = "Magyarország Alaptörvénye - Nemzeti Jogszabálytár.txt"
PROMPT  = "foglald össze a magyar alaptörvényt"

SAMPLING = {
    "default":  [],
    "qwen":     ["--temp", "1.0", "--top-k", "20", "--top-p", "0.95", "--min-p", "0", "--presence-penalty", "1.5"],
    "qwen-dry": ["--temp", "1.0", "--top-k", "20", "--top-p", "0.95", "--min-p", "0", "--presence-penalty", "1.5",
                 "--dry-multiplier", "0.8", "--dry-penalty-last-n", "16384"],
}

# a loop is a repeating unit of period 1..200 chars that repeats at least 3 times and spans >= LOOP_CHARS chars
LOOP_CHARS = 300


def loop_score(t):
    best = 0
    for per in range(1, 201):
        run = 0
        for i in range(per, len(t)):
            if t[i] == t[i - per]:
                run += 1
                if run > best and run >= per * 3:
                    best = run
            else:
                run = 0
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--dev", required=True, help="e.g. Vulkan2 or CUDA0")
    ap.add_argument("--sampling", default="qwen", choices=SAMPLING.keys())
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL")
    ap.add_argument("--out", default=os.path.join(os.environ.get("BENCH_OUT", os.path.expanduser("~/llama-bench-results")), "attach-test"))
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--no-ui", action="store_true", help="plain request, without the extra fields of the web UI")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"attach-{a.name}")

    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v

    mtp = [] if a.no_mtp else ["-ctkd", "q8_0", "-ctvd", "q8_0", "--spec-type", "draft-mtp", "--spec-draft-n-max", "1", "-bs"]
    cmd = [a.bin, "-hf", "unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL", "-c", "128000", "-fa", "on", "--parallel", "1",
           "-t", "16", "-tb", "16", "-dev", a.dev, "--port", str(a.port), "-ctk", "q8_0", "-ctv", "q8_0"] \
          + mtp + SAMPLING[a.sampling] + a.extra
    print("CMD:", " ".join(a.env), " ".join(cmd), flush=True)

    doc = open(DOC, encoding="utf-8").read()
    content = [
        {"type": "text", "text": f"\n\n--- File: {DOCNAME} ---\n{doc}"},
        {"type": "text", "text": PROMPT},
    ]

    log = open(base + ".srv.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    out  = open(base + ".txt", "w")
    recs = open(base + ".jsonl", "w")
    n_bad = 0
    seeds = [int(s) for s in a.seeds.split(",")]
    try:
        for _ in range(600):
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)).get("status") == "ok":
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                sys.exit(f"server died, see {base}.srv.log")
            time.sleep(1)

        for seed in seeds:
            body = {"messages": [{"role": "user", "content": content}], "max_tokens": a.max_tokens, "seed": seed,
                    "stream": True, "timings_per_token": True, "cache_prompt": False}
            if not a.no_ui:
                # the fields the web UI adds to every request; reasoning_control arms the (CPU) reasoning budget sampler
                body.update({"reasoning_format": "auto", "reasoning_control": True,
                             "return_progress": True, "sse_ping_interval": 1})
            req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            t0 = time.time()
            resp = urllib.request.urlopen(req, timeout=3600)
            rs, cs, tim, fin, ls, chk = [], [], {}, None, 0, 0
            for raw in resp:
                raw = raw.decode("utf-8", "replace").strip()
                if not raw.startswith("data: ") or raw == "data: [DONE]":
                    continue
                ev = json.loads(raw[6:])
                if ev.get("timings"):
                    tim = ev["timings"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("reasoning_content"):
                        rs.append(d["reasoning_content"])
                    if d.get("content"):
                        cs.append(d["content"])
                    fin = ch.get("finish_reason") or fin
                n_ch = sum(map(len, rs)) + sum(map(len, cs))
                if n_ch - chk > 400:
                    chk = n_ch
                    ls = loop_score(("".join(rs) + "".join(cs))[-1500:])
                    if ls >= LOOP_CHARS:
                        fin = "LOOP-ABORT"
                        break
            resp.close()

            r, c = "".join(rs), "".join(cs)
            loop_r = loop_score(r[-1500:])
            loop_c = loop_score(c[-1500:])
            # a run is not ok if it loops, or if it never got out of the thinking phase
            loop  = fin == "LOOP-ABORT" or max(loop_r, loop_c) >= LOOP_CHARS
            stuck = not c.strip()
            ok = not loop and not stuck and fin == "stop"
            n_bad += not ok

            rec = {
                "name": a.name, "seed": seed, "ok": ok, "finish": fin,
                "loop_reasoning": loop_r, "loop_content": loop_c, "stuck_in_thinking": stuck,
                "prompt_n": tim.get("prompt_n"), "pp_tps": tim.get("prompt_per_second"),
                "n": tim.get("predicted_n"), "tg_tps": tim.get("predicted_per_second"),
                "draft_n": tim.get("draft_n"), "draft_n_accepted": tim.get("draft_n_accepted"),
                "rchars": len(r), "cchars": len(c), "wall_s": round(time.time() - t0, 1),
            }
            acc = (rec["draft_n_accepted"] / rec["draft_n"]) if rec["draft_n"] else 0
            print(f"{'ok  ' if ok else 'BAD '} {a.name} seed={seed} pp={rec['pp_tps'] or 0:.1f} ({rec['prompt_n']} tok) "
                  f"tg={rec['tg_tps'] or 0:.2f} n={rec['n']} accept={acc:.2f} finish={fin} "
                  f"loop(r/c)={loop_r}/{loop_c} rchars={len(r)} cchars={len(c)}", flush=True)
            recs.write(json.dumps(rec, ensure_ascii=False) + "\n"); recs.flush()
            out.write(f"##### {json.dumps(rec, ensure_ascii=False)}\n{r}\n=====\n{c}\n\n"); out.flush()
    finally:
        os.killpg(proc.pid, 15)
        proc.wait()

    print(f"not ok: {n_bad}/{len(seeds)}")


if __name__ == "__main__":
    main()
