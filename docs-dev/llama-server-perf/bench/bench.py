#!/usr/bin/env python3
"""llama-server benchmark, Qwen3.5-9B-MTP (default device: Vulkan2 = B50; --dev or $VK_DEV to change).

Usage:
  bench.py --name ref-mtp3 --bin llama-server -- --spec-type draft-mtp --spec-draft-n-max 3
  bench.py --name own-base --bin build-vk/bin/llama-server --env GGML_VK_FOO=1 -- ...

Every config: fresh server start, warm-up, then each scenario --runs times
(interleaved), cache_prompt=false, temp=0. Results go to $BENCH_OUT/results/, medians to $BENCH_OUT/summary.csv
(default $BENCH_OUT: ~/llama-bench-results).
"""
import argparse, csv, glob, hashlib, json, os, signal, statistics, subprocess, sys, threading, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "corpus")
OUT = os.environ.get("BENCH_OUT", os.path.expanduser("~/llama-bench-results"))

BASE_ARGS = ["-hf", "unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL", "-c", "128000", "-fa", "on",
             "--parallel", "1", "-t", "16", "-tb", "16", "-dev", os.environ.get("VK_DEV", "Vulkan2"),
             "-ctv", "q8_0", "-ctk", "q8_0"]


def read(name):
    with open(os.path.join(CORPUS, name), encoding="utf-8") as f:
        return f.read().strip()


SCENARIOS = {
    # short prompt, long answer -> tg dominated
    "story": lambda: (read("story.txt"), 1024),
    # pp only
    "wikipp": lambda: (read("long-wiki.txt") + "\n\n" + read("analysis-question.txt"), 1),
    # quick sanity check
    "quick": lambda: (read("story.txt"), 128),
    # ~100 KB wiki article + analysis -> pp dominated, then tg on long ctx
    "wiki": lambda: (read("long-wiki.txt") + "\n\n" + read("analysis-question.txt"), 512),
}


def post(port, path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def vram_kib(pid):
    """Resident VRAM of the process on the xe device (sum of distinct DRM clients)."""
    seen, total = set(), 0
    for fd in glob.glob(f"/proc/{pid}/fdinfo/*"):
        try:
            txt = open(fd).read()
        except OSError:
            continue
        if "drm-driver:\txe" not in txt:
            continue
        info = dict(l.split(":", 1) for l in txt.splitlines() if ":" in l)
        cid = info.get("drm-client-id", "").strip()
        if cid in seen:
            continue
        seen.add(cid)
        for k, v in info.items():
            if k.startswith("drm-resident-vram"):
                total += int(v.split()[0])
    return total


B50_GT = glob.glob("/sys/class/drm/card*/device/tile0/gt0/freq0/act_freq")
B50_GT = [p for p in B50_GT if "xe" in os.path.realpath(p.split("/tile0")[0] + "/driver")]


def gpu_cycles(pid):
    """Per engine class (busy cycles, total cycles) for the process on the xe device."""
    seen, acc = set(), {}
    for fd in glob.glob(f"/proc/{pid}/fdinfo/*"):
        try:
            txt = open(fd).read()
        except OSError:
            continue
        if "drm-driver:\txe" not in txt:
            continue
        info = dict(l.split(":", 1) for l in txt.splitlines() if ":" in l)
        cid = info.get("drm-client-id", "").strip()
        if cid in seen:
            continue
        seen.add(cid)
        for k, v in info.items():
            if k.startswith("drm-cycles-"):
                cls = k[len("drm-cycles-"):]
                busy, tot = acc.get(cls, (0, 0))
                acc[cls] = (busy + int(v.split()[0]), max(tot, int(info.get("drm-total-cycles-" + cls, "0").split()[0])))
    return acc


def proc_cpu_ticks(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12])  # utime + stime


def sys_cpu():
    f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(f) - f[3] - f[4], sum(f)  # busy, total


class Monitor(threading.Thread):
    """1 s sampling: VRAM, GPU busy % (per engine class), GPU freq, process and system CPU %."""

    def __init__(self, pid, path):
        super().__init__(daemon=True)
        self.pid, self.peak, self.stop, self.tag = pid, 0, False, "idle"
        self.samples = []
        self.out = open(path, "w")
        self.out.write("t,tag,vram_mib,gpu_busy_pct,gpu_engine,gpu_mhz,proc_cpu_pct,sys_cpu_pct\n")

    def run(self):
        hz = os.sysconf("SC_CLK_TCK")
        ncpu = os.cpu_count()
        t0 = time.time()
        pc, sc, gc, pt = proc_cpu_ticks(self.pid), sys_cpu(), gpu_cycles(self.pid), time.time()
        while not self.stop:
            time.sleep(1)
            try:
                pc2, sc2, gc2, pt2 = proc_cpu_ticks(self.pid), sys_cpu(), gpu_cycles(self.pid), time.time()
            except (OSError, IndexError):
                break
            vram = vram_kib(self.pid) / 1024
            self.peak = max(self.peak, vram * 1024)
            busy, eng = 0.0, ""
            for cls, (b, t) in gc2.items():
                b0, t0c = gc.get(cls, (0, 0))
                if t > t0c:
                    pct = 100.0 * (b - b0) / (t - t0c)
                    if pct > busy:
                        busy, eng = pct, cls
            proc_pct = 100.0 * (pc2 - pc) / hz / (pt2 - pt)  # 100 = one full core
            sys_pct = 100.0 * (sc2[0] - sc[0]) / max(1, sc2[1] - sc[1])  # whole machine
            mhz = int(open(B50_GT[0]).read()) if B50_GT else -1
            s = dict(t=pt2 - t0, tag=self.tag, vram=vram, busy=busy, eng=eng, mhz=mhz, proc=proc_pct, sys=sys_pct)
            self.samples.append(s)
            self.out.write(f"{s['t']:.1f},{self.tag},{vram:.0f},{busy:.1f},{eng},{mhz},{proc_pct:.0f},{sys_pct:.1f}\n")
            self.out.flush()
            pc, sc, gc, pt = pc2, sc2, gc2, pt2
        self.out.close()

    def stats(self, tag):
        ss = [s for s in self.samples if s["tag"] == tag]
        if not ss:
            return {}
        avg = lambda k: sum(s[k] for s in ss) / len(ss)
        return {"gpu_busy_pct": avg("busy"), "gpu_mhz": avg("mhz"), "proc_cpu_pct": avg("proc"),
                "sys_cpu_pct": avg("sys"), "n_samples": len(ss)}


def chat(port, prompt, n_predict, seed=1):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict, "temperature": 0, "top_k": 1, "seed": seed,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # BENCH_SAMPLING=server: use the sampling given on the server command line instead of greedy
    if os.environ.get("BENCH_SAMPLING") == "server":
        del body["temperature"], body["top_k"]
    t0 = time.time()
    r = post(port, "/v1/chat/completions", body)
    wall = time.time() - t0
    t = r.get("timings", {})
    text = r["choices"][0]["message"].get("content") or ""
    out = {
        "wall_s": wall,
        "prompt_n": t.get("prompt_n"), "pp_tps": t.get("prompt_per_second"), "ttft_ms": t.get("prompt_ms"),
        "gen_n": t.get("predicted_n"), "tg_tps": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
        "out_sha": hashlib.sha1(text.encode()).hexdigest()[:12],
        "out_head": text[:120].replace("\n", " "),
    }
    if out["draft_n"]:
        out["accept_rate"] = out["draft_accepted"] / out["draft_n"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bin", default="llama-server")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--dev", default=None, help="replace the -dev value of BASE_ARGS (e.g. CUDA0)")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = [x for x in a.extra if x != "--"]

    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    log_path = os.path.join(OUT, "results", f"{a.name}.log")
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    base = list(BASE_ARGS)
    if a.dev:
        base[base.index("-dev") + 1] = a.dev
    cmd = [os.path.expanduser(a.bin)] + base + ["--port", str(a.port)] + extra
    print("CMD:", " ".join(a.env), " ".join(cmd), flush=True)

    if healthy(a.port):
        sys.exit(f"port {a.port} already serving - stop the other server first")

    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        t0 = time.time()
        while not healthy(a.port):
            if proc.poll() is not None:
                sys.exit(f"server died (rc={proc.returncode}), see {log_path}")
            if time.time() - t0 > 900:
                sys.exit("server start timeout")
            time.sleep(1)
        print(f"server up in {time.time() - t0:.1f}s, pid {proc.pid}", flush=True)

        poller = Monitor(proc.pid, os.path.join(OUT, "results", f"{a.name}.mon.csv"))
        poller.start()
        poller.tag = "warmup"
        chat(a.port, "Szia! Hogy vagy?", 32)  # warm-up

        scen = a.scenarios.split(",")
        runs = {s: [] for s in scen}
        for i in range(a.runs):
            for s in scen:
                prompt, n = SCENARIOS[s]()
                poller.tag = s
                r = chat(a.port, prompt, n, seed=1 + i)
                poller.tag = "idle"
                runs[s].append(r)
                print(f"[{a.name}] {s} #{i}: pp {r['pp_tps']:.1f} t/s ({r['prompt_n']} tok), "
                      f"tg {r['tg_tps']:.2f} t/s ({r['gen_n']} tok), "
                      f"accept {r.get('accept_rate', float('nan')):.2f}, sha {r['out_sha']}", flush=True)
        poller.stop = True
        poller.join()
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()

    # sanity: the server must really have used the GPU (the VRAM monitor only knows the xe driver, i.e. Intel)
    logtxt = open(log_path, errors="replace").read()
    # the -dev check: VRAM on the xe (B50) device must be in use
    on_b50 = poller.peak > 4 * 1024 * 1024
    if not on_b50 and B50_GT:
        print("WARNING: less than 4 GiB VRAM on the B50 - model not offloaded?", flush=True)

    res = {"name": a.name, "cmd": cmd, "env": a.env, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "vram_peak_mib": poller.peak / 1024, "on_b50": on_b50, "runs": runs,
           "load": {s: poller.stats(s) for s in list(runs) + ["idle"]}}
    with open(os.path.join(OUT, "results", f"{a.name}.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    summary = os.path.join(OUT, "summary.csv")
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "name", "scenario", "prompt_n", "pp_tps", "ttft_ms", "gen_n", "tg_tps",
                        "accept_rate", "vram_peak_mib", "gpu_busy_pct", "gpu_mhz", "proc_cpu_pct", "sys_cpu_pct",
                        "deterministic", "env", "extra"])
        for s, rs in runs.items():
            med = lambda k: statistics.median(r[k] for r in rs if r.get(k) is not None) if any(
                r.get(k) is not None for r in rs) else ""
            w.writerow([res["time"], a.name, s, rs[0]["prompt_n"], f"{med('pp_tps'):.1f}", f"{med('ttft_ms'):.0f}",
                        rs[0]["gen_n"], f"{med('tg_tps'):.2f}",
                        f"{med('accept_rate'):.3f}" if med('accept_rate') != "" else "",
                        f"{res['vram_peak_mib']:.0f}",
                        *(f"{res['load'][s].get(k, float('nan')):.0f}" for k in ("gpu_busy_pct", "gpu_mhz", "proc_cpu_pct", "sys_cpu_pct")),
                        len({r['out_sha'] for r in rs}) == 1,
                        " ".join(a.env), " ".join(extra)])
            ld = res["load"][s]
            print(f"MEDIAN {a.name} {s}: pp {med('pp_tps'):.1f} t/s, tg {med('tg_tps'):.2f} t/s | "
                  f"GPU {ld.get('gpu_busy_pct', 0):.0f}% @ {ld.get('gpu_mhz', 0):.0f} MHz, "
                  f"proc CPU {ld.get('proc_cpu_pct', 0):.0f}% (100=1 core), sys CPU {ld.get('sys_cpu_pct', 0):.0f}%", flush=True)
    print(f"VRAM peak {res['vram_peak_mib']:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
