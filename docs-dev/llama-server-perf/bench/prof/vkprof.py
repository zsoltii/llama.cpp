#!/usr/bin/env python3
"""Aggregate GGML_VK_PERF_LOGGER blocks from a server log.
Each "Vulkan Timings:" block is one graph. Graphs are classified by their LM-head mat-vec:
  verify  = head with n>=2 (target verify), decode1 = head n=1 (target, no MTP),
  draft   = MTP graph (contains the eh_proj-like m=4096 k=8192 matvec or the head with n=1 and few ops),
  prefill = any MUL_MAT with n>=64.
usage: vkprof.py LOG [kind ...]"""
import re, sys, collections

log = open(sys.argv[1], errors="replace").read().split("Vulkan Timings:")[1:]
want = sys.argv[2:]
line_re = re.compile(r"^(.*): (\d+) x ([\d.]+) us = ([\d.]+) us")

groups = collections.defaultdict(list)
for blk in log:
    ops = []
    for ln in blk.splitlines():
        m = line_re.match(ln.strip())
        if m:
            ops.append((m.group(1), int(m.group(2)), float(m.group(4))))
    if not ops:
        continue
    names = " | ".join(o[0] for o in ops)
    wn = [int(x) for x in re.findall(r"MUL_MAT(?:_VEC)? q\d_K m=\d+ n=(\d+)", names)]
    if wn and max(wn) >= 16:
        kind = "prefill"
    elif re.search(r"m=248320 n=([2-9])", names):
        kind = "verify"
    elif "m=248320 n=1" in names and len(ops) < 40:
        kind = "draft"
    elif "m=248320 n=1" in names:
        kind = "decode1"
    else:
        kind = "other"
    groups[kind].append(ops)

for kind, blks in groups.items():
    if want and kind not in want:
        continue
    tot = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    for ops in blks:
        for name, n, t in ops:
            # strip shapes for small ops so they aggregate
            key = name
            tot[key] += t
            cnt[key] += n
    nb = len(blks)
    total = sum(tot.values()) / nb
    print(f"\n##### {kind}: {nb} graphs, avg GPU {total/1000:.2f} ms/graph")
    for key, t in sorted(tot.items(), key=lambda kv: -kv[1])[:40]:
        print(f"{t/nb/1000:8.3f} ms {100*t/nb/total:5.1f}%  {cnt[key]/nb:6.1f}x  {key}")
