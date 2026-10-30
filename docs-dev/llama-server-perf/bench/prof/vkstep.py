#!/usr/bin/env python3
"""Sum GGML_VK_PERF_LOGGER op times over all non-prefill blocks and divide by the number of MTP steps.
usage: vkstep.py LOG N_STEPS [top]"""
import re, sys, collections
log = open(sys.argv[1], errors="replace").read().split("Vulkan Timings:")[1:]
steps = int(sys.argv[2]); top = int(sys.argv[3]) if len(sys.argv) > 3 else 45
line_re = re.compile(r"^(.*): (\d+) x ([\d.]+) us = ([\d.]+) us")
tot = collections.defaultdict(float); cnt = collections.defaultdict(int); pre = 0.0
for blk in log:
    ops = [(m.group(1), int(m.group(2)), float(m.group(4))) for m in (line_re.match(l.strip()) for l in blk.splitlines()) if m]
    names = " | ".join(o[0] for o in ops)
    wn = [int(x) for x in re.findall(r"MUL_MAT(?:_VEC)? q\d_K m=\d+ n=(\d+)", names)]
    if wn and max(wn) >= 16:
        pre += sum(o[2] for o in ops); continue
    for name, n, t in ops:
        key = re.sub(r"dst\(.*", "", name) if name.startswith("FLASH") else name
        if name.startswith("FLASH"):
            key = "FLASH_ATTN_EXT " + re.search(r"q\(\d+,(\d+),", name).group(1) + " rows, kv " + re.search(r"k\(\d+,(\d+),", name).group(1)
        tot[key] += t; cnt[key] += n
T = sum(tot.values())
print(f"prefill GPU total {pre/1000:.1f} ms; decode GPU {T/1000:.1f} ms over {steps} steps = {T/steps/1000:.2f} ms/step")
for k, t in sorted(tot.items(), key=lambda kv: -kv[1])[:top]:
    print(f"{t/steps/1000:7.3f} ms {100*t/T:5.1f}%  {cnt[k]/steps:6.2f}x/step  {k}")
