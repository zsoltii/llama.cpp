#!/usr/bin/env python3
"""Kernel micro-bench of the Qwen3.5-9B decode shapes via test-backend-ops perf (Vulkan device $VK_IDX, default 2 = B50).
usage: kbench.py NAME [KEY=VAL ...]   -> $BENCH_OUT/kernels/NAME.txt, prints us/run per shape"""
import os, re, subprocess, sys
name, envs = sys.argv[1], sys.argv[2:]
env = dict(os.environ, GGML_VK_VISIBLE_DEVICES=os.environ.get("VK_IDX", "2"))
for kv in envs:
    k, v = kv.split("=", 1); env[k] = v
bin_ = os.path.expanduser(os.environ.get("TBO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../build-vk/bin/test-backend-ops")))
shapes = [("q4_K",12288,4096),("q4_K",4096,12288),("q5_K",12288,4096),("q5_K",4096,4096),("q6_K",4096,12288),
          ("q6_K",8192,4096),("q6_K",248320,4096),("q8_0",4096,4096),("q8_0",4096,8192)]
out = subprocess.run([bin_, "perf", "-b", "Vulkan0", "-o", "MUL_MAT,FLASH_ATTN_EXT", "-p", "type_a=(q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(12288|4096|8192|248320),n=[123],k=(4096|12288|8192),|kv=20736"],
                     env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
dev = re.search(r"ggml_vulkan: 0 = (.*?) \|", out)
print("device:", dev[1] if dev else "?", file=sys.stderr)
res = {}
for line in out.splitlines():
    m = re.search(r"MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1.*?([\d.]+) us/run", line)
    if m and (m[1], int(m[2]), int(m[4])) in shapes:
        res[f"{m[1]} {m[2]}x{m[4]} n={m[3]}"] = float(m[5])
    m = re.search(r"FLASH_ATTN_EXT\(.*kv=20736,nb=(\d+).*?type_KV=(\w+).*?([\d.]+) us/run", line)
    if m:
        res[f"FA {m[2]} kv=20736 nb={m[1]}"] = float(m[3])
out_dir = os.path.join(os.environ.get("BENCH_OUT", os.path.expanduser("~/llama-bench-results")), "kernels")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, name + ".txt"), "w") as f:
    for k, v in res.items():
        f.write(f"{k}\t{v}\n")
for k, v in res.items():
    print(f"{k:32s} {v:9.1f} us")
