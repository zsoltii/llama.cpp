#!/usr/bin/env python3
# Hybrid bge-m3 embeddings of one llama-server against another (a reference backend, e.g. the CPU one):
#   bge-backend-cmp.py encode URL TEXTS.txt OUT.json [BATCH]   texts: paragraphs separated by blank lines
#   bge-backend-cmp.py compare REF.json OUT.json
# compare prints the dense cosine (min / mean), the sparse relative L1 and the top-1 / top-5 retrieval overlap
# (every text as a query against all others, dense and sparse scores).
import json, sys, time, urllib.request


def encode(url, path, out, batch=16):
    texts = [t.strip() for t in open(path).read().split("\n\n") if t.strip()]
    res = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        req = urllib.request.Request(url + "/embeddings", json.dumps({"content": texts[i:i + batch]}).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        if isinstance(r, dict):
            sys.exit(f"server error: {r}")
        for x in sorted(r, key=lambda x: x["index"]):
            sp = x.get("sparse", {})
            res.append({"dense": x["embedding"][0], "sparse": dict(zip(map(str, sp.get("indices", [])), sp.get("values", [])))})
    dt = time.time() - t0
    json.dump({"texts": len(texts), "seconds": dt, "res": res}, open(out, "w"))
    print(f"{len(texts)} texts in {dt:.2f} s")


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cos(a, b):
    return dot(a, b) / (dot(a, a) ** 0.5 * dot(b, b) ** 0.5)


def sdot(a, b):
    return sum(v * b[k] for k, v in a.items() if k in b)


def ranks(res, kind):
    n = len(res)
    out = []
    for i in range(n):
        if kind == "dense":
            s = [cos(res[i]["dense"], res[j]["dense"]) if j != i else -9 for j in range(n)]
        else:
            s = [sdot(res[i]["sparse"], res[j]["sparse"]) if j != i else -9 for j in range(n)]
        out.append(sorted(range(n), key=lambda j: -s[j])[:5])
    return out


def compare(ref_path, out_path):
    ref = json.load(open(ref_path))["res"]
    out = json.load(open(out_path))["res"]
    assert len(ref) == len(out), (len(ref), len(out))
    c = [cos(a["dense"], b["dense"]) for a, b in zip(ref, out)]
    l1 = sum(abs(a["sparse"].get(k, 0) - b["sparse"].get(k, 0)) for a, b in zip(ref, out)
             for k in set(a["sparse"]) | set(b["sparse"]))
    tot = sum(sum(a["sparse"].values()) for a in ref)
    same_keys = sum(set(a["sparse"]) == set(b["sparse"]) for a, b in zip(ref, out))
    print(f"texts {len(ref)}  dense cos min {min(c):.6f} mean {sum(c) / len(c):.6f}  "
          f"sparse rel L1 {100 * l1 / tot:.3f}%  same sparse tokens {same_keys}/{len(ref)}")
    for kind in ("dense", "sparse"):
        r, o = ranks(ref, kind), ranks(out, kind)
        top1 = sum(a[0] == b[0] for a, b in zip(r, o)) / len(r)
        top5 = sum(len(set(a) & set(b)) for a, b in zip(r, o)) / (5 * len(r))
        print(f"  {kind:6} top-1 {top1:.3f}  top-5 overlap {top5:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "encode":
        encode(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 16)
    else:
        compare(sys.argv[2], sys.argv[3])
