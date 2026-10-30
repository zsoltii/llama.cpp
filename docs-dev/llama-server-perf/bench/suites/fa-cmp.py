import sys,collections
# compare fa-bench.sh labels: fa-cmp.py FILE (first label is the baseline)
d=collections.defaultdict(dict); labels=[]
for l in open(sys.argv[1]):
    p=l.split()
    if len(p)<5: continue
    d[(p[1],p[2],p[3])][p[0]]=float(p[4])
    if p[0] not in labels: labels.append(p[0])
print("kv        nb      type    " + "  ".join(f"{x:>10}" for x in labels))
for k in sorted(d, key=lambda k:(k[0],int(k[1].split('=')[1]),k[2])):
    b=d[k].get(labels[0])
    row=[f"{d[k].get(x,float('nan')):10.1f}" for x in labels]
    rel=[f"{100*(d[k][x]/b-1):+6.1f}%" for x in labels[1:] if x in d[k] and b]
    print(f"{k[0]:9} {k[1]:7} {k[2]:7} " + "  ".join(row) + "  " + " ".join(rel))
