#!/usr/bin/env python3
# post-hoc loop scoring of loop-test.py output files.
# A run is LOOP if it was aborted on the fly (finish=LOOP-ABORT), or if the last 200 chars of its reasoning or its
# content occur >= 3 times within the last 12000 chars of that part. A run that hit the token limit without either
# is RUNAWAY (e.g. paraphrased repetition such as "ensure the story is X" with a different X every line).
import sys

def rep(part):
    part = part.rstrip("\n")
    if len(part) < 400:
        return 1
    return part[-12000:].count(part[-200:])

n_bad = n_run = 0
for f in sys.argv[1:]:
    for blk in open(f).read().split('##### ')[1:]:
        h, _, body = blk.partition('\n')
        reasoning, _, content = body.partition('\n=====\n')
        r = max(rep(reasoning), rep(content))
        aborted = 'finish=LOOP-ABORT' in h
        at_limit = 'finish=length' in h
        verdict = 'LOOP   ' if aborted or r >= 3 else ('RUNAWAY' if at_limit else 'ok     ')
        n_run += 1
        n_bad += verdict != 'ok     '
        print(verdict, h.split(' finish')[0], 'finish' + h.split(' finish')[1].split(' LOOP')[0], f'rep={r}')
print(f'not ok: {n_bad}/{n_run}')
