#!/usr/bin/env python3
"""Grounded QA / hallucination check on the long wiki corpus (level 3, custom).

The whole Roman Empire article (~20.7k tokens) is the system context; each question is answerable from the
text (expected answer = any of the given substrings) or deliberately NOT answerable from it (expected answer:
the model says NOT_IN_TEXT).  The article prefix is identical for every question, so the server's prompt cache
makes only the first request pay the full prompt processing.
  wiki_qa.py --name NAME [--port 8080]  -> $BENCH_OUT/eval/wikiqa-NAME.json, prints score
"""
import argparse, json, os, re, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.join(ROOT, "..", "corpus", "long-wiki.txt")

# (question, [accepted answers, case-insensitive substrings]) ; None = not answerable from the text
QA = [
    ("In which year did the Western Roman Empire collapse?", ["476"]),
    ("Who deposed Romulus Augustus?", ["odoacer"]),
    ("In which battle were three legions wiped out in 9 AD?", ["teutoburg"]),
    ("After that defeat, to roughly how many was the number of legions increased?", ["30", "thirty"]),
    ("How many officials did the central government send out annually to govern outside Italy in the 1st and 2nd centuries?", ["160"]),
    ("What census valuation qualified a man as an equestrian?", ["400,000", "400000", "400 000"]),
    ("At its mid-2nd-century peak, what was the estimated Roman silver stock?", ["10,000", "10000", "10 000"]),
    ("In which year was universal citizenship granted to all free inhabitants?", ["212"]),
    ("In which battle did Octavian defeat Mark Antony and Cleopatra?", ["actium"]),
    ("In which year was the Battle of Philippi?", ["42"]),
    ("What name did Hadrian give to the refounded Jerusalem?", ["aelia capitolina"]),
    ("According to Tacitus, roughly how many auxiliaries were there?", ["125,000", "125000", "125 000"]),
    ("About how many soldiers did the army have in the 1st century?", ["300,000", "300000", "300 000"]),
    ("Which years does the text give for the Decian persecution?", ["246"]),
    ("Which riots of 532 does the text mention?", ["nika"]),
    ("In which city did a persecution take place in 177?", ["lugdunum", "lyon"]),
    ("Which pope crowned Charlemagne?", ["leo iii", "leo the third", "leo 3"]),
    ("In which year was there a serious credit shortage in the early Empire?", ["33"]),
    ("Who emerged as the victor of the Year of the Four Emperors?", ["vespasian"]),
    ("What was the traditional population estimate of the Empire?", ["55", "60 million"]),
    ("What range does the text give for the average literacy rate?", ["5", "30"]),
    ("In which year did Constantine move the imperial seat to Byzantium?", ["330"]),
    ("Who was the last emperor to rule over both East and West?", ["theodosius"]),
    ("In which year did Italy's unification (Risorgimento) happen according to the text?", ["1861"]),
    ("What was the grain dole for citizens called?", ["cura annonae", "annona"]),
    # not answerable from the article -> expect NOT_IN_TEXT
    ("What was the name of Hadrian's favourite horse?", None),
    ("How many seats did the Roman Senate have under Diocletian, exactly?", None),
    ("What was the population of the city of Lugdunum in 177 AD?", None),
    ("Which Roman general first used war elephants in Britain?", None),
    ("What was the exact length in kilometres of Hadrian's Wall as given in the text?", None),
]

SYSTEM = ("Answer questions using ONLY the article below. Reply with a short answer (a few words). If the article "
          "does not contain the answer, reply exactly NOT_IN_TEXT.\n\n=== ARTICLE ===\n")


def ask(port, article, q):
    body = {"messages": [{"role": "system", "content": SYSTEM + article},
                         {"role": "user", "content": q}],
            "temperature": 0, "max_tokens": 48, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    article = open(ARTICLE, encoding="utf-8").read()
    rows, ok_ans, ok_abst, n_ans, n_abst, halluc = [], 0, 0, 0, 0, 0
    t0 = time.time()
    for q, gold in QA:
        out = ask(a.port, article, q)
        refused = "NOT_IN_TEXT" in out.upper().replace(" ", "_")
        if gold is None:
            n_abst += 1
            ok = refused
            ok_abst += ok
            halluc += not ok
        else:
            n_ans += 1
            ok = (not refused) and any(g in out.lower() for g in gold)
            ok_ans += ok
        rows.append({"q": q, "gold": gold, "out": out, "ok": bool(ok)})
        print(f"{'OK ' if ok else 'BAD'} {q[:60]:60s} -> {out[:70]!r}", flush=True)
    res = {"name": a.name, "answerable": f"{ok_ans}/{n_ans}", "abstain": f"{ok_abst}/{n_abst}",
           "hallucinated_on_unanswerable": halluc, "seconds": round(time.time() - t0, 1), "rows": rows}
    out = os.path.join(os.environ.get("BENCH_OUT", os.path.expanduser("~/llama-bench-results")), "eval")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"wikiqa-{a.name}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"WIKIQA {a.name}: answerable {ok_ans}/{n_ans}, abstained {ok_abst}/{n_abst}, {res['seconds']} s")


if __name__ == "__main__":
    main()
