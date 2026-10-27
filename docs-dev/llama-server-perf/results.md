# Mérési eredmények

## B-baseline (brew 0.6.0, Vulkan, pontos user-parancs)

`llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -fa on -ctk iq4_nl
-ctv iq4_nl --parallel 1 -c 80000`, temp=0, 2026-10-16.

| case | prompt (tok) | pp tok/s | TTFT | tg tok/s (medián) | VRAM csúcs | trunc |
|------|-------------|----------|------|-------------------|------------|-------|
| R1 köszönés (32 gen) | 8 | 25,1 | 0,32 s | 41,66 | 7756 MiB | nem |
| R2 MoE magyarul (512 gen) | 61 | 199,6 | 0,31 s | 37,46 | 7756 MiB | nem |
| R3 60k doksi (256 gen) | 65578 | 1024,6 | 64,46 s | 32,79 | 7756 MiB | nem |

Megjegyzés: tg kontextussal esik (42 → 37,5 → 32,9). VRAM: 7756 MiB
használt / 11264 MiB → ~3,5 GB szabad, van hely hangolásra.
Nyers JSON: `~/llama-bench-results/results/B-baseline.json`.

## B-opt rács (brew 0.6.0, Vulkan)

| konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| B-baseline | 41,66 | 37,46 | 1024,6 | 32,79 | 7756 |
| B-ub1024 | 47,33 | 43,79 | 944,0 | 30,40 | 7963 |
| B-ub2048 (-b 4096) | 45,30 | 42,52 | 924,6 | 30,38 | 8353 |
| B-nommproj (ub1024, mmproj CPU) | 45,42 | 42,47 | 929,7 | 30,56 | 6836 |
| B-mtp2 | 38,17 | 35,94 | 697,3 | 27,57 | 8468 |
| B-mtp3 | 34,83 | 31,50 | 742,8 | 23,99 | 8518 |

Tanulságok:
- Drift-vizsgálat (B-baseline2, azonos parancs ismételve): R1 tg 47,39,
  R2 tg 43,38, R3 pp 964,2 / tg 32,37 — az első baseline volt a kilógó,
  az ub1024 "javulás" mérési zaj (±5-15% futások között).
- **B-opt döntés**: brew+Vulkan oldalon nincs valós ub-nyereség a zaj
  felett; B-opt = baseline + `--no-mmproj-offload` (ingyen -1,1 GB VRAM).
- MTP brew-val ELVETVE: acceptance jó (mtp2 R3: 143/221 = 65%),
  mégis lassabb (draft-overhead > verifikációs nyereség 0.6.0-ban).

## q4_1 újramérés (2026-10-16 este, órajel-lock aktív)

ATTORÉS CUDA-n. `-ctk q4_1 -ctv q4_1` mindenhol.

| konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| B-base-q41 (brew) | 41,13 | 45,37 | 760,1 | 26,94 | 8506 |
| V-nommproj-q41 | 50,65 | 47,20 | 789,6 | 33,08 | 7275 |
| C-base-q41 | 62,08 | 66,11 | 1322,5 | 30,27 | 8803 |
| C-nommproj-q41 | 68,99 | 66,14 | 1325,8 | 30,21 | 7647 |
| C-mtp2-q41 | 70,49 | 73,34 | 1173,9 | **54,82** | 7814 |
| C-mtp3-q41 | 57,93 | 62,32 | 1176,4 | 54,34 | 7863 |

- pp bajnok: C-nommproj-q41 (1325,8, +42% Vulkanhoz képest).
- tg bajnok: C-mtp2-q41 (R3 54,82 = +81% MTP nélkülhöz, +66% Vulkanhoz).
  Acceptance R3: 150/209 = 72%. n-max 3 nem jobb mint 2.
- CUDA R3 tg MTP nélkül (30,2) gyengébb mint Vulkan (33,1) — hosszú
  kontextusú FA Turingon lassabb; MTP ezt túlkompenzálja.
- B-base-q41 R2-rep0 stall (pp 4,7, TTFT 13 s) — tranziens zaj, medián OK.

## 2. kör: n-max sweep széle + CPU-spin + nagy ubatch (CUDA, q4_1)
| konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| C-mtp1 (n-max 1) | 64,43 | 77,79 | 1207,0 | 49,99 | 7763 |
| C-mtp4 (n-max 4) | 54,65 | 58,33 | 1194,5 | 48,32 | 7913 |
| C-mtp2-t1 (-t1 -tb1 --poll 0) | 62,78 | 75,97 | 1185,2 | **57,61** | 7793 |
| C-ub4096 (-b 8192 -ub 4096, MTP nélkül) | 69,75 | 65,61 | **1346,3** | 31,34 | 8173 |

- n-max sorrend R3 tg-ben: 2 (54,82) > 3 (54,34) > 1 (49,99) > 4 (48,32).
- `-t 1` hozott R3-ban (+5%), R1-ben visszaesett — a hosszú kontextus számít.
- C-mtp2-ub4096: 48,44 tg / 1118,5 pp / 9773 VRAM — nagy ub MTP-vel ront.
- Döntés: győztes = **C-mtp2-t1** (mtp2 + t1/tb1/poll0, default batch).
  q4_1 KV marad. Következő: kódpatchek (BlockingSync, rm_kq).

## V-Head (source 0.7.0-dev build 11697, Vulkan, +`--offline`, iq4_nl KV)

| konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| V-base | 47,76 | 42,96 | 451,2* | 29,66 | 7769 |
| V-nommproj | 46,03 | 42,60 | 929,4 | 30,58 | 6642 |
| V-mtp2 | 37,97 | 36,28 | 687,7 | 27,06 | 7341 |
| V-mtp3 | 33,72 | 30,90 | 766,2 | 23,76 | 7391 |
| V-sparsedis | 45,73 | 42,32 | 936,8 | 30,20 | 6642 |

\* V-base R3 pp első-futás pipeline-fordítást tartalmaz (R1 rep0 1,3 s,
R2 rep0 13,5 s egyedi költség; driver-cache csak a 2. processtől él).
Valós V-Head: pp ~930, tg ~30,5.
Tanulságok:
- HEAD Vulkan ≈ brew (zajsávon belül). MTP HEAD-en is lassít Vulkanon.
- `FA_SPARSE_DISABLE` hatástalan itt. `--no-mmproj-offload` ingyen -1,1 GB.

## 3. kör: kombinációk + q8_0 + ngram (CUDA, győztes-alap)

| konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| C-mtp2-ub2048 | 70,33 | 73,88 | 1212,5 | 52,76 | 8649 |
| C-q80 (q8_0 KV) | 60,02 | 69,73 | 1111,6 | 49,29 | 8361 |
| C-ngram (mtp+ngram-mod) | 64,89 | 69,97 | 1122,6 | 49,72 | 7815 |
| C-mtp2-ub4096 | 68,42 | 63,03 | 1118,5 | 48,44 | 9773 |

- Nagy ub MTP-vel monoton ront (52,8 / 48,4 vs 57,6). q8_0 KV lassabb
  mint q4_1 (sávszélesség). ngram-lánc ront. q4_1 + default batch marad.

## Kódpatch-ítéletek (méréssel)

| patch | eredmény | döntés |
|-------|----------|--------|
| CUDA `BlockingSync` | R3 tg 57,6 → 42,9 (−25%, reprodukálható) | REVERT |
| Vulkan `rm_kq=1` | pp +10%, de tg −11–14% minden case-ben | REVERT |
| `-march=native` | nem készült (CPU-oldal t1 mellett minimális, ROI nincs) | ELVETVE |

A fa mindkét revert után tiszta (`git diff` üres).

## 4. kör: SSL-verifikáció + kv-unified + f16 KV (CUDA)

| konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| C-verify (győztes ismételve) | 65,54 | 74,35 | 1135,7 | 51,69 | 7773 |
| C-kvunified | 68,12 | 71,22 | 1122,5 | 51,90 | 7773 |
| C-f16kv | 75,64 | 71,91 | 1132,9 | **64,02** | 9221 |

- kv-unified: semmi (±zaj) → default marad.
- f16 KV: R3 tg +24% (51,69 → 64,02), R1 +15%. A dekvant-költség
  nagyobb mint a sávszélesség-többlet. VRAM 9221 MiB (2 GB szabad).
- Ismétlés (C-f16kv2): R3 tg **70,75**, pp 1191,5 → sáv 64–71, ROBUSZTUS.
  **ÚJ GYŐZTES: f16 KV + mtp2 + t1.**

## Mérési zaj: hő-drift (2026-10-16 éjjel, fontos tanulság)

A kártya szabad órajelen 87 °C-on termikusan throttlingol: a `-lgc 1410,1650`
lock **alatt is** lemegy 1365–1445 MHz-re hosszú terhelésnél (hidegen 1650+).
Session-ek között ez ±20–25% látszólagos különbséget okoz ugyanazon a konfigon:

- ugyanaz a konfig (f16 + mtp2 + t1): R3 tg 64,0 / 70,8 (egyik session),
  majd 51,1 / 51,3 (másik, forró session).
- Fix órajel-lock (`-lgc 1250,1250`) sudo-t igényelne → nincs; helyette
  kontrollált protokoll: interleaved A/B anchorok ugyanabban a sessionben,
  60 s hűtés a futások között, és futásonkénti `nvidia-smi` órajel-log
  (`clocks-*.csv`, kiértékelés: `clockstat.py`).

Emiatt a 4. kör "f16 ROBUSZTUS" kijelentése felülvizsgálatra szorul:
a +24% egy része session-szerencse lehetett. A 6. kör (kontrollált)
dönti el a q4_1 vs f16 kérdést.

## 5. kör (félbeszakítva — drift, alacsony értékű adat)

| konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| C-f16-mtp1 | 50,85 | 63,55 | 1036,1 | 51,14 | 9116 |
| C-f16-mtp3 | 49,53 | 54,57 | 1002,6 | 51,34 | 9216 |

- Forró session; az mtp1/mtp3 különbség a zajon belül. A kör
  félbeszakítva a kontrollált protokoll javára.

## 6. kör: kontrollált A/B (interleaved, órajel-loggol, 60 s hűtés)

GPU busy-órajel 1533–1577 MHz között stabil minden futásban (clocks-*.csv),
tehát az összehasonlítások érvényesek.

| konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| C-a-q41 (q4_1 anchor) | 55,84 | 64,82 | 1021,0 | 46,83 | 7718 |
| C-a-q41b (q4_1 anchor 2) | 63,92 | 61,81 | 1006,9 | 47,03 | 7718 |
| C-b-f16 (f16 anchor) | 68,50 | 66,89 | 1045,8 | 58,38 | 9166 |
| C-b-f16b (f16 anchor 2) | 69,43 | 62,75 | 1031,6 | 54,69 | 9166 |
| C-b-f16c (záró anchor) | 69,26 | 65,22 | 1047,2 | 57,92 | 9166 |
| C-f16vec1 (VEC ≤1 tok) | 69,12 | 66,08 | 1045,0 | 55,09 | 9166 |
| C-f16vec3 (VEC ≤3 tok) | 63,20 | 64,59 | 1043,5 | **40,90** | 9166 |
| C-f16-bsampling (draft -bs) | 64,28 | 64,70 | 1040,9 | 58,24 | 9166 |
| C-f16-bs-both (draft+main -bs) | **72,41** | **71,87** | 1033,1 | **61,98** | 9198 |
| C-f16-pmin30 (p-min 0.3) | 59,15 | 60,42 | 1041,1 | 54,82 | 9166 |

- **f16 vs q4_1 (interleaved): R3 tg 46,9 ± 0,2 vs 56,5 ± 1,9 → +20%.**
  Kontrollált körülmények között is f16 a nyertes (a 4. kör +24%-a
  valós volt, csak a session-ök átlagában kisebb a hatás).
- **FA VEC kernel Turingon: negatív.** A verify batch (3 tok) VEC-kel
  R3 tg 58,4 → 40,9 (−26%): az MMA kernel jobb, a stock heurisztika
  helyes. Batch-1-en (draft lépések) neutrális. Patch revertelve.
  (A `GGML_CUDA_FATTN_DEBUG` log nem látszott a server logban — a szerver
  a sima ggml INFO sorokat kiszűri; a következtetés viselkedés-alapú.)
- **`--spec-draft-backend-sampling -bs`: +9–13%** (R3 62,0 vs 54,7–58,4;
  R2 71,9 vs 62,8–66,9; R1 72,4). A draft_n/acceptance számok bitre
  azonosak (499/261, 214/147) → ugyanazok a tokenek, gyorsabb sampling.
  Részlegesen (csak draft) semleges → a fő útvonal sampling a nyereség.
- `--spec-draft-p-min 0.3`: nincs nyereség → kihagyva.
- Megjegyzés: `-bs` kísérleti kapcsoló; grammar/speciális samplerek
  esetén a szerver automatikusan visszavált CPU-samplingre (fallback).

## 7. kör: megerősítés (interleaved A/B, végleges bináris, tiszta fa)

| run | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|-----|-------|-------|-------|-------|------|
| C-r7-base1 | 67,11 | 66,26 | 1051,8 | 57,80 | 9166 |
| C-r7-bs1 | 73,37 | 73,14 | 1038,6 | **60,83** | 9198 |
| C-r7-base2 | 55,58 | 61,77 | 1034,4 | 55,45 | 9166 |
| C-r7-bs2 | 73,45 | 72,21 | 1031,2 | **60,68** | 9198 |

- `-bs` + `--spec-draft-backend-sampling`: R3 +7,6% / +9,4% (mindkét pár),
  R2 +10% / +17%, R1 +9% → bekerül a végleges parancsba.
- A base2-R1 (55,58) tranziens zaj; a bs-párok egymással konzisztensek.

Vulkan-fallback frissítés (f16 KV):

| run | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|-----|-------|-------|-------|-------|------|
| V-f16 | 43,02 | 43,21 | 910,0 | 30,06 | 8354 |
| V-f16-mtp | 33,67 | 35,53 | 721,4 | 25,92 | 9053 |

- V-f16 ≈ V-q4_1 (zajon belül), de +1,6 GB VRAM → a Vulkan-tartalék
  marad `q4_1`-gyel (több kontextus-tartalék). MTP Vulkanon továbbra is ront.

## Finálé — VÉGLEGES (2026-10-17 hajnal, kontrollált mérések)

**Győztes CUDA-parancs (mindennapi használat):**

```
build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 -t 1 -tb 1 --poll 0 \
  --spec-draft-backend-sampling -bs
```

**Vulkan-tartalék (változatlan):**

```
build-vulkan/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk q4_1 -ctv q4_1 --parallel 1 -c 80000 --no-mmproj-offload
```

## Végső összevetés (eredeti vs győztes)

Eredeti: brew 0.6.0 Vulkan, iq4_nl KV, user-parancs.
Győztes: CUDA HEAD, f16 KV, mtp2, t1/poll0, -bs (kontrollált r7 mérések).

| | pp tok/s | tg tok/s (R3) | tg (R2) | VRAM |
|---|---|---|---|---|
| Eredeti | ~960–1025 | ~32,8–33 | ~43–45 | 7756 |
| Győztes (kontrollált) | ~1035 | **60,7** | **72,7** | 9198 |
| Győztes (hűvös session csúcs) | ~1190 | **70,8** | ~75 | 9221 |

- tg hosszú kontextuson: +85% (kontrollált) … +115% (csúcs) az eredetihez képest.
- pp: a kontrollált session ~1035 (a hűvös session ~1190). Az órajel/hő
  függvényében 1010–1190 között szór.
- VRAM: 9198 MiB / 11264 → ~2 GB szabad; OOM-kockázat nincs.
- A `docs-dev/llama-server-perf/plan.md` tiltólistája szerint a
  `corpus/` fájlok érintetlenek; a fán a végén nincs lokális módosítás.

## Ami a végén bent maradt (kód)

Semmi. Minden kódkísérlet vagy hatástalan vagy rontó volt:
BlockingSync (−25%), Vulkan rm_kq=1 (−11% tg), FA VEC Turingon (−26% R3).
A nyereség mind konfigurációból jött (KV-típus, MTP, threads/poll,
sampling-hely, mmproj-offload).

> **Frissítés (2b. fázis, 2026-10-19):** ez a szakasz a korábbi (Okt 17-i)
> kampányra vonatkozik. A 2b. fázisban **két kódváltozás bent maradt**:
> a `GGML_CUDA_FA_QUANTS` += `q4_1-q4_1` build-opció (nem-MTP R3 +59%)
> és a vektorizált q4_1→f16 konverziós kernel a `convert.cu`-ban
> (MTP R3 +9%); mindkettő méréssel igazolt, default-on. Részletek:
> 13–15. kör + a `plan.md` "2b. fázis" szakasza.

## 8. kör — GRAPH_OPT kód-kapcsoló (2026-10-17 reggel)

Újabb kód-szintű jelölt: `GGML_CUDA_GRAPH_OPT=1` — a fában létező, alapból
kikapcsolt stream-overlap optimalizáció (env-var, rebuild nélkül mérhető).

| run | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|-----|-------|-------|-------|-------|------|
| C-r8-base1 (hideg start) | 73,78 | 75,28 | 981,1 | **68,87** | 9201 |
| C-r8-gopt1 | 71,60 | 70,95 | 1037,9 | 62,06 | 9215 |
| C-r8-base2 (meleg) | 65,10 | 69,36 | 1032,7 | 61,40 | 9201 |
| C-r8-gopt2 (meleg) | 72,60 | 71,60 | 1036,6 | 61,44 | 9215 |

- Meleg, azonos hőállapotú pár (base2 vs gopt2): R3 tg 61,40 vs 61,44 →
  **GRAPH_OPT nem hoz nyereséget**; elvetve.
- A hideg base1 (68,87) és a meleg futások (~61,4) közti ~11% a termikus
  drift — ez motiválta az új, fix teljesítmény-sapkás rezsimet.
- Busy fogyasztás a körben: átlag ~190 W, csúcsok 258–289 W; 81–88 °C.

## Új mérési rezsim: fix 170 W power cap (2026-10-17)

- Kártya limitek: default 250 W, max 280 W, min 100 W → **170 W** beállítva
  (root szükséges: `nvidia-smi -pl 170 -i 0`; permanens systemd service:
  `nvidia-powerlimit.service`).
- Cél: a 81–88 °C-os termikus throttling (és a session-ök közti ±15–20%
  szórás) megszüntetése. A sapka alatt a hő alacsonyan stabilizálódik.
- Az órajel-lock (reboottal elmúlt) NEM kerül vissza; a power cap az
  egyetlen szabályzó (egyszerűbb, determinisztikusabb).
- Az e rezsim alatti számok önálló referenciát alkotnak; a korábbi
  (throttlingos) mérésekhez abszolút értékben nem hasonlítandók.
- **Egységes mérési szabvány: q4_1 KV** (CUDA és Vulkan, minden futás);
  iq4_nl nem használatos. Az f16-os CUDA-előny (+20%) a régi rezsimből
  dokumentált, de a mérések összehasonlíthatósága miatt felfüggesztve.
- Lefutott: r9 referencia-sorozat 170 W alatt, q4_1 KV-val —
  win ×3 (stabilitás) + nomtp/nobs (MTP és `-bs` hatás) + Vulkan-alap és
  Vulkan-MTP + brew-anchor (lásd a 9. kör szakaszt).

## 9. kör — 170 W-os referencia, egységes q4_1 KV (2026-10-17 este)

8/8 futás rc=0, minden futás q4_1 KV-val. Futásrend:
win1, nomtp, nobs, win2, V-alap, V-mtp, brew-anchor, win3.

**CUDA (build-cuda, patch nélküli referencia-bináris):**

| run | konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| win1 | q4_1+MTP+BS | 58,6 | 74,9 | 1165 | 48,1 | 7945 |
| win2 | q4_1+MTP+BS | 61,2 | 76,4 | 1160 | 56,3 | 7947 |
| win3 | q4_1+MTP+BS | 64,0 | 64,7 | 1005 | 47,1 | 7958 |
| nomtp | q4_1 | 53,6 | 52,5 | 1131 | 23,8 | 7215 |
| nobs | q4_1+MTP (BS nélkül) | 68,7 | 68,0 | 1075 | 52,9 | 7909 |

**Vulkan / brew (0.6.0):**

| run | konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| V-alap | q4_1 | 41,3 | 40,0 | 805 | 29,2 | 6815 |
| V-mtp | q4_1+MTP+BS | 35,6 | 37,3 | 714 | 25,7 | 7575 |
| B-brew | q4_1 | 43,0 | 40,3 | 801 | 29,0 | 7953 |

**Megállapítások:**

- MTP-hatás (CUDA): hosszú-ctx tg ~2× (win 47–56 vs 23,8), R2 tg +25–45%,
  pp semleges (1131 → 1160, a futás-zajon belül). Az MTP marad.
- A draft-elfogadás minden CUDA MTP-futásban **bitre azonos**
  (0,71770; 150/209; mean len 2,43) — a win-anchorok R3 tg szórása
  (47–56, ±9%) üzemállapot-zaj (klumpa/hő a sapka mellett), nem
  elfogadás-különbség. Az r10 A/B ezért interleaved lesz.
- BS (`-bs`) vs nélküle: R1/R2/R3 mind a nobs javára szól (R2 kivétel:
  win1/win2 ~75–76 vs nobs 68), de a szórás (±9%) miatt egyértelmű
  verdikt nincs; a win-konfig marad BS-sel (a korábbi kontrollált r7
  mérések is ezt hozták ki).
- Vulkan: az MTP **ront** (R3 tg 29,2 → 25,7; pp 805 → 714; R1/R2 tg is),
  és VK-n az elfogadás is gyengébb (mean len 2,17 vs 2,43). A VK-alap
  hosszú-ctx tg-ben jobb, mint a CUDA MTP nélkül (29,2 vs 23,8), de
  rövid-ctx tg-ben és pp-ben jóval rosszabb.
- brew 0.6.0 ≈ build-vulkan (pp 801 vs 805, tg 29,0 vs 29,2) —
  a brew-anchor stabil, a két Vulkan-építés gyakorlatilag egyforma.
- Teljesítmény-rezsim a körben: átlag ~150 W, rövid csúcsok 176–193 W,
  max 86 °C, busy órajel-átlag 1430–1530 MHz; a sapka a busy-minták
  25–50%-ában aktívan fog.
- VRAM q4_1 KV-val: 7,2–8,0 GB (f16: 9,2 GB) → ~1,3 GB több szabad hely.

**Tájékoztató összevetés a régi rezsimmel** (nem tiszta párok:
f16↔q4_1 és sapka keveredik): r8 meleg f16 sapka nélkül pp ~1036 /
tg ~61,4; r9 meleg q4_1 170 W alatt pp ~1160–1165 / tg ~48–56. A pp
javult (+12%), a tg csökkent; a szétválasztás (f16 vs q4_1 a sapka
alatt) külön mérés lenne.

## 10. kör — CUDA MMVQ↔MMQ dekód A/B (2026-10-17 éjjel)

Env-gated kísérleti patch (`GGML_CUDA_DISABLE_MMVQ=1` → MMVQ helyett MMQ
a dekód batch-ekre), interleaved win/mmq párok, q4_1 + MTP + BS, 170 W.
Env nélkül a patch inaktív — a win-futások a referencia-útvonalat mérik.

| run | konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| C-r10-win1 | MMVQ (alap) | 80,1 | 83,4 | 1240 | 57,4 | 7800 |
| C-r10-mmq1 | MMQ (kényszerített) | 66,1 | 69,5 | 1138 | 50,7 | 7966 |
| C-r10-win2 | MMVQ (alap) | 73,5 | 75,5 | 1139 | 53,5 | 7888 |
| C-r10-mmq2 | MMQ (kényszerített) | 62,9 | 68,7 | 1126 | 48,1 | 7890 |

Páronkénti eltérés (MMQ − MMVQ): R1 −17% / −15%, R2 −17% / −9%,
R3 tg −12% / −10%; pp a futászajon belül (MMVQ csak ≤8 batch-re hat).

- **Verdikt: az MMVQ (alapértelmezett) út nyer minden mutatóban, mindkét
  párban** — a Turingon „hangolatlan" heurisztika a dekód batch-ekre
  valójában a jó utat választja. Kódváltozás nem marad (patch
  visszaállítva, a `mmvq.cu` tiszta).
- Az MMQ alatt a draft-elfogadás is esett (mean len 2,29 vs 2,43): a más
  kernel-numerika eltérő argmaxot ad, ez a tg-romlás egy részét
  magyarázza; a többi a kernel-sebesség.
- Az r10-es win-számok az r9-es sáv felső szélén (R1/R2 tg 73–83 vs
  58–76) — session-szintű variancia; az A/B érvényességét az interleaved
  párok adják. Az `env`-átadást minden mmq-futásnál a futó szerver
  `/proc/<pid>/environ`-jából ellenőriztük.

## 11. kör — Vulkan: MTP-izolálás + env-kapcsolók (2026-10-18 hajnal)

6 futás, q4_1 KV, 170 W, `build-vulkan`. Anchorok: base vs base2.

| run | konfig | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| V-r11-base | q4_1 (alap) | 47,0 | 46,9 | 901 | 31,6 | 6605 |
| V-r11-base2 | q4_1 (alap) | 45,7 | 46,1 | 910 | 31,2 | 6605 |
| V-r11-spec | +MTP (BS nélkül) | 35,4 | 38,6 | 741 | 26,6 | 7353 |
| V-r11-fmmvq | `FORCE_MMVQ=1` | 41,7 | 40,0 | 897 | 29,0 | 6605 |
| V-r11-dmmvq | `DISABLE_MMVQ=1` | 47,5 | 47,6 | 896 | 31,2 | 6610 |
| V-r11-nodes | `MAX_NODES=1000` | 46,9 | 45,0 | 912 | 31,5 | 6605 |

Megállapítások:

- Az anchorok ±2%-on belül — Vulkanon kiváló a futáspontosság.
- **Az MTP Vulkanon önmagában is ront** (BS nélkül: R3 tg −16%, R2 −18%,
  pp −18%) → nem a backend-sampling a hibás; a VK spec-út inherensen
  lassabb (a CUDA-n mért ~2× MTP-nyereség VK-n nem jön ki). MTP VK-n
  elvetve.
- `GGML_VK_FORCE_MMVQ=1` ront (R2 −15%, R3 −8%): a VK auto-mix
  (n=1, Q4_K, k≤4096 → MMQ) jobb az erőltetett MMVQ-nál.
- `GGML_VK_DISABLE_MMVQ=1` és `GGML_VK_MAX_NODES_PER_SUBMIT=1000`
  semleges (±1–3%) → nincs mit hangolni.
- **VK-optimum: base (MTP nélkül)** — de a CUDA+MTP-től elmarad
  (R3 tg 31,6 vs 48–57; pp 901 vs 1140–1240; R2 tg 47 vs 68–83).

## PCIe-link terhelés alatt (2026-10-18)

- 18 perces minta (2 s-enként) az r11 futásai alatt: 375 minta **Gen3 x8**,
  161 Gen1 (üresjárat/ASPM), 14 Gen2. Max képesség: Gen3 x16, a link
  aktuális szélessége x8 (fizikai/slot-korlát).
- Következtetés: terhelés alatt felmegy Gen3-ra; a token-méretű GPU↔CPU
  forgalomhoz az x8 bőven elég → **nincs GPU↔CPU szűk keresztmetszet**,
  ezen a téren nincs teendő.

## 10b. kör — Turing MMVQ nwarps variáns (elvetve, 2026-10-18 hajnal)

A Turing MMVQ-tábla dekód-útjának hangolása: a k-kvantok (Q2_K..Q6_K)
`ncols_dst=1`-nél 2 warps-ot kapnak — a variánsban 4-et (mint a többi
típus). Fordítási variáns + külön, tiszta referenciabináris
(`bin-cuda-ref/`), interleaved ref/var párok, win config, q4_1, 170 W.

| run | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|
| C-r10b-ref1 | 64,9 | 77,9 | 1193 | 55,3 | 7737 |
| C-r10b-var1 | 72,6 | 76,0 | 1148 | 53,7 | 7737 |
| C-r10b-ref2 | 73,2 | 75,1 | 1149 | 54,4 | 7737 |
| C-r10b-var2 | 73,5 | 76,9 | 1139 | 54,4 | 7737 |

- Páronkénti eltérés (var − ref): R1 +11,8% / +0,4%, R2 −2,4% / +2,4%,
  R3 tg −2,9% / ±0,0% → **nincs konzisztens irány**; a var1 R1-nyeresége
  a ref1 első-futás outlier (64,9; a másik három 72,6–73,5) rovására írható.
- Az elfogadás mind a 4 futásban bitre azonos (0,71770, mean len 2,43) —
  a variáns numerikailag ugyanaz, csak a warpfelosztás más; a tg-eltérések
  tiszta futászaj.
- **Verdikt: nincs mérhető nyereség → a variáns elvetve** (forrás
  `git checkout`, `build-cuda` újraépítve tiszta forrásból).
- Módszertani megjegyzés: az azonos forrásból újraépített
  `libggml-cuda.so.0.27.0` md5-je futásonként eltér, de `cmp` szerint
  mindössze 2 bájt (az nvcc temp-fájlnevének PID-je) — a build
  gyakorlatilag determinisztikus, a különbség kozmetikai.

## 12. kör — CPU-affinitás / NUMA-pin (2026-10-18 hajnal)

`taskset` wrapperek: pin0 = 0-11 mag (0. NUMA-node), pin1 = 12-23 (1.
node); a pinnelést futás közben `taskset -cp`-vel igazoltuk (0-11 ill.
12-23 aktív). Win config, q4_1, 170 W; anchorok: base és base2 (pin
nélkül).

| run | R1 tg | R2 tg | R3 pp | R3 tg |
|---|---|---|---|---|
| C-r12-base | 70,9 | 76,4 | 1153 | 55,4 |
| C-r12-base2 | 66,1 | 77,0 | 1145 | 55,8 |
| C-r12-pin0 | 73,2 | 76,8 | 1176 | 55,1 |
| C-r12-pin1 | 73,9 | 76,6 | 1158 | 55,3 |

- Anchorok: R2/R3/pp ±1% — szoros; R1 ±5–7% (a szokásos szórás).
- pin0/pin1 vs anchor-átlag: R2 ±0,2%, R3 tg −0,6..−0,9%, pp a zajban →
  **a NUMA-pin a hosszú-ctx metrikákon semleges**, a node választása sem
  számít (node0 ≈ node1).
- R1-nél a két pinnelt futás +7–8%, de ez a R1 ±7%-os szórásán belül van
  (ugyanez a konfig r9/r10 során 58,6–80,1 között szórt) → nem döntő;
  affinitás nem kerül a napi parancsba.
- Elfogadás mind a 4 futásban azonos (0,71770, mean len 2,43).
- A 170 W-os rezsim stabil: avgP 148,5–149,6 W, maxT 80–84 °C.

## 13. kör — natív q4_1 FA VEC kernelek A/B (2026-10-19 reggel)

`GGML_CUDA_FA_QUANTS` += `q4_1-q4_1` (build-cuda); ref = az okt. 17-i
tiszta runtime (`bin-cuda-ref`), var = az új build. **Lib-izoláció:**
a ref bináris RUNPATH-ja az élő `build-cuda/bin`-re mutat, ezért minden
futás `LD_LIBRARY_PATH=$(dirname $bin)`-nel indult; bizonyíték:
`LD_TRACE_LOADED_OBJECTS` (ref → bin-cuda-ref, var → build-cuda/bin) és
a szerver-log `no FlashAttention vector kernel compiled ... q4_1-q4_1`
sora: ref futásoknál 1, var futásoknál 0. Win config + MTP, ill. nem-MTP
(`--spec-*` nélkül); 170 W, q4_1 KV, interleaved.

| run | R1 tg | R2 tg | R3 pp | R3 tg | R3 acc |
|---|---|---|---|---|---|
| C-r13-ref1 | 67,3 | 76,6 | 1148 | 55,7 | 0,7177 |
| C-r13-var1 | 73,0 | 75,5 | 1166 | 54,2 | 0,7177 |
| C-r13-ref2 | 72,8 | 73,6 | 1130 | 54,5 | 0,7177 |
| C-r13-var2 | 74,5 | 76,6 | 1120 | 53,9 | 0,7177 |
| C-r13-var3 | 67,1 | 75,9 | 1141 | 54,7 | 0,7177 |
| C-r13-refN (nem-MTP) | 57,6 | 61,8 | 1301 | **26,6** | – |
| C-r13-varN (nem-MTP) | 63,7 | 62,4 | 1311 | **42,3** | – |

**Verdikt: a q4_1-q4_1 FA VEC kernelek MARADNAK.**
- Nem-MTP, hosszú kontextus: R3 tg 26,6 → 42,3 t/s (**+59%**) — a batch-1
  dekód eddig minden lépésben f16-ra konvertálta a teljes KV-t (a VEC
  kernelnek nem volt q4_1 instance-a); most natívan olvassa.
- MTP: nincs érdemi változás (54-56 t/s, -1..-3% a zajon belül) — a
  verify batch=3 Turingon MMA-ra megy, ami f16-only, tehát továbbra is
  konvertál.
- pp: nincs változás (1120-1166 MTP, 1301-1311 nem-MTP).
- Az acceptance bitre azonos (0,7177 / 0,5313 / 0,4783) → a numerika nem
  változott, csak a konverzió tűnt el.

Mellékes megfigyelés: az r9-es nem-MTP R3 tg (23,8) és a mostani tiszta
ref (26,6) eltér — session-szintű szórás; az azonos session-ön belüli
A/B (26,6 vs 42,3) a mérvadó.

## 14. kör — FA kernel-kényszerítés + konverziós kernel (2026-10-19 délelőtt)

7 futás a q4_1-FA build-en (win config + MTP, 170 W, q4_1 KV).

| run | R1 tg | R2 tg | R3 pp | R3 tg | megjegyzés |
|---|---|---|---|---|---|
| C-r14-base | 64,4 | 74,6 | 1124 | 55,1 | anchor |
| C-r14-vec4 | 73,6 | 77,6 | 1148 | **41,1** | `vec:4` — MTP verify batch=3 → VEC |
| C-r14-convfast | 73,6 | 75,6 | 1164 | **59,6** | vektorizált q4_1→f16 konverzió |
| C-r14-mmqprec | 74,4 | 76,3 | 1139 | 55,0 | `MMQ_PREC=q4` |
| C-r14-ub1024 | 72,9 | 74,3 | 1140 | **50,8** | `-ub 1024` |
| C-r14-mma1 | 74,2 | 76,7 | 1163 | 54,8 | `mma:1` kontroll |
| C-r14-base2 | 74,2 | 74,9 | 1128 | 54,2 | anchor |

**Verdikt:**
- **`vec:4` ELUTASÍTVA** (R3 tg −25%): a batch-3 verify-nél a VEC (SIMT)
  sokkal lassabb, mint az MMA (tensor core), a f16-konverzió megspórolása
  nem kompenzálja. Numerikai mellékhatás is van: R3 acceptance 0,71770 →
  0,70283 (a VEC és MMA összegzési sorrendje eltér) — a sebességvesztés
  viszont ettől független (a mean len csak 2,43 → 2,41).
- **`CONV_FAST` NYERT (+8,1% R3 tg, +3,5% pp)**: a q4_1→f16 konverzió a
  jelenlegi kernelben 8× 2 bájtos store-t ír (fél hatékonyságú szektorok);
  a v2 kernel 1× 32-bites load + 2× 8 bájtos store. Helyesség: detcheck
  (MMA-kényszerítés, azonos prompt+n_predict, temp=0) → **bitre azonos
  kimenet** a régi és az új kernellel.
- `mmqprec=q4` semleges (55,0), `-ub 1024` ront (−7,7%, VRAM +276 MiB),
  `mma:1` semleges (a draft batch-1 dekódjai rövid KV-n futnak).
- R1 nagyon zajos (64,4 vs 74,2 mindkét anchor); R2/R3 a mérvadó.

**Módszertani megjegyzés (log-csapda):** a FA-kényszerítő kapcsoló
`GGML_LOG_WARN`-ja azért nem látszott a szerver-logban, mert a wrapper
statikus inicializálása **már a modellbetöltés alatt** megtörténik
(0,5 s), ahol a ggml-log üzenetek elvesznek; a `fprintf(stderr, ...)`
viszont mindig megjelenik. A kapcsoló hatását eközben a viselkedés
igazolta (acceptance-eltérés + tg).

**Következő (r15):** a v2 konverziós kernel default-ra állítva
(`GGML_CUDA_CONV_LEGACY=1` a régi); igazoló A/B ugyanazon a binárison.

## 15. kör — konverziós kernel igazoló A/B (2026-10-19 délelőtt)

Ugyanaz a bináris (md5 25fe9bb9…), két ág: `GGML_CUDA_CONV_LEGACY=1`
(régi kernel) vs default (v2 vektorizált). Interleaved, 170 W, q4_1 KV.
Bizonyíték: a "legacy (unvectorized) kernel" sor csak a legacy
futásokban jelenik meg (1 vs 0) — a kernelválasztás igazolva.

| pár | R3 pp | R3 tg | d tg | acc |
|---|---|---|---|---|
| legacy (1) | 1146 | 54,32 | – | 0,7177 |
| base (2) | 1160 | 58,24 | **+7,2%** | 0,7177 |
| legacy2 | 1115 | 54,24 | – | 0,7177 |
| base2 | 1144 | 60,01 | **+10,7%** | 0,7177 |
| legacyN (nem-MTP) | 1321 | 41,20 | – | – |
| baseN (nem-MTP) | 1298 | 41,95 | +1,8% | – |

- **A v2 konverziós kernel marad defaulton**: MTP R3 tg +7,2…+10,7%
  (~+9% átlag), pp +1…+3%; a nem-MTP dekód nem használ konverziót
  (VEC natív), ott a különbség zaj.
- Az acceptance bitre azonos (0,7177) — a konverzió számszerűen
  ugyanazt adja (a detcheck is ezt mutatta).

## CPU-profil — hol megy a CPU-idő? (2026-10-19 10:00, cpuprobe)

Módszer: `/proc/<pid>/stat` utime+stime mintavétel kérés előtt/után,
szálankénti bontással; R1/R2/R3 × 3 ismétlés (cache off + cached), MTP win
config, `-bs`-szel (A) és anélkül (B), 170 W.

| eset | falidő s | CPU s | CPU% | cpu_ms/token |
|---|---|---|---|---|
| R1 cached | 0,62–0,72 | 0,63–0,73 | ~102% | 19,7–22,8 |
| R2 cached | 6,7–7,0 | 6,9–7,2 | ~102% | 13,5–14,0 |
| R3 cached (65k) | 4,9–5,3 | 4,9–5,4 | ~101% | 19,3–20,9 |
| R3 nocache (65k pp) | 59,6 | 60,2 | ~101% | 235 |

- A fő szál CPU-ideje ≈ a falidővel **minden** esetben → a szál
  gyakorlatilag 100%-ban fut; ez azonban **sync-spin** (CUDA event poll a
  fázisok között), nem valódi munka: a KV-maszk írás (~0,1 ms/lépés) és a
  tokenizálás eltörpül mellette.
- `-bs` **nélkül** (B): R2 15,3–16,3 ms/token, falidő 7,6–8,1 s; `-bs`-szel
  (A): 13,5–14,0 ms/token, 6,7–7,0 s → a CPU-sampler ~1,5–2 ms/token,
  ami R2-n **+12,8% falidő** (átlag 6,85 s vs 7,85 s). A `-bs` érdemben
  számít, a napi configban marad.
- Következtetés: CPU-oldali mikrooptimalizálás (maszk, CPU-sampler) nem
  hozható ki — a GPU a szűk keresztmetszet, a CPU a szinkron-spin miatt
  „mindig elfoglalt”. Nagyobb átalakítással (host-sync-ek számának
  csökkentése az MTP-lépésben: jelenleg ~4 graph-végrehajtás/lépés)
  lehetne elvi tartalék, de az nem gyors, kockázatmentes nyereség.

## 16. kör — Vulkan env-kapcsolók (2026-10-19 délelőtt)

9 futás, build-vulkan (md5 a0fcf84e, `GGML_VK_RM_KQ` bizonyíték-loggal),
nem-MTP base config, q4_1 KV, 170 W. **Mérési anomália:** az R3 *cached*
(rep1/rep2) szakasz néhány futásban belülről degradált (base2:
31,0 → 28,3 → 27,9, azaz a medián −10%), miközben az órajelek csak ~2%-ot
változtak. Ezért az értékelés a **rep0** adatra épül (minden futásban
azonos: friss 65k pp + 256 generált token).

| run | R3 pp (rep0) | tg rep0 | tg rep1 | tg rep2 | verdikt |
|---|---|---|---|---|---|
| base | 914 | 31,49 | 31,60 | 31,78 | anchor |
| nocm2 | **432** | **29,43** | 29,61 | 29,51 | ELUTASÍTVA (coopmat2 nélkül a pp feleződik) |
| nodot2 | 895 | 31,10 | 29,33 | 29,05 | semleges/gyenge loss |
| nomadd | 905 | 31,50 | 31,65 | 31,61 | semleges |
| nodint | 908 | 31,34 | 31,11 | 29,68 | semleges |
| memprio | 886 | 30,36 | 30,12 | 31,18 | semleges (zaj) |
| nodes32 | 892 | 31,96 | 31,91 | 32,46 | +1,5% rep0-n → a zajon belül |
| rmkq1 | 886 | **29,80** | 29,44 | 29,56 | ELUTASÍTVA (−5%) |
| base2 | 889 | 31,02 | 28,27 | 27,87 | anchor (rep0-drift −1,5%) |

**Verdikt:** a Vulkan base config marad. A coopmat2 és a dot2 utak
hasznosak (kikapcsolásuk ront), az `rm_kq=1` ront (−5%), a többi
kapcsoló semleges. Session-ön belüli rep0-drift: tg ~1,5%, pp ~2,7%.

## 2b. fázis — VÉGLEGES győztes konfiguráció (2026-10-19)

> **FRISSÍTÉS (2026-10-20, Part 1):** a KV-típus **q4_1 → f16** (+14–16% MTP,
> +12–17% no-MTP; lásd a „Part 1 — VÉGLEGES konfiguráció" szekciót).
> Az alábbi q4_1-es parancs a **VRAM-korlátos** változat marad.

**CUDA (napi parancs, 170 W, q4_1 KV, MTP):**

```
build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk q4_1 -ctv q4_1 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 -t 1 -tb 1 --poll 0 \
  --spec-draft-backend-sampling -bs
```

Mért értékek (r15, ugyanaz a bináris): MTP R3 tg **58,2–60,0 t/s**,
R3 pp 1144–1160; nem-MTP R3 tg **42,0 t/s**, pp 1298.

**Vulkan (tartalék, MTP nélkül):**

```
build-vulkan/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk q4_1 -ctv q4_1 --parallel 1 -c 80000 --no-mmproj-offload
```

R3 tg 31,5 t/s, pp 914.

**Bent maradt kódváltozások (méréssel igazolva, default-on):**

1. `GGML_CUDA_FA_QUANTS` += `q4_1-q4_1` (build-opció) — natív q4_1 VEC FA
   kernel; nem-MTP R3 tg 26,6 → 42,3 (+59%, r13).
2. Vektorizált q4_1→f16 konverziós kernel (`ggml/src/ggml-cuda/convert.cu`)
   — MTP R3 tg +8…+11% (r14/r15), `GGML_CUDA_CONV_LEGACY=1` adja a régit.

**Env-kapcsolók (default off):** `GGML_CUDA_DISABLE_MMVQ`,
`GGML_CUDA_BLOCKING_SYNC`, `GGML_CUDA_FA_FORCE_KERNEL=vec|tile|mma[:N]`,
`GGML_CUDA_CONV_LEGACY`, `GGML_CUDA_MMVQ_TURING_KQ_NWARPS4` (fordítási),
`GGML_VK_RM_KQ=1..4`.

**Függőben lévő sudo-teendők:**

```bash
sudo install -m 755 ~/llama-bench-results/nvidia-powerlimit.sh  /usr/local/bin/nvidia-powerlimit.sh
sudo install -m 644 ~/llama-bench-results/nvidia-powerlimit.service /etc/systemd/system/nvidia-powerlimit.service
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-powerlimit.service
```

(Enélkül a 170 W csak a jelenlegi bootig él.)

**Build-konfiguráció (CUDA):** a `build-cuda` cache-e tartalmazza a
`-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q4_1-q4_1`
opciót — friss build-könyvtárnál újra meg kell adni, különben a natív
q4_1 FA kernel nem fordul be.

**Forrásváltozások helye (uncommitted, working tree):**
`ggml/src/ggml-cuda/convert.cu` (v2 kernel), `ggml/src/ggml-cuda/fattn.cu`
(FA-kényszerítő + bizonyíték-log), `ggml/src/ggml-cuda/mmvq.cu` +
`ggml-cuda.cu` (P0 env-kapcsolók), `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
(RM_KQ).

## Konverziós kernel mikrobenchmark — v3 elvetve (2026-10-19 délután)

Izolált CUDA-mikrobenchmark (`/tmp/opencode/convbench.cu`, sm_75) a valós
MTP-verify konverziós volumenen (65 536 sor × 256 dim × 8 layer × 2 tenser
= 537 MB f16 kimenet, 168 MB q4_1 bemenet):

| kernel | idő | effektív sávszélesség |
|---|---|---|
| legacy (8× 2B store) | 2,103 ms | 335 GB/s |
| v2 (jelenlegi default) | **1,819 ms** | 387 GB/s |
| v3 (1× 16B store) | **1,819 ms** | 387 GB/s |
| memcpy D2D referencia | 2,629 ms | 408 GB/s |

- A v2 **gyakorlatilag a memória-sávszélesség határán** van (387 vs 408
  GB/s), a v3 (16 bájtos store) **nem ad semmit** (0,0%) → nem kerül be.
- A teljes konverziós szett egy lépésben ~1,82 ms (a korábbi ~2,9 ms
  becslés helyett), azaz a 41 ms-os MTP-lépés ~4,4%-a.
- Következtetés: a konverziós irány lezárva; a további CUDA-nyereséghez
  a verify batch matvec-kernelje (MMVQ, ~240 GB/s effektív) lenne a cél,
  de az már mély kernelmunka.

## 17. kör — Vulkan rm_kq 3/4 (2026-10-19 este)

A Turing default `rm_kq=2`; az r16-ban az `1` vesztett, a 3/4-et még nem
mértük. 4 futás, base config, q4_1, 170 W, bizonyíték-log stimmelt
(rm_kq_log 70 a variánsokban, 0 a base-ekben).

| run | R3 pp (rep0) | R3 tg rep0 | rep1 | rep2 |
|---|---|---|---|---|
| base | 826,7 | 28,61 | 28,57 | 29,03 |
| rmkq3 | 813,4 | 28,95 | 28,63 | 28,94 |
| rmkq4 | 811,0 | 28,67 | 28,89 | 28,75 |
| base2 | 849,4 | 29,75 | 32,36 | 28,57 |

**Verdikt:** rm_kq 3/4 semleges (a +1,2%/+0,2% a base2 anchor +4%-os
eltérésén belül van) → **nem lesz belőle kódváltozás**; marad a default 2.

**Környezeti megjegyzés:** ebben a (esti) sessionben az abszolút VK
számok ~10%-kal alacsonyabbak a délelőttinél (28,6 vs 31,6), miközben az
SM-órajel *magasabb* (1533 vs 1497 MHz) — tehát a különbség nem
órajel/hőmérséklet, hanem külső interferencia/latencia; **abszolút
számokat csak azonos sessionben szabad összevetni**, a verdikteket a
körön belüli rep0 + anchorok adják.

## 18. kör — MTP verify batch MMQ-routing (2026-10-19 este)

A CUDA MTP-lépés domináns komponense a verify (batch 3) matvec-jei
(~23 ms a 41 ms-os lépésből). Turingon a `ne11 <= 8` szabály miatt erre
MMVQ fut; az új, env-gated `GGML_CUDA_MMQ_MIN_BATCH=3` kapcsolóval az
MMQ-ra (tensor core) route-olható — ezt eddig külön nem mértük (a korábbi
MMQ-teszt a batch-1 dekódokat is érintette).

| run | R3 pp | R3 tg |
|---|---|---|
| base | 1184 | 59,90 |
| mmq3 | 1144 | **52,05 (−13,1%)** |
| base2 | 1175 | 59,52 |
| mmq3b | 1125 | **51,90 (−12,8%)** |

**Verdikt:** az MMQ a batch-3 verify-nél **13%-kal lassabb** (a nagy MMQ
tile 3 oszlopnál pazarló) → marad az MMVQ. A kapcsoló env-gated
kísérleti eszközként bent marad (default off), mint a többi P0-kapcsoló.
Bizonyíték: routing-log 1 a mmq futásokban, 0 a base-ekben; acceptance
azonos (0,4375).

**Eszköz-megjegyzés:** a kernel-szintű nsys-profil **nem elérhető** ezen a
gépen (a telepített nsys 2023.4-ből hiányzik a QDSTRM importer:
„importer binary and its dependencies were not found”) — sudo/telepítés
kellene hozzá; a `--` elválasztót sem fogadja el. A dekompozíció ezért
kivonásos/ablation módszerrel készült.

## 19. kör (Part 1 / WP-B1) — KV-típus: f16 vs q4_1 (2026-10-20)

8 futás, MTP és no-MTP, interleaved q4_1↔f16 párok + anchorok, 170 W,
ugyanaz a bináris (md5 `c05aa45e`), CUDA. Bizonyíték: CMD-sorok (-ctk) + VRAM.

| config | R3 tg rep0 | medián | R3 pp |
|---|---|---|---|
| MTP q4_1 | 65,27 | 64,15 | 1232,9 |
| **MTP f16** | **74,56 (+14,2%)** | 74,56 (+16,2%) | 1271,0 |
| MTP q4_1 (2) | 57,44 | 57,54 | 1147,6 |
| **MTP f16 (2)** | **65,77 (+14,5%)** | 66,03 (+14,8%) | 1134,8 |
| no-MTP q4_1 | 41,07 | 41,03 | 1293,1 |
| **no-MTP f16** | **47,95 (+16,8%)** | 47,82 (+16,6%) | 1314,9 |
| no-MTP q4_1 (2) | 41,42 | 41,40 | 1295,0 |
| **no-MTP f16 (2)** | **46,39 (+12,0%)** | 47,32 (+14,3%) | 1301,4 |

- **Verdikt: az f16 KV +14–16% MTP-n és +12–17% no-MTP-n**, pp változatlan,
  acceptance bitazonos (0,4375); a rep-szórás kicsi (MTP f16: 74,56/72,78/75,12).
- **Ára: +1,42 GB VRAM** (MTP 7876→9316 MB; no-MTP 7162→8610 MB) — 80k-nál
  belefér a 11,2 GB-ba, de a headroom ~1,9 GB-ra csökken.
- Ok: q4_1 KV-nál a batch-1 út SIMT VEC (elemenkénti dequant), f16-nál a
  VEC/MMA f16-tal dolgozik (nincs dequant); az MTP verify-nél elmarad a
  teljes-KV f16 konverzió (700 MB/lépés).
- **B2 (CUDA KV-shadow) ELVETVE**: az f16 árnyék-másolat a q4_1 cache fölött
  +2,1 GB lenne (összesen ~2,75 GB KV) — rosszabb, mint a tiszta f16 KV
  (+1,42 GB), miközben az f16 út egyszerűbb és gyorsabb.
- **Napi konfig: `-ctk f16 -ctv f16`** (a q4_1 megmarad VRAM-korlátos /
  nagyobb kontextusú esetre).
- Megjegyzés: a jelenlegi session ~10%-kal gyorsabb a korábbi estinél
  (q4_1 MTP 64–65 vs 59–60), ezért csak a körön belüli párok érvényesek.

## 20. kör (Part 1 / WP-E) — n-max és -ub újramérés f16 KV-val (2026-10-20)

8 futás, MTP, f16 KV (az új győztes), interleaved anchorok (minden variáns
mellett nmax2 futás), 170 W, ugyanaz a bináris.

| run | R3 tg rep0 | medián | pp |
|---|---|---|---|
| nmax2 (anchor) | 66,43 | 67,05 | 1166,4 |
| nmax3 | 60,88 | 60,88 | 1143,7 |
| nmax2b (anchor) | 65,21 | 65,97 | 1167,0 |
| nmax3b | 59,32 | 60,03 | 1132,4 |
| nmax1 | 64,04 | 66,88 | 1146,9 |
| nmax2c (anchor) | 64,22 | 64,60 | 1136,7 |
| ub1024 | 63,49 | 63,49 | 1140,7 |
| nmax2d (anchor) | 64,62 | 65,12 | 1143,3 |

- **n-max 3: −8,4% / −9,0% → elvetve** (a korábbi 2 > 3 sorrend megerősítve).
- **n-max 1: rep0 −0,3% (medián +3,5%, az anchor-szórás 3,8%-a) → semleges**;
  a 2. draft-pozíció mért haszna (mean len 1,88 vs 1,55) épp a lépésköltséggel
  egyezik ki. Marad **n-max 2** (robusztusabb változó acceptance mellett).
- **`-ub 1024`: −1,7% / −2,5%, VRAM +316 MB → elvetve** (a korábbi −7,7% iránya
  megerősítve).
- Minden futás acceptance-e konzisztens (nmax2: 0,4375; nmax1: 0,55; nmax3: 0,3125).

## 21. kör (Part 1 / WP-C0) — GEMV mikrobenchmark: a 160 B-os formátum ELVETVE (2026-10-20)

Cél: eldönteni, hogy a szélesebb (8/16 B) betöltés emeli-e a GEMV
sávszélességét (a 160 B-os igazított q4_K formátum indoklása).
Módszer: önálló kernel (`/tmp/opencode/gemvbench.cu`, sm_75), azonos
4-bites adatforgalom, 1 warp/sor, dp4a a nibble-ökön; 2,5 GiB buffer.

| variáns | ms | GB/s (read) |
|---|---|---|
| memcpy D2D (r+w) | 13,373 | 401,5 |
| pure-read 16 B | 6,306 | 425,7 |
| **gemv-4B (jelenlegi séma)** | **4,972** | **539,9** |
| gemv-8B | 5,311 | 505,4 (−6,8%) |
| gemv-16B | 5,794 | 463,3 (−16,5%) |

- **A hipotézis megdőlt: a 16 B-os betöltés 16,5%-kal LASSABB.** A 4 bájtos
  minta 540 GB/s-ot ér el (88% a 616 GB/s csúcsból) → a betöltés-szélesség
  nem szűk keresztmetszet.
- **WP-C (160 B-os igazított q4_K + 16 B betöltés) ELVETVE** — nem indokolt.
- Következmény: a valós decode effektív ~231 GB/s-a (5,5 GB / 20,8–23,8 ms)
  nem a súly-streamelésen, hanem a **többi komponensen** múlik (FA-számítás
  hosszú kontextuson, small ops, sync-ek) → következő kísérlet: FA-kernelcsalád
  A/B f16 KV-val (VEC vs MMA; a meglévő `GGML_CUDA_FA_FORCE_KERNEL` kapcsolóval).

## 22. kör (Part 1) — FA kernelcsalád A/B f16 KV-val (2026-10-20)

A C0 tanulság (a maradék idő nem a súly-streamelés) után a FA-t vizsgáltuk:
f16 KV-nál nincs konverzió, ezért a batch-1 VEC út helyett az MMA (tensor core)
lehetne jobb. A meglévő `GGML_CUDA_FA_FORCE_KERNEL` kapcsolóval, 8 futás.

| config | R3 tg rep0 | medián | pp |
|---|---|---|---|
| no-MTP VEC (default) | 49,18* | 49,18 | 1311,5 |
| no-MTP MMA (`mma:1`) | 49,11 | 49,11 | 1311,7 |
| no-MTP VEC (2) | 49,02 | 49,02 | 1309,7 |
| no-MTP MMA (2) | 48,59 | 48,59 | 1300,0 |
| MTP default | 67,33 | 67,33 | 1166,2 |
| MTP `mma:4` (draft is MMA) | 65,29 | 65,29 | 1142,2 |
| MTP default (2) | 65,33 | 65,33 | 1144,9 |
| MTP `mma:4` (2) | 66,22 | 66,22 | 1130,7 |

- **Verdikt: semleges** (no-MTP ±0,5%; MTP a két pár ellentétes előjelű, a
  szórán belül) → **marad a default** (batch 1: VEC, batch ≥2: MMA).
- Bizonyíték: `FA_FORCE_KERNEL: forcing` log 1 a kényszerített futásokban, 0 a
  defaultokban; acceptance mindenhol 0,4375.
- Ezzel a FA-útvonal is lefedve: f16 KV + default kernelválasztás a nyertes.

## 23. kör (Part 1 / WP-D) — CPU FA split-KV kvantált KV-ra: ELVETVE (2026-10-20)

A CPU FA split-KV útja eddig csak f16/f32 KV-ra ment (`kv_is_f32_or_f16`),
noha a chunk-függvény típus-generikus (K: `vec_dot`, V: `to_float`). A kaput
`GGML_CPU_FA_SPLIT_QUANT=1` env-gated kapcsolóval lazítottuk (default off).

**Helyesség**: CPU (9B, `-ngl 0`, `-fa on`, q4_1 KV, 1217 token kontextus,
greedy, 16 token) → a kimenet **bitazonos** a default úttal; a bizonyíték-log
mutatja, hogy a split út tényleg fut (`K=q4_1, V=q4_1, n_kv=1280`).

**Sebesség — közvetlen CPU FA mikrobenchmark** (`/tmp/opencode/fa_cpu_bench.c`,
n_kv=65536, DK=128, 8 KV-head, 32 Q-head, 16 thread):

| út | ms/hívás | KV-read |
|---|---|---|
| default (chunk-loop) | **36,41** | 2,3 GB/s |
| split-KV (kvantált) | 61,90 | 1,4 GB/s (**+70% lassabb**) |

- **Ok**: a split út a chunkon belül `for q_head` ciklussal **minden Q-headre
  újra végigolvassa ugyanazt a KV-chunkot** → GQA-nál 32× forgalom-növelés
  (a chunk már nem fér az L3-ba) → a többlet megeszi a jobb terheléselosztást.
- **Verdikt: elvetve** — a napi CPU-út marad a chunk-loop (a kapcsoló env-gated
  kísérleti eszközként marad, default off; egy jövőbeli, head-fúziós
  újraírásnak lehet értelme, de az már nagyobb munka).
- Megjegyzés: a CPU FA abszolút lassú (2,3 GB/s KV-read) — a q4_1 dequant a
  tokenenkénti vec_dotban dominál; ez a CPU hosszú kontextusú gyorsítás igazi
  korlátja, nem a párhuzamosság.

## Part 1 — VÉGLEGES konfiguráció (2026-10-20)

**CUDA napi parancs (f16 KV, MTP, 170 W):**

```
build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 -t 1 -tb 1 --poll 0 \
  --spec-draft-backend-sampling -bs
```

Mért (r19–r24, kontrollált A/B): **MTP R3 tg +14,2% / +14,5%** a q4_1-hez
képest (abszolút 64–75 t/s a session hőmérsékletétől függően), **no-MTP
+12–17%** (48–49 t/s), pp változatlan (~1150–1310), acceptance bitazonos.
**VRAM @80k: 9336 MB** (a q4_1-nél +1,42 GB).

**VRAM-korlátos / nagyobb kontextusú változat:** ugyanez `-ctk q4_1 -ctv q4_1`-gyel
(R3 tg ~65 t/s, VRAM 7876 MB).

**Text-only szolgáltatás (opcionális):** `--no-mmproj` — a 879 MB-os vision
projektor nem töltődik be, a sebesség változatlan (r24: 65,6 vs 66,1 t/s).

**Part 1 eredménytábla:**

| WP | eredmény | döntés |
|---|---|---|
| B1 f16 KV | +14,2% / +14,5% MTP; +12–17% no-MTP | **default: f16** |
| B2 KV-shadow | nem futott: az f16 árnyék (+2,1 GB) drágább, mint a tiszta f16 KV (+1,42 GB) | elvetve |
| E n-max 3 | −8,4…−9,0% | elvetve |
| E n-max 1 | −0,3% (rep0), medián +3,5% (anchor-szórás) | marad 2 |
| E `-ub 1024` | −1,7…−2,5%, VRAM +316 MB | elvetve |
| C0 GEMV 16 B | **−16,5%** (a 4 B-os séma 540 GB/s = 88% csúcs) | WP-C elvetve |
| FA kernelcsalád (f16 KV) | semleges | default (VEC batch 1, MMA ≥2) |
| D CPU split-KV kvantált KV | **−70%** (per-head KV-újraolvasás) | env-gated, default off |
| F `--no-mmproj` | −879 MB RAM, sebesség azonos | opcionális |

**Kódváltozások (Part 1):**
- `ggml/src/ggml-cpu/ops.cpp`: `GGML_CPU_FA_SPLIT_QUANT` env-gated kapcsoló
  (split-KV kvantált KV-ra; **default off**, mért −70% → nem használni).
- Minden más kísérlet a meglévő env-kapcsolókkal futott (nincs új default).

## 24. kör (Part 1) — záró validáció: f16 KV napi konfig + `--no-mmproj` (2026-10-20)

| run | R3 tg | R3 pp | VRAM | acceptance |
|---|---|---|---|---|
| napi (f16 KV, MTP) | 66,13 | 1145,2 | 9336 | 0,4375 |
| + `--no-mmproj` | 65,59 | 1163,4 | 9336 | 0,4375 |

- A napi konfig reprodukálja a +14–16%-os nyereséget (a q4_1-hez képest).
- `--no-mmproj`: a logban **0** „loaded multimodal model" sor (a 879 MB-os
  projektor nem töltődik be), a sebesség a szórán belül → opcionális
  text-only kapcsoló.

## 25. kör (Vulkan SDK 1.4.363 + újrafordítás) — 2026-10-20

- Letöltve a **Vulkan SDK 1.4.363.0** (2026-09-28, 366 MB tarball, sha256
  `197962f5...`), részlegesen kicsomagolva (glslc, vulkaninfo, spirv-dis,
  fejlécek, libshaderc — 66 MB), a tarball törölve.
- A `build-vulkan` újrakonfigurálva az SDK glslc-jére (`shaderc 2026.4`,
  fejlécek 1.4.363); a rendszerbeli loader maradt (`libvulkan.so`).
- **Build-hiba és javítás**: az új glslc már támogatja az
  `GL_EXT_float_e2m1/e4m3` kiterjesztéseket → a C++ hivatkozik az FP4
  (`_ocp`) shader-tömbökre, de a shader-generátor (ExternalProject)
  binárisa elavult volt → hiányzó szimbólumok. Javítás: a generátor
  prefixének és a generált fejlécnek a törlése, tiszta újragenerálás
  (1570 SPIR-V, 12 `_ocp` tömb). Ez egy upstream build-rendszer hiba
  (elavult ExternalProject + új glslc kombinációja).
- Új `libggml-vulkan` md5: `a8a14bd9b313bd8a5023036917fa14f3`
  (régi: `1a9808237a8111343c225cd93882a913`).

## 26. kör (Vulkan eszközlánc A/B: régi vs SDK-s könyvtár) — 2026-10-20

4 futás, csak a `libggml-vulkan` cserélve, q4_1 KV, MTP nélkül, 170 W.

| futás | könyvtár | R3 tg | R3 pp | VRAM |
|---|---|---|---|---|
| old | 1a980823 | 28,04 | 808,0 | 7146 |
| new | a8a14bd9 | 34,02 | 954,0 | 6762 |
| oldb | 1a980823 | 34,36 | 949,0 | 6755 |
| newb | a8a14bd9 | 34,19 | 948,8 | 6755 |

- **A tiszta pár (oldb vs newb): −0,5% tg, −0,02% pp, azonos VRAM → semleges.**
  Az első pár +21%-a a session hidegindítási hatása (az anchor `oldb` maga
  +22,5%-ot ugrott az `old`-hoz képest; a VRAM is 7146 → 6755).
- **Döntés: az SDK-s build marad** (újabb eszközlánc, azonos sebesség).
- Tanulság a méréshez: a Vulkan session első futása hidegen akár 20%-kal
  lassabb; a párokat mindig a bemelegedett futásokból kell venni.

## 27. kör (Vulkan KV-típus: q4_1 vs f16) — 2026-10-20

4 futás, SDK-s Vulkan könyvtár, MTP nélkül, 170 W, váltogatott párok.

| futás | KV | R3 tg | R3 pp | VRAM |
|---|---|---|---|---|
| q41 | q4_1 | 34,30 | 951,9 | 6755 |
| f16 | f16 | **37,99** | **1077,5** | 8480 |
| q41b | q4_1 | 33,77 | 956,6 | 6755 |
| f16b | f16 | **37,65** | **1076,5** | 8480 |

- **tg: +10,8% / +11,5%** (konzisztens pár), **pp: +13,2% / +12,5%**.
- Ár: **+1725 MB** VRAM (6755 → 8480 MB @ 80k).
- A pp nyereség oka: a prefill FA f16 KV-vel dequantizálás nélkül olvas
  (CUDA-n ez semleges volt, ott a FA már jól kezelte a q4_1-et).
- **Döntés: a Vulkan napi profil `-ctk f16 -ctv f16`** (a 8,5 GB belefér).
- Megjegyzés: ez a kör tiszta volt (a beragadt szerver csak később, a
  pipeline-cache tesztnél jelent meg). Tanulság a későbbi körökhöz: minden
  mérés előtt `ps ax | grep -i llama-server` + PID-es kill, és GPU-ellenőrzés.

## 28. kör — tartós Vulkan pipeline-gyorsítótár (betöltési sebesség) — 2026-10-21

Implementáció (környezeti kapcsolóval, alapból KI):
- `GGML_VK_PIPELINE_CACHE=<fájl>`: a device létrehozásakor betölti a
  `VkPipelineCache` blobot, a pipeline-ok létrehozásakor használja, kilépéskor
  (`atexit`) visszaírja. A device destruktor nem fut le (statikus regiszter
  tartja életben), ezért kell az exit-handler.
- Érintett fájlok: `ggml/src/ggml-vulkan/ggml-vulkan-types.h` (2 tag),
  `ggml-vulkan.cpp` (betöltés/mentés + `createComputePipeline` hívás).
- A shader-fordítás (SPIR-V → SASS) a driverben, CPU-n történik; a cache
  ezt spórolja meg a következő indításoknál.

Mérés (8 futás, váltogatott párok, f16 KV, 651 tokenes prompt, 170 W):

| futás | cache | startup | req1 pp (ms) | req2 pp (ms) |
|---|---|---|---|---|
| cold | hideg (ír) | 10850 | 1279 | 571 |
| noc1 | nincs | 10735 | 659 | 616 |
| warm1 | meleg | 9089 | 730 | 604 |
| noc2 | nincs | 9763 | 781 | 605 |
| warm2 | meleg | 10094 | 651 | 663 |
| noc3 | nincs | 12761 | 780 | 666 |
| warm3 | meleg | 9818 | 636 | 677 |
| nocend | nincs | 9523 | 747 | 588 |

- **Első prefill (req1)**: nincs-cache átlag 742 ms, meleg cache 672 ms
  (−9,4%); a párok: 1× rosszabb, 2× jobb (−17%, −18%) → irány helyes, de a
  szórás nagy. A gyorsítótár-fájl 3,33 MB.
- **Startup**: átlag −1,0 s (−9,6%), páronként −15% / +3% / −23% → zajos.
- **Hol megy el a betöltés** (log-időbélyegek): a modell+mmproj betöltése
  ~9,8 s a ~10,7 s-ból (0,7 s → 8,5 s súlyok, 8,5 → 10,5 s mmproj), a
  shader/pipeline létrehozás csak ~70 ms-ot tesz hozzá az első kérésnél.
  Vagyis a betöltés főleg I/O + GGUF-feldolgozás, nem shader-fordítás.
- **CPU-mérés** (új `cpu-watch.sh`, a `bench.py`-ba építve): a szerver a
  teljes futás alatt átlag **0,72–0,80 magot** használ, sys 2,6–5,0% —
  a teszt nem CPU-korlátos; a CPU-t főleg a modellbetöltés (egy szál) és a
  tokenizálás/mintavétel használja.
- **Döntés**: a kód marad, **környezeti kapcsolóval, alapból kikapcsolva**
  (a nyereség kicsi és zajos; nem teszünk alapértelmezett viselkedést kockára).
- Tanulság: a korábbi (szennyezett) mérés 2–9 s-os „prompt_ms" értékeit egy
  beragadt szerver okozta — a tiszta mérés 0,6–0,8 s.

## 29. kör — Vulkan: FORCE_MMVQ újraellenőrzés (f16 KV) — 2026-10-21

4 futás, váltogatott párok, f16 KV, 170 W, CPU-figyeléssel.

| futás | env | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| def | — | 44,90 | 42,11 | 952,3 | 35,14 | 8459 |
| mmvq | FORCE_MMVQ=1 | 37,51 | 39,13 | 1036,2 | 32,29 | 8459 |
| defb | — | 49,23 | 49,56 | 960,1 | 37,15 | 8449 |
| mmvqb | FORCE_MMVQ=1 | 45,16 | 42,05 | 1012,7 | 32,81 | 9091 |

- **tg: mindkét párban ront** (R1 −8…−16%, R2 −7…−15%, R3 −8…−12%).
- pp kicsit javul (+5,5…+8,8%) — de a tg a fő cél → **elvetve**
  (egyezik a 11. kör eredményével, most f16 KV-n is).
- CPU: a szerver R1-nél 0,87–0,93 mag, R3-nál 0,47–0,49 mag, sys 1,5–3,8%
  → nem CPU-korlátos; az MMVQ-kényszerítés nem CPU-többlet miatt lassabb.
- VRAM: az mmvqb futásnál +642 MB (MMVQ prealloc pufferek) — újabb ok az elvetésre.

## 31. kör — master merge (82 commit) hatásának mérése — 2026-10-21

Két backend, egy-egy váltogatott pár, majd **fordított sorrendű ismétlés**
(r31c) a drift kiszűrésére. Csak a backend-könyvtár cserélve (libllama azonos).

| futás | R1 tg | R2 tg | R3 pp | R3 tg |
|---|---|---|---|---|
| CUDA pre (hideg, 1. futás) | 59,85 | 78,82 | 1196 | 55,61 |
| CUDA merge | 82,51 | 78,01 | 1230 | 70,37 |
| CUDA mergeR (fordított pár) | 71,30 | 79,46 | 1265 | 71,86 |
| CUDA preR (fordított pár) | 72,83 | 82,81 | 1283 | 75,14 |
| Vulkan pre | 50,96 | 53,15 | 1074 | 37,75 |
| Vulkan merge | 51,55 | 48,93 | 972 | 30,63 |
| Vulkan mergeR | 51,99 | 51,44 | 1088 | 37,36 |
| Vulkan preR | 50,23 | 53,56 | 1065 | 37,99 |

- **Az első futás csapdája:** a session első CUDA futása **35%-kal lassabb**
  (55,61 vs 75,14 ugyanazzal a könyvtárral) — a frissen épített CUDA
  könyvtár driver-JIT/melegedés miatt. Ez okozta a látszólagos „+26,5%”-ot.
- **Kiegyensúlyozott (meleg) összehasonlítás:**
  - **CUDA: merge −4,4%** (R3 tg), R2 −4,0%, pp −1,5% → kis lassulás,
    valószínűleg a `99db719bd` FA vec WAR helyességjavítás ára (a vec FA
    a batch-1 dekód útja, hosszú kontextuson meleg).
  - **Vulkan: merge −1,7%** (R3 tg), pp +2,2% → zajhatáron belül, **semleges**.
    (A −19%-os első mérés VRAM/SM-órajel-drift műtermék volt; a mem-órajel
    rögzítése ezt most kizárta.)
- **Döntés: a merge marad** (upstream alignáció + helyességjavítások;
  a CUDA −4% elfogadható ár a WAR-hazard fixért). A `qwen35.cpp` 2D-reshape
  hatása ebben a mérésben nem izolálható (libllama mindkét futásban azonos).
- **Módszertani tanulság:** a session első futását (főleg CUDA-nál) ki kell
  dobni vagy külön „hideg" anchor-ként kezelni; a párokat fordított
  sorrendben is meg kell ismételni.

## 30. kör — Vulkan coopmat A/B (`GGML_VK_DISABLE_COOPMAT`) — 2026-10-21

4 futás, váltogatott párok, f16 KV, 170 W, CPU-figyeléssel.

| futás | env | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| def | — | 50,38 | 47,42 | 981,3 | 34,26 | 8461 |
| nocm | DISABLE_COOPMAT=1 | 50,29 | 47,29 | 986,8 | 34,74 | 8461 |
| defb | — | 47,40 | 47,79 | 988,0 | 34,57 | 8461 |
| nocmb | DISABLE_COOPMAT=1 | 48,62 | 48,22 | 986,3 | 34,84 | 8461 |

- R3 tg: +1,4% / +0,8% (coopmat nélkül), pp ±0,6%, VRAM azonos
  → **zajhatáron belül, semleges**. Marad az alapértelmezés (coopmat be).
- Megjegyzés: a Turing FA amúgy is `scalar` út (a kódban rögzített NVIDIA
  compiler-bug miatt), a coopmat csak a GEMM-eket érinti → kicsi a hatás.

---

## Vulkan — VÉGLEGES konfiguráció (2026-10-21)

```
LD_LIBRARY_PATH=build-vulkan/bin build-vulkan/bin/llama-server \
  -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload -ub 512
# opcionális: GGML_VK_PIPELINE_CACHE=/tmp/vk-pipeline-cache.bin (kis indulási nyereség)
```

- **f16 KV: +10,8–11,5% tg, +12,5–13,2% pp** (ár: +1725 MB VRAM @80k) — a fő nyereség.
- Elvetve (mérve, nem nyerő): FORCE_MMVQ (−8…−16% tg), DISABLE_COOPMAT (semleges),
  MTP Vulkanon (−16%), DISABLE_MMVQ (semleges), MAX_NODES (semleges).
- Pipeline-gyorsítótár: env-gated, alapból ki (első prefill −9% átlag, zajos).
- SDK-csere (1.4.363): semleges; a `-O` shader-kísérlet (rope/coopmat)
  előkészítve (`GGML_VULKAN_OPT_ALL_SHADERS=1` build-időben) — kis várható
  nyereség, helyességi kockázat miatt egyelőre nem kapcsoltuk be.
- A Vulkan továbbra is ~2× lassabb tg-ben a CUDA+MTP napi konfignál
  (R3: 34–38 vs 70–75), ezért a napi használat CUDA.

## 32. kör — master merge #2 (31 commit, 84e66d032 → e3797b184) — 2026-10-22

Fast-forward + stash pop, **konfliktus nélkül**; a 7 helyi módosítás bitre
azonosan megmaradt (235 insertions / 4 deletions), a kulcs-markerek
(`GGML_CPU_FA_SPLIT_QUANT`, `GGML_CUDA_BLOCKING_SYNC`, `GGML_VK_PIPELINE_CACHE`,
`GGML_VK_RM_KQ`) ellenőrizve.

**Upstream tartalom (relevancia):**
- `eb182d85c` d2t GPU-n tartása — **EAGLE3-only**, a Qwen3.5-MTP nem használ d2t-t.
- `d454a204e` új fused `top_k_mask` op (QSA/DSA/GLM) — a qwen35 nem használja;
  a CUDA regisztráció +6 sor, a meglévő utakat nem érinti.
- `824eaecf4` vocabulary loading optimalizálás — betöltési idő, nem dekód.
- `261700b89`, `9a3b91c68` server API/accounting — funkcionális.
- `479a52779` CPU gelu_quick — CPU-only.
- Többi: új modellek (Flint, SenseNova U1), SYCL/OpenCL/Metal/Hexagon/WebGPU,
  jinja, UI → **nincs CUDA/Vulkan dekód-perf változás**.
- Az egyetlen érdemi core-változás a K-shift feltétel (`llama-kv-cache.cpp`),
  ami csak context-shiftnél fut (a mérésekben nincs shift).

**Build:** CUDA `a73534b18fdc293a953e5ca948390d70`,
Vulkan `df3700a40572825e1d6f6909ad2de48d`.

**Sanity-mérés (170 W, CPU-figyelés, f16 KV):**

| futás | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|
| V-r32-new | 48,73 | 47,96 | 1034,3 | 34,82 | 8388 |
| C-r32-newA | 76,04 | 73,79 | 1188,1 | 64,82 | 9226 |
| C-r32-newB | 76,16 | 72,52 | 1169,7 | 66,53 | 9226 |
| C-r32-newC | 70,32 | 74,78 | 1250,8 | **70,16** | 9226 |
| C-r32-newD | 83,93 | 81,23 | 1250,4 | **72,29** | 9226 |

- **CUDA: semleges** — a 3.–4. futás platója (70,2 / 72,3 R3; R2 81,2;
  pp 1250) a r31-es meleg tartományban van (R3 70,4–75,1; R2 78–83; pp 1230–1283).
- **Vulkan: semleges** — 34,82 R3 a r30-as session klaszterében (34,3–34,8);
  a r31-es 37,4–38,0 session-szintű drift (ld. 31. kör).
- **Bemelegedési effektus megerősítve:** egy session első 1–2 CUDA futása
  5–12%-kal lassabb (64,8 / 66,5 → 70,2 / 72,3), akkor is, ha a könyvtár
  ugyanaz. **Méréshez az első 2 futást ki kell dobni** (vagy anchor kell).
- **VRAM:** CUDA 9290 → 9226 MiB (−64), Vulkan 8461 → 8388 MiB (−73) — valószínűleg
  a vocab/tokenizer változás; nem regresszió.
- **Load time:** 6,4–10,4 s, page-cache dominált → nem értékelhető.
- **Funkcionális check:** greedy `llama-cli` kimenet koherens (chat template OK).

**Döntés: a merge marad, a napi konfigurációk változatlanok.**

## 33–34. kör — MTP config sweep + szigorú A/B (Vulkan, f16 KV) — 2026-10-22

**33. kör (screen-r33, 60k prompt, 128 gen, cooldown ≤65 °C induláskor):**

| config | pp | tg |
|---|---|---|
| base-a / base-b | 917,2 / 927,7 | 31,08 / 29,44 |
| mtp2 (dbs+bs) | 737,6 | 29,04 |
| mtp1 (dbs+bs) | 722,3 | 30,38 |
| mtp3 (dbs+bs) | 742,6 | 24,92 |
| mtp2 (nincs -bs) | 769,2 | 31,52 |
| **mtp2 (-bs)** | 866,4 | **34,26** |
| mtp2 (ismétlés) | 730,9 | 28,35 |

**Kritikus korrekció:** a `--spec-draft-backend-sampling` **alapból BE van
kapcsolva** (`common.h`: `draft.backend_sampling = true`), ezért a „dbs" és
a „nincs dbs" feliratú futások **azonos konfigurációk** voltak. A köztük
mért 29,04 vs 34,26 (17%) tehát tiszta futászaj, nem config-hatás. A 33. kör
„+13%-os nyertese" így érvénytelen; a 34. kör ezt meg is cáfolta.

**34. kör (ab-60k.sh: szigorúan váltogatott, cooldown ≤62 °C, 2 kör):**

| config | pp | tg futás1 / futás2 | értékelés |
|---|---|---|---|
| base | 917,5 / 917,3 | 30,75 / 31,00 | stabil |
| base (anchor, később) | 908,6 | 29,88 | stabil |
| mtp2 (dbs default ON) | 735,3 / 736,4 | **23,66 / 29,15** | instabil, −23…−5% |
| mtp2 (dbs explicit ON, ismétlés) | 729,7 / 743,5 | 28,63 / 27,65 | −9% |
| **base + -bs** | 904,9 / 915,2 | **32,53 / 31,96** | **+5,6% tg**, pp semleges |

- Az azonos acceptance (70/113, mean len 2,23) és azonos GPU-statisztika
  (88–90% util, 154–158 W) mellett **ugyanaz a konfiguráció 23,66–29,15 t/s
  között szórt** (17%) → a szórás nem config-, hanem futásfüggő.
- **MTP Vulkanon 60k-n nem nyerő**: a 4 azonos-config futás átlaga 27,3 t/s
  a base 30,5-tel szemben (**−11%**), nagy szórással. A 33. kör +13%-a
  futászaj volt.
- **Napi Vulkan config: `-bs` hozzáadva** (+5,6% tg, pp azonos, VRAM azonos).

**Rövid kontextus (2,8k prompt, 256 gen):**

| config | tg |
|---|---|
| base | 41,6 (40,55 / 42,61) |
| base + -bs | 45,59 |
| base + `-t 1 -tb 1 --poll 0` | 46,5 (47,26 / 45,68) |
| base + `-bs -t 1 -tb 1 --poll 0` | 50,9 (48,80 / 53,11) |
| mtp2 (-bs) | 40,84 |
| mtp2 (+dbs) | 46,35 |
| mtp1 (+dbs) | 48,26 |

- `-t 1 -tb 1 --poll 0` rövid kontextuson +11…+22%, de 60k-n pp −5%,
  MTP-vel −9% → csak rövid kontextusú használatra.
- CUDA ugyanazon a prompton: base 52,73; MTP2 73,90 (+40%) → a CUDA MTP
  előnye valós, a Vulkan draft-gráf költsége aránytalanul nagy.

**Gyökérok-mérések (új, env-gated instrumentáció):**

- `LLAMA_GRAPH_TIMING=1` (`llama-context.cpp`): decode-nál
  **build = 0,06 ms, set_inputs ≈ 0, submit = 14,84 ms, sched sync = 0,03 ms**
  → a Vulkan backend a `graph_compute` hívásban blokkol (nincs CPU/GPU overlap),
  a submit idő = GPU végrehajtás.
- `llama-bench` (szerver nélkül) rövid kontextuson **tg 47,64 t/s**;
  a szerver 42,6 → a szerver-oldali overhead ~2,5 ms/token.
- Effektív sávszélesség decode-nál: 5,7 GiB / 21 ms ≈ **291 GB/s**
  (a 616 GB/s 47%-a); a C0 GEMV-mikrobench 540 GB/s (88%) → **~1,85× elvi
  headroom**, amit a sok kis kernel közti részek/dispatch-viselkedés visz el.
- Prefill: 1425–1570 t/s ≈ 26 TFLOP/s (~96% fp16 csúcs) → a GEMM út jó.
- `GGML_VK_SERIALIZE_SUBMISSIONS=1`: 10,75 t/s (−75%) → a submit/fence
  várakozás VFIO alatt drága, de a normál aszinkron út ezt elrejti.

**Új env-gated kapcsolók (default = változatlan viselkedés):**

- `GGML_VK_FLOPS_PER_SUBMIT_DIV` (default 40): a gráf submit-felosztása.
  Mérés: 40 → 42,61; 8 → 40,61; 4 → 37,08; 1 → 43,20 t/s → **nincs
  konzisztens hatás**, default marad.
- `LLAMA_GRAPH_TIMING` (default off): fázis-időmérés a logban.

**Döntések:**
1. Vulkan napi parancs: **`-bs` bekerül** (`--backend-sampling`).
2. MTP a Vulkanon 60k-n **elvetve** (instabil vagy −9%); a CUDA+MTP marad
   a gyors út, ha a felhasználó CUDA-t futtat.
3. `-t 1 -tb 1 --poll 0`: csak rövid kontextusú (≤3k) munkához.

**Vulkan napi parancs (2026-10-22-től, f16 KV + backend sampling):**

```
build-vulkan/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  -ub 512 -bs
```

Mért (60k, szigorú A/B): tg 32,25 t/s (`-bs`) vs 30,54 (base) = **+5,6%**,
pp 910 vs 917 (azonos), VRAM 8388 MiB. MTP nem kerül bele (lásd 33–34. kör).
Rövid kontextusú (≤3k) munkához opcionálisan `-t 1 -tb 1 --poll 0`
(+11…+22% tg, de 60k-n pp −5%).

> **FIGYELEM — a fenti MTP-verdikt érvénytelen, lásd a 35. kört.**

---

## 35. kör — 41k kontextus: MTP újraértékelése, CUDA vs Vulkan, `-t`/`-ub` (2026-10-26)

**Előzmény:** a 33–34. kör „MTP Vulkanon veszít" verdiktje **128 tokenes**
próbákon alapult. A hosszú, valós használat (41k prompt, 2000–8000 generált
token) ezt megcáfolja: a rövid próba nem méri a MTP bemelegedését, és 128
tokennél az acceptance szórása nagyobb, mint a hatás.

Minden futás ugyanaz a beszélgetés, `pp = 41173` token, Vulkan `f16` KV,
`-bs`, kézi futtatás (nem 170 W-os protokoll; a >10%-os effektek egyértelműek).

| config | pp t/s | gen tok | tg t/s | acc | mean len | ms/lépés |
|---|---|---|---|---|---|---|
| CUDA + MTP2, `-t 24` (#1) | 1034 | 4123 | 58,39 | 0,794 | 2,59 | 44,4 |
| CUDA + MTP2, `-t 24` (#2) | – | 2308 | 48,34 | 0,658 | 2,32 | 48,0 |
| Vulkan + MTP2, `-t 24` (#1) | 1075 | 8118 | 48,73 | 0,714 | 2,43 | 49,9 |
| Vulkan + MTP2, `-t 24` (#2) | – | 2317 | 49,13 | 0,702 | 2,40 | 48,8 |
| Vulkan, nincs MTP, `-t 24` | 1213 | 2172 | 37,39 | – | 1 | 26,7 |
| Vulkan, nincs MTP, `-t 1` | 1202 | 1382 | 37,94 | – | 1 | 26,4 |
| Vulkan, nincs MTP, `-t 1 -b 4096 -ub 4096` | 1145 | (8 t/s) | ~8,0 | – | 1 | **125** |

**Következtetések**

1. **MTP a 41k-n +30% tg, ára 4,37 s pp** (38,31 s vs 33,94 s, −11,4% pp).
   Megtérülés: `4,37 s / 6,5 ms` ≈ **670–700 generált token**. Egy 41k prompt
   + 2000 token válasz: 79,3 s vs 87,7 s (−9,5%); 5000 tokennél −16%.
   → hosszú válaszoknál a MTP egyértelműen nyerő, rövid válaszoknál (≤700
   token) pp-ben veszít.
2. **CUDA vs Vulkan 41k-n: egy szint.** A CUDA 58,39 t/s-a **acceptance-
   artefakt** (0,794 vs 0,714); lépésidőben CUDA 44,4/48,0 ms, Vulkan
   49,9/48,8 ms → a különbség a zajon belül van. **pp-ben a Vulkan jobb**
   (1075 vs 1034). A CUDA előnye rövid kontextuson valós (2,2k: 73,9 vs 46,4
   a 33. körben), 41k-n eltűnik.
3. **A szövegfüggés ±9%.** Ugyanaz a Vulkan+MTP config 41k-n 48,7 / 49,1 /
   53,4 t/s-ot ad attól függően, mit generál (acceptance 0,66–0,79). Ezért
   MTP-t csak **ms/lépés + acceptance** párossal, vagy ≥2000 generált
   tokennel, több ismétléssel szabad összevetni. A t/s önmagában félrevezet.
4. **`-t 1 -tb 1` 41k-n (MTP nélkül) semleges**: pp −1%, tg +2%. Nem ront,
   de nem is hoz; hosszú kontextushoz marad a `-t 24 -tb 24`.
5. **`-b 4096 -ub 4096` katasztrófa: 8,0 t/s decode (125 ms/lépés), és a pp is
   romlik (1145 vs 1213).** Az `-ub` marad 512 (ami egyben a default).
   Mechanizmus nem tisztázott (feltevés: a 4096 széles ubatch prefill-gráfja
   ~4 GB tranzens puffert kér a ~2,6 GB szabad VRAM-ba, és a decode pufferei
   kiszorulnak); a gyakorlati szabály: **soha ne emeld 512 fölé**.
6. **Korrekció:** a 34. kör „MTP Vulkanon 60k-n elvetve" verdiktje **visszavonva**
   (128 tokenes próba + 60k-s VRAM-nyomás). A 60k-s MTP-viselkedést hosszú
   generálással újra kell mérni (nyitott).

**Vulkan napi parancs (2026-10-26-tól, MTP-vel, hosszú kontextushoz):**

```
build-vulkan/bin/llama-server --port 8080 \
  -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-backend-sampling -bs \
  -t 24 -tb 24 --poll 0
```

CUDA fallback ugyanez a `build-cuda/bin/llama-server` binárisal.
Rövid (≤3k) csevegéshez `-t 1 -tb 1` tehető a `-t 24 -tb 24` helyére.

---

## 36. kör — Part 2/A: egyedi kvant-mix (2026-10-26)

**Cél:** a decode lépésenkénti súlyforgalmának csökkentése. Kiindulás
(UD-Q4_K_XL): a lépés forgalma **7,22 GB**, ebből a `output.weight`
(Q6_K, 834 MB) **háromszor** olvasódik (target verify + 2 MTP draft lépés)
= 2,50 GB, azaz a forgalom **34,7%-a**. A qwen35 MTP-gráf a
`model.output`-ot használja headként (`nextn.shared_head_head` nincs a
GGUF-ban), ezt a forrás is megerősíti (`qwen35.cpp:656`).

**Módszer:** `Qwen3.5-9B-BF16.gguf` (18,41 GB) + `imatrix_unsloth.gguf_file`
(248 bejegyzés, 77 chunk) → `llama-quantize`, **explicit tenzor-táblával**
(`mix-gen.py` a kiadott fájlból). A regexes `--tensor-type` önmagában nem
volt reprodukálható: a default Q4_K-mix belső számlálói (i_ffn_down,
i_attention_wv) csak a nem-felülírt tenzorokon nőnek, ezért a felülírások
száma eltolja, mely rétegek kapnak Q6_K-t. A teljes tábla ezt kiküszöböli:
az `a0` replica **5840,36 MiB = a kiadott fájl adata (6124 MB)** pontosan.

**Variánsok** (`quant-mix.sh`, 24 szál, 3–7 min/db):

| verzió | változás | fájl | lépés-forgalom |
|---|---|---|---|
| a0 | a kiadott mix pontos replikája | 6,14 GB | 7,22 GB (0%) |
| v1 | head Q6_K → Q5_K | 6,00 GB | 6,82 GB (−5,6%) |
| v2 | head Q6_K → Q4_K | 5,87 GB | 6,43 GB (−10,9%) |
| v3 | v2 + ffn_down Q6_K→Q5_K, ffn_down Q5_K→Q4_K, ssm_out Q8_0→Q5_K | 5,59 GB | 6,15 GB (−14,8%) |
| v4 | minden kvantálható tenzor Q4_K | 5,21 GB | 5,77 GB (−20,1%) |

**Nyers decode screen** (`llama-bench -p 128 -n 128 -r 3`, Vulkan, f16 KV,
nincs MTP, 6 modell interleaved; a `base` első futása hideg):

| modell | méret | pp t/s | tg t/s | Δ tg vs a0 |
|---|---|---|---|---|
| base (UD-Q4_K_XL) | 5,70 GiB | 1539 ± 173 | 54,02 ± 1,79 | +2,2% (zaj) |
| a0 (replica) | 5,70 GiB | 1632 ± 32 | 52,85 ± 0,76 | – |
| v1 (head Q5_K) | 5,58 GiB | 1654 ± 29 | 56,93 ± 1,54 | +7,7% |
| v2 (head Q4_K) | 5,46 GiB | 1652 ± 18 | 59,24 ± 1,22 | +12,1% |
| v3 | 5,20 GiB | 1601 ± 105 | 59,17 ± 0,05 | +12,0% |
| **v4 (mind Q4_K)** | **4,84 GiB** | **1839 ± 22** | **63,82 ± 1,05** | **+20,8%** |

**Következtetés a screenből:** a byte-modell pontos. v2-nél a számított
+8,9% (1x head olvasás) helyett +12,1% jött — a Q4_K a natív, gyorsabb
Vulkan-út is, nem csak kevesebb byte. v4 pp-je is +13% (1839 vs 1632),
tehát a kisebb kvant nem lassítja a prefillt. v3 gyakorlatilag egy szinten
van v2-vel (a `ssm_out` Q8_0→Q5_K nem hoz mérhetőt), viszont kisebb.

> **FIGYELEM:** az alábbi MTP-s A/B első olvasata („a kvantálás 15%-ot ront")
> **hibás volt** — a 60k-s futás thermálisan throttol, és a drift nagyobb,
> mint a mért hatás. A helyes értékelés a lenti két szakasz.

**MTP-s A/B, 60k, 2 kör (base / a0 / v2 / v4), 250 W:**

| futás | tg r1 | tg r2 | ms/lépés r1 → r2 |
|---|---|---|---|
| base | 45,06 | **36,99** | 62,6 → 76,2 |
| a0 | 45,02 | **37,39** | 63,1 → 76,0 |
| v2 | 38,06 | 37,64 | 75,7 → 76,5 |
| v4 | 36,34 | 36,58 | 77,6 → 77,1 |

**Ugyanaz a modell (base) 45,06 → 36,99 (−18%) a két kör között**, miközben
a hőmérséklet (83→84 °C), az SM-órajel (1561→1575 MHz) és a teljesítmény
(185→179 W) azonos. A 60k-s futás ~140 s folyamatos terhelés; a 2080 Ti
**88 °C-os thermál limitje** ekkor már aktív, a kártya throttol. A „gyors"
sáv (45 t/s) csak az első ~4–5 percben tart. A stabil sávban a sorrend
v2 37,64 > a0 37,39 > base 36,99 > v4 36,58 = **±1,5%, azaz nincs
modellhatás**. A 60k-s A/B tehát **nem alkalmas** a kvant-mix eldöntésére.

**Rövid kontextusú, stabil A/B (2,8k prompt, 3 kör, MTP2, 250 W):**

| modell | tg medián | **ms/lépés** | acc |
|---|---|---|---|
| base | 53,98 | **49,3** | 0,833 |
| a0 | 51,91 | 50,5 | 0,815 |
| v1 | 58,24 | 47,9 | 0,898 |
| v2 | 55,43 | 50,9 | 0,911 |
| v3 | 54,80 | 51,6 | 0,915 |
| v4 | 53,44 | 51,1 | 0,863 |

A t/s sorrend itt is **acceptance-artefakt** (0,815–0,915: modellenként más
szöveget generál). A lépésköltség (ms/lépés) alapján a kvantált változatok
**−3…+5%**, a futásszóráson (±2–4%) belül → **az MTP-útban nincs nyereség**,
sem rövid, sem hosszú kontextuson.

**Miért:** a base MTP-lépés 7,22 GB-ot mozgat 49,3 ms alatt = **146 GB/s**,
miközben az MTP nélküli út ugyanazon a kártyán 345 GB/s (54,0 t/s → 18,5 ms,
6,39 GB). Az MTP-út a sávszélesség **~42%-át** használja, tehát
overhead-kötött (draft gráf + 2 submit + szinkron), nem byte-kötött. Egy
byte-csökkentés ezért legfeljebb a maradék 58%-on hatna — a mért hatás a
zajban marad. **Ez egyben a legnagyobb nyitott lehetőség: ~+40% elvi
decode, ha az MTP-út overheadje csökken.**

**Döntés:** a napi config marad **UD-Q4_K_XL + MTP2**. A kvant-mix csak
MTP nélkül nyerő (+12…+21%), illetve VRAM-célra (v4: 5,21 vs 6,14 GB,
+0,9 GB szabad hely) — a minőségi kaput (perplexity) csak akkor kell
lefuttatni, ha ilyen célra bevezetjük.

**Módszertani tanulságok (a kör mellékterméke):**

- **A 170 W-os limit nincs bekapcsolva** — a kártya 250 W-on fut (a
  `nvidia-powerlimit.service` nem aktív). Ezért ebben a körben minden
  250 W-on készült; rövid kontextuson ez +17% a korábbi 170 W-os
  értékekhez képest (base MTP2: 54,0 vs 46,35 t/s). A régi és az új
  abszolút számok **nem összevethetők**.
- **60k fölött a futás thermálisan throttol** (88 °C limit). Hosszú
  kontextusú A/B-hez vagy előmelegített steady state kell (2–3 eldobott
  futás), vagy rövid kontextuson kell mérni.
- A harness most **`clocks.mem`-et is mintáz** (a teljes körben 6800 MHz,
  tehát a throttling nem memóriaóra-csökkenés, hanem SM/power oldali).
- ~~Az `LLAMA_GRAPH_TIMING` csak az újonnan **épített** gráfokra ír
  (`built=1`)~~ — **téves (38. kör):** a számláló közös volt a két kontextus
  között és minden 16. hívást írt ki; a 4 gráf/lépés ciklussal aliasolt, így
  mindig az újraépülő catch-up gráfot mintázta. Javítva.

---

## CTranslate2 — lezárt irány (összegzés, 2026-10-26)

**Mi volt:** a `plan-ctranslate2.md` (rev. 2, archív) a CTranslate2
sebességforrásait bontotta ki, és work package-eket származtatott belőlük.
A CTranslate2 **nincs és nem is volt beépítve** a llama.cpp-be: nem függőség,
nem backend, nem kód — csak kutatási forrás (`~/git/CTranslate2` checkout +
a rev. 2 elemzés). Két kísérlet származott belőle, mindkettő **mérve**:

| WP | CTranslate2-elem | Mérés | Verdikt |
|---|---|---|---|
| C0 | AWQ int4 GEMV: 16 B-os (`float4`) súlybetöltés + split-K | 4 B **540 GB/s**; 8 B −6,8%; 16 B −16,5% (21. kör) | **elvetve** → a 2.2 (160 B-os q4_K repack) nem indokolt |
| D | split-KV a CPU FA-ban kvantált KV-ra | +70% lassabb GQA alatt (23. kör) | **elvetve**, csak env-gated kísérlet |

**Ami megmaradt:** `GGML_CPU_FA_SPLIT_QUANT=1` env-kapcsoló
(`ggml/src/ggml-cpu/ops.cpp:9586`, default **off**) — kísérleti eszköz, a napi
úton nem fut.

**Amit a saját mérések azóta felülírtak:** a rev. 2 premisszája az 1,7×
GEMV-headroom volt (231 GB/s a 408-as plafonhoz). Ma: no-MTP **345 GB/s**
(a GEMV-mikrobench 540), az MTP-lépés **146 GB/s** (36. kör) → a szűk
keresztmetszet nem a betöltésszélesség, hanem az MTP-út overheadje (draft gráf
+ 2 submit + szinkron). A 16 B-os irány ezzel tárgytalan; a nyitott kérdés
átkerült az MTP-overheadre.

**Ami hasznos maradt:** a rev. 2 „mi már megvan" listája (split-KV FA mindhárom
backenden, fuzionált QKV, caching allocator, CUDA graph) — ezek upstream elemek,
nem a CTranslate2-ből kerültek be; a CT2 csak magyarázatot adott arra, miért
gyors a konkurens implementáció.

---

## 37. kör — int8 (Q8_0) vs Q4_K: a CTranslate2-féle int8 nem átvihető (2026-10-26)

**Kérdés:** a CTranslate2 int8-a (per-sor szimmetrikus int8 + cuBLAS IMMA GEMM)
átvihető-e? Először kód nélkül: a repóban van kész `Qwen3.5-9B-Q8_0.gguf`.

**Mérés** (`llama-bench`, Vulkan, 170 W, `-p 4096 -n 64 -r 3`, f16 KV, `-ub 512`):

| modell | méret | pp4096 | tg64 |
|---|---|---|---|
| UD-Q4_K_XL | 5,70 GiB | 2040,4 ± 9,4 | **60,87** ± 0,32 |
| Q8_0 | 9,10 GiB | 2077,7 ± 7,5 | **41,30** ± 0,41 |

- **Prefill +1,8%**, **decode −32,2%**. Az int8 súly a meglévő úton (Vulkan
  integer-dot / CUDA dp4a) nem hoz számottevő prefill-nyereséget, a decode-ot
  pedig a byte-többlet (8,5 vs 4,5 bpw) viszi el.

**Miért más a CTranslate2-nél** (forrás: `~/git/CTranslate2`,
`docs/quantization.md`, `src/cuda/primitives.cu`):

| szempont | CTranslate2 int8 | llama.cpp Q8_0 |
|---|---|---|
| skála | 1 f32 / **sor** (`127/max(abs(W[i,:]))`) | 1 f16 / **32 súly** (128× finomabb) |
| layout | sűrű `int8_t` mátrix, BLAS-kész | GGUF blokk, skálák beágyazva |
| számítás | `cublasGemmEx` `CUDA_R_8I` → `CUDA_R_32I` = **IMMA** | kézi MMVQ/MMQ, **dp4a** (IMMA csak sm_80+) |
| lefedettség | csak embedding + lineáris rétegek | minden kvantálható tenzor |
| workload | **batch-elt** (nagy M) → számítás-kötött | 1–3 token → **memória-kötött** |

**Következtetés:** a CT2 int8 előnye nem a „8 bitből", hanem a **vendor IMMA
GEMM-ből és a batch-elt workload-ból** jön. A mi decode-unknál az int8
strukturálisan veszteséges (2× byte). A prefill viszont számítás-kötött
(~37 TFLOP/s ≈ a 2080 Ti fp16 tensor csúcs ~70%-a), ezért **elméletileg** egy
IMMA-út akár ~2× pp-t adhatna — de:
- **Vulkanon Turingon nincs int8 tensor core** (a `VK_KHR_cooperative_matrix`
  Ampere+, és nincs vendor BLAS) → ott nincs mit implementálni a dp4a fölött;
- **CUDA-n** van sm_75-ös int8 IMMA helper (`mma.cuh`, `m8n8k16.s8`), de a
  MMQ konfigok Ampere-től használják → új MMQ-konfig + helyesség-ellenőrzés
  kellene, és csak a pp-t gyorsítaná (a teljes falidő ~9%-a 41k-n).

**Döntés:** int8 implementálás nem indokolt (Vulkanon nem lehet, CUDA-n csak a
pp ~9%-a, nagy kockázattal). A meglévő Q4_K marad.

---

## 38. kör — MTP-lépés felbontás, előzetes (G0, 2026-10-26 este)

**Javított `LLAMA_GRAPH_TIMING`** (kontextusonként, `(n_tokens, n_outputs)`
szerint aggregálva), füstteszt: Vulkan, MTP2, f16 KV, `-bs`, `-c 8192`,
8 tokenes prompt, 300 token, temp 0, **170 W**. Egyetlen futás → előzetes.

| kontextus | gráf | n | újraépítve | build | submit (ms) |
|---|---|---|---|---|---|
| main | verify, 3 token / 3 kimenet | 64 | 1 | 0,23 | **14,95** |
| mtp | draft, 1 token / 1 kimenet | 42 | **22** | 0,33 | 1,71 |
| mtp | catch-up, 3 token / **0 kimenet** | 22 | **22** | 0,73 | 1,61 |

- sched sync: main ~0,8–1,1 ms, mtp ~0,25 ms.
- Szerver: 52,6 t/s, acceptance 0,673, mean len 2,35 → 127 lépés,
  **~44,8 ms/lépés**.
- Gráfokban töltött idő ≈ 15,2 + 2×2,1 + 2,4 + ~1,5 (sync) ≈ **23 ms/lépés**
  → **~22 ms/lépés a `process_ubatch`-en kívül** (szerver-ciklus, sampling,
  embedding-readback, `yield_to_queue`).
- A catch-up gráf **minden** lépésben újraépül, az 1. draft-lépés is (slot-váltás
  a 0 ↔ 1 kimenet miatt) — a plan-part3 hipotézise igazolva.

~~**Következtetés:** a legnagyobb tétel nem a GPU, hanem a gráfokon kívüli
~22 ms.~~ **Téves, lásd 39. kör:** a Vulkan `graph_compute` nem blokkol végig;
a „submit” csak a parancsrögzítés, a GPU a maradékot a szerver
`llama_synchronize` hívásában fejezi be. A rés GPU-idő volt.

---

## 39. kör — a „22 ms” gyökéroka: a verify n=3 k-quant matvec (2026-10-26 este)

**Új instrumentáció (env-gated, default off):**
- `LLAMA_SERVER_TIMING=1` (`server-context.cpp`): lépésenkénti bontás 64
  lépés átlagával — gap, pre_decode, draft(+work), tgt_decode(+work),
  spec_process(+work), post_decode, accept, step.
- `LLAMA_GRAPH_TIMING=1` most a `llama_context::decode` fázisait is bontja
  (reserve, mem_update, init_batch, out_reserve, ubatch, extract, apply).

**Lépés-bontás (Vulkan, MTP2, rövid ctx, 170 W):** step 44,5 ms =
tgt_work **32,4** + draft_work 8,0 + spec_work 3,0 + kb. 1 ms yield/egyéb.
A szerver maga elhanyagolható (gap 0,02, yield ~0,3/fázis, sampling 0,1 ms).
A target `decode()` CPU-oldala 15,9 ms (szinte mind a `graph_compute`
parancsrögzítése), a maradék ~16 ms a `llama_synchronize`-ban a GPU-ra vár.
→ **A verify-gráf GPU-ideje ~30,6 ms** (`GGML_VK_PERF_LOGGER`, 59 gráf átlaga).

**Op-szint:** ~26 ms a súly-matvec. Ugyanaz a mátrix (`q4_K 12288×4096`):
n=1 (draft) 65 µs (~435 GB/s), **n=3 (verify) 168 µs (~165 GB/s)**. Ok: a
float k-quant shader workgroupja `rm_kq=2` sort számol, és oszloponként
újraolvassa az aktivációs vektort → n=3-nál ~300 MB L1/L2-forgalom a 28 MB
súly mellett. Az MMVQ (egész-pontszorzat) út, ami n>1-re az alapértelmezés,
ugyanilyen lassú (167–171 µs).

| q4_K 12288×4096, n=3 | µs |
|---|---|
| alap (MMVQ) | 168 |
| float, rm=2 (`GGML_VK_DISABLE_MMVQ`) | 156 |
| float, rm=4 | 116–122 |
| float, rm=8 | 112 |

A `GGML_VK_DISABLE_MMVQ` globálisan nem jó: a q8_0 n=3-at rontja (56 → 101 µs).

**Új kapcsoló: `GGML_VK_KQ_BATCH_RM=N`** (`ggml-vulkan.cpp`, default off):
a q4_K/q5_K/q6_K float mat-vec pipeline-ok n≥2-re N sort kapnak (q6_K
legfeljebb 4, mert 8-nál romlik: fej 1,67 → 5,16 ms), és n>1-nél a
q4_K/q5_K a float-útra megy MMVQ helyett. A q8_0 marad MMVQ-n.

**Mérés** (interleaved, 2 kör × 3 config, 600 token, 30 s cooldown):

| config | tgt_work | lépés | t/s |
|---|---|---|---|
| alap | 34,2–35,0 ms | 48,5–49,1 ms | 47,1–48,7 |
| **RM=4** | **29,1–29,6 ms** | **43,1–44,4 ms** | **52,8–53,6** |
| RM=8 (q6 ≤4) | 28,4–31,0 ms | 42,3–48,8 ms | 48,7–55,6 |

(A 2. sorozat első köre mindhárom confignál lassabb volt — drift.)
→ **RM=4: verify −14%, lépés −9…−10%, t/s +10%**. RM=8 a zajon belül
azonos, ezért a 4 az ajánlott.

**Kimenet:** konfigurációnként determinisztikus (ismétlésenként bitre azonos),
az alaptól a 2532 karakterből az 1499.-nél tér el (lebegőpontos összegzési
sorrend, greedy dekódolásnál várható); acceptance 345 vs 349.

**Maradék a lépésben (RM=4):** draft_work ~8 ms, ebből GPU ~4,6 ms
(2 draft-gráf, a fej 1,55 ms/lépés, ~538 GB/s — sávszélesség-kötött);
spec_work ~3 ms (GPU ~1,3 ms). A különbség CPU-oldali (gráf-újraépítés,
sampling, szinkronok) → ez a G1 célpontja.
**Nyitott anomália:** az MTP-kontextusban `q4_K m=8192 n=1 k=4096` 235 µs
(~80 GB/s), miközben a törzsben ugyanez n=3-mal 97–118 µs (~0,4 ms/lépés).

**Teendő:** 41k kontextuson, ≥2000 tokennel megerősíteni; ha ott is nyer,
`GGML_VK_KQ_BATCH_RM=4` kerüljön a napi parancsba. CUDA-n külön kell
megnézni, hogy az MMVQ n=3 hasonlóan rossz-e.

### 39b — igazolás hosszú kontextuson (2026-10-26 este)

`long-60k.txt` első 129k karaktere + „foglald össze részletesen” kérés →
**45 253 tokenes prompt**, 2000 generált token, Vulkan + MTP2 + f16 KV + `-bs`,
`-t 24 -tb 24 --poll 0`, 170 W, interleaved (alap, RM=4, alap, RM=4), 60 s hűtés,
max. 79 °C (nincs throttling). Script: `ab41k.sh` (session scratchpad).

| futás | pp t/s | tg t/s | acc | mean len | ms/lépés | tgt_work | draft_work | spec_work |
|---|---|---|---|---|---|---|---|---|
| alap #1 | 998,5 | 49,70 | 0,929 | 2,86 | 57,4 | 41,7 | 9,0–9,5 | 3,3–3,5 |
| RM=4 #1 | 1011,6 | **54,06** | 0,939 | 2,88 | **53,1** | **37,5** | 9,1–9,3 | 3,3–3,4 |
| alap #2 | 991,8 | 49,06 | 0,929 | 2,86 | 58,1 | 42,3 | 9,6 | 3,5–3,6 |
| RM=4 #2 | 998,8 | **53,80** | 0,939 | 2,88 | **53,5** | **37,0** | 9,3–9,8 | 3,3–3,5 |

- **ms/lépés −7,6%** (57,8 → 53,3), verify −11,9% (42,0 → 37,0), tg **+9,2%**
  (ebből ~1 pont az eltérő szövegből adódó kicsit jobb acceptance).
- pp változatlan (+0,7%, zaj) — a prefill nagy batch-e nem érintett.
- Kimenet ismétlésenként bitre azonos (alap: 5684, RM=4: 5652 karakter).
- A draft (~9,3 ms) és a catch-up (~3,4 ms) nem változik → G1 a következő.

**Döntés:** a Vulkan napi parancs kiegészül a `GGML_VK_KQ_BATCH_RM=4`
env-vel:

```
GGML_VK_KQ_BATCH_RM=4 build-vulkan/bin/llama-server --port 8080 \
  -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-backend-sampling -bs \
  -t 24 -tb 24 --poll 0
```

---

## 40. kör — CUDA vs Vulkan 45k-n + javítások mindkét oldalon (2026-10-26/27 éjjel)

Protokoll: ugyanaz a 45 253 tokenes prompt + 2000 token, MTP2, f16 KV, `-bs`,
`-t 24 -tb 24 --poll 0`, 170 W, interleaved, 60 s hűtés (`ab45k.sh`).

### CUDA vs Vulkan (javítások előtt, Vulkan már RM=4-gyel)

| futás | pp t/s | tg t/s | lépés | verify | draft | catch-up |
|---|---|---|---|---|---|---|
| CUDA #1 (zavart) | 978 | 46,1 | 61,5 | 50,0 | 7,1 | 2,2 |
| CUDA #2 | 1404 | 96,4 | 29,9 | 22,4 | 6,1 | 0,6 |
| CUDA #3 / #4 | 1327 / 1265 | 87,8 / 87,8 | 34,1 / 33,0 | 24,1 / 24,5 | 7,2 / 6,7 | 1,4 / 0,9 |
| Vulkan RM=4 #1 / #2 | 983 / 986 | 53,4 / 54,7 | 53,4 / 54,2 | 38,1 / 37,9 | 9,0 / 9,6 | 3,2 / 3,5 |

- **A CUDA 45k-n ~1,6–1,8× gyorsabb a Vulkannál** (tg és pp is). A 35. kör
  „egy szint” megállapítása valószínűleg egy zavart CUDA-futáson alapult.
- **Zavarforrás:** a `gnome-remote-desktop-daemon` CUDA-kontextust tart
  (C+G, 289 MiB). Aktív távoli asztal mellett a CUDA-futás a felére eshet
  (#1), a Vulkan ezt kevésbé érzi. Mérés alatt ne legyen aktív távoli munka.

### Op-szint (`test-backend-ops perf`, a modell alakjaival, µs)

| op | CUDA | Vulkan |
|---|---|---|
| FA 45k KV, head 256, GQA 4, batch 1 | 367 | 465 |
| **FA 45k KV, batch 3** | **380** | **1262** |
| q4_K 12288×4096 n=1 / n=3 | 75 / 100 | 81 / 140 (RM=4) |
| q6_K fej 248320×4096 n=1 / n=3 | 2020 / 2498 | 1723 / 2561 |

**Vulkan FA batch 3 gyökérok:** a GQA-ágon (`ggml_vk_flash_attn`, `N <= 8`)
minden token külön workgroup-csoportot kap (`workgroups_x = neq1`), mindegyik
a teljes K/V-t olvassa → 3 token = 3×185 MB. 8 attention-rétegen ez
~7 ms/verify. A „split-major” workgroup-sorrend (L2-megosztás reményében)
nem hozott semmit (1245 vs 1262 µs) → visszaállítva. A valódi javítás a
token×fej sorok egy csempébe vonása (3D Q/maszk/kimenet-címzés a scalar, cm1
és cm2 FA shaderben) — nyitott shader-projekt (plan-part3 G6).

### CUDA: MMVQ rows_per_block 2 → 4 Turingon (ncols_dst 2–4) — ALAPÉRTELMEZETT

Ugyanaz a mechanizmus, mint Vulkanon (az y-vektor újraolvasása). Op-szint,
n=3: q4_K 100 → 84 µs (−17%), q4_K 4096×12288 101 → 79 (−22%), q5_K −16%,
q6_K −20%, fej 2498 → 1991 (−20%); rpb=8 összességében rosszabb.
`test-backend-ops -o MUL_MAT`: OK. 45k A/B (interleaved, 2+2):

| | lépés | verify | tg t/s | pp t/s |
|---|---|---|---|---|
| rpb=2 | 29,5 / 29,5 | 22,0 | 97,7 / 97,3 | 1449 / 1380 |
| **rpb=4** | **27,0 / 26,8** | **19,3** | **105,8 / 107,2** | 1439 / 1439 |

→ **lépés −8,8%, verify −12%, tg +9,3%**, pp változatlan, a kimenet mind a
4 futásban bitre azonos (acceptance is). Kód: `mmvq.cu`
`GGML_CUDA_MMVQ_TURING_RPB_SMALL` (default 4, `-D...=2` a régi).

### G1: catch-up beolvasztása a draftba (`LLAMA_MTP_DEFER_CATCHUP=1`, default off)

- Megvalósítva (`common/speculative.cpp`). Közben **meglévő hiba** került elő:
  a qwen35 MTP-gráf a `t_h_nextn`-t a `get_rows(inp_out_ids)` *előtt* rögzítette,
  így maszkolt módban kevert (kimenetes + kimenet nélküli) batchnél rossz sor
  h-ját adta vissza → acceptance 0,68 → 0,52. Javítva (`src/models/qwen35.cpp`:
  maszkolt módban a crop után rögzít, mint a törzs). Más modellek MTP-gráfjai
  (deepseek2, glm4-moe, …) ugyanezt a mintát követhetik — a G1 csak qwen35-ön
  tesztelt.
- Javítás után: kimenet bitre azonos, acceptance azonos; rövid ctx-en
  Vulkan 38,1 → 37,8 ms/lépés, CUDA 26,6 → 26,3 (~1%). A catch-up munkája
  átkerül az első draft-lépésbe; az indítási overhead kicsi → **nem éri meg
  bekapcsolni**, kísérleti kapcsolóként marad.

### Egyéb

- G2 (mohó draft-sampler) elvetve mérés alapján: a draft `TOP_K` Vulkanon
  ~26 µs/lépés, nincs mit nyerni.
- `build-vulkan` glslc-útvonala a törölt `/tmp/opencode/vulkansdk`-ra mutatott
  (shader-módosítás nem fordult) → átállítva:
  `-DVulkan_GLSLC_EXECUTABLE=~/vulkan-sdk/1.4.363.0/x86_64/bin/glslc`.
- `tests/test-backend-ops.cpp`: a perf-esetek közé felvéve a modell alakjai
  (FA 45k batch 1/3, a fő matvec-ek n=1/3).
- A `build-cuda` és `build-vulkan` most `LLAMA_BUILD_TESTS=ON`.

### Napi parancs (2026-10-27-től): CUDA

45k-n a CUDA ~2× gyorsabb (27 vs 53 ms/lépés), ezért a napi sofőr a CUDA:

```
build-cuda/bin/llama-server --port 8080 \
  -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-backend-sampling -bs \
  -t 24 -tb 24 --poll 0
```

Vulkan tartalék: ugyanez `GGML_VK_KQ_BATCH_RM=4` env-vel a `build-vulkan`-ból.
