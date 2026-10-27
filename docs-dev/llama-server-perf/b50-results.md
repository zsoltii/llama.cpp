# Intel Arc Pro B50 (Vulkan2) — llama-server gyorsítás, Qwen3.5-9B-MTP

## Környezet

- GPU: Intel Arc Pro B50 (BMG G21, Xe2), 16 GB, Mesa ANV + xe kernel driver (Fedora 44).
- **Thunderbolt eGPU**: a B50 egy TB-dokkon ül (PCIe 3.0 x4 a dokk felé), **nincs ReBAR**,
  a host-visible VRAM-ablak 256 MiB (a heap budgetje induláskor ~2 MiB szabad).
- CPU: i7-11850H (16 szál), 31 GB RAM. Más GPU-k: Intel UHD (Vulkan0), NVIDIA T1200 (Vulkan1).
- Referencia: brew `llama-server` 0.7.0 (build 11669, Vulkan).
- Modell: `unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL` (+ mmproj), 128k ctx, q8_0 KV, FA on.
- Coopmat-alakok a B50-en: csak **M=8 N=16 K=16** (f16/bf16), int8: 8x16x32.

## Mérési protokoll

- `~/llama-bench-results/bench.py`: szerver-újraindítás konfigonként, warm-up, temp=0,
  `cache_prompt=false`, thinking off, medián; kimenet-hash (determinizmus); 1 s-es monitor
  (GPU busy % az xe fdinfo ciklusaiból, GPU MHz, VRAM, folyamat- és rendszer-CPU).
- Korpusz (`~/llama-bench-results/corpus/`): `story` = rövid magyar mesekérés, 1024 token
  generálás; `wiki` = Római Birodalom Wikipédia-cikk (99,7 KB, 20 726 token) + elemzés, 512 token.
- Kernel-szint: `~/llama-bench-results/kbench.py` és `test-backend-ops perf` (Qwen3.5 alakzatok).

## Referencia (brew)

| konfig | story pp | story tg | wiki pp | wiki tg | elfogadás (story/wiki) |
|---|---|---|---|---|---|
| MTP nélkül | 306 | 25,6 | 423 | 18,3 | – |
| MTP n=1 | 238 | **27,6** | 376 | **19,6** | 0,58 / 0,73 |
| MTP n=2 | 234 | 27,6 | 376 | 16,9 | 0,47 / 0,55 |
| MTP n=3 | 234 | 24,5 | 377 | 16,8 | 0,34 / 0,46 |
| MTP n=4 | 213 | 19,9 | 375 | 11,5 | 0,25 / 0,39 |

Mind `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` mellett: enélkül MTP-vel **2,5 tok/s** (lásd lent).
Referencia-parancs: a felhasználói parancs + `--spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0`.

## Megállapítások és javítások

1. **Small BAR → compute bufferek rendszermemóriában.** A backend a device-local+host-visible
   típust kérte; az ablak betelt, a driver a többit sysmem-be tette, a GPU TB-n át olvasta
   (matvec 2–5 GFLOPS). Javítás: `ggml-vulkan.cpp` — ha a host-visible device-local heap
   < VRAM/2, csak device-local memória (`GGML_VK_ALLOW_SMALL_BAR_VIDMEM` kikapcsolja). 10× MTP-vel.
2. **Deferred MTP catch-up** (`LLAMA_MTP_DEFER_CATCHUP=1`, meglévő): -1 decode/lépés, ~+5%.
3. **Backend sampling** (`-bs`): nincs 2 MB logit-visszaolvasás TB-n, post_decode 3,5 → 0,75 ms,
   ~+9% tg, kimenet bitre azonos. Rövid prompt pp-je csökken (238 → 167) — vizsgálandó.
4. **FA split-k** (decode/verify): a régi heurisztika ~2 WG/mag → a B50-en 32 workgroup, latency-
   bound (q8_0 KV 20k: 24 GB/s). Új, gyártófüggetlen: 16 WG/mag, de split >= 256 kulcs
   (`GGML_VK_FA_WG_PER_CORE`, `GGML_VK_FA_MIN_KV_SPLIT`). FA nb=1/2/3: 2,1× / 2,3× / 3,1×.
5. **FA coopmat1 8x16x16-tal**: a shader eddig csak 16x16x16-ot ismert → Intel Xe2-n a prefill FA
   skalár úton futott. A shader most a `FaCmM` spec-constanttal az M méretet az eszközből veszi
   (16 = változatlan NVIDIA/AMD kód). Prefill FA (512 × 8,7k): f16 65 → 21,6 ms, q8_0 80 → 29 ms.
   `test-backend-ops -o FLASH_ATTN_EXT`: 5368/5368 OK.
6. Diagnosztika: `GGML_VK_FA_DEBUG=1` (FA út/split, coopmat-alakok); a perf-logger a GET_ROWS/ARGSORT/TOP_K
   alakját is kiírja.
7. **`-bs` + rövid prompt (pp 238 → 175).** Minden kérés `llama_set_sampler`-t hív (reset: nullptr, új task: új
   lánc) → teljes `sched_reserve()` (63 ms, 3 graph_reserve + új scheduler, a gráf-újrahasznosítás is elvész).
   Javítás (`src/llama-context.cpp`): a sampler-csere csak ellenőrzést kér; újrafoglalás csak akkor, ha a láncok
   szerkezete (seq → sampler-nevek) eltér a legutóbb foglalttól. Az új sampler-pointer miatt a gráf úgyis újraépül
   (a `can_reuse` pointerre hasonlít).
8. **Prompt-checkpoint (recurrent állapot, 50 MiB) 106 ms → 39 ms.** Ebből 64 ms egy friss 50 MiB-os `std::vector`
   page-faultja volt. Javítás (`common/common.cpp`): a checkpoint-pufferek egy (max. 512 MiB-os) poolba kerülnek
   vissza, `update_tgt/dft` onnan vesz. A maradék 39 ms a 48 db TB-s visszaolvasás.
   7+8 együtt: story pp **175 → 294 tok/s** (ref 238).
9. **Gráf-bemenetek feltöltése (scheduler).** Minden user-bemenet (tokenek, pozíciók, maszk, idx-ek, s_copy …) és a
   CPU-split kimenete (token-embedding) külön, blokkoló feltöltés volt: draft 9 db = 1,7 ms, verify 12 db = 2,7 ms
   lépésenként, közben a GPU áll. Javítás (`ggml/src/ggml-backend.cpp`, backend-független): splitenként egyetlen
   szinkronizálás, a bemenetek a backend pinned host-pufferébe másolódnak és `set_tensor_async`-kal mennek fel
   (Vulkan: a parancsfolyamba kerülő copy, CUDA: aszinkron pinned memcpy). 2,6–2,8 ms → 0,3–0,4 ms lépésenként.
   Kikapcsolható: `GGML_SCHED_SYNC_INPUTS=1`.
10. Nem vált be: Br=32 a coopmat1 FA-ban (több query/workgroup a K/V-újrahasznosításért) — Xe2-n lassabb
    (f16 21 → 38 ms, occupancy); n=2/3 draft (a draft LM-head lépésenként 4,3 ms).

## KV-típusok — FA kernel (µs, B50, a 4. pont után, az 5. előtt)

| típus | decode nb=1 | nb=2 | nb=3 | prefill 512@8,7k |
|---|---|---|---|---|
| f16 | 584 | 1280 | 1839 | 64 995 |
| q8_0 | 978 | 1533 | 2301 | 80 457 |
| q5_1 | 642 | 1230 | 1824 | 80 761 |
| q5_0 | 946 | 1637 | 2295 | 96 656 |
| q4_1 | **538** | **1015** | **1499** | 69 181 |
| q4_0 | 733 | 1193 | 1736 | 76 791 |
| iq4_nl | 767 | 1405 | 2230 | 94 772 |

Hardveres út: q4_0/q4_1/q5_0/q5_1/q8_0 K → int8 dot (DP4A) decode-ban; prefill: XMX coopmat
(5. pont után). iq4_nl: LUT, nincs gyorsítás. q6 KV-típus nincs a llama.cpp-ben.

## Saját build — eddigi legjobb (MTP n=1, q8_0 KV, defer + `-bs`, 4. pontig)

| | ref | saját | |
|---|---|---|---|
| story tg | 27,6 | 32,9 | +19% |
| wiki tg | 19,6 | 25,5 | +30% |
| wiki pp | 376 | 386 | +3% (az 5. pont előtt) |

## Saját build v3 (1–5. pont, MTP n=1, defer + `-bs`) — KV-típus rács (szerver)

| KV (target+draft) | story pp | story tg | wiki pp | wiki tg | elfogadás story/wiki | VRAM csúcs |
|---|---|---|---|---|---|---|
| q8_0 | 175 | 32,3 | **604** | 26,7 | 0,59 / 0,72 | 9,5 GB |
| f16 | 171 | 33,1 | **605** | **31,5** | 0,58 / 0,76 | 11,5 GB |
| q4_1 | 181 | 32,8 | 579 | 27,8 | 0,62 / 0,74 | 8,4 GB |
| q4_0 | 166 | 33,0 | 568 | 28,3 | 0,63 / 0,78 | 8,3 GB |
| q5_1 | 170 | 32,7 | 542 | 27,6 | 0,62 / 0,77 | 8,7 GB |
| iq4_nl | 177 | 33,4 | 553 | 27,8 | 0,65 / 0,78 | 8,3 GB |

Minden futás determinisztikus (2/2 azonos hash). GPU-terhelés: story ~82%, wiki ~91%.
A KV-típus a sebességben kicsi különbség (tg: a q4-ek ~+5% a q8_0-hoz képest, pp: −4…−10%);
**az f16 a leggyorsabb hosszú kontextusban** (wiki tg +18% a q8_0-hoz), +2 GB VRAM árán.
A minőségi ár (KLD) még mérendő.

## Saját build v4 (1–9. pont, q8_0 KV, MTP n=1, defer + `-bs`)

| | ref (brew) | v3 | **v4** | v4 vs ref |
|---|---|---|---|---|
| story pp | 238 | 175 | **294** | +24% |
| story tg | 27,6 | 32,3 | **33,5** | +21% |
| wiki pp | 376 | 604 | **598** | +59% |
| wiki tg | 19,6 | 26,7 | **27,2** | +39% |

Kimenet bitre azonos a v3-mal (story és wiki hash). GPU-terhelés: story 82 → 89%, wiki 93%.
Egy MTP-lépés (story): draft 8,2 ms + verify 38,1 ms + egyéb ~1,3 ms = 47,6 ms, 1,6 token/lépés.
MTP n=2: 31,7 / 26,9, n=3: 28,8 / 23,8 (story / wiki tg) → n=1 marad.

## Pontosság 1. szint — KL-divergencia (wikitext-2, 6×4096 token)

Referencia: brew, f16 KV (PPL 6,6991 ± 0,151). Minden futás ugyanazon a 24k tokenen, `-fa on`, B50
(`~/llama-bench-results/eval/kld.sh`).

| build / KV | PPL | Mean KLD | 99,9% KLD | Max KLD | RMS Δp | azonos top-1 |
|---|---|---|---|---|---|---|
| brew q8_0 | 6,6972 | 0,00040 | 0,0113 | 0,52 | 0,62% | 99,03% |
| **saját q8_0** | 6,6911 | 0,00044 | 0,0189 | 0,26 | 0,67% | 98,67% |
| saját f16 | 6,6928 | 0,00042 | 0,0155 | 0,37 | 0,63% | 98,93% |
| saját q5_1 | 6,7028 | 0,00329 | 0,0845 | 12,8 | 1,58% | 98,21% |
| saját q4_0 | 6,7102 | 0,00510 | 0,397 | 6,9 | 1,96% | 96,92% |
| saját q4_1 | 6,7158 | 0,00510 | 0,266 | 15,1 | 1,97% | 97,06% |
| saját iq4_nl | 6,7094 | 0,00584 | 0,260 | 17,9 | 2,00% | 97,34% |

- A saját build (XMX coopmat FA, split-k, bemenet-staging) a brew-val **azonos pontosságú**: a saját f16 eltérése a brew
  f16-tól (KLD 0,0004) a lebegőpontos összegzési sorrend különbsége, akkora, mint a q8_0 KV teljes ára.
- KV-kvantálás: q8_0 gyakorlatilag veszteségmentes; q5_1 ~8×, a 4 bites típusok ~12–14× nagyobb KLD, és ritka, de
  nagy egyedi eltérések (max KLD 7–18) — hosszú generálásnál ezek terelhetik el a kimenetet. Ajánlás: q8_0 (vagy
  f16, ha a VRAM elég; ld. v3-tábla: hosszú kontextusban f16 a leggyorsabb).

## Pontosság 2. szint — feleletválasztós (llama-perplexity, log-likelihood)

Azonos (rögzített seedű) feladatkészlet minden futásnál (`~/llama-bench-results/eval/mc.sh`). A szerver ezt nem tudja
mérni (nem ad prompt-token logprobot); a számítási út (FA, matmul, KV-kvantálás) ugyanaz a ggml/Vulkan kód.
Megj.: a `llama-perplexity` n_ctx × n_vocab float puffert foglal (248k szótár → `-c 2048 -np 16` = 32 GB, OOM), ezért
a 4-nél több válaszos feladatok (TruthfulQA, ARC) `-c 256 -np 16`-tal futnak.

| | HellaSwag (400) | Winogrande (600) | TruthfulQA mc1 (300) | ARC-C (299) | MMLU (300) |
|---|---|---|---|---|---|
| brew q8_0 | 78,0 | 72,0 | 32,3 | 50,5 | 42,7 |
| **saját q8_0** | 78,0 | 72,3 | 32,3 | 50,5 | 43,0 |
| saját q4_1 | 78,5 | 72,2 | 31,7 | 50,2 | 43,7 |

Stat. hiba ±1,8–2,9 pont → nincs mérhető különbség.

## Pontosság 3. szint — generatív, a valódi szerver-konfiggal (MTP n=1, q8_0 KV, 128k, thinking off)

`~/llama-bench-results/eval/server_eval.sh` (lm-eval 0.4.13, `llama-eval` conda env, local-chat-completions, temp 0)
+ `wiki_qa.py`: a teljes Római Birodalom cikk kontextusban, 25 a cikkből megválaszolható kérdés + 5 olyan, amire a
cikk nem ad választ (elvárt: NOT_IN_TEXT → hallucináció-teszt).

| | wiki QA | wiki: nem talál ki választ | NQ-open (300, 5-shot) | GSM8K (150, 5-shot) flex/strict | IFEval (150) prompt strict/loose |
|---|---|---|---|---|---|
| brew ref | 25/25 | 5/5 | 23,0 | 86,0 / 83,3 | 85,3 / 89,3 |
| **saját v4** | 25/25 | 5/5 | 22,7 | 85,3 / 83,3 | 85,3 / 88,7 |

Páros összevetés (azonos feladatok): NQ-open 288/300 szövegre azonos kimenet, 1 eredmény-flip; GSM8K 99/150 azonos,
5 flip (3 brew, 2 saját javára); IFEval 32/150 azonos (hosszú válaszok), 6 flip (3–3). A flipek szimmetrikusak → a
greedy dekódolás egy-egy közel-döntetlen tokenen eltér (KLD ~0,0004), minőségi romlás nincs. A wiki QA a saját
builden 55 s vs 96 s (gyorsabb pp + tg).
Megj.: NQ-open 0-shot chat-módban 0 pont (a modell teljes mondatban válaszol, az exact match elbukik) → 5-shot
multiturn + rövid-válasz system prompt.

## Tesztek

- `test-backend-ops -o FLASH_ATTN_EXT` (B50, v4): 5368/5368 OK.
- `test-llama-context` (`GGML_SCHED_DEBUG_REALLOC=1`, Vulkan) és `test-backend-sampler` (`get_sub_batch`) a
  **módosítatlan HEAD-en is ugyanígy bukik** → nem a változtatások okozzák.

## NVIDIA GTX 1080 Ti (Pascal, Thunderbolt, Vulkan2) — 2026-10-27

Meghajtó: 580.178.04 zárt modul (RPM Fusion `580xx`; a 615-ös ág csak nyílt modult ad, az GSP nélkül nem kezeli a
Pascalt). Eszköz: fp16: 0, matrix cores: none, int dot: 1, 48 KB shmem, BAR 246 MiB → a small-BAR felismerés
(1. pont) itt is bekapcsol.

Javítások (eredeti llama.cpp hibák, a módosítatlan HEAD-en is jelentkeznek):
11. **Skalár FA GPU-hiba (Xid 13, `SKEDCHECK18_L1_CONFIG_TOO_SMALL`) → a folyamat végtelenül vár.** NVIDIA-n a K/V
    shmem-staging mindig be van kapcsolva hsk < 256-nál; fp16 nélkül (fp32 FLOAT_TYPE) hsk=192-nél a tile 50 KB > 48 KB.
    A méretellenőrzés dekódolásnál le sem fut, és csak block_rows-t felez (a tile Bc-től függ). Javítás: ha nem fér be,
    staging kikapcsolva (globális memóriából olvas). fp16-os eszközökön változatlan.
12. **block_rows 1 → 0 felezés** (n_rows=1, n_kv ≥ 1024, nem fér be) → `wg_denoms > 0` assert. Felezés csak > 1-nél.
13. `GGML_VK_PERF_LOGGER` + bemenet-staging (9. pont): assert, mert a compute_ctx már tartalmazza az aszinkron
    feltöltéseket; az assert helyett az első timestamp a feltöltések után kerül be.

`test-backend-ops` (teljes, 1080 Ti): 21161/21161 OK.

| | story pp | story tg | wiki pp | wiki tg | elfogadás |
|---|---|---|---|---|---|
| brew, MTP nélkül | 137 | 42,3 | 275 | 32,8 | – |
| brew, MTP n=1 (ref) | 180 | 39,6 | 253 | 32,1 | 0,57 |
| **saját v4** (defer + `-bs`) | 163 | **42,7** | 256 | **38,2** | 0,60 |

GPU 97% foglalt, átlag 178 W (max 278 W), max 78 °C, 1911 MHz. A Pascal számítás-korlátos: MTP-vel a brew lassabb,
mint nélküle; a saját build wiki tg +19% a refhez. llama-bench: pp512 374, pp512@d8192 291, tg64 44,9.
pp512 bontás: 98% MUL_MAT (MMQ, q8_1 + DP4A): q4_K 6,7, q5_K/q6_K 4,3–5,0 TOPS — az int8 csúcs (~45 TOPS) töredéke.

14. **Int MMQ tile tensor core nélküli NVIDIA-n.** Az alap `l` tile 128 szál, szálanként 64–128 akkumulátor → kevés
    aktív warp. Söprés után: 512 szál, WM=WN=32, WMITER=2, TM=2, TN=1 (`l_warptile_mmq_int(_k)`, csak
    `NVIDIA && !coopmat_support`). MUL_MAT 4096×512×14336: q4_K 11,1 → 5,6 ms, q8_0 12,1 → 7,3, q5_K 14,0 → 6,6,
    q6_K 13,6 → 6,7 ms. MUL_MAT teszt 2275/2275 OK.
15. **Skalár FA prefill fp16 nélkül.** fp32 FLOAT_TYPE mellett a Br×Bc=16×32 tile regiszter-túlcsordul. `!fp16 &&
    n_rows > 8` esetén 16×16 tile, row_split 4, 128 szál (ha a shmem befér). 512 q × 8,7k kulcs: hsk=256 f16 48,5 →
    21,4 ms, q8_0 56 → 25,5; hsk=128 q8_0 39 → 28; hsk=64 q8_0 37 → 15,5 ms. Dekódolásnál (1 sor) a Bc nem számít
    (64/32/16 ±3%), változatlan. Az int8 MMQ FA-út (dequant nélkül) itt lassabb (59 vs 48 ms) → marad a dequant.
    FA teszt 5368/5368 OK.

llama-bench v5: pp512 **656** (374), pp512@d8192 **536** (291), pp512@d16384 449, tg64 45,0 / 42,2 / 38,9.

| 1080 Ti | story pp | story tg | wiki pp | wiki tg | elfogadás |
|---|---|---|---|---|---|
| brew, MTP n=1 (ref) | 180 | 39,6 | 253 | 32,1 | 0,57 / 0,73 |
| saját v4 | 163 | 42,7 | 256 | 38,2 | 0,60 / 0,73 |
| **saját v5** (14–15. pont) | **307** | **43,7** | **467** | **38,4** | 0,62 / 0,73 |
| v5 vs ref | +71% | +10% | +85% | +20% | |

(story pp: az 1. futás 104 a bemelegedés miatt, a 2. futás értéke.) GPU 95%, átlag 232 W, max 295 W, max 79 °C.
KLD (a B50-es brew f16 referenciához, 1080 Ti-n): saját v5 q8_0 0,0022, brew q8_0 0,0026 → azonos pontosság
(a ~0,002 a két eszköz eltérő számítási útja: int8 MMQ vs coopmat).

## Hátra

- FA prefill (coopmat1, Xe2): ~3,5 TFLOPS — a wiki pp kb. harmada. A Br=32 nem segít, a kernel szerkezete (kis
  tile-ok, shmem-softmax, sok barrier) a korlát; mélyebb átírás kellene.
- Draft LM head (248320×4096 q6_K, 834 MB, 4,3 ms/lépés): kisebb bitszámú másolat vagy szűkített szótár.
- Verify CPU-rögzítés ~7 ms/lépés (nagyrészt átfedésben a GPU-val).
- Pontosság: kész (1–3. szint, fent) — a saját build a brew-val azonos pontosságú.
