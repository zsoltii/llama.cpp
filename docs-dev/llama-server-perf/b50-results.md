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

## CUDA — GTX 1080 Ti (sm_61)

Toolchain: conda env `llama-cuda` (CUDA 12.9 — a 13-as már nem támogatja az sm_61-et — + GCC 14, a rendszer GCC 16-ját
az nvcc 12 nem fogadja el). Build: `build-cuda`, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;75;86"` (1080 Ti,
2080 Ti/T1200, 3090). Viszonyítás: upstream `de5122d87` CUDA-build. `test-backend-ops` (CUDA0): 19051/19051 OK.

llama-bench (q8_0 KV): upstream / saját pp512 980 / 966, pp512@d8192 778 / 764, tg64 41,0 / 41,0. 1–4 tokenes batch
(saját): CUDA pp1 40,7, pp2 76, pp3 106, pp4 127 t/s vs Vulkan 45, 69, 83, 92 → a CUDA verify-batch olcsó.

16. **A felhasználói parancs + MTP CUDA-n: a `-fit` csak 31/34 réteget tesz GPU-ra** (MTP nélkül, ill. Vulkanon 34/34).
    Ok: a CUDA FA a kvantált K/V f16 másolatát a compute bufferben foglalja (`ggml_cuda_flash_attn_ext_get_alloc_size`,
    TILE/MMA kernelhez), a teljes kontextusra: 128k-nál ~0,5 GB kontextusonként. Compute buffer CUDA vs Vulkan: main
    689 vs 229 MiB, MTP 681 vs 171 MiB. Az 1 GB-os alap fit-tartalékkal 3 réteg (súlyok + recurrent állapot) a CPU-ra
    kerül → CPU-splitek, 13 pörgő CPU-szál, story tg 34 (MTP nélkül 40). Valójában befér (teljes offload: 10,46 / 11,26 GB).
    - `-fitt 512` → 34/34; `-ctkd f16 -ctvd f16` → MTP compute 681 → 169 MiB, de csak 33/34.
17. **Javítás: darabolt K/V f16 konverzió a CUDA FA-ban** (`fattn-common.cuh`, `fattn.cu`). A TILE és VEC kernel
    (parallel_blocks, stream-k nélkül) a kvantált KV-t 64 MiB-os darabokban (Qwen3.5-9B: 16384 sor) konvertálja és
    dolgozza fel; minden darab ≥2 parallel blockkal fut, a részeredményeket (VKQ, KQ max/sum) egy új
    `flash_attn_combine_kv_chunk` kernel olvasztja a dst-ben futó eredménybe, az utolsó darab normalizál. A sinks csak
    az első darabnál, a mask darabonként eltolva. A compute buffer csak egy darabnyi f16 helyet foglal → CUDA0 compute
    689 → 253 MiB, a fit 8284 MiB-ot vetít 11017 MiB szabadra → **a felhasználói parancs módosítás nélkül 34/34**.
    Az MMA (Turing+, stream-k) és a sparse út változatlan (2080 Ti/3090-en még teendő). `GGML_CUDA_FA_KV_CHUNK_MB`
    felülírja a darabméretet (0 = kikapcsolva).
    - Helyesség: `test-backend-ops` teljes CUDA 19051/19051, FLASH_ATTN_EXT 256 soros darabokkal
      (`GGML_CUDA_FA_KV_CHUNK_MB=0.0001`) 4040/4040.
    - Költség (FA prefill, kv=8704, nb=512, q8_0, us/run, darab nélkül / 8 MiB / 2 MiB): hsk256 16612 / 17981 / 22013,
      hsk128 14793 / 17247 / 22699 → darabonként ~0,3 ms fix költség; 64 MiB-os darabnál (≥16k sor) ~1%.
      f16 KV-nél nincs változás (nincs konverzió).

| CUDA 1080 Ti, MTP n=1, `-fitt 512` | story pp | story tg | wiki pp | wiki tg |
|---|---|---|---|---|
| upstream | 472 | 49,7 | 667 | 42,3 |
| saját (defer + `-bs`) | 499 | 51,9 | 662 | 44,2 |
| saját + darabolt FA, alap fit (34/34, `-fitt` nélkül) | 501 | 52,1 | 663 | 43,5 |
| + Pascal mmvq (18.) | 505 | 53,7 | 666 | 44,5 |
| **+ Pascal FA GQA (19.), végleges** | **506** | **53,2** | **677** | **45,1** |
| (alap fit, 31/34 réteg) upstream / saját | 231 / 220 | 31,7 / 34,0 | 507 / 505 | 25,1 / 26,0 |
| (MTP nélkül, upstream) | 525 | 40,0 | 726 | 35,9 |

18. **Pascal mmvq paraméterek** (`mmvq.cu`, új `MMVQ_PARAMETERS_PASCAL` tábla, sm_60–69): ncols_dst=1-nél 4 warp / 2 sor
    blokkonként, 2–4 oszlopnál (MTP verify) 4 sor. llama-bench tg128 41,1 → 42,1, pp2 76,8 → 77,9. Izolált kernelben a
    4 sor/blokk a legjobb (q6_K −9%), de a fúzionált gate/up úton lassabb (tg 40,5), ezért 2. L2 prefetch (DGX Spark
    mintájára) +15..+70% lassabb Pascalon. `-DGGML_CUDA_MMVQ_PASCAL_{NW1,RPB1,RPB_SMALL}` felülírja.
19. **Pascal FA: GQA-optimalizáció 4-es GQA aránynál prefillben is** (`fattn-tile.cuh`): eddig Pascalon gqa ≤ 4 és
    Q > 16 sor esetén ki volt kapcsolva. Mért: minden [4,1] perf-eset −5..−10% (pl. hsk256 kv8704 nb512 q8_0
    16,3 → 15,1 ms), a többi változatlan. gqa ≤ 2 és pre-Pascal marad a régi. pp2048@d16384 625 → 644 (upstream: 625).
    KLD (wikitext, q8_0 KV) bitre azonos az upstream CUDA-val; a szerver rövid promptjainál a KV-felosztás más, ezért
    a greedy kimenet eltérhet.
20. **CUDA alatt a GeForce P2 állapotban fut: memória 5005 MHz vs Vulkan P0 5508 MHz** (−10% sávszélesség). Ez a
    Vulkan 45 vs CUDA 42 tg különbség nagy része; a kernelek csúcshoz mért hatásfoka hasonló (q4_K 12288×4096: CUDA
    61%, Vulkan 66%). `nvidia-smi --cuda-clocks` / `-lmc` csak Volta+, Pascalon nincs támogatott felülírás.
    Rootként kipróbálva (driver 580.178): `-lgc 1936,1936`, `-lmc 5505,5505` és `-ac 5505,1936` mind
    „not supported”; `-pm 1` mellett terhelés alatt továbbra is P2, SM 1911 MHz, memória 5005 MHz (tg256 42,8).
    Az `nvidia-settings` Coolbits-os memória-offset X szervert igényel (Wayland + fej nélküli eGPU), és a
    legfelső P-szintre hat, nem a P2-re. MTP nélküli generáláshoz Pascalon a Vulkan a gyorsabb.
    MTP-vel ennek ellenére a CUDA nyer, mert a 2–3 tokenes verify olcsó (pp2 78 vs Vulkan 69).
    - MTP draft hossz: n=1 a legjobb (n=2: story 52,4 / wiki 44,8; n=3: 43,7 / 42,7).
    - `-ub 1024/2048` nem gyorsít prefillt (945 / 944 / 928).

A kimenetek bitre azonosak (story/wiki sha upstream = saját). Pascalon a CUDA a gyorsabb backend (Vulkan v5: 307 /
43,7 / 467 / 38,4).

## NVIDIA RTX 2080 Ti (Turing, sm_75, Thunderbolt) — 2026-10-29

Ugyanaz a parancs, MTP n=1, defer + `-bs`. Vulkan: `-dev Vulkan2` (NV_coopmat2), CUDA: `CUDA_VISIBLE_DEVICES=1 -dev CUDA0`.
CUDA alatt P2 / 6800 MHz memória (Vulkan P0 / 7000 MHz, csak −3%).

| 2080 Ti | story pp | story tg | wiki pp | wiki tg |
|---|---|---|---|---|
| brew Vulkan, MTP nélkül | 360 | 61,6 | 2061 | 54,6 |
| brew Vulkan, MTP n=1 (ref) | 562 | 69,2 ¹ | 1796 | 53,6 |
| saját Vulkan v5 | 341 | 64,2 | 1827 | 58,8 |
| **saját Vulkan (22–23.)** | **690** | **74,1** | **1821** | **65,4** |
| upstream CUDA, MTP nélkül | 1018 | 74,7 | 1949 | 65,5 |
| upstream CUDA, MTP n=1 (29/34 réteg) | 187 | 19,3 | 723 | 16,6 |
| saját CUDA + MMA darabolás (34/34) | 973 | 95,3 | 1786 | 61,6 |
| **saját CUDA + vec verify (21.)** | **910–939** | **96,5–97,2** | **1773–1794** | **83,9–84,2** |

¹ A brew a story-ban más szöveget generál (elfogadás 0,91, ismétlődő szöveg), ezért a story tg nem összevethető;
a quick tesztben (elfogadás 0,56 vs 0,62) brew 56,8 vs saját 61,9.

21. **CUDA darabolt FA az MMA (stream-k) úton is** (`fattn-mma-f16.cuh`, `fattn-common.cuh`, `fattn.cu`): darab módban
    a kernel tile-onként egy blokkal fut, és a meglévő `needs_fixup` úton írja a nem normalizált tile-t + KQ max/rowsum
    párost (`kv_chunk_mode`: meta-puffer + gridDim.x == tile-szám, ami normál indításnál nem fordul elő); az új
    `flash_attn_combine_kv_chunk_tiles` fésüli össze a darabokat. A sparse variáns (op_params[4] > 0) nem darabol.
    2080 Ti: target compute buffer 964 → 253 MiB, draft 681 → 245 MiB, 29/34 → 34/34 réteg. FA 4376/4376 (normál és
    256 soros darabokkal). Darabonként ~0,1 ms (kv=8704, 256 soros darabok: 2,99 → 6,78 ms; 64 MiB-nál elhanyagolható).
    **Kvantált KV + 2 tokenes verify Turing/Ampere-en a vec kernelt használja** (eddig csak Ada+): az MMA minden
    hívásnál a teljes KV-t f16-ra konvertálja. FA kv=20736 nb=2 q8_0: 1266 → 279 us; pp2@d20480 83 → 120 t/s.
22. **CUDA FA iq4_nl K/V**: eddig nem volt támogatott (a FA a CPU-ra esett). Új `vec_dot_fattn_vec_KQ_iq4_nl`,
    `dequantize_V_iq4_nl`, f16-konverzió (`dequantize_iq4_nl_pair`), vec példányok; a `GGML_CUDA_FA_QUANTS`
    alapértéke most minden azonos típusú KV-t tartalmaz (q4_0, q4_1, q5_0, q5_1, iq4_nl, q8_0, f16, bf16) — eddig a
    q4_1/q5_x lassú f16 visszaesést kapott. FA (kv=20736, us, nb=1 / 2): f16 159/161, q8_0 219/287, q4_0 227/314,
    q4_1 203/311, q5_0 344/421, q5_1 313/407, iq4_nl 415/517. (A q6_0 nem létező KV-típus.)
23. **Vulkan, NVIDIA Turing+ mat-vec**: 2–3 oszlopnál (verify) a fp K-quant shader 4 sorral gyorsabb az MMVQ-nál
    (`kq_batch_rm` = 4 alapból): n=2 q4_K 12288×4096 108 → 76 us, q5_K 121 → 87, q6_K 248320×4096 1620 → 1476.
    1 oszlopnál k > 4096 esetén is a fp shader (q4_K 4096×12288 MMVQ 70 → 60 us). **q4_K/q5_K shader**: a min-tag
    B-összegei soronként helyett egyszer számolódnak (gyártófüggetlen): q4_K n=2 −6..−10%, n=3 −5..−8%, n=1 −2..−3%;
    q5_K −1..−3%. llama-bench pp2 101 → 129–135, tg128 68,6 → 70,4–70,8.
    - Vulkan FA: nb=2 ≈ 2× nb=1 (a GQA-összevonás csak a fejeket vonta össze, a tokeneket külön munkacsoportok
      olvasták végig a KV-n) — kv=20736 q8_0 215 vs 418 us. → 24. pont.
    - Vulkan pp2@d20480 GPU-idő 21 ms: mat-vec ~12,9 ms, FA 3,3 ms, apró műveletek (ADD/CPY/RMS_NORM/SCALE,
      5–8 us/db) ~4 ms.

24. **Vulkan FA: tokenek × GQA-fejek egy munkacsoportban** (`flash_attn.comp`, `ggml_vk_flash_attn`): a skaláris
    shader sora r = token r / gqa, fej r % gqa; a munkacsoport gqa_T tokent kezel (gqa_T·gqa ≤ a blokk sorai), így a
    tokenek egyszer olvassák a KV-t. Q/maszk/kimenet és a split_k részeredmények soronkénti címzéssel; gqa_T = 1
    esetén a régi viselkedés. Alapból be (`GGML_VK_FA_GQA_TOKENS`, 1 = ki), kivéve coopmat1 (nincs mérve) és sparse.
    kv=20736, 2 token, us: f16 395 → 219, q4_0 415 → 245, q8_0 409 → 305, q4_1 462 → 302, q5_0 511 → 372, q5_1 492 →
    473, iq4_nl 619 → 583 (3 tokennél iq4_nl 872 → 947, a többi gyorsul). FA 5368/5368, teljes Vulkan 21371/21371.
    Szerver (MTP n=1): q8_0 story 76,3 → 79,2, wiki 66,4 → 73,4; q4_0 wiki 68,4 → 72,9.

K/V-típusok a szerveren (2080 Ti, MTP n=1, a 21–24. pont után; t/s story / wiki tg):

| KV | CUDA | Vulkan (24. előtt) |
|---|---|---|
| q8_0 | 97,9 / 85,3 | 76,3 / 66,4 |
| q4_0 | 95,6 / 84,7 | 74,4 / 68,4 |
| q4_1 | 96,3 / 84,3 | 79,1 / 67,2 |
| iq4_nl | 94,7 / 75,5 | 75,6 / 64,6 |
| q5_0 | 95,2 / 76,0 | 75,3 / 63,6 |
| q5_1 | 95,4 / 79,0 | 77,4 / 67,1 |
| f16 | 11 / 9 (nem fér el: a fit CPU-ra tesz rétegeket) | nem indult el |

Vulkanon az első futás egy új KV-típussal ~5 s shader-fordítás (a story pp ~20 t/s), utána a meghajtó gyorsítótárából
tölt. A q6_0 nem létező KV-típus.

## Pontosság 4. szint: ismétlési hurok (végtelen token-ismétlés), 2026-10-29

**Miért:** a felhasználó hosszú magyar meséknél gyakran kapott végtelen ismétlést Vulkanon és CUDA-n is. Két minta:
- „épp csak épp csak …”;
- bekezdésnyi blokkok újra és újra.

A kérdés az volt, hogy a gyorsítások (FA darabolás, vec verify, Vulkan GQA-összevonás, MTP defer, `-bs`) rontják-e a
numerikát vagy a mintavételezést.

**Mit csináltam:** saját tesztet írtam (`~/llama-bench-results/loop-test/`).
- `loop-test.py` elindítja a szervert a megadott konfiggal, 128k kontextussal, q8_0 KV-val.
- A felhasználó két promptját küldi el streamelve, fix seedekkel (1–6), legfeljebb 12000 tokenig, alapértelmezett
  mintavételezéssel.
- A rövid periódusú hurkot menet közben felismeri, és megszakítja a generálást.
- `loop-score.py` utólag pontoz: hurok, ha a kimenet utolsó 200 karaktere legalább háromszor előfordul az utolsó
  12000 karakterben. Ez a bekezdésnyi (~1100 karakteres) periódust is elkapja.
- `loop-run.sh` futtatja a konfigokat: cu-full, cu-mtp, cu-nomtp, up-nomtp, up-mtp, vk-full.

Promptok:
- p0: „tervezz meg egy 10000 szavas gyerekenek szóló mesét, amit később meg kell írnod”
- p1: „csinálj 10000 szavas mesét avgy hosszabbat”

**Eredmény (2080 Ti, CUDA, hurok / 6 futás):**

| konfig | mintavételezés | p0 | p1 | p1 tg (t/s) |
|---|---|---|---|---|
| saját CUDA, MTP n=1, defer + `-bs` | alapértelmezett (temp 0,8, top_k 40, top_p 0,95, min_p 0,05, büntetés nélkül) | 0/6 | 4/6 | ~99–106 |
| **módosítatlan upstream CUDA, MTP nélkül** | alapértelmezett | 0/6 | **4/6** | ~72 |
| saját CUDA, MTP n=1, defer + `-bs` | Qwen-ajánlás: `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5` | – | **0/6** | ~82 |

- A változtatások nem okoznak ismétlést: a módosítatlan upstream MTP nélkül ugyanannyiszor hurkol.
- Ez a 9B Q4 modell viselkedése hosszú magyar szövegnél, büntetés nélküli mintavételezéssel. Az upstream hurkai
  ugyanolyanok, mint a felhasználó példái, például „A toll a toll volt. A toll a toll volt. …”.
- A hurkok alatt a draft elfogadás magas (~0,87, a szokásos ~0,7 helyett), ezért a tg is magasabb (~105 t/s). Ez a
  bench-számokat torzíthatja (lásd a brew story ¹ megjegyzését).
- A presence penalty 1,5 megszünteti a hurkot, és a szöveg értelmes marad. A `-bs` mellett is a GPU-n fut (a
  penalties sampler backend-képes).
- Ára: a magasabb temp és a büntetés miatt az elfogadás ~0,6-ra esik, a tg ~100 → ~82 t/s. Ez még mindig gyorsabb az
  MTP nélküli upstreamnél (~72).
- **Ajánlás:** a szerverparancshoz `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5`.
  Ha a webes felületen saját mintavételezési értékek vannak beállítva, ott a Presence penalty-t kell 1,5-re állítani.

**Sebesség a hurokmentes mintavételezéssel** (2080 Ti, 2026-10-29). Mérés: `bench.py` `BENCH_SAMPLING=server`, a fenti
Qwen-ajánlással, 3 futás medián, seed = 1 + futás, thinking off. t/s.

| | story pp | story tg | wiki pp | wiki tg |
|---|---|---|---|---|
| brew Vulkan, MTP nélkül | 696 | 59,8 | 2055 | 54,6 |
| brew Vulkan, MTP n=1 | 582 | 52,0 | 1801 | 49,8 |
| eredeti kód, saját Vulkan build, MTP nélkül | 718 | 59,0 | 2123 | 54,2 |
| eredeti kód, saját Vulkan build, MTP n=1 | 591 | 52,0 | 1856 | 51,1 |
| eredeti kód, CUDA, MTP nélkül | 1035 | 70,6 | 1969 | 63,1 |
| eredeti kód, CUDA, MTP n=1 (nem fér el, CPU-ra tesz rétegeket) | 208 | 18,4 | 737 | 16,6 |
| saját Vulkan, MTP nélkül | 767 | **63,8** | 2062 | 57,5 |
| saját Vulkan, MTP n=1, defer + `-bs` | 683 | 61,7 | 1821 | **58,4** |
| saját CUDA, MTP nélkül | 1100 | 70,6 | 1983 | 62,9 |
| **saját CUDA, MTP n=1, defer + `-bs`** | 1034 | **77,6** | 1798 | **71,2** |

- Az eredeti kódban az MTP ezzel a mintavételezéssel lassít (Vulkan: 59 → 52 tg).
- A saját buildben:
  - CUDA-n az MTP gyorsít: +10% story, +13% wiki az MTP nélküli futáshoz képest.
  - Vulkanon az MTP nem ad gyorsulást: 61,7 / 58,4 MTP-vel, 63,8 / 57,5 nélküle. Mivel MTP nélkül a pp is
    nagyobb, Vulkanon ennél a mintavételezésnél az MTP nélküli futás az ajánlott.
- MTP nélkül:
  - CUDA-n a saját és az eredeti kód azonos (70,6 / 62,9 vs 70,6 / 63,1), mert a CUDA-változtatások az MTP verify
    lépést célozzák.
  - Vulkanon a saját build +8% story és +6% wiki tg-t ad a Turing mat-vec beállítások (23. pont) miatt.
- A wiki pp MTP-vel ~10%-kal kisebb, mint MTP nélkül, mert a draft fej is feldolgozza a promptot. Ez az eredeti
  kódban is így van (2123 → 1856).
- A greedy mérésekhez (fent: CUDA 97/85, Vulkan 79/73) képest a tg ~20%-kal kisebb, mert a draft elfogadás ~0,6.

**Validálási szabály mostantól:** minden gyorsítás után a pontossági és hallucinációs tesztek (1–3. szint) mellett ezt
a hurok tesztet is le kell futtatni, minden érintett backenden, az upstreammel azonos mintavételezés mellett.
- A gyorsítás csak akkor elfogadható, ha a hurok-arány nem rosszabb az upstreaménél (p1: 4/6 alapértelmezett, 0/6
  Qwen-ajánlással).
- Futtatás: `~/llama-bench-results/loop-test/loop-run.sh cu-full vk-full up-nomtp`.
- Csak p1: `ONLY_P=1`.
- Ismert hiba: a `loop-run.sh` `UP` útvonala ideiglenes könyvtárra mutat; ha hiányzik, újra kell építeni az
  upstream buildet.

## Hátra

- FA prefill (coopmat1, Xe2): ~3,5 TFLOPS — a wiki pp kb. harmada. A Br=32 nem segít, a kernel szerkezete (kis
  tile-ok, shmem-softmax, sok barrier) a korlát; mélyebb átírás kellene.
- Draft LM head (248320×4096 q6_K, 834 MB, 4,3 ms/lépés): kisebb bitszámú másolat vagy szűkített szótár.
- Vulkan FA token-összevonás coopmat1-en (Intel B50, RDNA3) mérése; iq4_nl/q5_1 lassú dequant a skaláris FA-ban.
- Vulkan apró műveletek fúziója (~4 ms / verify lépés a 2080 Ti-n).
- A 23. pont Turingon mérve; Ampere-en (3090) ellenőrizni.
- Pascal FA prefill: a TILE kernel fp32 Q-tile-ja a 48 KB shared memory határán van (hsk=256), mélyebb átírás kellene.
- Verify CPU-rögzítés ~7 ms/lépés (nagyrészt átfedésben a GPU-val).
- Pontosság: kész (1–4. szint, fent) — a saját build a brew-val azonos pontosságú, és nem hurkol gyakrabban az
  upstreamnél. Minden új gyorsítás után mind a négy szintet újra kell futtatni.
- Hurok teszt Vulkanon (`vk-full`) is: a 2080 Ti-n csak CUDA-n mértem, a felhasználó mindkettőn látta.
- Köztes mintavételezés mérése: csak `--presence-penalty 1.5` a régi temp 0,8 mellett (vagy DRY, `--dry-multiplier
  0.8`). Cél: hurok nélkül megtartani a ~100 t/s-ot, mert a Qwen-ajánlás ~18%-ot veszít a draft elfogadáson.
- A hurok teszt kiterjesztése a többi KV-típusra (q4_0, q4_1, q5_x, iq4_nl) és MTP n=2-re.
- Vulkan mat-vec n=2 újratervezése (~400 GB/s, CUDA-n ~510), apró műveletek fúziója mindkét backenden (~700 db, ~5 µs
  egyenként; pl. RMS_NORM+SCALE a GDN l2-normban).
- iq4_nl/q5_x FA dequant gyorsítása CUDA-n és Vulkanon.
- Hardver: RTX 3090 (Ampere), RX 6900 XT (ROCm és Vulkan; a csomagokhoz és csoportokhoz sudo kell a felhasználótól).
- Stabil méréshez: `sudo nvidia-smi -i 1 -lgc 1800,1800` (a felhasználónak kell futtatnia), a zaj most ±10–15%.
