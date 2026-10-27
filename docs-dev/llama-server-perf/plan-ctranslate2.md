# llama.cpp gyorsítás — CTranslate2-inspirált terv (rev. 2)

> **ÁLLAPOT: ARCHÍV (2026-10-26).** A tartalom beolvadt a
> `plan-part1-nodisk.md` / `plan-part2-disk.md` tervekbe, és a két belőle
> származó kísérlet **le van zárva**:
> - **WP-C0** (16 B-os AWQ-GEMV betöltés + split-K) → 21. kör: **elvetve**,
>   a 4 B-os betöltés a leggyorsabb (540 GB/s), a 16 B −16,5%.
> - **WP-D** (CPU FA split-KV kvantált KV-ra) → 23. kör: **elvetve** (+70%).
>
> A rev. 2 premisszája (decode 231 GB/s a 408-as plafonhoz → 1,7× headroom)
> a későbbi mérésekkel **részben felülíródott**: no-MTP 345 GB/s, az MTP-lépés
> 146 GB/s (36. kör) → nem a betöltésszélesség, hanem az MTP-út overheadje a
> szűk keresztmetszet.
>
> Eredeti hely: `~/.opencode/plan/llamacpp-perf-ctranslate2.md`. Az alábbi
> szöveg változatlan (rev. 2), csak a státusz-header került elé.

---


> **Végrehajtható változat két részre bontva (lemezkorlát miatt):**
> - `llamacpp-perf-part1-nodisk.md` — ma: `plan-part1-nodisk.md` (mérés + kód)
> - `llamacpp-perf-part2-disk.md` — ma: `plan-part2-disk.md` (quant-mix, repack)
> A két fájl a `docs-dev/llama-server-perf/` alá kerül az implementáció első lépéseként.

Állapot: **terv, jóváhagyásra** (Plan mód; implementáláshoz agent-váltás kell)
Cél: Qwen3.5-9B-MTP, RTX 2080 Ti, 170 W, 80k kontextus, q4_1 KV — minél nagyobb tg/pp nyereség, mérve, hordozhatóan (CPU/CUDA/Vulkan/ROCm).

---

## 0. Vezetői összefoglaló — mi változott a rev. 1-hez képest

1. **A prioritás megfordult.** A rev. 1 a CUDA-s GEMV/újraformázott súlyokat tette elsőnek (nagy, kockázatos, nem bizonyított). A kritikus átvizsgálás kimutatta, hogy a modellben **~1,2 GB/lépés "felesleges" precizitás** van, ami **nulla kóddal** kivehető, és **minden backenden azonnal hat** → ez lett a WP-A (első).
2. **A 16 bájtos betöltés hipotézise kaput kapott**: előbb egy 30 perces szintetikus GEMV-mikrobenchmark dönt (4/8/16 B betöltés, azonos forgalom). Csak nyerés esetén indul a formátum-munka.
3. **A formátum-terv egyszerűsödött**: nem AWQ-stílusú szétválasztott skálák, hanem **160 bájtos, 16 B-ra igazított q4_K blokk** (+8% méret, veszteségmentes repack) — ez egyszerre adja az igazítást és a széles betöltést.
4. **Új tétel: a KV-típus kérdése** — a korábbi mérés szerint **f16 KV + MTP ≈ +4-5%** (62,18 vs 59-60 t/s), kód nélkül; a q4_1-et a user írta elő mérési standardként → **döntés kell**. Ha marad a q4_1, akkor jön a CUDA "KV-shadow" (a redundáns konverzió megszüntetése).
5. **Új tétel: n-max újramérés** a WP-A után (a draft most 3× olvassa a headet; olcsóbb head mellett a 3. draft ismét megérheti).
6. **Új apróság: `--no-mmproj`** (a vision projektor feleslegesen töltődik: RAM + load-idő).

---

## 1. Tények (kód + mérés + GGUF-elemzés)

### A modell összetétele (UD-Q4_K_XL, GGUF fejlécből)
| rész | méret |
|---|---|
| rétegek 0-31 | 4431,7 MB |
| MTP blokk (blk.32, 15 tensor) | 150,7 MB |
| **LM head (`output.weight`, Q6_K, 248 320 szókincs)** | **795,7 MB** |
| token_embd (Q4_K, gather → forgalom ~0) | 560,8 MB |
| **összesen** | **5939 MB** |

Típus-mix: Q4_K 2963 MB, Q6_K 1678 MB (ebből head 796, `ffn_down` 883), Q5_K 827 MB, Q8_0 442 MB (`ssm_out` 408 + `eh_proj` 34), F32 28 MB.
A modell **hibrid**: 24 SSM-réteg (`ssm_out` Q8_0!) + 9 attention-réteg.

### MTP lépés forgalma (súlyok)
- verify (32 réteg + head): **5228 MB**
- 2 draft dekód (MTP blokk 151 + head 796): 2 × 947 = **1894 MB**
- **összesen: 7120 MB/lépés; a head 34% (3× olvasás: verify + 2 draft)**
- ehhez jön az FA konverzió (~700 MB) + f16 KV olvasás (~534 MB) → ~8,3 GB/lépés

### Mért számok (170 W, q4_1)
- no-MTP: 42 t/s → 231 GB/s effektív súly-sávszélesség
- MTP R3: 59-60 t/s, 41,1 ms/lépés, acceptance 0,7177, mean len 2,43
- pp: 1180-1300 t/s
- memóriarendszer felső korlát (memcpy D2D, convbench): **408 GB/s** → a matvec ~1,7× a csúcs alatt

### CTranslate2 — mivel magyarázza a sebességet
- Saját README/szerzői fórum: **precizió (fp16/int8) + vendor BLAS (MKL/cuBLAS, `CUBLAS_COMPUTE_16F` = 2× GeForce-on) + rétegfúzió (masked softmax, fuzionált QKV) + padding-kiküszöbölés + batch-rendezés + in-place/caching**.
- Saját **AWQ int4 GEMV**: 16 bájtos (`float4`) súlybetöltés + split-K + warp-redukció (`src/ops/awq/gemv_gpu.cu`).
- Saját FlashAttention: **csak sm_80+**, Turingon nem fut; a Python csomagból kivették („minimal performance gain").
- Benchmark (opus-mt, 4 CPU thread): transformers 1023 → CT2 fp16 9297 tok/s; memória 4097→909 MB.

### Mi már megvan a llama.cpp-ben
split-KV FA mindhárom backenden (CUDA VEC `parallel_blocks` / MMA stream-K; Vulkan `fa_split_k_reduce`; CPU chunk-partials — **csak f16/f32 KV-ra**), fuzionált QKV/FFN, kvantált súlyok, caching allocator, CUDA graph, `GGML_CUDA_CUBLAS_COMPUTE_TYPE` env-kapcsoló.

---

## 2. Kritikus átvizsgálás — a rev. 1 hibái

1. **Prioritás**: a legnagyobb, legkockázatosabb tétel volt az első; a legolcsóbb, legnagyobb hozamú (quant-mix) ki sem volt emelve. → WP-A lett az első.
2. **Bizonyítatlan hipotézis**: a 16 B-os betöltés haszna nem bizonyított → **C0 mikrobenchmark-kapu** kell előtte.
3. **Túl nagy formátum-terv** (AWQ-stílus) → egyszerűbb 160 B-os igazított blokk (veszteségmentes, +8% méret).
4. **Hiányzó KV-döntés**: f16 KV vs q4_1 → B1/B2 szétválasztva.
5. **Hiányzó mérési kockázat**: esti drift ±10% → csak session-ön belüli, interleaved A/B + anchor; modellváltásnál külön anchor.
6. **A head 3× olvasása** a rev. 1-ben nem szerepelt — ez a legnagyobb egyszeri tétel.
7. **mmproj** felesleges betöltése → `--no-mmproj`.

---

## 3. Work package-ek (prioritás szerint)

### WP-A — Custom quant mix (0 kód, minden backenden hat) ★ első
- **Mit**: a head és a túlprecíz tensorok újrakvantálása.
- **Eszköz**: `llama-quantize --output-tensor-type q5_K|q4_K --tensor-type 'ffn_down=q5_K' --tensor-type 'ssm_out=q5_K'` (tensor-szintű felülírás létezik).
- **Változatok / várható forgalom-csökkenés (MTP lépés)**:
  - A1: head q5_K (700 MB) → −402 MB (−5,7%)
  - A2: head q4_K (546 MB) → −750 MB (−10,5%)
  - A3: A2 + ffn_down q5_K + ssm_out q5_K → ~−1014 MB (−14%)
  - A4: mind q4_K → ~−1197 MB (−17%)
- **Várható sebesség**: +8-15% tg (MTP), +4-8% no-MTP; pp kb. változatlan (compute-bound); VRAM 5,9→~5,2 GB (szabad hely f16 KV-hez!).
- **Minőség-kapu**: `llama-perplexity` + KL a baseline-hoz, és **acceptance-ellenőrzés** (a draft ugyanazt a headet használja → az acceptance változhat; a mean len-t is nézni kell).
- **Hordozhatóság**: automatikus (CPU/CUDA/Vulkan/ROCm ugyanazt a GGUF-ot eszi).

### WP-B — KV: f16 vs q4_1, illetve CUDA KV-shadow
- **B1 (config, 0 kód)**: `-ctk f16 -ctv f16` újramérés MTP-vel; várható +4-5%; VRAM +450 MB @80k (7,75→8,2 GB, belefér; a WP-A még szabadít is rajta).
- **B2 (kód, ha q4_1 marad)**: a CUDA MMA-út **KV-shadow**-ja: a KV-write mellé f16 másolat készül, az FA azt olvassa → a teljes-KV konverzió (700 MB/lépés) eltűnik. Kockázat: cache-shift/rollback invalidáció (epoch-számláló), a VEC út (batch 1) változatlan.
- **Várható**: +4-7% MTP.

### WP-C — GEMV-mikrobenchmark → (ha nyer) 160 B-os igazított q4_K + 16 B betöltés
- **C0 (kapu, ~30 perc)**: szintetikus kernel-benchmark: 4/8/16 B-os betöltés, azonos 5,5 GB forgalom → GB/s; eldönti, van-e egyáltalán sávszélesség-nyereség.
- **C1**: új típus: q4_K, **160 bájtos blokk** (128 B kvantum + 12 B skála + d/dmin + pad), 16 B-ra igazítva, **veszteségmentes repack** (+8,1% méret).
- **C2**: CUDA MMVQ (+MMQ fallback nagy batchre) 16 B-os betöltéssel; ROCm automatikus (hipify).
- **C3**: CPU `vec_dot` (16 B/AVX2) és Vulkan matvec shader (`vec4`) ugyanarra az elrendezésre.
- **Várható**: +20-40% a decode-on, **ha** a hipotézis áll; nagy munka (új ggml-típus + konverter + 3 backend).

### WP-D — CPU FA split-KV kvantált KV-ra (portabilitás)
- `ggml/src/ggml-cpu/ops.cpp:9566`: a `kv_is_f32_or_f16` feltétel kiterjesztése kvantált KV-ra (a chunk-partial + LSE reduce út már megvan).
- Várható: CPU hosszú kontextus +10-30%; CUDA/Vulkan változatlan; ROCm nem érintett.

### WP-E — Újramérések a WP-A után
- `--spec-draft-n-max 3` (olcsóbb head mellett), `-ub`, FA-kernel választás, f16 KV kombináció.
- Esetleg a draft **külön, kisebb headje** (extra GGUF-tensor + kód) — csak ha a közös head Q4_K minősége nem elég a verify-nak.

### WP-F — Apróságok
- `--no-mmproj` (RAM + load-idő).
- Batch-reordering (hossz szerinti slot-rendezés) — csak több párhuzamos kérésnél; nálunk (`--parallel 1`) elhanyagolható.

---

## 4. Hordozhatósági mátrix

| WP | CPU | CUDA | Vulkan | ROCm | Megjegyzés |
|---|---|---|---|---|---|
| A quant-mix | ✓ | ✓ | ✓ | ✓ | csak GGUF, kód nélkül |
| B1 f16 KV | ✓ | ✓ | ✓ | ✓ | config |
| B2 KV-shadow | – | ✓ | – | ✓ (hipify) | CUDA-specifikus optimalizálás |
| C repack | ✓ | ✓ | ✓ | ✓ | új típus; minden backenden kernel kell |
| D CPU split | ✓ | – | – | – | CPU-only |
| F mmproj | ✓ | ✓ | ✓ | ✓ | config |

ROCm: a `ggml-hip/CMakeLists.txt` a `../ggml-cuda/*.cu` forrásokat fordítja (hipify a buildben) → a CUDA-változás automatikusan átvihető, ha nincs CUDA-only API; **a fordulás az elfogadási kritérium**.

---

## 5. Mérési protokoll (kötelező, a kampány szabályai szerint)

- 170 W ellenőrzés; lib-út + md5 + env bizonyíték-log; interleaved A/B + anchor; 60 s cooldown; nincs forrás-szerkesztés/fordítás mérés közben.
- Értékelés: rep0 (Vulkan R3), medánok, delta-párok; `results/summary.csv` sorok; acceptance + mean len ellenőrzés.
- **Modellváltásnál** (WP-A): régi modell anchor → új modell → régi anchor újra; különben a ±10%-os session-drift elnyeli a hatást.
- Minden WP csak **mért győzelem** esetén marad; a vesztesek dokumentálva (results.md).

---

## 6. Nyitott döntések (a user válasza kell)

1. **WP-A head**: mehet-e a **q4_K** (legnagyobb nyereség, ~−10,5%), vagy a biztonságosabb **q5_K** (−5,7%)?
2. **WP-A forrás**: a meglévő UD-Q4_K_XL-ből újrakvantáljunk (gyors, kész), vagy töltsük le a **BF16 originált** (jobb minőség, ~18 GB letöltés)?
3. **WP-B**: elfogadható-e a **f16 KV** (+450 MB VRAM, +4-5%) a q4_1 helyett, vagy marad a q4_1 + CUDA shadow?
4. **WP-C**: ha a C0 mikrobenchmark nyer, belevágunk a **160 B-os formátumba** (nagy munka: új típus + 3 backend + konverter)?
5. **WP-E**: kell-e a draft külön headje (extra modell-tensor + kód), vagy elég a közös head?

---

## 7. Sorrend (javasolt)

1. **WP-A1/A2** (quant-mix, 0 kód) — 1-2 óra mérés + minőség-kapu → várhatóan a legnagyobb olcsó nyereség.
2. **WP-B1** (f16 KV) — 1 óra, ha a user engedi.
3. **WP-E** (n-max, -ub újramérés) — 1 óra.
4. **WP-C0** (mikrobenchmark-kapu) — 30 perc, döntés a nagy munkáról.
5. **WP-D** (CPU split-KV) — fél nap, portabilitási demonstráció.
6. **WP-C1-C3** (csak ha C0 nyer) — nagy munka.

**Implementáláshoz agent-váltás szükséges.**
