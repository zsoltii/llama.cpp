# llama-server gyorsítás: Vulkan + CUDA (RTX 2080 Ti, Qwen3.5-9B-MTP)

Cél: a `llama-server` gyorsítása egy GPU-n futó modellre úgy, hogy a modell
nem lóg ki a GPU memóriájából. Mindkét backenden (Vulkan, CUDA) javítani
kell a prompt processing (pp) és a token generálás (tg) sebességét is.

## Környezet (rögzített, mert minden mérés erre épül)

- GPU: NVIDIA GeForce RTX 2080 Ti, 11 GB VRAM (Turing, sm_75, tensor core van)
- Driver 595.99.02 (CUDA driver 13.2). `nvcc` NINCS telepítve (pótolva: apt).
- OS: Ubuntu 26.04.1 LTS (QEMU/KVM vendég, VFIO GPU-passthrough, host: Dell R720),
  2×Xeon E5-2670 = 24 logikai CPU (2 NUMA node), 15 GB RAM.
  Lemez: induláskor 20 GB szabad, 2026-10-26 óta ~142 GB (Part 2 ezért indulhatott).
- Modell (fix, nem változtatható) — a kiinduló parancs; a KV-típus azóta
  `f16` (19./27. kör), az aktuális napi parancs: results.md 35. kör:
  `llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -fa on -ctk iq4_nl -ctv iq4_nl --parallel 1 -c 80000`
- Modell-architektúra (Qwen/Qwen3.5-9B config + `src/models/qwen35.cpp`):
  32 réteg, ebből 24 gated delta net (lineáris figyelem) + 8 full-attention
  (16 Q / 4 KV fej, head_dim 256). MTP fej be van égetve a GGUF-ba (`nextn`).
- Kiinduló állapot: brew llama.cpp 0.6.0 (build 11429, Vulkan-only),
  7,3 GB VRAM-ot foglal. Hosszú promptnál kb. 35 tok/s tg.
- VRAM matek (80k ctx, iq4_nl KV): ~5,8 GB súlyok + ~0,9 GB KV + ~0,5 GB
  compute + ~0,9 GB mmproj = ~7,3 GB. Cél: mindig min. ~800 MB szabad VRAM.
- Repó: friss HEAD (f3363593f), tartalmazza a Qwen3.5 beépített MTP
  speculative decoding támogatást (`draft-mtp`), ami a brew 0.6.0-ban is
  megvan (ellenőrizve: `git show d81235049`).

## Fázisok

### F0 - telepítések (felhasználó futtatja sudo-val) - KÉSZ

- `cmake glslc libvulkan-dev spirv-headers gcc-13 g++-13 nvidia-cuda-toolkit`.
- CUDA buildhez: `CUDAHOSTCXX=/usr/bin/g++-13` (CUDA 12.4 + gcc-13).
- Mérések alatt Firefox és más GPU-s app LE (VRAM-zaj), lemez-ellenőrzés.

### FA - referencia felmérés (szabványos protokoll, minden binary-val azonos)

- Fix korpusz (egyszer generálva, NEM módosítható):
  - `greeting.txt` (R1: köszönés, `n_predict:32`)
  - `moe-hu.txt` (R2: MoE-magyarázat magyarul, `n_predict:512`)
  - `long-60k.txt` (R3: 65 578 token + kérdés, `n_predict:256`,
    `/tokenize`-zal pontosítva; 65k/80k = ~82% ctx-töltés)
- Protokoll konfigonként: szerver-újraindítás, warm-up, 1x
  `cache_prompt:false` (pp+tg) + 2x `cache_prompt:true` (hosszú-ctx tg),
  3 futás medián. Sampling: temp=0 (determinisztikus).
  Metrikák: TTFT, pp tok/s, tg tok/s, VRAM-csúcs
  (1 s-es `nvidia-smi` polling), OOM-ellenőrzés.
- Indítás után **kötelező futás-ellenőrzés**: folyamat + szerver-log
  (`server is listening`) + valós GPU-terhelés (util/VRAM), és a végén
  a várt mérések megléte a `summary.csv`-ben. A puszta processz-jelenlét
  nem bizonyíték (a bench halott szerverre is vár).
- Eredmények: `~/llama-bench-results/` (`bench.py`, `results/*.json`,
  `summary.csv`).

### FB - brew mérések (Vulkan, 0.6.0) - KÉSZ

- B-baseline: pontosan a felhasználói parancs.
- B-opt rács: `-b`/`-ub`, `--no-mmproj-offload`, `--cache-reuse`,
  `--spec-type draft-mtp` + `n-max`.
- Eredmény: nincs valós ub-nyereség a ±5-15% futásközi zaj felett
  (drift-vizsgálattal igazolva). B-opt = baseline + `--no-mmproj-offload`.
  MTP brew-val elvetve.

### FC - Vulkan build a repó HEAD-ből (`build-vulkan/`) - KÉSZ, kiértékelve

- 0.7.0-dev, build 11697. Valós számok: pp ~930, tg ~30,5.
- HEAD Vulkan ≈ brew (zajsávon belül). MTP HEAD-en is lassít Vulkanon.
- `FA_SPARSE_DISABLE` hatástalan. `--no-mmproj-offload` ingyen -1,1 GB.
- Módszertani tanulság: új bináris ELSŐ futása Vulkan pipeline-fordítást
  fizet (driver-cache csak a 2. processtől él) — első futást nem értékelünk.

### FD - CUDA build a repó HEAD-ből (`build-cuda/`) - KÉSZ

- `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=75`, `CUDAHOSTCXX=g++-13`.
- Eredmény: a HEAD CUDA-binary lett a mindennapi győztes (MTP-vel,
  `-bs`-sel); lásd results.md 6.–7. kör + 9. kör.
- Az env-gated kísérleti kapcsolók listája: a „2b. fázis” táblája lent.

### FE - MTP a legjobb Vulkan és CUDA konfigon

- Feltétel: tg >= +15% és pp nem romlik, VRAM belefér. Különben eldobva.
- `n-max` 1-6 sweep (`2` a kiindulópont Unsloth szerint),
  `--spec-draft-p-min` marad 0 GPU-n, `--parallel 1` kötelező.
- Láncolt variáns: `--spec-type draft-mtp,ngram-mod`.
- Jegyzőkönyv: attempted/accepted/mean-len a szerver-statból + VRAM-csúcs.

### FF - dokumentáció és végső konfigok

- `~/llama-bench-results/` + `docs-dev/llama-server-perf/results.md`:
  táblák binary-nként és konfigonként.
- 2 végső indítóparancs (Vulkan + CUDA) + opcionális indítóscript.
- Követelmény a győztesekre: R1/R2/R3 hibátlanul 80k-n, min. ~800 MB
  szabad VRAM.

## Tiltólista (méréssel bizonyítottan árt)

- `GGML_VK_DISABLE_FUSION` (és bármely fúzió-kikapcsolás): -5%..-18%.
- `p-min` gate gyors GPU-n: -17%..-21% (mért adatok).
- `--mtp` download-flag: nincs sidecar, a fej be van égetve. Helyes:
  `--spec-type draft-mtp`.
- Saját GDN/CUDA-kernel írása: tensor-core út megvan, nem éri meg.

## Rizikók

- CUDA 12.4 + Ubuntu 26.04 glibc súrlódás (eddig OK: configure átment).
- R3 futások lassúak: összmérés ~1,5-2 óra volt eddig is.
- Futásközi zaj ±5-15%: csak e feletti, ismételt különbséget hirdetünk ki.

## 2. fázis — optimalizációs körök (2026-10-17-től)

Rezsim: fix 170 W power cap (systemd: `nvidia-powerlimit.service`);
minden mérés **q4_1 KV-val** (CUDA és Vulkan is) — nincs iq4_nl, és nincs
f16 a mérésekben (az f16 előnye a korábbi rezsim dokumentuma marad).

Célterületek (az induló felderítés alapján):
- **CUDA GPU**: MMVQ↔MMQ dekód-döntés (Turinghoz nincs hangolt küszöb;
  env-gated patch kész), MMVQ-nwarps/konfig A/B, FA-konfig, GDN-kernel.
- **Vulkan GPU**: env-kapcsolók A/B (FORCE/DISABLE_MMVQ,
  MAX_NODES_PER_SUBMIT, DISABLE_ASYNC, DESCRIPTOR_REUSE,
  MEMORY_PRIORITY, GRAPH_OPTIMIZE); MTP/`-bs` újrateszt q4_1-en.
- **CPU**: affinitás (2×Xeon E5-2670, 24 logikai CPU, 2 NUMA node; a GPU
  `numa_node=-1`), `taskset`/szál-kísérletek, szerver-oldali hot path.
- **GPU↔CPU**: PCIe link terhelés alatti ellenőrzése (idle-ben Gen1 x8,
  max Gen3 x16), szinkron/másolás útvonalak, gráf-lefedettség,
  pinned-memória.

Módszertan: 60 s cooldown, winner-anchor futások, power-guard; egy
változó egyszerre; minden forduló után docs-frissítés. Indítás után
kötelező futás-ellenőrzés (lásd FA protokoll).

Körök:
- r9 (KÉSZ): 170 W-os referencia, egységes q4_1 KV — CUDA win ×3, nomtp,
  nobs, VK-alap, VK-MTP, brew-anchor (results.md 9. kör).
- r10 (KÉSZ): CUDA MMVQ↔MMQ A/B — MMQ minden mutatóban rosszabb
  (R1 −15..17%, R2 −9..17%, R3 tg −10..12%) → marad az MMVQ; patch
  visszaállítva (results.md 10. kör).
- r10b (KÉSZ): Turing MMVQ nwarps variáns — nincs nyereség, elvetve
  (results.md 10b. kör).
- r11 (KÉSZ): Vulkan — MTP önmagában ront, FORCE_MMVQ ront,
  DISABLE_MMVQ/nodes semleges → VK-optimum a base (results.md 11. kör).
  PCIe terhelés alatt Gen3 x8 → nem bottleneck.
- r12 (KÉSZ): CPU-affinitás/NUMA-pin — semleges (results.md 12. kör).
  Nyitott: systemd 170 W service telepítés (sudo).
- r13 (KÉSZ, nyert — results.md 13. kör): FA q4_1-q4_1 VEC kernelek A/B — ref = archivált tiszta CUDA
  bináris, var = új build (q4_1 FA lefordítva + env-gated kísérleti kód).
  Várható: a batch-1 dekód FA f16-konverziója megszűnik → MTP tg +3..8%,
  nem-MTP dekód +10..20%.
- r14 (KÉSZ — results.md 14. kör): CUDA env A/B-k — `GGML_CUDA_MMQ_PREC=q4`, FA-kernel force,
  visszaállított kapcsolók kontextus-tesztje, `-t/-tb` sweep.
- r15 (KÉSZ — konverziós A/B, results.md 15. kör; CPU-profil: „CPU-profil” szakasz):
  CPU profilozás (perf, GGML_SCHED_DEBUG) + párhuzamosítás.
- r16 (KÉSZ — results.md 16. kör): Vulkan profilozás (`GGML_VK_PERF_LOGGER`) + env-sweep + `rm_kq`.

## 2b. fázis — kísérleti kódok visszaállítása + új kísérletek (2026-10-19-től)

A visszaállított kísérleti kapcsolók **env-gated, default off** módon élnek
(a napi parancs változatlanul gyors); a mérés csak a szükséges env-eket állítja.

| kapcsoló | hely | működés | default |
|---|---|---|---|
| `GGML_CUDA_DISABLE_MMVQ=1` | `ggml/src/ggml-cuda/mmvq.cu` | minden kvantált mul_mat MMQ-n megy | off (MMVQ) |
| `GGML_CUDA_BLOCKING_SYNC=1` | `ggml/src/ggml-cuda/ggml-cuda.cu` | `cudaDeviceScheduleBlockingSync` a spin helyett | off (spin) |
| `GGML_CUDA_FA_FORCE_KERNEL=vec\|tile\|mma[:N]` | `ggml/src/ggml-cuda/fattn.cu` | FA-kernel család kényszerítése (opcionális `:N` = csak `Q->ne[1] <= N` batchre) | off (auto) |
| `GGML_CUDA_MMQ_MIN_BATCH=N` | `ggml/src/ggml-cuda/mmvq.cu` | `ne11 >= N` batch MMQ-n (tensor core), az alatta lévők MMVQ-n; az MTP verify (batch 3) célzott vizsgálatára | off (MMVQ ≤8-ig) |
| `GGML_VK_RM_KQ=1..4` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | k-quant MMVQ shader rows-per-shader felülírás | off (2) |
| `-DGGML_CUDA_MMVQ_TURING_KQ_NWARPS4` | `mmvq.cu` (fordítási variáns) | Turing k-quant `ncols_dst=1`: 2→4 warp | off (2 warp) |
| `GGML_CUDA_FA_QUANTS` += `q4_1-q4_1` | build-konfiguráció | natív q4_1 VEC FA kernel (nincs f16-konverzió) | **BE** (r13 igazolta: nem-MTP R3 +49%) |
| `GGML_CUDA_CONV_LEGACY=1` | `ggml/src/ggml-cuda/convert.cu` | a *régi* q4_1→f16 konverzió kényszerítése; a vektorizált v2 az alapértelmezett | **BE** (r14/r15: MTP R3 +8…+11%) |
| `GGML_CPU_FA_SPLIT_QUANT=1` | `ggml/src/ggml-cpu/ops.cpp` | CPU FA split-KV kvantált KV-ra | off (23. kör: −70%) |
| `GGML_VK_BLOCKING_SYNC=1` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | fence-várakozás kernelben (spin helyett) | off |
| `GGML_VK_FLOPS_PER_SUBMIT_DIV=N` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | a gráf submit-felosztása | 40 (34. kör: nincs hatás) |
| `GGML_VK_PIPELINE_CACHE=<fájl>` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | tartós pipeline-cache | off (28. kör) |
| `LLAMA_GRAPH_TIMING=1` | `src/llama-context.cpp` | gráf-időmérés kontextusonként, `(n_tokens, n_outputs)` szerint, 64 gráfonként + `decode()` fázisai | off (38. kör óta aliasing-mentes) |
| `LLAMA_SERVER_TIMING=1` | `tools/server/server-context.cpp` | lépésenkénti szerver-bontás (draft, verify, catch-up, sampling, yield), 64 lépésenként | off (39. kör) |
| `GGML_CUDA_MMVQ_TURING_RPB_SMALL` | `ggml/src/ggml-cuda/mmvq.cu` (fordítási) | Turing MMVQ rows_per_block ncols_dst 2–4-re | **4 (BE)** — 40. kör: verify −12%, tg +9,3%; `-D...=2` a régi |
| `LLAMA_MTP_DEFER_CATCHUP=1` | `common/speculative.cpp` | MTP catch-up beolvasztása az első draft-lépésbe | off (40. kör: ~1%, nem éri meg) |
| `GGML_VK_KQ_BATCH_RM=N` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | q4_K/q5_K/q6_K float mat-vec n≥2-re N sor/workgroup (q6_K ≤4), q4_K/q5_K n>1 MMVQ helyett float-úton | off — **39. kör: N=4 → MTP verify −14%, t/s +10%; 45k-n igazolva: lépés −7,6%, tg +9,2%** → napi parancsban env-vel |

**Állapot (2026-10-19 délelőtt):** az r13–r15 igazolta a két bent maradó
kódváltozást (natív q4_1 FA + vektorizált konverzió). Az r14 elutasította a
`vec:4`-et, a `-ub 1024`-et; semleges a `MMQ_PREC=q4` és az `mma:1`.
Nyitott: r16 Vulkan env-kör, CPU-profil kiértékelése.


**Mechanizmus (r13):** Turingon a kvantált KV-s FA batch=1-nél VEC,
batch≥2-nél MMA (f16-only) → az MMA út *minden lépésben* az egész KV-t
f16-ra konvertálja (60k ctx-nél ~3 GB/token a 8 attention-layeren).
A natív q4_1 VEC kernel ezt a batch-1 dekódnál megszünteti: nem-MTP R3
tg 27,7 → 41,3 t/s (+49%). Az MTP verify batch=3 → MMA → a konverzió
megmarad: ezért `vec:4` (r14) és `CONV_FAST` (r15) a jelölt.

**Mérési csapda:** a `bin-cuda-ref/llama-server` RUNPATH-ja az élő
`build-cuda/bin`-re mutat → futásonként `LD_LIBRARY_PATH=$(dirname $bin)`
kell, különben a "ref" is az új libeket tölti.


Tiltólista-kiegészítés: iq4_nl (nem támogatott a CUDA FA-ban + user
döntés); f16 KV a mérésekben (felfüggesztve); `GGML_VK_DISABLE_FUSION`
(korábban mérve: −5..−18%).
