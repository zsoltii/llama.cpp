# Státusz: llama-server gyorsítás (Vulkan + CUDA)

Utolsó frissítés: 2026-10-26 este (doksi-rendrakás + 38. kör: MTP-lépés felbontás, előzetes).
**Az aktuális napi parancs: results.md 35. kör** (Vulkan + MTP2 + f16 KV + `-bs`).

## Kész

- [x] F0: telepítések. FA: korpusz + `bench.py`. FB–FD: baseline + Vulkan +
      CUDA mátrix. FE: MTP (CUDA-n).
- [x] 6. kör (kontrollált): f16 +20% vs q4_1 (interleaved anchorok);
      FA VEC Turingon negatív → revert; `-bs` + draft-backend-sampling +9–13%
      (azonos tokenek: 499/261, 214/147); p-min 0.3 nem hozott.
- [x] 7. kör: `-bs` megerősítve mindkét párban (R3 60,8/60,7 vs 57,8/55,5);
      Vulkan f16 ≈ q4_1 de +1,6 GB → Vulkan marad q4_1.
- [x] Kódkísérletek mérve: BlockingSync (−25%), rm_kq=1 (−11% tg),
      FA VEC Turingon (−26% R3) → mind revert. **A fa tiszta.**
- [x] Hő-drift kezelés: kontrollált protokoll (interleaved anchor,
      `clocks-*.csv` + `clockstat.py`, 60 s hűtés).
- [x] Végső dokumentáció (results.md: 6.–7. kör + VÉGLEGES szakasz).
- [x] 8. kör: `GGML_CUDA_GRAPH_OPT=1` (dormant stream-overlap) tesztelve →
      meleg párban 61,40 vs 61,44 → nincs nyereség, elvetve.
- [x] Új mérési rezsim: fix 170 W power limit (systemd service
      `nvidia-powerlimit`); a termikus drift megszüntetése.
- [x] 9. kör (KÉSZ): referencia-újramérés 170 W alatt, egységes q4_1
      KV-val (CUDA+Vulkan), 8/8 rc=0. Fő számok: CUDA win R3 pp ~1160 /
      tg 47–56, R2 tg 65–76; MTP ~2× hosszú-ctx tg (23,8 → 47–56);
      Vulkan MTP ront (29,2 → 25,7); brew ≈ build-vulkan. Részletek:
      results.md 9. kör szakasz.
- [x] 10. kör (KÉSZ): CUDA MMVQ↔MMQ A/B — MMQ minden mutatóban rosszabb
      (R1 −15..17%, R2 −9..17%, R3 tg −10..12%) → marad az MMVQ; a patch
      visszaállítva, kódváltozás nincs. Részletek: results.md 10. kör.
- [x] 10b (KÉSZ): Turing MMVQ nwarps variáns (k-kvant 2→4 warps) —
      nincs konzisztens nyereség (R1 +11,8%/+0,4%, R2/R3 ±3% = zaj),
      elfogadás bitre azonos; variáns elvetve, forrás+build visszaállítva
      (results.md 10b. kör).
- [x] 11. kör (KÉSZ): Vulkan izoláció + env-kapcsolók — az MTP önmagában
      ront (BS nélkül is), FORCE_MMVQ ront, DISABLE_MMVQ/nodes semleges
      → VK-optimum a base; CUDA marad a napi sofőr. PCIe: terhelés alatt
      Gen3 x8, nem szűk keresztmetszet. Részletek: results.md 11. kör.
- [x] 12. kör (KÉSZ): CPU-affinitás/NUMA-pin — a hosszú-ctx metrikákon
      semleges (R2 ±0,2%, R3 tg −0,6..−0,9%), node0 ≈ node1; R1 +7–8% a
      szórásán belül → nem kerül a napi parancsba (results.md 12. kör).
      A kampány lezárult; nyitott: 170 W-os systemd service (sudo).

## Győztesek — TÖRTÉNETI (2026-10-18-i állapot)

> **Elavult.** Azóta: f16 KV nyert (19. kör), a Vulkan + MTP2 lett a napi
> config (35. kör). Az alábbi parancsok a 170 W-os q4_1-es rezsim lenyomatai.

- CUDA — mért referencia a 170 W-os rezsimben (q4_1, ajánlott):
  `build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL
  -fa on -ctk q4_1 -ctv q4_1 --parallel 1 -c 80000 --no-mmproj-offload
  --spec-type draft-mtp --spec-draft-n-max 2 -t 1 -tb 1 --poll 0
  --spec-draft-backend-sampling -bs`
  (r9–r12 mérések: R3 pp ~1150–1240 / tg 53–57; R2 tg 75–77; R1 tg 66–80;
   VRAM ~7,7–8,0 GB; draft-elfogadás 2,43; MTP hosszú-ctx tg ~2×.)
- CUDA — f16 változat (a sapka előtti rezsim győztese, 170 W alatt nem
  újramérve):
  `build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload
  --spec-type draft-mtp --spec-draft-n-max 2 -t 1 -tb 1 --poll 0
  --spec-draft-backend-sampling -bs`
  (kontrollált: R3 pp ~1035 / tg 60,7; R2 tg ~72,7; VRAM 9198 MiB.
   Hűvös session csúcs: pp ~1190, tg ~70,8.)
- Vulkan (tartalék):
  `build-vulkan/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL
  -fa on -ctk q4_1 -ctv q4_1 --parallel 1 -c 80000 --no-mmproj-offload`
  (r11: R3 pp ~895–912, tg ~31,2–31,6; R2 tg ~46–48; VRAM 6605 MiB.
   MTP-vel NEM érdemes: a spec-út Vulkanon ront — lásd r11.)

Megjegyzések:
- `-bs` kísérleti: grammar/speciális sampler esetén auto-fallback CPU-ra.
- Órajel/termika: a kártya 87–88 °C-on ~1550 MHz-re esik (lock alatt is);
  hűvös session-ben 10–20%-kal jobb számok jönnek. A kontrollált mérések
  a megbízható viszonyítási alap.
- KV-szabvány (mérések): q4_1 (CUDA és Vulkan) a 170 W-os rezsimben;
  az f16 változat a régi rezsim győztese (170 W alatt nem újramérve).
  iq4_nl elvetve (CUDA FA nem támogatja).
- `corpus/` érintetlen. A fán azóta vannak lokális, env-gated (default off)
  módosítások — lista: plan.md kapcsolótábla; `git diff --stat` 10 fájl.

Megjegyzések (környezet):
- `--offline` KIKERÜLT: libssl-dev telepítve + rebuild OpenSSL 3.5.5-tel,
  a `-hf` működik (cache-ből tölt, füsttesztelve).
- Órajel-lock: a 2026-10-16-i reboot óta NINCS (és az új rezsimben nem
  is kerül vissza); helyette fix 170 W power cap (systemd:
  `nvidia-powerlimit.service`). Visszaállítás: `nvidia-smi -pl 250 -i 0`.
  **Figyelem:** a 35–36. kör 250 W-on futott (a limit akkor nem volt aktív);
  2026-10-26 este `nvidia-smi` szerint ismét **170 W** él. Mérés előtt mindig
  ellenőrizni: `nvidia-smi --query-gpu=power.limit --format=csv`.
- KV-szabvány: ma **f16** (19. és 27. kör); q4_1 a VRAM-korlátos változat;
  iq4_nl ELVETVE (CUDA FA nem támogatja).
- Sampling a mérésekben temp=0 volt; élesben a saját temp/top-p
  beállításod érvényes (tg ±10% eltérés lehet).

## Eredmény számokban (R3, 65k prompt)

| | pp | tg | VRAM |
|---|---|---|---|
| Eredeti (brew, iq4_nl) | ~960 | ~33 | 7756 |
| Győztes (CUDA, f16 KV, MTP, régi rezsim) | ~1150 (+20%) | ~67 (+103%) | ~9220 |
| 170 W rezsim (CUDA, q4_1, MTP) | ~1150–1240 | ~53–57 | ~7700–8000 |

## 2b. fázis — P0 kód-visszaállítás + P1 natív q4_1 FA (2026-10-19)

**P0 — kísérleti kapcsolók visszaállítva** (mind env-gated, default off;
a fában uncommitted, 4 fájl, +60/−1 sor):
`GGML_CUDA_DISABLE_MMVQ`, `GGML_CUDA_BLOCKING_SYNC`,
`GGML_CUDA_FA_FORCE_KERNEL=vec|tile|mma[:N]` (N = max batch, új),
`GGML_VK_RM_KQ=1..4`, `-DGGML_CUDA_MMVQ_TURING_KQ_NWARPS4` (compile-time).

**P1 — `GGML_CUDA_FA_QUANTS` bővítés `q4_1-q4_1`-gyel** (build-cuda
újrafordítva; a `fattn-vec-instance-q4_1-q4_1.cu` létezik).

**Módszertani csapda (fontos!):** a `bin-cuda-ref/llama-server` RUNPATH-ja
`/home/zsoltii/git/llama.cpp/build-cuda/bin` — NEM `$ORIGIN`. Az első r13
kör ezért a "ref" futásoknál is az ÚJ libeket töltötte (A/B érvénytelen).
Megoldás: futásonként `LD_LIBRARY_PATH=$(dirname $bin)` (a RUNPATH-nál
erősebb). Ellenőrzés: `LD_TRACE_LOADED_OBJECTS=1` + a szerver-logban a
`no FlashAttention vector kernel compiled for K/V types q4_1-q4_1`
figyelmeztetés megléte (ref: van, var: nincs).

**Gyors igazolás (nem-MTP, R3, 170 W):**
| lib | R3 tg | megjegyzés |
|---|---|---|
| tiszta (ref, okt. 17) | 27,7 t/s | FA warning = 1, f16 konverzió fut |
| új (q4_1 FA) | 41,3–41,6 t/s | FA warning = 0, natív VEC kernel |

→ a batch-1 dekódnál a f16 K/V konverzió megszűnése **~+49%** tg-t ad
(nem-MTP út). Az MTP verify batch=3 Turingon MMA-ra megy (f16-only), ott
a konverzió megmarad — ezt a `vec:4` kapcsolóval támadjuk (r14).

### r13 végeredmény (javított lib-izolációval, 2026-10-19 08:30-08:50)

| run | R1 tg | R2 tg | R3 pp | R3 tg | R3 acc |
|---|---|---|---|---|---|
| ref1 (MTP) | 67,3 | 76,6 | 1148 | 55,7 | 0,7177 |
| var1 (MTP) | 73,0 | 75,5 | 1166 | 54,2 | 0,7177 |
| ref2 (MTP) | 72,8 | 73,6 | 1130 | 54,5 | 0,7177 |
| var2 (MTP) | 74,5 | 76,6 | 1120 | 53,9 | 0,7177 |
| var3 (MTP) | 67,1 | 75,9 | 1141 | 54,7 | 0,7177 |
| refN (nem-MTP) | 57,6 | 61,8 | 1301 | **26,6** | – |
| varN (nem-MTP) | 63,7 | 62,4 | 1311 | **42,3** | – |

**Verdikt:** `GGML_CUDA_FA_QUANTS` += `q4_1-q4_1` **MARAD** (nem-MTP R3
+59%, MTP/pp semleges, acceptance bitre azonos). FA-warning bizonyíték:
ref futások 1, var futások 0.

**r14 (fut, ~09:35-ig):** 7 futás a q4_1-FA build-en, win config + MTP:
base, `vec:4` (MTP verify batch=3 → natív VEC, nincs konverzió, ez a fő
jelölt), `CONV_FAST=1` (vektorizált konverziós kernel, detcheck: bitre
azonos kimenet), mmqprec=q4, ub1024, mma:1 (kontroll), base2 (anchor).

### r14 végeredmény (2026-10-19 08:55-09:15)

| run | R3 pp | R3 tg | verdikt |
|---|---|---|---|
| base | 1124 | 55,1 | anchor |
| vec:4 | 1148 | **41,1** | ELUTASÍTVA (SIMT VEC lassabb batch-3-nál, mint az MMA) |
| CONV_FAST | 1164 | **59,6** | **NYERT (+8,1%)** |
| MMQ_PREC=q4 | 1139 | 55,0 | semleges |
| -ub 1024 | 1140 | 50,8 | ELUTASÍTVA (−7,7%, VRAM +276 MiB) |
| mma:1 | 1163 | 54,8 | semleges (kontroll) |
| base2 | 1128 | 54,2 | anchor |

- A v2 konverziós kernel (1× 32-bites load + 2× 8 bájtos store a 8×
  2 bájtos helyett) **default-ra állítva**; `GGML_CUDA_CONV_LEGACY=1` a
  régi. Helyesség: detcheck → bitre azonos kimenet.
- vec:4 numerikai mellékhatás: R3 acceptance 0,71770 → 0,70283 (nem ez
  okozta a −25%-ot; mean len csak 2,43 → 2,41).
- **Log-csapda:** a FA-kényszerítő wrapper statikus initje már a
  modellbetöltés alatt lefut (~0,5 s), ahol a `GGML_LOG_*` üzenetek
  elvesznek → a bizonyíték-logot `fprintf(stderr, ...)`-re írtuk át
  (a r13/r14 első bizonyíték-grepjei ezért adtak 0 találatot).

**r15 (fut, ~10:05-ig):** igazoló A/B ugyanazon a binárison:
legacy vs default konverziós kernel (6 futás, MTP + nem-MTP párok).
Utána: Vulkan rebuild (RM_KQ) + r16, CPU-profil (cpuprobe).

### r15 végeredmény (2026-10-19 09:29-09:46)

Ugyanazon a binárison (md5 25fe9bb9): MTP R3 tg legacy 54,32/54,24 vs
**v2 default 58,24/60,01 (+7,2…+10,7%)**; pp +1…+3%; nem-MTP ±2% (zaj).
A konverziós kernel-választás log-bizonyítéka stimmelt (1 vs 0).
→ A v2 kernel default marad; `GGML_CUDA_CONV_LEGACY=1` vészkapcsoló.

**Következő:** Vulkan rebuild (RM_KQ kapcsoló) → r16 (9 futás, nem-MTP
base config) → CPU-profil (cpuprobe, MTP win config, -bs vs nem).

### r16 végeredmény (Vulkan, 2026-10-19 10:06-10:40)

9 futás, base config + env-kapcsolók. **Fő tanulságok:**
- nocm2: pp 914 → 432 (coopmat2 nélkül feleződik) → a coopmat2 kell.
- rmkq1: tg 31,5 → 29,8 (−5%) → marad a default rm_kq.
- nodot2/nodint/nomadd/memprio/nodes32: semleges (rep0 ±1,5%, a
  nodes32 +1,5%-a a base/base2 anchor-szóráson belül).
- **Mérési anomália:** az R3 cached rep1/rep2 néhány futásban belülről
  degradált (base2 rep1/rep2 −10%), ezért ezentúl a rep0 (azonos friss
  pp) a mérvadó; a medián-alapú értékelést óvatosan kell használni VK-n.
- A `GGML_VK_RM_KQ` bizonyíték-log (fprintf, 70 sor a rmkq1 futásban,
  0 máshol) működik.

### 2b. fázis — LEZÁRVA (2026-10-19)

Végleges: CUDA MTP + q4_1 KV + `-bs` (R3 tg 58-60), Vulkan base (31,5);
2 kódváltozás bent (q4_1 FA quants, vektorizált konverzió); minden más
kísérlet elutasítva vagy semleges. CPU-profil: a fő szál sync-spin miatt
~100%, valódi CPU-munka elhanyagolható; a CPU-sampler (`-bs` nélkül)
~1,5-2 ms/token. Sudo-teendő: 170 W systemd service telepítése.

### 2b. fázis folytatás (2026-10-19 este) — további kódkísérletek

- **CUDA konverziós v3** (16 bájtos store): mikrobenchmarkban implementálva,
  de **azonos a v2-vel** (1,819 ms; a kernel a memória-sávszélesség
  határán: 387 GB/s vs memcpy 408) → nem került be.
- **Vulkan rm_kq 3/4** (r17): semleges (28,95/28,67 vs 28,61; anchor-szórás
  ±4%) → marad a default 2. Környezeti drift: esti session ~10%-kal
  lassabb abszolút értékek, magasabb SM-órajel mellett.
- **CUDA MMQ_MIN_BATCH=3** (r18, új env-gated kapcsoló): a verify batch
  MMQ-routingja **−13%** → elutasítva, MMVQ marad.
- nsys kernel-profil: telepítési hiba miatt nem használható (sudo kellene).
- Összegzés: a maradék nyereség csak mély kernelmunkával (MMVQ verify-út
  ~240 GB/s effektív sávszélessége, ill. Vulkan matvec-shaderek) érhető el;
  a kapcsolókkal elérhető tér lezárt.

### CTranslate2-inspirált terv — 2 részre bontva (2026-10-20)

- **plan-part1-nodisk.md**: 2,1 GB szabad hellyel futtatható (mérés + kód):
  B1 f16 KV A/B, E újramérések (n-max, -ub), C0 GEMV-mikrobenchmark (kapu),
  D CPU FA split-KV kvantált KV-ra, B2 CUDA KV-shadow (ha kell), F konfig-higiénia.
- **plan-part2-disk.md**: ~24-25 GB csúcsigény (BF16 originál 18 GB + kimenet ~6 GB):
  custom kvant-mix (A1-A4, mindet mérni, memóriatudatosan), 160 B-os q4_K repack
  (csak ha a C0 kapu nyer), E2 újramérések.
- User döntések: minden kvant-változat mérése; BF16 forrás; KV-típus A/B;
  kód-sorrend C0 → D → B2; vegyes GPU+CPU üzemmód optimalizálása is cél.

### Part 1 / B1 — KV-típus (2026-10-20) — LEZÁRVA

- **f16 KV vs q4_1**: +14,2% / +14,5% (MTP R3, két pár), +16,8% / +12,0% (no-MTP),
  pp változatlan, acceptance bitazonos.
- Ár: +1,42 GB VRAM (MTP 9316 MB @80k — belefér).
- **Döntés: napi konfig `-ctk f16 -ctv f16`**; q4_1 marad VRAM-korlátos esetre.
- **B2 (KV-shadow) elvetve** (az árnyék +2,1 GB lenne a q4_1 fölött — rosszabb, mint az f16).
- Következik: E kör (n-max/-ub, f16 KV-val), C0 mikrobenchmark, D (CPU split-KV).

### Part 1 — LEZÁRVA (2026-10-20)

- **B1: f16 KV +14,2% / +14,5% MTP, +12–17% no-MTP** → napi konfig **f16 KV**
  (VRAM 9336 MB @80k); a q4_1 a VRAM-korlátos változat (7876 MB).
- **E**: n-max 3 (−8,4…−9,0%) elvetve, n-max 1 semleges (marad 2),
  `-ub 1024` (−1,7…−2,5%) elvetve.
- **C0**: a 16 B-os betöltés −16,5%; a 4 B-os GEMV 540 GB/s (88% csúcs) →
  **WP-C (160 B-os igazított formátum) elvetve**.
- **FA kernelcsalád** f16 KV-val: semleges → default (VEC batch 1, MMA ≥2).
- **D** (CPU split-KV kvantált KV): helyes (bitazonos kimenet), de **−70%**
  a per-head KV-újraolvasás miatt → env-gated (`GGML_CPU_FA_SPLIT_QUANT`),
  default off, nem használni.
- **F**: `--no-mmproj` opcionális (−879 MB RAM, sebesség azonos).
- Záró validáció: MTP R3 tg 66,1 t/s, pp 1145, acceptance 0,4375.
- Napi parancs: results.md → „Part 1 — VÉGLEGES konfiguráció".

### Vulkan körök (2026-10-20/21) — SDK + merge + mérések

- **Vulkan SDK 1.4.363.0** letöltve/részlegesen kicsomagolva, `build-vulkan`
  az SDK glslc-jével újraépítve (elavult shader-generátor miatti hibával —
  javítva). Eszközlánc-hatás mérése: **semleges** (26. kör).
- **27. kör: Vulkan f16 KV → +10,8/11,5% tg, +12,5/13,2% pp**, ára +1725 MB
  → a Vulkan napi profil most **`-ctk f16 -ctv f16`**. (Vulkan továbbra is
  ~25% lassabb a CUDA-nál; MTP Vulkanon nem nyerő.)
- **Pipeline-gyorsítótár** implementálva (`GGML_VK_PIPELINE_CACHE=<fájl>`,
  `atexit`-mentéssel; a device-destruktor nem fut le). Mérés: első prefill
  −9% átlagban, de zajos; **alapból KI**.
- **CPU-mérés bevezetve** (`cpu-watch.sh` + `bench.py`): minden futás rögzíti
  a szerver magszámát/sys%/iowait%-át. Normál érték: 0,5–0,9 mag, sys 1,5–4%
  → a mérések nem CPU-korlátosak. Egy beragadt szerver okozta korábban a
  furcsa CPU/VRAM értékeket.
- **29. kör: FORCE_MMVQ ront** (f16 KV-n is) → elvetve.
- **Master merge**: 82 commit fast-forwarddal, a 7 helyi módosítás bitre
  azonosan visszaállítva. Releváns új kód: `qwen35.cpp` kimenet-reshape
  (2D), CUDA MMVQ low-bit, MMQ Stream-K cap, FA vec/tile javítások,
  CUDA-graph tisztítás. A CUDA-s FA-oob ötlet Vulkanon már megvan
  (`kv_active` gyors út).
- **Unsloth**: Dynamic 2.0 mix használatban (nincs MXFP4, Q4_K+Q5_K+Q6_K+Q8_0);
  Dynamic 3.0 a 9B-MTP-hez még nincs kiadva, de az **imatrix elérhető**
  (5,2 MB) → a 2. rész A0 lépéseként saját v3-stílusú mix (plan-part2-disk.md).
- Következik: merge-ellenőrző A/B (31. kör), coopmat A/B (30. kör),
  `-O` shader-kísérlet (32. kör).
- **30. kör:** coopmat A/B semleges → default marad.
- **31. kör:** master merge mérése — a hideg első futás +35%-os csapdája miatt
  fordított sorrendű ismétlés kellett; meleg alapon **CUDA −4,4%** (FA vec WAR
  helyességjavítás ára), **Vulkan −1,7%** (zaj) → a merge marad.
- **Lezárva:** Vulkan végleges konfig (results.md), ReBAR terv rev.3,
  Unsloth Dynamic 3.0 a 2. rész A0 lépéseként.
- **32. kör:** master merge #2 (31 commit → `e3797b184`), konfliktus nélkül,
  helyi módosítások érintetlenek. Új libek: CUDA `a73534b1`, Vulkan `df3700a4`.
  Sanity: CUDA meleg plató 70,2/72,3 R3 tg, pp 1250; Vulkan 34,82 → **semleges**.
  Bemelegedési szabály megerősítve: az első 2 CUDA futás 5–12%-kal lassabb.

### 33–34. kör — MTP újramérés + gyökérok (Vulkan) — 2026-10-22

- **Modell-review:** a `src/models/qwen*.cpp` (qwen35, qwen35moe, qwen3next,
  qwen4exp) lineáris-attention blokkja és MTP-gráfja gyakorlatilag azonos;
  a qwen4exp `build_conv_state_at` csak a rollback-slot logikában tér el
  (nálunk `n_rs_seq = 0`), a qwen3next a régebbi, több `cont`-os változat.
  **Nincs átemelhető gráfszintű nyereség** — a Vulkan gráf-optimalizáló
  amúgy is fúziózza a silu+mul, sigmoid+mul, rms_norm+mul, ssm_conv+silu
  láncokat.
- **MTP (60k, szigorú A/B, cooldown ≤62 °C):** base 30,5; ugyanaz az
  mtp2 config 4× mérve: 23,66 / 29,15 / 28,63 / 27,65 (átlag 27,3, −11%,
  17%-os futásszórással) → **MTP Vulkanon 60k-n elvetve.**
- **Korrekció:** a `--spec-draft-backend-sampling` default ON, ezért a
  33. kör „dbs vs nincs dbs" összevetése két azonos config volt; a +13%-os
  „nyertes" futászaj volt. A korábbi checkpoint-elemzés („a dbs a fő
  lassító") ezzel érvénytelen.
- **`-bs` (backend sampling) a Vulkan napi configon: +5,6% tg**
  (32,25 vs 30,54), pp azonos → bekerült a napi parancsba.
- **`-t 1 -tb 1 --poll 0`:** rövid kontextuson +11…+22%, de 60k-n pp −5%,
  MTP-vel −9% → csak rövid kontextusú munkához.
- **Gyökérok:** a decode a Vulkanon **nem sávszélesség-kötött** (291 GB/s
  effektív a 616-ból; a GEMV-mikrobench 540), hanem a sok kis kernel közti
  rés + egyszeri gráf-költség. A `LLAMA_GRAPH_TIMING` szerint a build
  0,02–0,06 ms, a backend `graph_compute` (GPU-végrehajtás, blokkoló)
  13–18 ms. CUDA ugyanott 52,7 (base) / 73,9 (MTP2) t/s a Vulkan 41,6 /
  46,4-jével szemben.
- **Új env-kapcsolók (default off / változatlan):**
  `GGML_VK_FLOPS_PER_SUBMIT_DIV` (default 40; mérés szerint nincs hatás),
  `LLAMA_GRAPH_TIMING` (fázis-időmérés).
- **Eszköz-javítás:** a probe/screen script guardja `pgrep -x llama-server`
  (a `BIN=.../llama-server` szöveg a wrapper parancssorában hamis pozitívot
  adott a `ps ax | grep` alapú ellenőrzésnél).

### 35. kör — 41k valós kontextus: MTP újraértékelés — 2026-10-26

- **A 33–34. kör MTP-verdiktje visszavonva.** 41k prompton, 2000–8000 tokenes
  generálással a Vulkan+MTP2 **+30% tg** (48,7–49,1 vs 37,3–37,9 t/s),
  ára 4,37 s pp (38,31 vs 33,94 s). Megtérülés ~670–700 generált token.
- **CUDA vs Vulkan 41k-n egy szint** (lépésidő 44,4/48,0 vs 49,9/48,8 ms);
  a CUDA 58,39 t/s acceptance-artefakt (0,794 vs 0,714). pp-ben Vulkan jobb
  (1075 vs 1034). A CUDA rövid-kontextus előnye 41k-n eltűnik.
- **Módszertani tanulság:** a t/s erősen szövegfüggő (±9%, acceptance
  0,66–0,79). MTP-összevetéshez **ms/lépés + acceptance**, vagy ≥2000 token,
  több ismétlés. A `probe.sh` most kiírja az `acc` / `mean_len` /
  `ms/step` / `ms/token` értékeket, és `MODEL=`-lel tetszőleges GGUF-ra
  futtatható (`SAVE=` a válasz szövegét is menti).
- **`-t 1 -tb 1` 41k-n (MTP nélkül) semleges** (pp −1%, tg +2%) → hosszú
  kontextushoz `-t 24 -tb 24` marad.
- **`-b 4096 -ub 4096` = 8,0 t/s decode** (125 ms/lépés) és lassabb pp is;
  az `-ub 512` (default) marad. Ok feltáratlan (VRAM-hipotézis).
- **Napi config frissítve:** Vulkan + MTP2 + `-bs` + `-t 24 -tb 24 --poll 0`
  (results.md 35. kör). CUDA ugyanezzel a flagsorral fallback.

### 36. kör — Part 2 / A: egyedi kvant-mix (elindult) — 2026-10-26

- **Lemezhely megjött** (142 GB szabad) → a Part 2 (plan-part2-disk.md) indulhat.
- BF16 alap letöltve (`~/models/qwen35mtp/Qwen3.5-9B-BF16.gguf`, 18,41 GB)
  + `imatrix_unsloth.gguf_file` (5,2 MB, 248 bejegyzés, 77 chunk).
- **Eszközök:** `~/llama-bench-results/gguf-tensors.py` (tenzor-leltár,
  méret/forgalom), `mix-gen.py` (explicit `--tensor-type-file` tábla a
  UD-Q4_K_XL-ből), `quant-mix.sh` (variánsok), `quant-ppl.sh`, `quant-mix-ab.sh`.
- **Fontos tanulság:** a `--tensor-type` regex-alapú felülírás + a default
  Q4_K-mix belső számlálói miatt a „replica" nem volt reprodukálható
  (9 tenzor eltért). Megoldás: **explicit, teljes tenzor-tábla**
  (`^blk\.N\.tensor\.weight$=TÍPUS`), így az a0 pontosan 5840,36 MiB
  (= a kiadott fájl 6124 MB adata).
- **Lépcső:** a0 (replica) → v1 (head Q5_K) → v2 (head Q4_K) → v3
  (+ffn_down/ssm_out) → v4 (mind Q4_K). Számított lépés-forgalom:
  −5,6% / −10,9% / −14,8% / −20,1%.
- **Nyers screen (llama-bench, nincs MTP):** v1 +7,7%, v2 +12,1%, v3 +12,0%,
  **v4 +20,8%** tg; v4 pp +13%. A byte-modell pontos, a Q4_K a natív
  gyorsabb Vulkan-út is.
- **MTP-vel viszont nincs nyereség.** Rövid kontextusú, stabil A/B
  (3 kör, ms/lépés): base 49,3; v1 47,9; v2 50,9; v3 51,6; v4 51,1 ms
  → a szórás határán. Ok: az MTP-lépés 7,22 GB / 49,3 ms = **146 GB/s**,
  az MTP nélküli út 345 GB/s → az MTP-út **overhead-kötött**, nem byte-kötött.
- **Mérési buktató (fontos):** a 60k-s MTP A/B első olvasata szerint a v2
  15%-ot rontott — **hamis**. Ugyanaz a base modell 45,06 → 36,99 t/s a két
  kör között, azonos órajel/hőmérséklet mellett: a 60k-s futás (~140 s)
  eléri a 2080 Ti 88 °C-os limitjét és throttol. Stabil sávban ±1,5%.
- **A 170 W-os limit nincs bekapcsolva** (250 W a default) → ez a kör
  250 W-on készült; rövid kontextuson +17% a 170 W-os értékekhez képest.
- **Harness:** `clocks.mem` mintázás hozzáadva; `MODEL=` (tetszőleges GGUF),
  `SAVE=` (válasz mentése), `ms/step` + `ms/token` kiírás.
- **Eszközök:** `state-test.sh` (thermál állapot), `quant-short-ab.sh`
  (stabil rövid-kontextusú A/B), `quant-diag.sh`.
- **Döntés:** MTP-vel a napi config marad az UD-Q4_K_XL; a mix csak MTP
  nélkül (+12…+21%) vagy VRAM-célra (v4: −0,93 GB) hasznos. A legnagyobb
  nyitott lehetőség az MTP-út overheadje (~+40% elvi).

### CTranslate2-vonal lezárva (2026-10-26)

- **A CTranslate2 nincs beépítve** — soha nem is volt függőség/backend/kód,
  csak kutatási forrás (`~/git/CTranslate2` + a rev. 2 elemzés).
- **Rev. 2 archiválva a repóba:** `docs-dev/llama-server-perf/plan-ctranslate2.md`
  (státusz-headerrel); a `plan-part1-nodisk.md` / `plan-part2-disk.md`
  hivatkozásai erre a fájlra mutatnak (eddig `~/.opencode/plan/`-ra mutattak,
  ami a repón kívül van).
- **Mindkét származtatott kísérlet lezárva:** C0 (16 B-os AWQ-GEMV betöltés)
  → 21. kör, elvetve (a 4 B a leggyorsabb, 540 GB/s); D (CPU FA split-KV
  kvantált KV-ra) → 23. kör, elvetve (+70%).
- **Maradék kód:** `GGML_CPU_FA_SPLIT_QUANT=1` env-kapcsoló
  (`ggml/src/ggml-cpu/ops.cpp:9586`, default off).
- **A premiszsza felülírva:** a rev. 2 1,7× GEMV-headroomja helyett ma
  no-MTP 345 GB/s, MTP-lépés 146 GB/s (36. kör) → nem a betöltésszélesség,
  hanem az MTP-overhead a szűk keresztmetszet.
- **plan-part2-disk.md státusza frissítve:** 2.1 kész (36. kör), 2.2 elvetve
  (C0), 2.3/2.4 nyitott.

### 37. kör — int8/CTranslate2 kvantálás: nem átvihető (2026-10-26)

- Kód nélküli mérés kész Q8_0 fájllal (170 W, Vulkan, llama-bench):
  **pp4096 +1,8%**, **tg64 −32,2%** a Q4_K-hoz képest.
- A CT2 int8 per-soros skálát + `cublasGemmEx` IMMA-t használ, batch-elt
  workload-ra; a mi decode-unk 1–3 tokenes és memória-kötött → az int8
  2× byte-tal strukturálisan veszteséges.
- Vulkanon Turingon nincs int8 tensor core (coopmat Ampere+), CUDA-n a
  `mma.s8` helper létezik, de a MMQ csak sm_80+ alatt használja; egy ilyen út
  csak a prefill ~2×-ét adná (a falidő ~9%-a), új típus + kernelek árán.
- **Döntés: nincs int8 implementáció.** Részletek: results.md 37. kör.

### Doksi-rendrakás + 38. kör: MTP-lépés felbontás (2026-10-26 este)

- **Új terv:** `plan-part3-mtp-overhead.md` (G0–G4: az MTP-lépés overheadje).
- **Instrumentáció-hiba javítva:** a `LLAMA_GRAPH_TIMING` számlálója közös volt
  a fő és az MTP kontextus között, és minden 16. hívást írt ki → a 4 gráf/lépés
  ciklussal aliasolt (mindig ugyanazt a gráfot mintázta). Most kontextusonként,
  `(n_tokens, n_outputs)` szerint aggregál, 64 gráfonként ír. A szerver a draft
  kontextus `graphs reused` értékét is kiírja.
- **`mmvq.cu`:** a két kísérleti env-kapcsoló `getenv`-je statikusan cache-elve
  (eddig minden mul_mat dispatchnél lefutott).
- **38. kör (előzetes, 1 futás):** a dft-kontextus 0-kimenetes catch-up gráfja
  minden lépésben újraépül (22/22), az 1. draft-lépés is (22/42). A gráfokban
  töltött idő ~23 ms/lépés, a mért lépés **~45 ms** → **~22 ms/lépés a gráfokon
  kívül** (szerver/sampling/readback). Részletek: results.md 38. kör.
- Doksi-ellentmondások javítva: plan.md (környezet, kör-státuszok,
  kapcsolótábla), plan-part1 (állapot, f16 VRAM-ár), plan-part2 (head-méret,
  2.0 recept, 2.4 → G3), results.md 36. kör (timing-megjegyzés).

### 39. kör — a „22 ms” gyökéroka (2026-10-26 este)

- **A 38. kör olvasata téves volt:** a Vulkan `graph_compute` nem blokkol
  végig, a rés GPU-idő. Új szerver-időzítő (`LLAMA_SERVER_TIMING`): a szerver
  maga ~1 ms/lépés; a target verify ~32 ms, ebből a GPU ~30,6 ms.
- **Gyökérok:** a verify n=3 k-quant matvec (`q4_K 12288×4096`: n=1 65 µs,
  n=3 168 µs) — a float shader 2 soronként oszloponként újraolvassa az
  aktivációt; az MMVQ út sem jobb.
- **Új kapcsoló `GGML_VK_KQ_BATCH_RM=4`:** verify −14%, lépés −9…−10%,
  **t/s +10%** (rövid ctx, 2×3 interleaved futás). Default off; 41k-n kell
  igazolni, utána mehet a napi parancsba.
- Maradék: draft ~8 ms (GPU ~4,6) + catch-up ~3 ms (GPU ~1,3) → G1.
- **39b — 45k-n igazolva:** ms/lépés 57,8 → 53,3 (−7,6%), tg 49,4 → 53,9
  (+9,2%), pp változatlan, kimenet determinisztikus. A napi Vulkan parancs
  `GGML_VK_KQ_BATCH_RM=4` env-vel bővült (results.md 39b).

### 40. kör — CUDA vs Vulkan + javítások (2026-10-27 éjjel)

- **CUDA 45k-n ~1,7× gyorsabb a Vulkannál** (29,5 vs 53,8 ms/lépés); a 35. kör
  „egy szint” olvasata zavart CUDA-futásból jött (távoli asztal CUDA-t használ).
- **CUDA: MMVQ rows_per_block 4 (Turing, ncols 2–4) — alapértelmezett:**
  45k-n lépés −8,8%, tg 97,5 → 106,5 t/s, kimenet bitre azonos.
- **Vulkan FA batch 3 = 3× KV-olvasás** (1262 vs CUDA 380 µs) — ~7 ms/verify;
  javítása shader-munka (plan-part3 G6). Split-major kísérlet: nincs hatás.
- **G1** kész (env, default off, ~1%); közben a qwen35 MTP-gráf maszkolt
  `t_h_nextn` hibája javítva.
- Napi sofőr: **CUDA** (results.md 40. kör).
