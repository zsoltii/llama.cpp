# llama.cpp gyorsítás — 1. RÉSZ: lemezhely nélkül végrehajtható terv

> **STÁTUSZ: VÉGREHAJTVA (2026-10-20).** Eredmények: `results.md` 19–24. kör,
> „Part 1 — VÉGLEGES konfiguráció" és `progress.md`. Összefoglalva:
> B1 (f16 KV) nyert (+14–16%), B2 elvetve, E (n-max 3, ub1024) elvetve,
> C0 → WP-C (160 B formátum) **elvetve**, FA kernelcsalád semleges,
> D (CPU split-KV kvantált KV) **env-gated, default off** (mért −70%),
> F (`--no-mmproj`) opcionális.

Állapot: **végrehajtva** (lásd a fenti státusz-blokkot; az alábbi szöveg az eredeti terv)
Cél: Qwen3.5-9B-MTP, RTX 2080 Ti, 170 W, 80k kontextus — tg/pp nyereség, mérve, hordozhatóan (CPU/CUDA/Vulkan/ROCm).
Korlát: **2,1 GB szabad lemezhely** → ebben a részben **nincs új modellfájl**, csak mérés + kód (inkrementális build).

**Döntések (user, 2026-10-20):**
- A kvant-mix **minden változatát** meg kell mérni (q5_K, q4_K, A1–A4), **memóriatudatosan** (a VRAM-lábnyom is szempont: a kisebb mix több réteget enged a GPU-n).
- A kvant-mix forrása a **BF16 originál** (~18 GB letöltés) → a 2. részben, lemezhely birtokában.
- A KV-t **mérjük mindkettőt** (q4_1 vs f16), a gyorsabb marad.
- A Part 1 kódmunkáinak sorrendje: **C0 → D → B2**.
- **Vegyes GPU+CPU üzemmódra is optimalizálni kell** (részleges offload esetére is legyen gyors).

Háttér: `plan-ctranslate2.md` (rev. 2, archív — a tartalom beolvadt ide), `docs-dev/llama-server-perf/results.md`.

---

## 1.0 — Dokumentáció (ELSŐ LÉPÉS, implementáló agentnek)

A két terv behelyezése a repóba:
- `docs-dev/llama-server-perf/plan-part1-nodisk.md` ← ez a fájl
- `docs-dev/llama-server-perf/plan-part2-disk.md` ← a 2. rész
- `docs-dev/llama-server-perf/progress.md`: új bejegyzés „CTranslate2-inspirált terv, 2 részre bontva (lemezkorlát miatt)".
- A meglévő `plan.md`/`results.md` változatlan; a rev. 2 elemzés
  (`plan-ctranslate2.md`, archív) tartalma a két részbe olvadt.

---

## 1.1 — B1: f16 KV vs q4_1 (0 kód)

**Miért**: a CUDA MMA-út (batch ≥2, azaz az MTP verify) minden lépésben az egész KV-t f16-ra konvertálja (~700 MB/lépés 65k-nál). f16 KV-val ez a konverzió eltűnik; a korábbi mérés szerint **+4-5% MTP** (62,18 vs 59-60 t/s). Cserébe a batch-1 dekód több KV-t olvas (534 vs 157 MB/token) és +450 MB VRAM (@80k) — *becslés; a mért ár +1,42 GB lett (19. kör)*.

**Lépések**
1. Interleaved A/B: `q4_1` anchor → `f16` → `q4_1` anchor → `f16` (4 futás, 170 W, MTP win config).
2. Ugyanez no-MTP-vel (2+2 futás) — itt a f16 várhatóan veszít (nagyobb KV-olvasás).
3. VRAM-ellenőrzés @80k (bele kell férnie: 7,75 GB + ~0,45 GB ≈ 8,2 GB).
4. Acceptance/mean len ellenőrzés (f16 KV numerikailag jobb, az acceptance változhat).

**Elfogadás**: ha az MTP tg javul és a VRAM befér → a napi konfig KV-típusa f16 lesz (a no-MTP külön konfigban maradhat q4_1).
**Ha nem nyer** → 1.5 (B2 KV-shadow) lép életbe.

---

## 1.2 — E: újramérések a jelenlegi modellel (0 kód)

**Miért**: a korábbi sweep-ek (n-max 2/3/1/4) a *drága* Q6_K headdel és a régi konverziós kernellel készültek; a konverzió azóta +9%-ot javult (r15), ami eltolhatja az optimumot.

**Lépések**
1. `--spec-draft-n-max 3` vs `2` (interleaved, 4 futás) — a 3. draft-pozíció acceptance-e 0,478 volt.
2. `--spec-draft-n-max 1` ellenőrző futás (a régi 49,99 vs 54,82 viszony validálása).
3. `-ub` újraellenőrzés (512 default vs 1024) — a korábbi −7,7% megerősítése vagy felülírása.
4. FA-kernel választás a verify-re: `GGML_CUDA_FA_FORCE_KERNEL=mma` (default) vs `vec:4` — a korábbi −25% megerősítése (a konverzió javulása után).

**Elfogadás**: bármelyik eltérés >2% és konzisztens (2 pár) → napi konfig frissítése + docs.

---

## 1.3 — C0: GEMV-mikrobenchmark (kapu a 2. rész formátum-munkájához)

**Miért**: a decode matvec ~231 GB/s (a memóriarendszer ~408 GB/s-ot bír) → ~1,7× headroom. A hipotézis: a 4 bájtos betöltések (VDR=2) korlátozzák a memória-szintű párhuzamosságot; a 16 bájtos (uint4) betöltés + split-K (CTranslate2 AWQ GEMV minta) javít.

**Lépések**
1. Önálló .cu (`/tmp/opencode/gemvbench.cu`, ~10 MB): azonos 5,5 GB forgalom, 3 variáns:
   - 4 B betöltés (jelenlegi MMVQ-séma, VDR=2)
   - 8 B betöltés (VDR=4)
   - 16 B betöltés (uint4, VDR=8) + opcionális split-K
2. Mérték: GB/s és ms; ismétlés hideg L2-vel (nagy buffer).
3. Összevetés: 231 GB/s (MMVQ tényleges) vs 408 GB/s (memcpy referencia).

**Kapu**: ha a 16 B-os variáns **≥15%-kal** gyorsabb → a 2. rész C (160 B-os igazított q4_K + kernelfordítás) indokolt. Ha nem → a formátum-munka **elvetve**, a decode marad a jelenlegi.

---

## 1.4 — D: CPU FA split-KV kvantált KV-ra (kód, CPU-only)

**Miért**: a `ggml/src/ggml-cpu/ops.cpp` split-KV útja ma **csak f16/f32 KV-ra** megy (`kv_is_f32_or_f16`, 9566. sor); kvantált KV-nál (q4_1) a lassabb, chunkolás nélküli út fut. A chunk-partial + LSE-reduce infrastruktúra már megvan.

**Lépések**
1. A feltétel kiterjesztése: kvantált K/V is (a chunk-számítás a dekvantált K/V-t használja, ahogy a generikus út).
2. Helyesség: `test-backend-ops -o FLASH_ATTN_EXT` (CPU) a referencia-összevetéshez; bitközeli egyezés a nem-split úttal (tolerancia szerint).
3. Mérés: CPU `llama-bench` (vagy server `-ngl 0`) 4k/32k/65k kontextuson, split nélkül vs splittel.
4. CUDA/Vulkan: **nem érintett** (saját FA-juk van) → a fordításuk változatlan.

**Elfogadás**: CPU hosszú kontextus +10% felett, helyes eredmény → megtartva; különben csak env-gated kísérlet.

---

## 1.5 — B2: CUDA KV-shadow (kód, csak ha B1 nem nyer)

**Miért**: ha a q4_1 KV marad, a verify (MMA) útvonalon az egész KV f16-ra konvertálása (~700 MB/lépés) megszüntethető egy perzisztens f16 árnyék-KV-vel, amely a KV-write-tal együtt frissül.

**Lépések**
1. Perzisztens f16 scratch allokálása a KV cache méretében (@80k: ~655 MB; a VRAM-ba belefér).
2. A KV-write (graph `cpy` a cache-be) mellé f16 írás — CUDA backenden belül, a q4_1 cache mellett.
3. Az MMA FA a shadow-t olvassa; a VEC út (batch 1) változatlanul a q4_1-et.
4. Invalidáció: cache-shift/rollback/defrag esetén epoch-számláló → teljes újrakonverzió.
5. Mérés: MTP R3 A/B (4 futás), helyesség: `detcheck.sh` (bitközeli kimenet) + acceptance.

**Elfogadás**: +4% feletti MTP tg, helyes kimenet; különben elvetve és dokumentálva.

---

## 1.6 — F: konfig-higiénia (0 kód)

1. `--no-mmproj` (a vision-projektor nem kell; RAM + load-idő) — ellenőrizni, hogy a szöveges inference változatlan.
2. Batch-reordering (hossz szerinti slot-rendezés) — **csak több párhuzamos kérésnél** számít; nálunk `--parallel 1` → dokumentálva, nem implementálva.
3. A napi konfig véglegesítése a nyertesekkel (f16 KV / n-max / -ub függvényében).

---

## Mérési protokoll (kötelező)

- 170 W ellenőrzés; lib-út + md5 + env bizonyíték; interleaved A/B + anchor; 60 s cooldown; nincs forrás-szerkesztés/fordítás mérés közben.
- Értékelés: rep0 (Vulkan), medánok, delta-párok; `results/summary.csv`; acceptance + mean len.
- Session-drift ±10% → csak azonos sessionben hasonlítunk; kétely esetén ismétlés.

## Lemezhatás

- **0 új modellfájl**; csak `/tmp/opencode` (~10 MB) és inkrementális build (a meglévő `build-cuda`/`build-vulkan` könyvtárakba).
- A CPU-teszthez nem kell új build-dir: a `build-cuda/bin/libggml-cpu.so` használható (`-ngl 0`).

## Vegyes GPU+CPU üzemmód (user követelmény)

- Cél: a konfig akkor is legyen gyors, ha nem fér be minden réteg a VRAM-ba (részleges offload, más modellek mellett futás).
- Mérési forgatókönyv: a győztes konfig + részleges offload (`-ngl` úgy, hogy ~8–16 réteg a CPU-n) → tg/pp + VRAM.
- A CPU-oldali attention a WP-D (split-KV kvantált KV-ra) miatt gyorsul — ez a vegyes mód kulcsa.
- A 2. rész kvant-mix választásánál a **VRAM-lábnyom is kritérium**: a kisebb (Q4_K) mix több réteget enged a GPU-n → vegyes módban nagyobb nyereség.
- A `-ngl` hangolás és a CPU FA külön méréssel validálandó (CPU-n és vegyes módban is).

## Sorrend (user döntése: C0 → D → B2)

1.0 docs → 1.1 B1 f16 KV → 1.2 E újramérések → 1.3 C0 kapu → 1.4 D CPU split-KV → 1.5 B2 (ha kell) → 1.6 F.
