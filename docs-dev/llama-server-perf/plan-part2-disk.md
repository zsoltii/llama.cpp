# llama.cpp gyorsítás — 2. RÉSZ: lemezhelyet igénylő terv

Állapot: **RÉSZBEN VÉGREHAJTVA (2026-10-26)** — lemezhely megvan (142 GB szabad).
- **2.1 (A0–A4) KÉSZ**, eredmény: **36. kör**. Rövid összefoglaló: a kvant-mix
  MTP nélkül +12…+21% tg (v4), **MTP-vel nincs nyereség** (az MTP-út 146 GB/s-on
  overhead-kötött) → a napi config marad az UD-Q4_K_XL; a mix csak MTP nélkül
  vagy VRAM-célra hasznos.
- **2.2 (C, 160 B-os repack) ELVETVE** — a C0 kapu nem nyert (21. kör).
- **2.3 (E2) NYITOTT** (n-max 3 már részben mérve: a 3. draft-pozíció acceptance-e
  0,478 volt), **2.4 → kiváltva** a `plan-part3-mtp-overhead.md` G3 pontjával
  (csökkentett vocabú draft head, FR-Spec minta).
Cél: a decode/MTP súlyforgalmának csökkentése új modellfájlokkal (custom kvant-mix), illetve — ha a mikrobenchmark nyer — új, 16 B-ra igazított súlyformátummal.
Háttér: `plan-part1-nodisk.md`, `plan-ctranslate2.md` (rev. 2, archív).

---

## Lemezigény (előfeltétel)

| tétel | méret | megjegyzés |
|---|---|---|
| **BF16 originál (forrás — user döntés)** | **~18 GB** | letöltés; a kvantálás után törölhető (18 GB felszabadul) |
| 2.1 kvant-mix GGUF-k (A1–A4) | ~5,2–5,9 GB / db | a BF16-ból kvantálva; egyszerre 1–2 variáns tárolható |
| 2.2 160 B-os repack (ha C0 nyer) | ~6,4 GB | +8,1% a jelenlegihez képest |
| minőség-mérés (perplexity log) | <10 MB | — |

→ **Csúcsigény a 2.1-hez: ~24–25 GB szabad hely** (BF16 18 GB + kimenet ~6 GB). A BF16 törlése a kvantálás után 18 GB-ot szabadít. **Alternatíva**: a kvantálás külső lemezen (a BF16 és a kimenet is ott), majd csak a kész ~5,5 GB-os GGUF visszamásolása.

---

## 2.1 — A: Custom kvant-mix (0 kód, minden backenden hat)

**Miért**: a GGUF-elemzés szerint a modellben ~1,2 GB/lépés „felesleges" precizitás van:
- **LM head (`output.weight`) Q6_K, 796 MB (*a 36. kör mérése szerint 834 MB*) — lépésenként 3× olvasva** (verify + 2 draft) = a lépés súlyforgalmának 34%-a;
- `ffn_down` Q6_K: 883 MB;
- `ssm_out` Q8_0: 408 MB (hibrid SSM-rétegek).

**Eszköz**: `llama-quantize --output-tensor-type ... --tensor-type 'ffn_down=...' --tensor-type 'ssm_out=...'` (tensor-szintű felülírás létezik).

**Változatok — MINDET meg kell mérni (user döntés), memóriatudatosan**

| verzió | tartalom | Δ forgalom | várható tg | VRAM |
|---|---|---|---|---|
| A1 | head q5_K (700 MB) | −402 MB (−5,7%) | +4–6% | −96 MB |
| A2 | head q4_K (546 MB) | −750 MB (−10,5%) | +8–10% | −250 MB |
| A3 | A2 + ffn_down q5_K + ssm_out q5_K | ~−1014 MB (−14%) | +9–12% | −290 MB |
| A4 | mind q4_K | ~−1197 MB (−17%) | +10–15% | −~700 MB |

**Memóriatudatos szempont (user)**: a kisebb mix **több réteget enged a GPU-n** részleges offloadnál, és jobban befér más modellek mellé. Ezért a méréshez két értékelési szempont kell: (1) teljes GPU-offload tg/pp, (2) **vegyes GPU+CPU** mód (pl. `-ngl` ~8–16 réteggel a CPU-n) tg/pp + VRAM. A nyertes mixet mindkét módra jelenteni kell.

**Vegyes GPU+CPU validáció**: a WP-D (CPU FA split-KV) és a `-ngl` hangolás külön méréssel; a kisebb mix előnye itt nagyobb lehet, mint teljes offloadnál.

**Minőség-kapu (kötelező)**
1. `llama-perplexity` a baseline (UD-Q4_K_XL) vs az új mix (ugyanaz a korpusz, pl. `corpus/`).
2. KL-divergencia a baseline logits-hoz (ha van rá eszköz: `llama-perplexity --kl-divergence`).
3. **Acceptance + mean len** ellenőrzés MTP-vel (a draft ugyanazt a headet használja → a head kvantálása az acceptance-et is érinti).
4. Küszöb: perplexity +≤2% és acceptance ≥0,68 → elfogadható; különben lépcsőzés (q5_K).

**Várható**: MTP +8–15%, no-MTP +4–8%; pp kb. változatlan; VRAM 5,9→~5,2 GB (szabad hely az f16 KV-hez!).

---

## 2.2 — C: 160 B-os, 16 B-ra igazított q4_K + 16 B-os betöltések (csak ha C0 nyer)

**Előfeltétel**: az 1.3 C0 mikrobenchmark ≥15%-os nyereséget mutat a 16 B-os betöltésre.

**Terv**
1. **C1 — új típus**: q4_K, **160 bájtos blokk** (128 B kvantum + 12 B skála + d/dmin + pad), 16 B-ra igazítva; **veszteségmentes repack** (+8,1% méret). Konverter: a meglévő GGUF-ból (llama.cpp tool vagy saját script).
2. **C2 — CUDA/ROCm**: MMVQ 16 B-os (`uint4`) betöltésekkel + opcionális split-K; MMQ fallback a nagy batchre; ROCm automatikus (hipify), a fordulás az elfogadási kritérium.
3. **C3 — CPU**: `vec_dot` 16 B-os (AVX2) betöltésekkel; **Vulkan**: matvec shader `vec4` betöltésekkel.
4. Mérés: MTP + no-MTP + pp, a szokásos protokollal; helyesség: `detcheck.sh`.

**Várható**: +20–40% decode (ha a hipotézis áll); pp akár +5–10% (kevesebb betöltési tranzakció).
**Kockázat**: új ggml-típus minden backenden (nagy felület) → csak a C0 kapu után.

---

## 2.3 — E2: újramérések a 2.1 után

- `--spec-draft-n-max 3` (olcsóbb head mellett a 3. draft ismét megérheti).
- f16 KV kombináció (a 2.1 VRAM-nyeresége miatt biztosan befér).
- `-ub` 512 vs 1024 az új súlyokkal.
- A nyertesek beépítése a napi konfigba + docs.

## 2.4 — (opcionális) A draft külön, kisebb headje — KIVÁLTVA

> Helyette: `plan-part3-mtp-overhead.md` G3 (a gyakori ~32k token sorai a
> meglévő headből, új GGUF-tenzor nélkül).

Ha a közös head q4_K minősége nem elég a verify-nak: a GGUF-ba egy külön `mtp.head` tensor (Q4_K) + kód, hogy a draft azt használja, a verify a Q6_K-t. Várható: −500 MB/lépés, de az acceptance csökkenhet → mérni kell.

---

## Mérési protokoll (ugyanaz, mint az 1. részben)

- 170 W; lib/md5/env bizonyíték; interleaved A/B + anchor; 60 s cooldown; nincs szerkesztés/fordítás mérés közben.
- **Modellváltásnál** (2.1): régi modell anchor → új modell → régi anchor újra (a ±10%-os session-drift miatt).
- Minden változat csak mért győzelem esetén marad; a vesztesek a `results.md`-be.

## Sorrend

0. **BF16 originál letöltése** (~18 GB) → a kvantálás helyének biztosítása (belső vagy külső lemez).
1. 2.1 **A1 → A2 → A3 → A4** (minden változat mérése, memóriatudatosan; egyszerre 1–2 fájl) → minőség-kapu.
2. 2.3 E2 (n-max 3, f16 KV kombináció, `-ub`) + **vegyes GPU+CPU** validáció.
3. (ha a C0 kapu nyert) 2.2 C1 → C2 → C3.
4. 2.4 opcionális (draft külön head).

---

## 2.0 — Unsloth Dynamic 3.0: mit jelent és hogyan alkalmazható

**Mi az**: az Unsloth következő generációs dinamikus kvantálása (tiszta PTQ,
QAT/QAD nélkül): sokkal jobb imatrix-kalibrációs adat (agentic coding, chat,
többnyelvű), finomabb layer-szelekció és több kvantálási technika. Állításuk
szerint **azonos méret mellett >10% top-1 nyereség** a 2.0-hoz/versenytársakhoz
képest (KLD, Divergence-300@32 is jobb). A GGUF szabványos → llama.cpp-ben
fut. Az MTP modult a `UD-Q2_K_XL` alatti (kisebb) kvantokból kiveszik
(~500 MB spórolás), külön Q4_0 MTP modullal pótolható; a Q4_K_XL-ben marad.

**Elérhető-e most nálunk?**
- A `unsloth/Qwen3.5-9B-MTP-GGUF` repó utolsó módosítása **2026-05-16**, minden
  fájl `UD-*` = **2.0-s mix** (Q4_K 3,02 + Q5_K 0,85 + Q6_K 1,76 + Q8_0 0,46 GB,
  MXFP4 nélkül). Hivatalos 3.0-s kiadás eddig **Qwen3.8-27B**-hez van → a mi
  modellünkhöz még nincs kész v3 fájl.
- **DE**: a repóban van a modellhez tartozó **imatrix** (`imatrix_unsloth.gguf_file`,
  5,2 MB) → a v3-stílusú mixet magunk is előállíthatjuk.

**Recept (amikor lesz ~25 GB szabad hely):**

> **Figyelem (36. kör):** a regexes `--tensor-type` felülírás **nem
> reprodukálható** (a default Q4_K-mix belső számlálói eltolják, mely rétegek
> kapnak Q6_K-t). Helyette teljes, explicit tábla kell:
> `~/llama-bench-results/mix-gen.py` → `--tensor-type-file` (`quant-mix.sh`).
> Az alábbi parancs csak a szándékot mutatja.
```bash
hf download unsloth/Qwen3.5-9B-MTP-GGUF \
  Qwen3.5-9B-BF16.gguf imatrix_unsloth.gguf_file --local-dir ~/models/qwen35mtp
# v3-stílusú dinamikus mix: imatrix + tensor-szintű felülírások
llama-quantize --imatrix imatrix_unsloth.gguf_file \
  --tensor-type attn_v=Q6_K --tensor-type ffn_down=Q6_K \
  --tensor-type ssm_out=Q8_0 --output-tensor-type Q6_K \
  Qwen3.5-9B-BF16.gguf out-UD-Q4_K_XL-v3.gguf Q4_K
```
(`--tensor-type tensor_név=típus`, `--tensor-type-file` teljes táblához;
a pontos mixet a 2.1-es mérési eredmények + az Unsloth v3 arányai alapján kell
kalibrálni, és a **MTP fejet** külön, legalább Q4_0/Q5_K szinten kell tartani,
különben az elfogadási arány és így a tg esik.)

**Előnyök**: azonos méret mellett jobb minőség (ingyenes pontosság, futásidejű
költség nélkül); nyilvános imatrix → reprodukálható; szabványos GGUF.
**Hátrányok**: nincs kész v3 fájl a 9B-MTP-hez (magunknak kell kvantálni);
~25 GB szabad lemez (BF16 18 GB + kimenet 6 GB) és több óra futás/validálás;
a meglévő Q4_K_XL-ből **nem** szabad újrakvantálni (kétszeres veszteség) —
BF16/Q8_0 alap kell; a minőség-nyereséget a saját harness-ünkkel kell igazolni.

**Döntés**: a 2.1-es custom kvant-mix terv **A0 lépése** legyen ez a v3-stílusú,
Unsloth-imatrixos mix (a mostani A1–A4 mellé/elé), és figyeljük a hivatalos
v3 megjelenését a 9B-MTP-hez.
