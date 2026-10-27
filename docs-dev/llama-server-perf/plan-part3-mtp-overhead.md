# llama.cpp gyorsítás — 3. RÉSZ: az MTP-lépés overheadje

Állapot: **terv** (2026-10-26). Előzmény: 36. kör — az MTP-lépés 146 GB/s-on
fut, az MTP nélküli decode 345 GB/s-on → a lépés nem byte-kötött. Ez a terv a
kódból (`common/speculative.cpp`, `tools/server/server-context.cpp`,
`src/llama-context.cpp`) azonosított konkrét overhead-forrásokat célozza.

## Mi történik egy MTP-lépésben (n_max = 2, qwen35)

| # | hívás | kontextus | batch | kimenet | gráf |
|---|---|---|---|---|---|
| 1 | verify (`llama_decode`) | tgt | 3 token | 3 | reuse |
| 2 | `process()` catch-up (`speculative.cpp:1656`) | dft | 3 token | **0** | **mindig újraépül** |
| 3 | `draft()` 1. lépés | dft | 1 token | 1 | **mindig újraépül** |
| 4 | `draft()` 2. lépés | dft | 1 token | 1 | reuse |

- Lépésenként **4 blokkoló `graph_compute`** + 3 CPU-oldali sampling/readback.
- A catch-up a verify-batch **mindhárom** tokenét dekódolja a dft KV-ba, a
  később elutasított draftokat is (az elfogadás ekkor még nem ismert).
- **Gráf-újraépítés:** `get_gf_res_prev()` (`llama-context.cpp:2610`) két
  slotot tart `n_outputs > 0` szerint, de reuse csak akkor van, ha a slot az
  utoljára aktív (`gf_res_prev_active == res`, mert a sched-allokáció közös).
  A 0-kimenetes catch-up és az 1-kimenetes draft váltakozása miatt a dft
  kontextus **lépésenként 2× újraépít** (sched reset + alloc + Vulkan
  gráf-optimalizálás / CUDA graph újracapture).
- Becslés (rövid ctx, Vulkan, 250 W): verify ~19–22 ms, 2 draft-lépés
  byte-alapon ~6 ms, catch-up ~1 ms → ~28–30 ms várható a mért **49,3 ms**
  helyett. A ~20 ms-os rés a fő célpont. (Becslés, mérni kell — G0.)

## Mérési hiba a meglévő instrumentációban — JAVÍTVA (38. kör)

`LLAMA_GRAPH_TIMING` (`llama-context.cpp`, lokális patch): a
`static int64_t n_graph` **függvény-statikus, tehát a tgt és a dft kontextus
közös számlálója**, és csak minden 16. hívást írja ki. MTP-vel a ciklus
pontosan 4 gráf/lépés → a 16-os mintavétel **mindig ugyanazt a pozíciót**
(itt: az újraépülő catch-up gráfot) látja. A 36. kör megfigyelése („csak
`built=1` sorokat ír") ennek az aliasingnak tűnik, nem a reuse hiányának
általános jele. Javítás: kontextusonkénti számláló, minden hívás naplózása
(vagy aggregált statisztika `(ctx_type, n_tokens, n_outputs, built)` szerint),
`n_outputs` kiírása. Ugyanez a `synchronize()`-beli `n_sync % 16`-ra.

## Első eredmény (38. kör, előzetes)

Az instrumentáció-javítás kész. Rövid füstteszt (Vulkan, 170 W, 300 token):
lépés ~44,8 ms, ebből a gráfokban ~23 ms (verify 15,0, 2 draft 2×2,1,
catch-up 2,4, sync ~1,5) → **~22 ms a gráfokon kívül**. A catch-up és az 1.
draft-gráf minden lépésben újraépül (22/22, 22/42). → A G0/2 (szerver-oldali
időzítő) a legsürgősebb; a G1 a GPU-oldali részt (~3 ms) célozza.

## 39. kör — a felbontás eredménye (felülírja a fenti „~22 ms a gráfokon kívül” olvasatot)

- A rés **GPU-idő** volt (a Vulkan `graph_compute` nem blokkol végig).
- Lépés (alap): verify ~32 ms (GPU 30,6), draft ~8 ms (GPU ~4,6),
  catch-up ~3 ms (GPU ~1,3), szerver ~1 ms.
- Fő ok: a verify n=3 k-quant matvec 2,6× lassabb az n=1-nél → új
  **G5: `GGML_VK_KQ_BATCH_RM=4`** — verify −14%, t/s +10% (results.md 39. kör).
- **G5 45k-n igazolva** (lépés −7,6%, tg +9,2%, results.md 39b) → napi parancsban.
- Új prioritás: G1 (draft/catch-up CPU-oldal, ~4–5 ms)
  → G2 → G3. CUDA MMVQ n=3 ellenőrzése.

## G0 — lépés-felbontás (instrumentáció, kód ~40 sor, env-gated)

1. ~~A `LLAMA_GRAPH_TIMING` javítása + a dft `n_reused` kiírása~~ — **KÉSZ** (38. kör).
2. ~~Szerver-oldali lépés-időzítő~~ — **KÉSZ** (`LLAMA_SERVER_TIMING`, 39. kör): verify decode,
   `common_speculative_process`, `common_speculative_draft`, verify-sampling +
   accept, `yield_to_queue` várakozás — ms/lépés átlag 64 lépésenként.
3. `GGML_VK_PERF_LOGGER=1` egy rövid MTP-futásra: a draft-gráf `TOP_K`
   (248k vocab), `GET_ROWS`, a head `MUL_MAT` és az FA ideje.
4. Ugyanez CUDA-n (`GGML_CUDA_DISABLE_GRAPHS=1` vs alap: a graph-recapture ára).

**Kimenet:** a 49,3 ms felbontása → ez dönti el a G1–G4 sorrendjét.

## G1 — catch-up beolvasztása a draft 1. lépésébe (kód, `speculative.cpp`)

**Ötlet:** a `process()` generálási batchnél ne dekódoljon, csak jegyezze fel
a (token, pos, h-sor) hármasokat; az `accept()` vágja le az elfogadott
prefixre; a `draft()` 1. lépése egy batchben küldje: `[elfogadott catch-up
tokenek (kimenet nélkül)] + [id_last (kimenettel)]`. Prefillnél (több ubatch)
marad az azonnali catch-up.

- **−1 `graph_compute` + 1 szinkron lépésenként**, és nem kerülnek elutasított
  tokenek a dft KV-ba.
- A „0-kimenetes" gráf eltűnik a dft kontextusból → a slot-váltás miatti
  egyik újraépítés is.
- Upstream is ismert TODO: `[TAG_SPEC_AVOID_DRAFT_REEVAL]`
  (`server-context.cpp:4619`).
- Figyelni: a szerver dft-oldali `seq_rm`-je a már nem létező pozíciókra
  (ártalmatlan kell legyen), a `pending_h` kezelése, `n_seq > 1`.
- Helyesség: azonos prompt + temp 0 → **bitre azonos draftok és acceptance**
  a régi úttal (a draft-tokenek nem változhatnak, csak a sorrend/batching).
- Várható: −3…−6 ms/lépés (**+6…+12% tg**), minden backenden.

## G2 — mohó draft-sampler a backend top-k(10) helyett (kód, ~10 sor)

A draft-lánc a backenden `top_k(10)`-et futtat a teljes 248k-s vocabon
(`speculative.cpp:1509`), de mohó draftolásnál (`!params.probabilistic`) csak
`cur_p->data[0]` kell. `llama_sampler_init_greedy()` → `ggml_argmax`
(egymenetes redukció) a top-k helyett, lépésenként 2×. A `p_min` ág 0-s
küszöbbel nem használja a `p`-t, de ellenőrizni kell.

- Olcsó kísérlet; a G0/3 `TOP_K` ideje mondja meg, mennyit ér.
- Helyesség: azonos draftok (az argmax = top-k első eleme).

## G3 — csökkentett vocab a draft headhez (FR-Spec minta; a 2.4 helyett)

A head 3× olvasódik lépésenként (2,50 GB, a forgalom 35%-a); ebből 2× a
draft. A draftnak elég a leggyakoribb ~32k token sora (FR-Spec: a vocab
~13%-a fedi a tokenek túlnyomó részét).

- A beágyazott `output.weight` sorai összefüggőek → a gyakori sorok
  bájtmásolással kivághatók egy új tenzorba (Q6_K is), +~110 MB VRAM.
- A visszaképzés mintája megvan: EAGLE-3 `d2t` (`src/models/eagle3.cpp:307`).
- Gyakorisági lista: a saját korpusz + általános HU/EN szöveg tokenizálva
  (`/tokenize`), vagy első közelítésként a BPE-sorrend.
- Ára: kicsit kisebb acceptance (ha a target token kívül esik a 32k-n, a draft
  biztosan elhibázza). Mérni: **ms/lépés + acceptance**, ≥2000 token.
- Várható: −1,45 GB/lépés; a hatás attól függ, mennyire byte-kötött a draft
  (G0/3) — ha a draft is overhead-kötött, ez keveset hoz. **G0 után döntsünk.**
- Kódméret: közepes (modell-betöltés + qwen35 MTP-gráf + id-visszaképzés).

## G4 — a dft gráf-újraépítés ára (csak ha G0 szerint jelentős)

Ha G1 után is marad váltakozás (az 1. lépés batch-mérete az elfogadástól
függően 2–4 token, a 2. lépésé 1), és a G0 szerint az újraépítés drága
(CUDA graph recapture, Vulkan graph_optimize), akkor: az 1. lépés
fix 1+n_max tokenre paddingolása (maszkolt, kimenet nélküli dummy sorokkal)
→ állandó gráf-topológia. Bonyolultabb, ezért csak mérés után.

## Mérési protokoll

- **170 W ellenőrzése minden mérés előtt** (36. kör: 250 W-on
  futottunk, 60k-n thermál-throttling; 2026-10-26 este ismét 170 W él).
- Rövid kontextus (2,8k) a stabil A/B-hez, 41k az igazoláshoz;
  **ms/lépés + acceptance**, nem t/s. Interleaved A/B, 60 s cooldown, első
  2 futás eldobva (bemelegedési szabály).
- Minden kódváltozás env-kapcsolóval (régi út visszahívható), a megszokott
  lib-izolációval (`LD_LIBRARY_PATH`).

## Sorrend

G0 (instrumentáció + a timing-aliasing javítása) → G2 (olcsó) → G1 → G0
újra → G3 / G4 a felbontás alapján.

## 40. kör után — állapot és következő lépések

- G1 kész (env, ~1%, nem éri meg), G2 elvetve, G5 Vulkanon env, CUDA-n
  (rows_per_block 4) alapértelmezett.
- **G6 (új, Vulkan, legnagyobb maradék):** FA több tokenre a GQA-ágon —
  a `ggml_vk_flash_attn` `N <= 8` ágán a token×fej sorok egy workgroupba,
  így a K/V egyszer olvasódik. Shader: `flash_attn_base.glsl` (`init_indices`,
  `gqaStore`), `flash_attn.comp`, `flash_attn_cm1.comp`, `flash_attn_cm2.comp`
  (Q/maszk 3D címzés, kimenet `(t, h)` szerint). Várható: 45k-n ~7 ms/verify
  (Vulkan lépés −13%). Teszt: `test-backend-ops -o FLASH_ATTN_EXT`.
- **G7 (CUDA):** a draft-lépés ~6 ms (2 draft, a fej 2× ~2 ms) — a G3
  (szűkített vocabú draft head) itt a legnagyobb tétel.
