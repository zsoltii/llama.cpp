# ReBAR (Resizable BAR) — TERVEZET (rev. 3, R720 + Turing tényekkel)

Állapot: **csak terv**; UEFI-hez nem nyúlunk. Dátum: 2026-10-21.

## Architektúra (mért/tények)

- **Host:** Dell PowerEdge R720 (12G, C600, Xeon E5) — Proxmox.
- **Guest:** QEMU/KVM (DMI: Q35/ICH9), Ubuntu, GPU VFIO passthrough.
- **GPU:** NVIDIA TU102 RTX 2080 Ti, BAR1 = **256 MiB**, `nvidia-smi` BAR1 Used = **11 MiB**.
- PCIe: Gen3 x8; korábbi mérés szerint nem szűk keresztmetszet.

## „Kernel szinten bekapcsolható-e a ReBAR?" — Nem (Turingon)

A Turing ReBAR mechanizmusa (NvStrapsReBAR, envytools dokumentáció alapján):
a BAR méretét a **GPU strap-regiszterei** választják ki a **PCIe felsorolás
közben, UEFI-ben** (DXE driver programozza, mielőtt az OS elindul). A vBIOS
aláírt/titkosított, nem módosítható, és **nem hirdeti** a nagyobb méreteket.

- A Linux `resourceN_resize` / `pci_resize_resource` csak a **hirdetett**
  méretek közül tud választani → Turingon ez a 256 MiB-ra korlátozott.
- A NvStrapsReBAR README szerint a „UEFI-módosítás nélküli" ötletek
  *„may not be technically possible and nothing is implemented"*.
- **Ampere/Ada** (RTX 30/40) kártyáknál, ahol a vBIOS hirdeti a méreteket,
  a host-oldali kernel/sysfs (vagy `nvidia-rebar` modul) működhet —
  Above 4G Decoding mellett.
- **VFIO guest:** a QEMU statikus ReBAR capability-t mutat, a vendég nem
  méretezhet (a host rögzíti a VM indításakor).

## R720 konkrétumok

- A BIOS-ban **van** „Memory Mapped IO > 4GB" opció (Dell közösségi
  bejegyzések; bizonyos BIOS-verzióknál hibás viselkedés előfordult).
  Ez az Above 4G előfeltétel tehát elvben adott.
- A Turinghoz mégis **UEFI DXE mod** kell (NvStrapsReBar / ReBarUEFI):
  a UEFI kép módosítása + flash → **kizárva** a mi korlátunk miatt
  (Dell szerver BIOS-nál különösen kockázatos).
- Következtetés: **R720 + 2080 Ti + Proxmox mellett ReBAR nem érhető el
  UEFI-módosítás nélkül.**

## Tévhitek pontosítása (a neten terjedő állítások)

1. **„ReBAR csökkenti az AI-modell VRAM-igényét" — HAMIS.**
   A VRAM-foglalás a modell/KV/allocator függvénye. A ReBAR a CPU által
   látható BAR-ablakot növeli, nem a VRAM-felhasználást. A B550+3090-en
   tapasztalt „kevesebb VRAM" más tényező (driver-verzió, display-foglalás,
   allocator-beállítás, más kontextus) következménye.
2. **„ReBAR nélkül a CUDA bounce buffereket használ" — HAMIS.**
   A device-local memóriát a GPU közvetlenül éri el; a BAR1 csak a CPU-s
   leképezésekhez kell. A llama.cpp-nél a BAR1-használat 11 MiB.
3. **„Az R720/R730 nem támogatja az Above 4G-t" — PONTATLAN.**
   Az R720 BIOS-ában ott a „Memory Mapped IO > 4GB" opció; a 13G/14G is
   támogatja. A ReBAR hiánya a GPU vBIOS-a miatt van, nem a 4G miatt.
4. **„pci=realloc / kernel-modul megoldja"** — csak ott, ahol a kártya
   hirdeti a nagyobb méreteket (Ampere+); Turingon nem.
5. **„x-vga / rombar kapcsolók gyorsítanak"** — nem ReBAR; a rombar az
   opciós ROM BAR-ját állítja, nem a VRAM-ablakot.

## Döntés

- **Nem alkalmazható** a mostani környezetben UEFI-módosítás nélkül.
- **Nem is érdemes**: a várható nyereség a mi workloadunkra ~0
  (BAR1 Used 11 MiB, a link nem szűk; a súlyok/KV device-local).
- Ha valaha mégis kell: a host-oldali resize csak Ampere/Ada kártyával
  (host sysfs, VM stop → vfio-pci unbind → resize → rebind → VM indítás),
  Above 4G mellett; Turinghoz UEFI DXE mod kellene (kizárva).
