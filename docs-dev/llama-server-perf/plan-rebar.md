# ReBAR (Resizable BAR) — DRAFT (rev. 3, with R720 + Turing facts)

Status: **plan only**; we do not touch UEFI. Date: 2026-10-21.

## Architecture (measured / facts)

- **Host:** Dell PowerEdge R720 (12G, C600, Xeon E5) — Proxmox.
- **Guest:** QEMU/KVM (DMI: Q35/ICH9), Ubuntu, GPU VFIO passthrough.
- **GPU:** NVIDIA TU102 RTX 2080 Ti, BAR1 = **256 MiB**, `nvidia-smi` BAR1 Used = **11 MiB**.
- PCIe: Gen3 x8; an earlier measurement showed it is not a bottleneck.

## "Can ReBAR be enabled at the kernel level?" — No (on Turing)

How ReBAR works on Turing (NvStrapsReBAR, based on the envytools documentation):
- The BAR size is selected by the **GPU strap registers** **during PCIe enumeration, in UEFI**. A DXE driver programs them before the OS starts.
- The vBIOS is signed and encrypted, cannot be modified, and **does not advertise** the larger sizes.

Consequences:
- Linux `resourceN_resize` / `pci_resize_resource` can only choose among the **advertised** sizes, so on Turing it is limited to 256 MiB.
- The NvStrapsReBAR README says the "no UEFI modification" ideas *"may not be technically possible and nothing is implemented"*.
- On **Ampere/Ada** (RTX 30/40), where the vBIOS advertises the sizes, host-side kernel/sysfs resizing (or the `nvidia-rebar` module) can work, provided Above 4G Decoding is enabled.
- **VFIO guest:** QEMU exposes a static ReBAR capability. The guest cannot resize; the host fixes the size when the VM starts.

## R720 specifics

- The BIOS **does** have a "Memory Mapped IO > 4GB" option, according to Dell community posts. Some BIOS versions have shown buggy behaviour.
  So the Above 4G prerequisite is in principle met.
- Turing still needs a **UEFI DXE mod** (NvStrapsReBar / ReBarUEFI): modifying and flashing the UEFI image. This is **ruled out** by our constraints, and it is especially risky on a Dell server BIOS.
- Conclusion: **with R720 + 2080 Ti + Proxmox, ReBAR is not reachable without modifying UEFI.**

## Common misconceptions (claims circulating online)

1. **"ReBAR reduces the VRAM needs of AI models" — FALSE.**
   - VRAM allocation depends on the model, the KV cache and the allocator. ReBAR enlarges the BAR window the CPU can see; it does not change VRAM use.
   - The "less VRAM" seen on a B550 + 3090 came from other factors: driver version, display allocation, allocator settings or a different context.
2. **"Without ReBAR, CUDA uses bounce buffers" — FALSE.**
   The GPU accesses device-local memory directly; BAR1 is only needed for CPU mappings. llama.cpp uses 11 MiB of BAR1.
3. **"The R720/R730 do not support Above 4G" — INACCURATE.**
   The R720 BIOS has the "Memory Mapped IO > 4GB" option, and 13G/14G servers support it too. The missing ReBAR comes from the GPU vBIOS, not from 4G decoding.
4. **"pci=realloc / a kernel module solves it"** — only where the card advertises the larger sizes (Ampere+); not on Turing.
5. **"The x-vga / rombar switches speed things up"** — these are not ReBAR. rombar sets the option ROM BAR, not the VRAM window.

## Decision

- **Not applicable** in the current environment without modifying UEFI.
- **Not worth it either:** the expected gain for our workload is about zero.
  BAR1 Used is 11 MiB, the link is not a bottleneck, and the weights and KV are device-local.
- If it is ever needed:
  - On Ampere/Ada, host-side resizing works with Above 4G enabled: host sysfs, stop the VM → unbind vfio-pci → resize → rebind → start the VM.
  - Turing would need a UEFI DXE mod, which is ruled out.
