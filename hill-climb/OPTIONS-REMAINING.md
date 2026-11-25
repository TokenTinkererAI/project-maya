# Remaining levers — exploration only (no tests applied)

Mapped against `inference-optimisation` skill taxonomy (levers/ceilings/gotchas) and the full
hill-climb record in PLAN.md + ledger.jsonl. Nothing below has been tested in this round.

## Where the machine stands (measured facts)

| metric | now | practical ceiling | gap |
|---|---|---|---|
| Windows decode c1 @131K | ~26.8-27.1 tok/s | ~36 (two-bus: PCIe 24.5 GiB/s + CPU lane 9.78 GB/s) | ~75% of ceiling |
| per-visit serial spine | routing/D2H ~0.38 ms + orchestration ~0.06 ms + lane wait ~0.12-0.19 ms | fetch would have to be free to see 36 | — |
| Windows KV | fp8 2.23 GiB; K8V4 option 1.75 GiB (default off) | 4-bit proven lossy at gen length | closed |
| lab decode | 18.7-18.9 @131K (frozen) | hardware floor | closed |

Ceiling note (skill formula): decode bytes/token now ≈ 580 MiB PCIe + 310 MiB CPU-lane.
The serial spine (route+dense ~18.7 ms/token) is nearly half the 37 ms token — the engine is
~50% serial GPU compute, ~50% overlapped fetch. That reframes the remaining headroom.

## A. Config-only, unexplored on WINDOWS (cheap, bit-exact)

1. **Headroom reserve sweep (IK2_ROOM_MB 64 vs 128/256 on Windows).** Skill gotcha: keep
   ≥1.5 GB free; we run with 0 free VRAM and have hit cudaHostAlloc OOM classes. Lab said
   room-128/256 ≈ noise; the Windows desktop-starved context may differ. One A/B, 12× medians.
2. **Graph-path re-test post-all-levers.** Skill ranks decode graphs "#1 lever" (2× on other
   engines). Our E7 measured graph vs host-DMA at +21% for host — but pre-CPU-lane. Design
   conflict: CPU_FRAC>0 forces the host path (E6 gate). A graph-with-lane design is the
   launched-block/captured-visit project (priced 2.7% Windows ceiling) — not a config flip.
   Low value; listed for completeness.

## B. Model/quant swaps (labelled lossy, one A/B each)

3. **UD-Q4_K_M on WINDOWS.** The sha-verified 16.6 GB file is on disk, all-supported types.
   Skill prior: "pick a quant the hardware decodes natively — on Ampere AWQ-INT4 beat FP8" —
   our NVFP4-heavy experts decode in *software* on sm_86 (byte_perm LUT). Q4_K/Q5_K blocks are
   GGML-native. The lab said −4% (Pascal, no native block advantage); the Ampere case was never
   benched. One Windows A/B (12× medians) decides.
4. **Expert-only native-block requant** (clean version of 3): Q4_K/Q5_K experts, rest unchanged.
5. **LM head precision** (Q6_K → Q5_K/Q4): the head is ~0.4 GiB of resident VRAM and ~2 ms/token
   (maxwell_sgemm 6% + gemv 3% of GPU time). Lowering it frees ~0.1-0.2 GiB for the tier and cuts
   its bytes — a labelled lossy trade on the final projection (top-1 impact measurable with the
   `--score` instrument).
6. **LM head vocab pruning** — only viable for constrained/agent decoding; lossy on open text.

## C. Engine work (bit-exact or labelled)

7. **int8/fp8 prefill GEMMs (labelled lossy).** Skill evidence: "all linears 1.89× faster,
   NLL +0.17%, top-1 97.2%, decode bit-exact — prefill is compute-bound so this pays."
   Our prefill is compute-bound at length (lab) and 781 tok/s at 102K (Windows). Untried.
8. **Grouped MoE padding cliffs / 64-row M-block.** Skill evidence (bit-exact): "A 64-row M-block,
   which halves weight decode per row: +7% end-to-end prefill." Our chunked prefill uses grouped
   kernels — the padding-cliff audit was never run.
9. **Captured visit / launch fusion.** Priced at 29-32 µs/visit × 48 ≈ 2.7% on Windows (the only
   box where it pays). Weeks of work, kernel-count evidence from the community RDNA4 delta (+74%
   from fusion at 8,000 → 2,138 kernels/token) suggests the upside may exceed our price — a
   re-price with the ExactLoop microbench method could justify it.
10. **Linux on this box.** System explorer measured +20-30% for H2D-heavy MoE offload (WDDM tax);
    skill gotchas agree Windows host overhead is real. Heavy: dual-boot/WSL2. The lab exists but
    is a different GPU class.
11. **LM head custom GEMV (ISA).** Skill: a planar-register rewrite took an LM head 4.05 → 1.72 ms.
    Ours runs cuBLAS maxwell_sgemm; a dedicated GEMV could shave part of the ~2 ms/token.

## D. Hardware (ceiling breakers)

12. **12-16 GB card** — tier triples/quadruples; Strata-fork scaling predicts ~25 tok/s+ on the
    lab class, more on Ampere. The only lever that changes the ceiling itself.
13. RAM OC: XMP already on; DDR4-3600 marginal.

## Skill-checklist gaps (not levers, but owed)

- Measure real device-copy bandwidth + large GEMM TFLOPS on the 3060 Ti (skill: "measure, don't
  trust spec sheets") — our ceilings use PCIe/DDR4 numbers, not a direct GEMM measurement.
- Needle at ~100% context depth (we did 51K; skill asks 50% *and* ~100%).
- Concurrent full-length soak: N/A (single user).

## Closed — do not revisit without new evidence

Drafting (acceptance 0.30-0.36), prefetch/prediction (WiSP + our own), early-exit loops,
skip-tail (lossy ~10%), K4V4 & K8V4 quality (all forms fail token-exactness; K8V4 kept as
option), i8r KV (fp8 already smaller), 8-warp tiling (falsified both arches), pool2 (−4.3%),
DPACK (−36% lab), DCBATCH (no-op), chunk-4096-for-live (−12.5% decode), launch-block
as-priced (2.7%), HAGS-on already, downclock moot (clocks full), XMP on, TCC/RBAR dead.
