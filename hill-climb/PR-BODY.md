Title: `maya-ik2: 8 more expert block types, CPU expert lane, offload deadlock fix`

## Summary

Reads the Inkling-2-Mini GGUFs the engine refused before on an 8 GB card on Windows (verified: unsloth UD-Q3_K_XL, UD-Q4_K_XL, UD-Q4_K_M — not every quant in the family) and speeds the offload path up 16.0 → 29.7 tok/s decode (UD-Q3_K_XL, RTX 3060 Ti, 131K context). Every new behavior is env-gated and default-off; with the env unset the binary is byte-identical (40-token greedy, `fc /b`).

```text
src/ik2/
├── ik2_kernels.cu/.cuh   +8 GGUF block types (Q5_0 Q3_K Q4_K IQ2_XXS IQ2_XS
|                          IQ4_NL IQ3_S IQ2_S), DS4-IQ ports + fresh Q5_0,
|                          mixed-type dense fallbacks in the fast paths
├── ik2_main.cu           ExpertCache pool floor 8 (no-slot-free deadlock fix,
|                          Q4_K_XL@131K), CPU-lane split (IK2_CPU_FRAC),
|                          IK2_CPU_BENCH, offload drafting (IK2_DRAFT_OFF),
|                          GGUF-drafter stats
├── ik2_draft.cu/.hpp     --draft <file>.gguf: quantized drafters load as
|                          blocks, no dequant at load (dspark Q3_K: 809 → 437 MiB)
├── ik2_off.cu/.cuh       df_prev same-visit predictor hooks (IK2_DF, default off)
├── ik2_cpu.cpp/.hpp      native CPU expert lane over pinned host memory
src/kernels/cpu/          ik2_rows.hpp/.cpp (AVX2 row-dot bench hooks, IK2_CPU_BENCH)
include/strata/kernels/cpu/native_expert.hpp   + src/kernels/cpu/native_expert.cpp
tests/ik2_kernels_test.cu  + entries for every new type
CMakeLists.txt             new sources, /arch:AVX2
```

- **Block types**: UD-Q3_K_XL, UD-Q4_K_XL and UD-Q4_K_M now load (previously `type X not read here`). Device decode bit-exact vs the ggml reference; host = ggml reference at 0.00 tolerance on 85k values/type; `ik2_kernels_test` all passed. Greedy 40-token output identical between UD-Q3_K_XL and UD-Q4_K_XL on the same binary.
- **Offload fix**: `ExpertCache::init` keeps a floor of 8 slots per pool. A starved pool (Q4_K_XL at 131K: one slot shared by four layers) deadlocked with `expert cache: no slot free` as soon as one visit pinned more slots than the pool had; lazy retry got more passes.
- **CPU expert lane** (`IK2_CPU_FRAC`, default off): a share of each visit's missed experts is computed on the CPU (ggml-cpu AVX2 dots over pinned host memory) while the rest DMA in — two buses in parallel. Adaptive split settles at the cap on Q3-heavy mixes.
- **Offload drafting** (`IK2_DRAFT_OFF`, default off) + **GGUF drafters**: `--draft <file>.gguf` keeps quantized tensors as blocks on device. Included for completeness; on the offload path drafting stays unprofitable (verify cost is target-side, acceptance ~0.3), so the levers default off.

## Evidence

- **Before:** UD-Q3_K_XL refuses to load — `type 11 (Q3_K) not read here`. UD-Q4_K_XL at 131K: `expert cache: no slot free` after prefill.
  **After:** UD-Q3_K_XL: decode 16.02 → 29.7 tok/s median (6× --gen 256, control/candidate/control, ±1.3% noise; end-user API 27-28 tok/s), prefill unchanged 42-44. UD-Q4_K_XL: 131K clean, 27.4 tok/s. `ik2_kernels_test`: all passed. Default-off: byte-identical greedy vs pre-change binary for every env gate (`fc /b`).
- Offload tuning context (env-only, no code): `IK2_ROOM_MB`, regenerated `IK2_WARM` profile, host-DMA decode path — full ledger in `hill-climb/PLAN.md` if wanted.

## Merge Danger

**Door:** two-way — every feature is env-gated, default off, byte-identical when unset; revert is safe.

**Blast Radius:** engine-wide-offload

- Pool floor 8 changes `ExpertCache` sizing slightly on every offload layout (budget guard dies early if floors can't fit; Q3_K_XL sizing unchanged byte-for-byte per gates).
- `ik2_main.cu` heavily conflicts with the 1.1.74 experts-out-before-the-cache rework (+1827 upstream lines in the same regions); this branch is based on 1.1.73.
