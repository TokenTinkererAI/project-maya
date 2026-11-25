# Ceilings — Inkling-2-Mini UD-Q3_K_XL on RTX 3060 Ti (offload path)

Model: 26 B total. 24 layers, layers 6..17 looped 3x → n_visit = 48 expert-layer visits per token.
Experts: 128/layer, top-8 per visit. Total expert bytes ≈ 11.4 GiB (Q3_K_XL, pinned in RAM).
Tier: VRAM cache, currently 3.0 GiB / 802 slots / 4 pools (~26% of experts resident).

## Decode ceiling (memory-bandwidth bound)
Per-token expert bytes if EVERY pick missed the tier:
  48 visits x 8 experts x (11.4 GiB / 24 layers / 128 experts)  ≈ 1.43 GiB/token
  (per layer: 0.475 GiB → 3.72 MiB/expert → 29.8 MiB/visit → 1.43 GiB/token)

Effective PCIe 4.0 x16 for small async staged copies: ~15-18 GiB/s practical (measure: bytes_up / time from engine logs).

- 0% tier hits: 1.43 GiB / 17 GiB/s ≈ 12 tok/s ceiling.
- Measured server 15.0 tok/s (150-tok) → already >0% hits (prefetch+tier working).
- Measured 10.5 tok/s (741-tok, 1-tok prompt) → effective ~17 GB/s with misses dominating: PCIe-bound, tier hit-rate is THE lever.
- Each +10% of expert bytes served from tier ≈ +1.2 tok/s near this operating point.

Upper bound if ALL experts were VRAM-resident (needs 11.4 GiB, impossible on 8 GB): dense-path speed ~29 tok/s (Q4 CLI 40-tok run) — the practical aspiration: push toward 20 tok/s by maximising tier coverage + prefetch.

## Prefill ceiling (compute bound)
2 x active params ≈ 2 x ~3.7 B active/token (48 visits x 8 experts x ~30 MiB expert + dense) ≈ 7-8 GFLOP/token.
RTX 3060 Ti FP16 tensor peak ≈ 20 TFLOPS → ~1250 tok/s compute ceiling; measured 43.6 tok/s → prefill is fetch-bound on this card, not compute-bound. Secondary; gains come from the same tier hit-rate lever.

## Numbers to measure (fill in)
- [ ] PCIe effective: engine log bytes_up over a timed decode window
- [ ] Tier hit rate at baseline: from IK2_TRACE/STATS if exposed, else inferred from tok/s

## Levers ranked (ceiling gap x likelihood)
1. Tier size: `--cache-gb` (default 60 GiB → budget = min(cache_gb, free - room)). Free after dense+KV ≈ 5.4 GiB, room default IK2_ROOM_MB=1024 → current tier 3.0 GiB is free-limited, NOT cache_gb-limited. Raising tier requires lowering room or... verify: budget=min(60, fr-room). If fr≈4.0 GiB then tier 3.0. Lever = IK2_ROOM_MB ↓ (512) and/or trim other VRAM use.
2. IK2_WARM profile: placement of top experts in tier at load (already on; profile from Q4 usage — could regenerate for Q3/this workload).
3. IK2_PF_P prefetch threshold (default 0.999) / IK2_PREFETCH ring: hide miss latency behind decode.
4. IK2_STAGE (default 2*K=16 slots): staging ring depth for in-flight fetches.
5. Prefill chunking: engine-internal, no env knob found yet.

## Link probe (2026-11-22, E5 evidence)

Standalone pinned H2D probe on the idle GPU (bw_test.cu, sm_86): peak 512 MB 24.47 GiB/s; 3.8 MB chunks 24.29; 1.3 MB chunks 23.51; 256 KB 20.43. Multi-stream (s4) changed nothing at >=1.3 MB. Conclusion: chunk size is NOT the engine's problem; the 17 GiB/s effective rate is the per-visit issue-time serialization (copies gated on routing), not the link. Fixing it requires predicted prefetch (E5), not stream count or chunking.
