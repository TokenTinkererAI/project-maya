# Inkling-2-Mini UD-Q3_K_XL hill-climb — RTX 3060 Ti 8 GB

## Lab port (GTX 1070, Pascal sm_61, 2026-11-22)

Port via STRATA_IK2_PASCAL: ik2_kernels.cu :924 guard `#if defined(STRATA_USE_HIP) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700)`, mma884 shuffle+fmaf emulation :1082-1152, ik2_off.cu lane_pause spin (no __nanosleep on Pascal), ik2_main.cu :4685 device admission under MAYA_IK2_PASCAL, CMakeLists :899 option (default OFF). CUDA 12.8 at /opt/cuda-12.8 + g++-14 (13.4 can't target Pascal). ik2_kernels_test ALL PASSED on the 1070 (emulated mma, attn_flash skipped by design).

Bench env: ~/maya, model /home/lab/lab/inkling/models/Inkling-2-Mini-UD-Q3_K_XL.gguf, ctx 16384, IK2_KV=fp8, WARM=natural profile, ROOM_MB=64, bench tokens as Windows. Build: cmake -DSTRATA_ENABLE_CUDA=ON -DSTRATA_IK2_PASCAL=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_COMPILER=/opt/cuda-12.8/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14.

### Lane-on result (18 replicates, bench2 ctrl arms contaminated — env var leaked into all arms, so all 18 runs were IK2_CPU_FRAC=0.5): generation 18.31-18.49 → median 18.40 tok/s ±0.5%; tier 1201 slots / 4.6 GiB; cache ~87k hits / ~11k misses (~88% tier hit at 16K); lane split 0.35-0.50 of misses, 75 MiB/token offloaded, 0.32 ms/visit; prompt 22.7-23.2 tok/s. vs abandoned llama.cpp attempt: 12.3 t/s @16K → +49%. No slot deaths, coherent, 18 clean exits. TRUE lane-off control pending (bench-nolane, env -u).

## E11 RESULT

### E11 width-long probe (2026-11-22, user question: dspark uses longer lengths)

Widths 5/6/7 with the Q3_K GGUF drafter, 16K ctx, live decode env + IK2_DRAFT_OFF=1, 2× --gen 256 each (log e11-width-long.log): w5 12.73/13.51 tok/s (verify 142-143 ms), w6 11.81/12.36 (160-161 ms), w7 10.95/10.74 (177-181 ms) — monotonic decline. Acceptance by position (w5 histogram 59 34 21 11 5 0 0 0): positions 5+ accepted ~5 of ~125 windows; tok/step flat ~2.0 at all widths; each extra position ≈ +18-20 ms verify for ~zero return. Optimum stays width 2 (17.0-17.4, E10 probe), 2× under plain decode 33.7. DSpark's long-window design assumes a RESIDENT target (verify = compute only); on the offload path every position pays full PCIe. Engine's adaptive SpecCtl (decay 0.85) already converges toward short windows — E11 default-mode 15.4 vs pinned w5 ~13. Long-window lever CLOSED. (2026-11-22): quantized drafter capability — DONE (user request; economics unchanged)

User asked for the capability despite the proven negative (@E10/@b33@: drafting 2.3× slower on offload, acceptance ~0.30).

- **Q3_K dspark GGUF built**: llama-quantize (vendored third_party/llama.cpp @3cf03257f, `tools/quantize/main.cpp`, CPU build) quantized the sha-verified unsloth BF16 dspark GGUF (606.4 MiB, 16 BPW) → `C:\LLM models\Maya-data\models\inkling-2-mini\dspark-Inkling-2-Mini-Q3_K.gguf` (142.6 MiB weights, 3.76 BPW; 1 F16 fallback tensor `dspark_fuse` [4,4]). Tensor mix: Q3_K×21, Q4_K×7, Q5_K×2, F32×23 (norms/biases/mask), F16×1; embeddings + LM head remain borrowed from target. Identity vs mtp.safetensors verified by content (norms max abs diff 0.0 under correct name pairing).
- **Engine GGUF-drafter support** (ik2_draft.cu): gg_open header reader :169, WMat blocks-or-BF16 :283, gemm_q :448 (activation widened to FP16 once per 512-row chunk → k::decode_f16 → existing FP16 GEMM; weights NEVER dequantized on device — blocks stay blocks), load_gguf :559 (GGUF names, gate+up concat gate-first, markov_w1 decoded once to BF16 for rnn_inputs row gathers), dispatch :730 (*.gguf → load_gguf); ik2_main.cu :5667 ON-only stats line. Safetensors path untouched.
- **Gates**: BF16 regression 40-token greedy byte-identical prefix vs pre-change binary (fc, mtp.safetensors, IK2_DRAFT_OFF=1, ctx 16384) ✔; Release rebuilt + re-copied (Release→build gotcha); pre-df backup untouched.
- **Numbers (ctx 16384, IK2_DRAFT_OFF=1, gen 256)**: Q3_K drafter 437 MiB resident (BF16 809) → tier 594 slots/2.3 GiB (BF16 499/2.0); drafting 15.38 tok/s (BF16 14.27); acceptance/drafted token 0.363 (BF16 0.353 — no material loss, slightly better within noise); coherent output; no slot deaths/OOM. Orchestrator spot (gen 64): loads, drafts, exits 0 — warm-in dominated short run as expected.
- **Verdict**: capability works — quantized drafter is smaller AND slightly faster while drafting, and the freed 372 MiB buys 95 tier slots. Drafting still loses to plain decode (~33.7) on the offload path; unchanged conclusion, now with the option usable on any GGUF drafter. Live config unchanged (no --draft).

## v1.1.74 comparison (2026-11-22): stock loses on this PC; our tree stays best

Cloned to `C:\LLM models\project-maya-new` (v1.1.74, 861e211/e8d4f29), built stock Release (BUILD_EXIT 0). Headline upstream changes: experts-out-before-cache + KV lending at long ctx; own offload CPU lane with 'decode mode: lane parity' auto-split; Windows officially supported; Answers at once (resident). Q3_K_XL still unsupported upstream (our kernels remain ours alone).

Q4_K_XL@131K bench, 6× --gen 256, log newvsours.log:
- A stock defaults: FAILS AT LOAD (experts-not-held could not be pinned; no bench ran).
- B stock + our env knobs (KV fp8, WARM natural, ROOM 64, POP_OUT live): tier 439/1.8 GiB, prefill 38 tok/s, then `expert cache: no slot free` EVERY run — the E9 pool-floor bug (pmin=0, one-slot pool) is still upstream. Lane parity measured 495 us/visit with lane vs 814 without, tier 49% of profile picks — never got to decode.
- C ours full env: 27.4 tok/s median (27.11-27.61). D ours no-frac: 20.5.
- Verdict: our tuned v1.1.73 tree wins by default; stock 1.74 cannot complete one decode here. Our patch saved at hill-climb\our-v1173-tuned.patch (12 files, +1614).
- Open question: upstream's lane-parity system + KV lending may beat our E6/E9 once the pool-floor fix is applied — port direction (our fixes onto 1.74) pending user decision.

### Fixed-stock re-bench (2026-11-22, pool-floor fix ported to the clone; log new-fixed-q4.log)

Minimal fix ported by orchestrator to `project-maya-new\src\ik2\ik2_main.cu` (13-line delta): kPoolFloor=8 in EC::init (`pmin[p]=min(max(min_base,8),pmax[p])` + budget-guard die) and lazy retry 2→6. BUILD_EXIT=0.

Arm E — fixed stock, arm-B env exactly (no IK2_CPU_FRAC; upstream lane parity left to itself), Q4_K_XL@131K, 6× --gen 256:
- Decode 19.73/19.98/20.06/20.02/19.89/19.94 → **median 19.98 tok/s** (±0.8% tight, no slot deaths — fix holds in anger).
- Lane parity engaged: 10 avx2 threads, expert 155/206 µs CPU vs 172/324 over bus; calibration printed ~495 µs/visit with lane vs ~817 without; tier 433/1.8 GiB at 49% profile coverage; decode = CUDA-graph path ('resident path's launches, a CUDA graph each'). Prefill 37.5-38.0 tok/s.
- vs our tree arm C (same model/env semantics + IK2_CPU_FRAC 0.5): **27.4 tok/s → our v1.1.73 tree wins by +37%** even against FIXED 1.74 machinery.
- Verdict: upstream's lane-parity + graph decode loses to our host-DMA + E9 adaptive CPU lane on this 8 GB card (their per-visit lane estimate ~495 µs vs actual ~1.04 ms/visit at 20 tok/s). Port direction CLOSED: keep our tree as the working engine; `project-maya-new` stays as a reference clone with the pool-floor fix in its tree. Revisit only if a future release changes the graph/laneparity architecture or reworks EC::init upstream.

## PREFILL PHASE (2026-11-22, opened after decode settled at ~29.7)

Targets: prefill tok/s at real prompt lengths; short-prompt TTFT secondary; decode env FIXED (IK2_UNSUPPORTED=1, IK2_KV=fp8, WARM natural profile, ROOM 64, POP_OUT live.bin, CPU_FRAC 0.5); 131072 ctx stays.

### Baselines (6× --gen 8, fresh process each; make_prefill_prompts.py built natural 512/2048/8192-token prompts from docs text; bench-prefill.ps1)
- 22-token (decode bench prompt): 42-44 tok/s, 2211 misses, 8.4 GiB — fetch-dominated single pass (~0.34 s of 0.55 s).
- 512: **~457 tok/s** (444-465), 4112 misses, 15.8 GiB, ~1.12 s.
- 2048: **~1011 tok/s** (985-1030), 4528 misses, 17.4 GiB, ~2.03 s (experts 0.40 s, uploads 0.01).
- 8192: first attempt failed (45 KB command line > Windows limit) — engine supports `--tokens @FILE` (ik2_main.cu:4674).

### Structural facts (code-verified)
- Offload prefill = token-major chunk loop (:6695, forward→exec grouped-GEMM chunk branch); layer-major super-chunk (forward_prefill_sup) needs T_sup>0 → ≥20 GiB free VRAM (:5633) → OFF on 8 GB. IK2_SUP dead here.
- prefetch_layer returns immediately on g_off (:1807) → IK2_PF_P irrelevant on this path.
- Per-chunk pick-union fetches essentially ALL experts once (17.4 GiB per 2048-chunk, content-independent) → n-chunk prompt refetches n×.
- CPU lane (IK2_CPU_FRAC) lives in the S<=64 token path (:3885) → NOT engaged at prefill (S=T·K≫64); grouped branch :3855-3883 runs GPU-only. **CORRECTED during P1 control run**: first c2048-a log shows the lane DOES engage at prefill with frac 0.5 (7 threads, 752 experts computed, **629 waited for**, 0.51 ms/visit, `cpu lane split: 0.50 … 2.95 experts a visit`) — the E9 adaptive split sits in the shared exec offload branch. Possible prefill regression source (GPU waits on late CPU experts). Isolate after P1 arms with a frac-0 arm at 8192.
- Compute ceiling empirically ~1000+ tok/s at 2048 (34 GFLOP/token × 1011 ≈ 34 TFLOPS effective — fp16 tensor fast paths); fetch floor at 2048: 17.4 GiB ÷ 24.5 GiB/s ≈ 0.71 s.

### Planned levers (one change each)
- P1: --chunk 2048→8192 at 8192-token prompt: one 17.4 GiB union fetch instead of 4 (~2.1 s of ~8 s). Watch VRAM (Tm buffers + sliding KV window+chunk ≈ +90 MiB fp8) for OOM/slot deaths.
- P2: short-prompt TTFT (22-512 tok): profile-weighted prefill prefetch (predict union from W.use before routing sync) — offload analogue of prefetch_layer, est −0.1-0.3 s.
- P3 (speculative): CPU-lane extension to prefill union — RAM-side compute ≈ saved fetch time, likely a wash; only if P1/P2 leave fetch critical.

### P1 A/B result (8192-token prompt, 6× each, chunk sweep)
- ctrl-a (chunk 2048): 1001.6-1045.4 → median 1032; ctrl-b: 1016.5-1062.1 → median 1056 (drift +2.3%).
- candidate (chunk 8192): 744.1, 1225.0, 1270.9, 1265.3, 1048.4, 811.4 → median 1137 (+8-10% vs ctrls) — mechanism PROVEN: copied 69.7 → 23.1 GiB, experts 1.55 → 0.6-0.9 s, tier hits 5112→61 (big-chunk pass evicts tier), but **variance ±25% disqualifies clean accept**; suspected CPU-lane contention at prefill (629 waits seen in ctrl).
- Isolation arms DONE (sh_203a8fed0001auMXEG68iiV75q): **lane OFF does NOT fix the variance — chunk 8192 itself is the problem.**
  - n2048 (chunk 2048, lane off): 1059.9/1059.8/1059.2/1024.5/1008.9/1017.6 → median ~1021, ±2.5% tight. Lane neutral at prefill/2048 (≈ lane-on ctrls 1032/1056).
  - n8192 (chunk 8192, lane off): 818.7/747.6/758.7/754.9/840.5/1181.8 → median ~789, ±25% persists.
  - **Structural find**: chunk 8192 collapses the tier to 32 experts / 0.1 GiB (vs 488 / 1.9 GiB) — chunk-scaled activation+KV buffers eat ~1.8 GiB of fr. Lane-on 8192 looked better (1137) only because the CPU lane covers the collapsed tier.
  - **Fetch dedup is real but compute-bound wall**: 23.1 vs 69.7 GiB copied, experts 0.6-1.2 s of 7-11 s total — long-prompt prefill is dense+attention compute-bound; saved fetch doesn't reach the wall clock.
  - Chunk 8192 REJECTED at lane-off (789 vs 1021). Middle arm chunk 4096 (2 fetch passes, ~half the tier shrink) benched before final P1 verdict.

### P1 FINAL VERDICT — chunk lever REJECTED for live config (prefill/decode tradeoff)

### P2 RESULT — REJECTED (2026-11-22)

IK2_PF_OFF implemented (ik2_main.cu only: g_pf_off/g_pf_n/g_pf_next_l :717-727, staged-slot consume :2008-2026, layer-ahead prefetch round :2033-2064, ring raise IK2_PF_SLOTS default 32 at :5810-5817). Default-off byte-identical to pre-df exe ✔ (40-token greedy). ON-mode results kill the lever:

- 22-token: 34.5 vs 40.5 tok/s off (−15%); copied 10.0 vs 6.3 GiB (+59%, stats: 1152 copies, 65.4% wasted).
- 512-token: 433.2 vs 449.0 (−3.5%); 17.1 vs 15.8 GiB.
- Decode regression: 27.0 vs 28.6 (−5.6%) — 32-slot ring took ~0.6 GiB from the tier (356 vs 440 slots).
- ON coherence FAILS at --gen 64 (text prefix diverges from drafting-off run) — staged-slot reuse hazard in core-layer repeat visits; not shippable even if speed had worked.

Why the max(fetch, compute) model was wrong: the 16-slot ring already pipelines a visit's fetch against its expert GEMMs — link idle is only the inter-visit gap (~4.4 of 11.5 ms/visit at T=22 ≈ ≤4% of prompt wall). Staged bytes are waste at chunk scale (union ≈ all experts; lowest-id choice is a bad pick proxy). The prefetch pays for staging out of the tier it tries to save. If ever revisited: no ring growth (stage ≤ st.n−8), rank by W.use, strict re-resolve rule — ceiling remains the inter-visit gap; not worth a round.

Verdict: IK2_PF_OFF stays default-off (inert). No config change. P1 chunk lever remains CLI-only conditional. Prefill phase closes at baseline: 22-tok 42-44 / 512-tok 457 / 2048-tok ~1011 / 8K ~1045 tok/s; long-prompt prefill is dense+attention compute-bound (34 TFLOPS effective), short-prompt is single-fetch-pass bound. P3 (CPU-lane prefill extension) dead per compute-bound wall.

## P2 (2026-11-22, dispatched): layer-ahead prefill prefetch (IK2_PF_OFF)

Rationale: at chunk scale the per-layer pick union ≈ ALL experts (T×K=176 picks vs E=128 at T=22) → layer l+1's non-tier experts are needed with near-certainty; overlap fetch(l+1) with compute(l). Exec prefill serializes route→sync→fetch→compute per visit today (22-tok: 0.34 s fetch of 0.55 s; 512-tok: 15.8 GiB of 1.12 s). Est 22-tok 42-44 → 55+ tok/s, 512-tok 457 → 600+ (toward max(fetch, compute)). Dispatched to fixer (same session as E5-B/E6/E9/E10 machinery): env IK2_PF_OFF default 0 byte-identical; prefill branch only; ring/tag/used consume-don't-recopy; ring may be raised when ON (env-gated alloc); CPU-lane double-compute exclusion; gates = default-off byte-identical, prefill spots (22-tok vs 42-44, 512-tok via @prefill-tokens-512.txt vs 457), decode spot vs 29.7, no deaths, copied-GiB not 2×, ON-only stats line. Orchestrator A/Bs after.

- Chunk 4096 prefill A/B (8192-token prompt, lane on, all 6× arms): fresh ctrl 1044.5 (1020-1060), candidate arms 1318.7 + 1320.3 (1252-1332, ±3%) → +26% prefill, mechanism confirmed (40.5 vs 69.7 GiB, 2 passes vs 4, tier 1.0 GiB intact).
- BUT decode A/B (interleaved ctrl/c4096/ctrl, 3× each, gen 256): ctrl 30.14/29.81, c4096 **26.37** (−12.5%) — `--chunk` scales activation buffers at INIT, permanently shrinking the tier (250 slots vs 488) for every request, including short-prompt decode. Cache hit rate 67.7% → 51.8%.
- Decision: decode is the primary metric; global chunk change = net loss for chat workloads. NO live-config change. CLI-only conditional lever: `--chunk 4096` for one-shot long-document ingestion runs (~1320 vs ~1045 tok/s at 8K prompts).
- Chunk 8192 stays REJECTED (tier collapse to 32 slots, ±25% variance, slower lane-off).

### E10 width probe (2026-11-22): drafter-precision levers (user proposals) closed

User proposed (a) quantized MTP drafter (Q3_K) to shrink resident cost, (b) DFlash drafter, same logic. Probe: 16K ctx, live decode env + IK2_DRAFT_OFF=1, mtp.safetensors, --gen 256 ×2 per width (log e10-width-probe.log):
- Width 2: 1.56-1.59 tok/step, drafting 4.6-5.2 ms, verify 85-89 ms → 17.0-17.4 tok/s. Width 3: 1.83 tok/step, verify 105 ms → 16.7 tok/s. (Width 4 earlier: 1.95 tok/step, 127 ms → 14.7.)
- Acceptance ~0.30 at EVERY width (w2 histogram 64 96 = 96/160 steps accepted 1 of 2). Not width-limited — it is the drafter↔Q3-target rejection rate on natural text.
- Arithmetic: at w2, even a zero-cost perfect drafter gives 3 tok / ~90 ms ≈ 33 tok/s = break-even vs drafting-off 33.7 (16K). Break-even needs acceptance ~0.65; measured 0.30.
- Verdict (a) quantized MTP drafter: NEGATIVE — drafter is 4-5% of step cost; quantization can only lower acceptance (noise alignment explains ≤ a few points of the 70% rejections); ceiling even at perfection is break-even. Not built.
- Verdict (b) DFlash: N/A — DFlash2 is Anbeeld's GLM-5.3-only drafter (CHANGELOG: upstream measured MTP faster everywhere); no Inkling-2-Mini drafter exists.
- Verdict (c) DSpark (user follow-up, 2026-11-22, pointing at unsloth/Inkling-2-Mini-GGUF `dspark/`): **SAME DRAFTER ALREADY TESTED.** Per unsloth's dspark README: "DSpark drafter modules converted from Inkling 2 Mini's own 3-layer block drafter (`mtp.safetensors`, block size 8)" — a GGUF format conversion of the identical weights for llama.cpp (`--spec-type draft-dspark`); maya-ik2 loads the original safetensors directly (ik2_draft.hpp: "maya-ik2's drafter: Inkling 2's DSpark draft model (the official checkpoint's…)", borrows target LM head, taps looped layers — same design llama.cpp uses). The width probe above therefore measured DSpark at its best precision (BF16 source): acceptance ~0.30 at every width. The repo's Q8_0 conversion (worst 0.62% RMSE on projections) can only match or lower it. Lever closed without new work.
- Side observation: at 16K ctx the drafter fits fine (0.96 GiB resident + tier 494/2.0 GiB intact) — the 131K capacity wall is separate and moot.

### E10 RESULT — REJECTED (lever does not pay; abort rule fired, code stays default-off)

Fixer round-1 delivered a mechanically complete port (`IK2_DRAFT_OFF`, default 0 byte-identical: 40-token greedy vs pre-df exe ✔). Spot-checked myself: spec_loop gate :6765 (drafter + g_fast||g_draft_off, loaded-but-unusable drafter decodes plainly) and floor-fits-budget guard :966-977 (NOTE + g_draft_off=false + floor back to 8).

- **Acceptance 0.32 per drafted token** (abort threshold 0.4): 65 windows, 1.95 tok/step, accepted histogram 30/15/12/8/0 (of 4 drafts) — real acceptances, healthy shape, but low.
- **Window byte cost ~3× a token's picks-union** (worse than the 1.5-2× prefill-overlap estimate): drafting 5.6 ms + verifying 127 ms/step → **14.74 vs 33.70 tok/s** (16K ctx, width 4) = 2.3× SLOWER.
- **At the live 131K ctx the drafter cannot run at all**: drafter 1.22 GiB + FP16 head copy 0.92 + KV fp8 2.23 + dense 0.9 → tier budget 0.00 GiB → NOTE fallback to plain decode (9.02 tok/s spot, 32-slot tier). Head copy added at :5585-5590 (offload layout keeps GGUF-block head).
- Divergence at 16K: coherent, prefix path diverges from plain greedy as disclosed (T>1 numerics ~1e-5).
- **Future revisit path (recorded, not done)**: (1) avoid the 0.92 GiB FP16 head copy (fold from GGUF blocks in place), (2) cut window byte cost (pick union ~3× a token's). Acceptance must reach ~0.6+ AND window cost ~1.5× before drafting pays on 8 GB. Resident-path drafting (24+ GB card) remains the clean unlock.

## E10 (2026-11-22, dispatched): MTP drafting on the offload path (IK2_DRAFT_OFF)

User question: give away expert slots to run mtp speculative drafting? Resident gate blocks it (:5041 — drafting needs every expert in VRAM), but a port is feasible: exec offload branch already handles T>1 (prefill), adjacent-position pick overlap measured high (prefill 386 vs decode 1135 MiB/token), host-path syncs amortize per STEP. Est +30-60% if MTP acceptance 0.6-0.75; drafter resident 0.64 GiB → tier auto-shrinks (~-4-6% base). Risks: acceptance on Q3 target unknown (E5-B lesson: intra-token prediction weak — but MTP is the model predicting itself); T>1 numerics ~1e-5 off (engine warns at :2851) → greedy may occasionally diverge, disclosed not hidden; same-visit slot pinning S=T·K (pool floor 8 may need drafting-only raise).

Dispatched: env IK2_DRAFT_OFF=1 gates drafter load (before tier budget) + verify window through exec offload branch + taps wiring + SpecCtl reuse. Gates: default-off byte-identical; ON run reports acceptance — if <0.4/drafted token the lever dies cheap (code stays default-off). Orchestrator A/B after.

### E9 A/B results (7 arms, bench-e9.ps1, 6× --gen 256 @131K, medians)

- Q3 ctrl-a 23.49 / ctrl-b 24.05 (lane off); **Q3 f05 (adaptive cap 0.5) 29.70** — +23-27% over controls, split held at cap; queue 0.34-0.60 ms vs link 0.59-1.05 ms → CPU lane has slack at cap 0.5 → extra arms q3-f075/q3-f09 launched to see if EMA finds a better share above 0.5 (static 0.75 collapsed to 22.18 in E6; adaptive should back off).
- Q4M ctrl 21.52/21.53 (131K CLEAN — pool-floor fix verified in anger); **Q4M f05 28.12 (link-bound at cap); Q4M f075 28.55** — EMA settled 0.59-0.75 exactly at queue/link balance → adaptive split works as designed.
- Model-swap verdict pending q3-f075/f09: Q3 f05 29.70 vs Q4M f075 28.55 → Q3 leads decode by ~4% (Q4M quality advantage stands, user's rule was swap-only-if-decodes-wins).

### E9 VERDICT (2026-11-22)

- **Extra arms**: q3-f075 median **29.45** (EMA 0.50-0.68), q3-f09 median **29.47** (EMA 0.55-0.73) — all Q3 caps 0.5/0.75/0.9 within ±1.3% of each other; **static 0.75 collapse (22.18) is gone** — adaptive split delivers robustness, not speed. Live config keeps IK2_CPU_FRAC=0.5 (best arm 29.70, lowest thread pressure under desktop load).
- **Lever 1 (hand AVX2 kernels): NEGATIVE** — ggml auto-vectorized loop already at parity; left unwired behind IK2_CPU_BENCH=1.
- **Lever 5 (Q4_K_M): NO SWAP** — Q4M best 28.55 vs Q3 29.70 (−4%); per user rule swap only if decode wins. Q4_K_M stays downloaded+verified on disk; slot-death fix (pool floor 8) makes ANY Q4 usable at full context. Q4_K_M = quality-first option (+~1 GB VRAM/RAM, −4% decode).
- Net live-config change from E9: **none** (same env, same model). Session best remains ~29.5-30.1 CLI decode.
- Remaining ranked levers: thread-budget knob (deferred by user), live-profile swap (deferred; file exists), RAM XMP/3600 (+6-12%, BIOS-side), 12+ GB card (ceiling breaker).

## E9 (2026-11-22, dispatched): Q4@131K slot-death fix + CPU-lane round-2 (user-greenlit)

## E9 round-1 (fixer): Task A pool-floor fix + Task B lane round-2 — all gates green, A/B launched

### Task A — Q4@131K slot-death ROOT CAUSE + fix
- Cause: EC::init tier_only sizing (pmin=0) left pool 1 = ONE slot for 4 layers (Q4_K_XL@131K); a visit pins up to S=T·K=8 slots of its pool at the same tick; evict skips used>=t → second same-visit miss deadlocks (`no slot free`). Not the staging ring.
- Fix (ik2_main.cu): kPoolFloor=8 at :952-958 (every pool keeps ≥8 slots; value-based sizing fills the rest); lazy-copy retry 2→6 passes (:867-876); die prints pool diagnostics (:890-900). Q3 pools already ≥8 → sizing byte-for-byte unchanged.
- Gates: Q4_K_XL@131K host path 3× gen-256: exit 0, identical output, 19.63/19.59/19.73 tok/s coherent (no frac). Q3@131K spot unchanged (23.06 @room256 / 23.53 @room64).

### Task B — lane round-2
- Type mix on Q3 frac 0.5: nvfp4 55.5%, iq3_xxs 17.0%, iq2_s 11.5%, iq3_s 7.9%, q3_K 4.3%, iq2_xs 3.8% → NVFP4 dominates; ggml's nvfp4 dot auto-vectorizes, hand AVX2 kernel (src/kernels/cpu/ik2_rows_avx2.cpp, selftest 7.9e-8) NOT faster → NOT wired (honest negative; reachable via IK2_CPU_BENCH=1 selftest only).
- Threads: 11 workers (SMT all) = 17.1-23.9 tok/s vs 7 workers = 28.4 → default threads-1 workers; `--threads N` sweeps.
- Adaptive split: g_cpu_share starts at IK2_CPU_FRAC cap, EMA feedback (over>1.35 → −0.05 floor 0.1; over<1.05 → +0.02 ceiling cap) at :3902-3910; n_cpu = clamp(round(share·n_miss), 0, cap). Stats line: `cpu lane split: … queue X ms against Y ms of the link`. Deterministic, not byte-identical when ON.
- Gates: default-off byte-identical vs pre-df exe ✔; Q3 frac 0.5: 28.10 tok/s single (off 23.53), split held at cap (CPU 0.46 ms < link 1.00 ms); Q4_K_XL frac 0.5 gen 128: 26.92 tok/s, at balance point (1.22 vs 1.16 ms) ✔.

### Q4_K_M downloaded + verified
- Inkling-2-Mini-UD-Q4_K_M.gguf 16,610,544,328 B, sha256 F90C5A4D…F3CDF7E1 = HF LFS pin (ledger row). Type mix all-supported (F32 508, Q8_0 150, BF16 32, NVFP4 40, Q5_K 4) — Q8_0/NVFP4-heavy → CPU dots ~2× Q3_K.

### A/B IN FLIGHT (sh_203679f4f0014mUUfFXYTrtvSG, bench-e9.ps1 param Model/Frac/Tag, env room64+POP_OUT+natural profile)
Arms: Q3 ctrl-a / Q3 f05 / Q3 ctrl-b / Q4M ctrl-a / Q4M f05 / Q4M f075 / Q4M ctrl-b; 6× gen 256 each. Acceptance: Q3 f05 vs E8 static 30.14 (same-session ctrls anchor); Q4M arms vs Q3 best → model-swap decision.
Two sequential tasks to the E6 fixer session (same code area, no parallel writers):
- **Task A (bug fix, first)**: UD-Q4_K_XL@131072 dies `expert cache: no slot free` on host path (pre-existing, reproducible on pristine exe; works at 8192 ctx; tier ~394 slots). Same class as the original junk-token deadlock: eviction (:860-876) dies when a burst of misses in a starved pool finds no free slot while staging entries are busy. Fix = graceful degradation (bounded backoff / staging steal / pool minimums), gates: Q4@131K 3× natural bench no-death, Q3@131K unchanged spot ~30.
- **Task B (performance)**: (1) hand-tuned AVX2 dequant-dot for the dominant Q3_K_XL expert types (batched dequant + FMA, ref 9.78 GB/s ggml baseline; selftests vs scalar decode_value), (2) verify 12-thread SMT pool (+35% probe), (3) dynamic queue-aware CPU/GPU split replacing static ceil(frac×n_miss) — IK2_CPU_FRAC becomes master enable + max-share cap; motivation: static 0.75 collapsed (22.18, 0.61 ms/visit) so the true balance point needs adaptive scheduling; CPU lane is co-critical at 0.5 (310 MiB/token ≈ token wall), every CPU GB/s buys tok/s until GPU-bus cap ~42.
- Gates: default-off byte-identical; frac 0.5 coherent on Q3 AND (post-A) Q4 @131K; no A/B by fixer — orchestrator benches: Q3 curve re-sweep (0.5/0.75/0.9) with new kernels, then Q4_K_M evaluation.
- Q4_K_M download started in parallel (background curl, sha256 to verify vs HF LFS pin; 16.6 GB → Maya-data models dir). Q4_K_M promise: Q8_0/NVFP4-heavy types dot ~2× faster on CPU (probe: Q4_K ~20 GB/s vs Q3_K 9.78) → CPU lane ceiling ~2×; possibly higher quality than Q3_K_XL. Was REJECTED earlier only as unrunnable (unsupported types) — now runnable after the E6 kernels exist.


Engine: maya-ik2 (project-maya, Windows port, Q3 kernel extension in working tree).
Skill: inference-optimisation. Started 2026-11-22.

## Targets & constraints (fixed for the whole run)
- Metric: c1 decode tok/s (primary); prefill tok/s (secondary).
- Memory cap: 8 GB VRAM / 32 GB RAM, offload path (experts 11.4 GiB pinned RAM, tier in VRAM, misses over PCIe).
- Context 131072 must stay.
- Quality: lossless preferred (KV already fp8 — that trade is already made and stays).
- Drafting: N/A (drafting window is resident-VRAM-only; off).
- One change per experiment; control/candidate/control; ≥6 decode samples, median; accept only if beats both neighbours, no regression >3%.

## Method
- Bench = engine CLI: `build\Release\maya-ik2.exe --model <Q3 gguf> --max-context 131072 --tokens "1,2,3,4,5" --gen 256`, env IK2_UNSUPPORTED=1, IK2_KV=fp8, IK2_WARM=data\expert-profile-ik2mini.bin. Prints tok/s per run. Server API (timings.predicted_per_second) as end-user spot check.
- Serialize all GPU jobs manually (no queue.sh on Windows). Host idle, AC power, no other GPU jobs.
- Longer gens (≥256) for steady state: short 8-token runs overestimate (36 tok/s) vs server reality (10.5-15 tok/s).

## Machine inventory
- RTX 3060 Ti 8 GB (sm_86), PCIe 4.0 x16, 32 GB RAM (experts pinned), AVX2 CPU (no AVX-512, CPU lane NVFP4-only → unusable here).
- Current best-known numbers: server 131K ctx: 15.0 tok/s (150-tok gen), 10.5-11 tok/s (741-tok gen), prefill 43.6 tok/s. CLI 40-tok greedy: 28.95 tok/s.

## Experiments

### 2026-11-22 — E8 RESULT: IK2_ROOM_MB 256 → 64 ACCEPTED
- ctrl-a(256) median 28.38 (warm-in trend 26.15→29.67) / r64 median **30.14** (488 slots, 1.9 GiB) / r0 median 30.44 (505 slots, 2.0 GiB) / ctrl-b(256) median 29.31.
- r64 beats both controls +2.8-6.2% (beyond ±1.3% noise). r0 only +1% over r64 (inside candidate noise) for 0.1 GiB that leaves zero VRAM margin under real UI load (user's browser at 7.8/8.0) → r0 rejected; r64 accepted. Tier budget self-adapts (fr measured at init), so heavy-desktop starts get proportionally smaller tiers — safe.
- Applied to live config (env IK2_ROOM_MB 256→64, json.load valid). New best CLI decode ~30.1 (+88% cumulative from 16.02). Ledger e8 rows + verdict.
- Driver gotcha: .ps1 must be invoked via PowerShell `&`, NOT python.exe (first launch died at SyntaxError, GPU untouched, relaunched sh_20346d464001w87wuYDAgc7g2a).

### 2026-11-22 — E8 (reopened on user request): VRAM headroom reclamation at frac 0.5
- User reports UI VRAM 7.8/8.0, Task Manager 7.6 — suspected headroom; UI short-prompt decode measured 21.4 tok/s vs our 27.7 API run (gap to diagnose: browser compositing contention / first-request warm-in / short-gen average / prompt-specific routing).
- Sweep: IK2_ROOM_MB 256(ctrl-a) / 64 / 0 / 256(ctrl-b) at IK2_CPU_FRAC=0.5 + IK2_POP_OUT=1, 6× --gen 256 each, `bench-roomfrac.ps1` → `e8-*.log`.
- Hypothesis: room 256→0 adds ~0.25 GiB tier (+~7% slots) ≈ +1-2% decode; watch cudaMalloc OOM / slot deaths (desktop apps fluctuate 0.3-1 GiB).
- Driver gotcha: .ps1 must be invoked via PowerShell `&`, NOT python.exe (first launch died at SyntaxError, GPU untouched, relaunched sh_20346d464001w87wuYDAgc7g2a).

### 2026-11-22 — E0 BASELINE (control)
- Prompt: 22 natural tokens (offloading question, IDs in ledger). --gen 256, max-context 131072, env IK2_UNSUPPORTED=1 IK2_KV=fp8 IK2_WARM=profile, default IK2_ROOM_MB=1024.
- Junk-token prompts (IDs 1-5) DEADLOCK at this tier size (`expert cache: no slot free`, ik2_main.cu:877) — rare-expert pools get 2-4 slots; do NOT bench with junk tokens.
- Tier: 250 slots / 4 pools / 1.0 GiB (fr≈2.0 GiB minus room 1.0 GiB).
- Decode (tok/s): 15.97, 16.08, 15.69, 15.96, 16.09, 16.10 → **median 16.02**. Noise ±1.3%.
- Prefill median 41.0 tok/s. Tier 22.4%, staging 2.8%, fetched 74.8% = 1135.1 MiB/token (282.68 GiB/run).
- Implied effective PCIe ≈ 17.4 GiB/s; fixed overhead ≈ 31 ms/token. All-tier ceiling ≈ 12 tok/s is for zero tier; with tier we're past it.

### 2026-11-22 — E1 CANDIDATE: IK2_ROOM_MB=256 (tier 1.0 → 1.8 GiB)
- Result: 438 slots, tier coverage 22.4→**31.5%**, fetch 1135.1→994.7 MiB/token.
- Decode (tok/s): 17.68, 16.96, 17.63, 17.69, 17.61, 16.79 → **median 17.62** (+10.0% vs 16.02 control).
- Prefill median 42.05 (42.4, 42.4, 39.1, 42.5, 41.7, 40.4) — no regression.
- No `no slot free`, no OOM. Coverage per slot saturating (top experts already resident).
- Trailing control (control-tail.log): 15.05, 15.97, 15.88, 16.09, 16.09, 15.88 → median 15.92 (−0.6% vs lead control, within noise; run 1 = cold outlier).
- **VERDICT: ACCEPT.** 17.62 beats both neighbours ~10%. New best: env + IK2_ROOM_MB=256.

### 2026-11-22 — E1b: decode is ~100% fetch-bound (analysis, from E1 numbers)
- 1/17.62 s = 56.8 ms/token; fetch 994.7 MiB ÷ 56.8 ms ≈ 17.5 GiB/s. Fixed overhead negligible → every MiB/token cut from fetch is direct tok/s.
- `IK2_STAGE` ruled out: decode misses copy through per-pool staging ring (ik2_main.cu:1940-1974), default 16 slots/pool ≥ K misses/layer; copies never span tokens (tag memset :1973). More ring = steals tier bytes, no overlap gain.
- `IK2_PF_P` is a PREFILL lever (prefetch_layer is layer-major chunk prefetch, ik2_main.cu:1746-1778; at decode T=1, threshold 0.999 ~never fires — staging 2.6% is prefill residue).

### 2026-11-22 — E2 CANDIDATE: IK2_ROOM_MB=128 (tier 1.8 → 1.9 GiB, 438 → 470 slots)
- Result: 470 slots, tier coverage 31.5→32.8%, fetch 994.7→977.1 MiB/token. No OOM, no `no slot free`.
- Decode (tok/s): 17.31, 17.91, 17.71, 17.72, 17.92, 17.74 → **median 17.73** (+0.6% vs E1's 17.62).
- Prefill median 42.6 — no regression.
- **VERDICT: REJECT.** +0.6% is inside the ±1.3% noise band and coverage-per-slot is saturating (438→470 slots bought only +1.3% tier). Room lever exhausted: fr ≈ 2.0 GiB is the hard ceiling at 131K ctx. Keep IK2_ROOM_MB=256.
- Negative result: further tier growth via room is done; remaining gains must come from better *ranking* of what fills the tier (E3: profile regen) or fewer bytes per expert (out of scope, requant).

### 2026-11-22 — E3 SCOPED: profile regeneration (workload-matched tier ranking)
- Shipped profile `expert-profile-ik2mini.bin` covers 31.5% at 438 slots; ranking drives pool sizing (off_value = picks-per-byte, ik2_main.cu:5593-5604), warm fill order (:5611-5643) and prefill prefetch seed (:5646-5655).
- `IK2_POP_OUT=FILE` (process exit) + `pop_count` at routing time (:313, cache-state independent) → profile from natural runs.
- Plan: 6 diverse natural prompts (story / TCP-UDP / French Revolution / meal plan / photosynthesis / Python code — all ≠ bench prompt), one CLI process each with IK2_POP_OUT=prof-N.bin, --gen 512; merge counts (merge_profiles.py, uint64 element-wise add); E3 candidate = IK2_WARM=<merged>, control/candidate/control vs 17.62.

### 2026-11-22 — E3 CANDIDATE: regenerated IK2_WARM profile (natural workload) — ACCEPTED
- Built `expert-profile-natural.bin` (hill-climb\): merged IK2_POP_OUT counts from 6 diverse natural prompts (story / TCP-UDP / French Revolution / meal plan / photosynthesis / Python code; bench prompt EXCLUDED), one process each, --gen 512; merged via merge_profiles.py (format n_layer=24, E=128, uint64 counts; 24,584 B file; 1.22M picks).
- Same bench as E0/E1/E2. Result: 440 slots (vs 438 — pool sizing is picks-per-byte driven), tier coverage 31.5→**39.5%**, staging 2.6→1.3%, fetch 994.7→**889.1 MiB/token**.
- Decode (tok/s): 19.06, 19.04, 18.52, 18.54, 19.10, 19.14 → **median 19.05**.
- Prefill 42.2-43.8 (no regression). No deadlock/OOM.
- Trailing control (shipped profile, same session): 17.59, 17.37, 17.70, 17.46, 17.45, 17.21 → median 17.46 (one prefill outlier 34.1, decode unaffected). Lead control E1: 17.62.
- **VERDICT: ACCEPT.** 19.05 beats both neighbours (+8.1% / +9.1%), far outside ±1.3% noise. Gain is honest: bench prompt excluded from profile-gen set; 6-prompt diversity ranks the general-text expert head, which transfers better than the shipped single-workload profile.
- Cross-check: 889.1 MiB ÷ 52.5 ms/token ≈ 16.9 GiB/s effective PCIe — consistent with E1b's fetch-bound model.

### 2026-11-22 — E3-generality: unseen-prompt check — PASSED (profile transfers)
- Second prompt: microservices question (in neither the bench nor the 6-prompt profile set).
- Candidate (natural profile): 16.49, 18.32, 19.13 → median 18.32; tier 40.6%.
- Control (shipped profile): 15.85, 16.14, 16.18 → median 16.14; tier 24.2%.
- **+13.5% on a fully unseen prompt** → not bench-overfit; the natural profile ranks the general-text expert head. Log e3-generality.log.

### 2026-11-22 — Live server config updated (best config applied)
- `expert-profile-natural.bin` copied to `C:\LLM models\project-maya\data\expert-profile-natural.bin` (stable location, survives hill-climb dir cleanup).
- `maya-inkling-2-mini.json` env: IK2_UNSUPPORTED=1, IK2_KV=fp8, IK2_WARM=data\expert-profile-natural.bin, **IK2_ROOM_MB=256**. Args unchanged (--model Q3 GGUF, --max-context 131072, --drafting off), port 8095. JSON validated (json.load). Server was stopped during the edit.
- Cumulative: 16.02 → 19.05 decode (+18.9%), prefill ~42 (+~5% vs 41.0 baseline... within/above noise, secondary metric unchanged in practice).

### 2026-11-22 — E4 CANDIDATE: 20-prompt profile (expert-profile-natural20.bin) — REJECTED
- 14 additional diverse prompts profiled (IK2_POP_OUT, --gen 512 each; all clean 21.8-24.5 tok/s), merged with prof-0..5 → natural20 (24,584 B, 4.04M picks).
- Bench 3×control / 6×candidate / 3×control, same session (e4-natural20.log, bench-e4.ps1):
  - Control (6-prompt natural profile): 19.09, 19.13, 19.11 | 18.25(outlier), 19.13, 19.14 → median 19.12; tier 39.5%, fetch 889.1 MiB/token.
  - Candidate (natural20): 18.71, 18.71, 18.71, 18.71, 18.70, 18.71 → median 18.71; tier 37.8%, fetch 918.5 MiB/token.
- **VERDICT: REJECT (−2.1%).** Adding breadth diluted the ranking: the 20-prompt mix (haiku, email, chess, …) spreads picks over a flatter head, while the 6-prompt tech/prose set ranks this workload family better. Profile quality is WORKLOAD-SIMILARITY-driven, not size-driven. Keep `expert-profile-natural.bin` (6-prompt).
- Practical corollary: the ideal profile is one generated from the user's own real chats — `IK2_POP_OUT` can be set in the server config env block; the engine writes the profile at server exit. Not done now (needs user's real workload accumulated).

## STATUS — PLATEAU (declared 2026-11-22)

| metric | baseline | best | ceiling | measurement |
|---|---|---|---|---|
| decode (c1, bench prompt) | 16.02 tok/s | **19.05** (+18.9%) | ~43 tok/s all-tier | median of 6, CLI --gen 256 |
| prefill (22 tok) | 41.0 tok/s | 42-44 | compute-bound, low priority | median of 6 |
| tier coverage | 22.4% | 39.5% | fr-limited 2.0 GiB | engine offload counters |
| fetch/token | 1135 MiB | 889 MiB | 0 if all resident | engine offload counters |

- Winning changes: `IK2_ROOM_MB=256` (E1, +10.0%) + regenerated natural profile as `IK2_WARM` (E3, +8.1%). Both applied to `maya-inkling-2-mini.json`; profile at `data\expert-profile-natural.bin`.
- Why it stops here (evidence): decode is ~100% PCIe-fetch-bound (889 MiB ÷ 52.5 ms ≈ 17 GiB/s effective); tier budget is hard-capped by fr ≈ 2.0 GiB at 131K ctx (E2: +0.1 GiB bought +0.6% tok/s, inside noise); profile ranking converged (E4: breadth dilutes, −2.1%); staging/prefetch knobs are prefill-side or steal tier bytes (E1b analysis).
- Remaining levers, ranked: (1) 12+ GB VRAM card → experts resident, ~2-3× decode (hardware); (2) requant experts to fewer bytes/pick (lossy, out of scope per constraints); (3) speculative drafting (needs resident VRAM, N/A on 8 GB); (4) profile from user's actual server chats via IK2_POP_OUT in server env (workload-matched, est. +1-3 tok/s, needs accumulated real usage); (5) IK2_STAGE 16→8 frees ~0.13 GiB for tier ≈ +0.5-0.7% decode — sub-noise, deadlock-risk, not attempted.
- Stability: no `no slot free` deaths on any natural prompt at 440 slots (deaths only with junk-token prompts at tiny tiers); no OOM; no output-quality change claimed or needed (lossless levers only). Not verified: long-context retrieval needles and concurrent soak (single-user box, drafting off, KV fp8 unchanged from before the climb).
- Files: PLAN.md, THEORY.md, ledger.jsonl (E0-E4 + generality + config-apply), best-config.json (e3 config), configs/ unchanged, logs baseline-natural / e1-room256 / control-tail / e2-room128 / e3-* / e3-generality / e4-natural20 .log.

## E5 ( reopened 2026-11-22, post-plateau ): decode-time predicted prefetch (IK2_DF)

User observed the CPU is fully idle during decode and suspected an unblocked resource. Investigation:

- CPU idle is by design (fetches are DMA on the dedicated copy stream; the CPU compute lane is NVFP4-only so Q3 experts cannot run there; an AVX2 CPU computing experts would be slower than the PCIe shuttle).
- BUT a standalone pinned-H2D probe (sm_86, GPU idle, the engine's own chunk sizes) measured the link at **24.3 GiB/s @ 3.8 MB chunks, 23.5 @ 1.3 MB, 24.5 peak**, vs the engine's effective **~17 GiB/s** -> ~30% of the link is unused. Probe source: `C:\Users\Nico\AppData\Local\Temp\opencode\pcie-probe\bw_test.cu`.
- Root cause (code analysis, ik2_main.cu:1868-1974): copies for a layer visit can only be issued after that visit's routing (D2H sync reveals picks), so `cst` idles during the previous visit's expert compute + the next visit's dense/attn/routing (~0.3 ms x 48 visits ~= 15 ms of the 52.5 ms token; transfers alone at real link speed are 37.8 ms -> the gap matches exactly).
- Lever: predict the next visit's misses from `W.use` (IK2_WARM profile + runtime routing counts) and copy them into the per-pool staging ring during the idle window. Ring-based, env-gated `IK2_DF` (default 0 = off, zero risk), reuses the existing rf/rc event handshake; predicted-and-picked experts skip their copy at issue time.
- Expected ceiling: idle window caps useful prediction at ~f=0.3-0.38 of fetch bytes; with hit-rate 0.4-0.8 on predicted experts -> est. decode 19.05 -> 23-25 tok/s. If prediction quality is poor (misses are by definition low-rank experts), reject and revert.
- Status: implementation delegated (fixer, ik2_main.cu only, gates: g_off && T==1 && cst!=g_s && IK2_DF>0). Acceptance: byte-identical 40-token greedy vs pre-change exe with IK2_DF=8 and =4 AND unset; then orchestrator benches control/candidate/control 6x medians.

### E5-B round-2 result (2026-11-22): same-token predictor AT BASELINE — REJECTED, lever closed

df_prev[l] (this layer's previous-visit picks) as predictor: precision 44.6% (k=2) / 45.4% (k=4). But the new prevhit counter shows **49.6% of all picks (7429/14976) repeat the layer's previous visit** — random-subset expectation for k=2 of an 8-pick set with 49.6% repeats is 49.6%. The predictor sits AT baseline; top[] order carries no ranking signal. Bytes: 902.9 → 1011.3 (+12%) / 1122.8 (+24%) MiB/token; spot 17.4/15.9/14.9. Break-even precision ≈ 59% (saved 0.161 ms/consumed vs exposed 3.91 MiB × (1−p) at saturated link). Fixer's residual idea — rank stored picks by routing weight — is speculative at a 9-pt margin; not pursued after two data-driven rejections.

Verdict: REJECT. Code default-off byte-identical (all three gates pass; unset run 902.9 MiB/token = pristine). Backup build\Release\maya-ik2.pre-df.exe. Lever B CLOSED.

### E7 (2026-11-22): host-DMA decode path — zero-code candidate

IK2_POP_OUT=1 → g_pop non-empty → :3440 graph short-circuit skipped → decode falls to exec host-MV branch (:3825, S≤64), misses via EC.get → cudaMemcpyAsync on copy engine, overlap stream g_cs ACTIVE (g_prof false, unlike IK2_PROF which forces single-stream + per-visit syncs). Spot: 22.6 vs 18.9 tok/s. A/B: bench-e7.ps1, control 3× / candidate 6× / control 3×, + 40-token greedy pair. Side effects: per-pick atomic, profile file "1" written at exit.

### E7 result (2026-11-22): host-DMA decode path — ACCEPTED, new best

- Control lead 3×: 19.09 / 19.09 / 19.11 → median 19.09. Candidate (IK2_POP_OUT=1) 6×: 23.13, 23.15, 23.05, 22.44, 23.08, 23.06 → **median 23.06 tok/s (+21%)**. Control tail 3×: 18.59 / 18.87 / 18.76 → median 18.76 (+23% vs candidate). Far beyond ±1.3% noise.
- Lossless gate: 40-token greedy stream TOKEN-IDENTICAL (fc /b diff is only the shifted prefill stat fields — host path populates `cache 768 hits, 1743 misses, 6.6 GiB copied` where the graph path prints zeros).
- Prefill unchanged (43.0 tok/s, experts 0.14 s). Tier unchanged (440 experts / 1.8 GiB, warm fill intact). No OOM, no slot deaths, coherent output.
- Mechanism (matches lever-A samples: link idle ~70% of token in graph mode): host path (:3825, S≤64) issues misses as cudaMemcpyAsync on the overlap copy stream g_cs while the GPU computes; the graph path's device copy kernel serializes per visit.
- Cumulative: baseline 16.02 → E1/E3 19.05 → **23.06 (+44.2%)**.
- Cost: zero code; env toggle only. Side effects: per-pick atomic (negligible); a profile file written at process exit (g_pop) — a BONUS: pointed at a real filename in the live config it continuously captures the user's real-chat expert profile (the parked E3b lever for free).
- Wart: cumulative `offload:` picks line prints zeros on this path (device counters unused; per-run stats live in the `cache N hits, M misses` fields instead). Cosmetic.

### Post-E7 status (2026-11-22)

New best decode 23.06 tok/s median. Remaining levers: E6 CPU lane (parked, 1.44× ceiling, big lift), profile refresh from accumulated live IK2_POP_OUT captures (free, passive), copy-vectorization of the graph path (superseded by E7's DMA path). Link now the binding constraint again: 889 MiB/token at ~24.5 GiB/s DMA ceiling ≈ 36 ms/token ≈ 27.5 tok/s practical ceiling on this path; headroom ~+19% before PCIe-saturated.

### Lever A (2026-11-22, folded into E7 session): nvidia-smi during live decode


Decode-phase steady state: 7667/8192 MiB VRAM used (~0.5 GiB headroom — tier hard-capped, confirms E1/E2 saturation), GPU util 100% (any-kernel metric), **memory controller busy only 27-34% of wall time → link idle ~70% of each token in graph mode**. That idle share is the E7 headroom: a path that keeps the copy engine fed during GEMM/routing windows converts it directly. VRAM accounting vs engine log still leaves ~1.4 GiB unexplained (graphs/logits/ring pools/fragmentation) — not actionable at 0.5 GiB free.

#User insight: CPU path is NVFP4-only; porting the 8 Q3 block types to the AVX2 CPU lane would let CPU COMPUTE a share of expert misses locally from pinned RAM (no PCIe), overlapped with GPU fetch of the rest. Math: token time = max((1-f)*900MiB/17GiB/s, f*900MiB/CPU_GBps). At 10 GB/s CPU -> f=0.37 -> ~30 tok/s (+58%); at 5 GB/s -> ~24.5 (+28%). Stacks with link fixes (E5 round-2). Cost: biggest lift yet — 8 CPU kernel ports + hybrid miss-scheduler + per-visit CPU-GPU sync. Probe dispatched (temp cpu-probe dir): faithful Q3_K/Q4_K AVX2 dequant-GEMM, single+all-core, RAM bw sanity. Verdict on probe numbers. No GPU use during probe (E5 round-2 diagnosis in flight).

### E6 UN-PARKED (2026-11-22, user decision): hybrid CPU expert lane now viable

E7 changed the calculus: decode now runs the host-MV exec branch (no CUDA graph), misses are host-side EC.get decisions → a per-visit CPU/GPU split needs no graph surgery. Probe numbers already in hand (Q3_K 9.78 GB/s all-core, RAM 42.4 GB/s; q3k_bench.c = tested AVX2 kernels). Equal-split math vs NEW baseline 23.06 (889 MiB/token, ~24.5 GiB/s DMA): f≈0.29 → ~36 tok/s ideal; f=0.2 → ~33 (+43%). Dispatched to fixer (ses_dfd19a739ffe6AVXFUpenUMvi8): `IK2_CPU_FRAC` env gate (default 0, byte-identical), AVX2 dot kernels for ALL expert types in BOTH UD-Q3_K_XL and UD-Q4_K_XL (host scalar decode_value = ground truth), persistent worker pool, per-visit miss split, activations D2H first so workers overlap GPU fetch, H2D result upload event-synced into combine. Gates: default-off byte-identical vs maya-ik2.pre-df.exe; per-type selftest <1e-3 rel err; coherent output at frac 0.25/0.5 both GGUFs; `cpu lane:` stats line. ON-mode NOT byte-identical (accumulation order) — correctness = selftest + coherence. A/B: orchestrator, IK2_CPU_FRAC 0.15/0.25/0.5 vs unset control, both models, accept only if median beats both controls beyond ±1.3% with no deadlock/OOM/degenerate output.

### E5 round-2 diagnosis (2026-11-22): where the link gap lives — fixer report accepted

- Copy is a device zero-copy kernel (ik2_off.cu:68-82: uint4 16 B/thread, 4 in flight, grid 2×SM×256 ≈ 25% occupancy), NOT copy-engine DMA — the 24.3 GiB/s probe was DMA. Per-expert sequential inside one kernel per visit; whole-visit barrier (experts wait for last byte of last miss, ik2_main.cu:2477).
- Plan kernel (ik2_off.cu:9-65, <<<1,32>>>): ring-wide tag scan; clock hand advances n_fetch/visit; host can't read hand inside a captured graph. Device-side prefetch trick: prefetched slot gets `used[s]=cnt->seq+1` (victim-protected, still tag-matchable).
- W.use is HOST-only (never uploaded, never touched by route_token/plan). Live in-graph signals exist: `g_offs.heat` (VRAM, atomicAdd per pick) + OffCount. Next-visit layer table derivable from static plan (ik2_main.cu:5104-5109).
- ALL decode is on g_s; g_cs 100% idle in fast_token_launches (verified 2200-2498). Per-visit: copy 0.757 ms + non-copy 0.35 ms = 1.104 ms; gap-vs-per-byte split inseparable from wall clock alone; host-MV (DMA, IK2_POP_OUT=1) spot runs 22.6 vs 18.9 tok/s single-run hint copy mechanism matters.
- Lever ranking: **A** tier-budget sanity (~1.4 GiB unexplained VRAM; 1.15-1.30x; diagnose first, may be config) > **B** device-side predicted prefetch k=1-2 (1.10-1.20x; design fully specified) > **C** per-expert copy split on g_cs (1.03-1.10x) > **D** copy hygiene 8×uint4/__ldg (≤1.03x) > **E** DMA predicted copies (medium-high risk) > **F** bigger IK2_STAGE ring (+3-6% but eats 0.6-1 GiB that A wants).
- Actions: B dispatched (revert round-1 inert exec patch; implement graph-path predicted prefetch, env IK2_DF=k, staging-only, cross-stream overlap with visit N+1 dense/attn, byte-identical gates k=0/1/2). A folded into next bench (nvidia-smi breakdown during decode). CPU probe (E6) running in parallel.

### E5-B round-1 result (2026-11-22): mechanism works, W.use predictor REJECTED on counters (no A/B needed)

- Implemented + all three byte-identical gates pass (unset/1/2); overlap layout proven from captured dependency graph: df kernel+copy on g_cs concurrent with visit N+1 dense/attn/plan, joined before experts; victim safety via used=seq+1; tier untouched (38.4% identical all runs).
- BUT predictor precision: k=1 → 194/1872 predictions consumed (10.4%), +162 MiB/token pure waste (902.9 → 1065.0); k=2 → 284/3744 (7.6%), +335 waste (1238.5). Stage hits 210→404→494 of 14976 picks (1.4%→2.7%→3.3%). Spot tok/s 17.4/15.9/14.9 — net negative both k. W.use top-k is token-independent; ceiling ~10% when 8 picks/128 experts vary per token.
- Structural insight (fixer): looped core layers 6-17 visited 3×/token (36/48 visits); previous pass's picks for the same layer are 1 device store away ([n_layer][K] table written by off_fetch_plan) → same-token predictor with much higher expected precision. Also explains low stage%: previous-pass expert is ~12 visits back, evicted from the 16-slot ring (n=16 holds ~5.5 visits).
- Round 2 dispatched: same-token predictor (IK2_DF=k = top-k of previous-pass picks for l_next, tier/duplicate/staged-excluded), same byte-identical gates (unset/2/4) + stage counts + precision report.
- Code state: ik2_off.cu:84-152 off_df_plan_kernel (W.use top-k, to be replaced), df_use upload 5710-5723, events/stream layout 2240-2471, df fields 1722-1729. Backup build\Release\maya-ik2.pre-df.exe = pristine pre-DF binary.

### E6 CPU-lane probe result (2026-11-22): ceiling 1.44x — PARKED unless link levers fail

- Probe: temp cpu-probe\q3k_bench.c (AVX2, MSVC /O2), block layouts copied from engine decode_value (Q3_K 110 B, Q4_K 144 B), selftest vs engine scalar relerr ~5e-8. Machine: Ryzen 5 5600X 6C/12T (3.7 base/4.6 boost), DDR4-3200 dual-channel; RAM read ceiling 42.4 GB/s (83% of 51.2).
- Q3_K expert GEMM: **9.78 GB/s** stream, 6 physical cores (ALU-bound — only 23% of DRAM; SMT adds just 35%; all-core clock droop 4.6→4.0). Q4_K ~20 GB/s.
- Hybrid math: ideal overlap 18.25+9.78=28 GB/s → 33.8 ms (1.53x, DRAM 66% — tight). Realistic f≤0.3 → GPU critical at 36.3 ms/token ≈ **1.44x (27.5 tok/s)**.
- Cost: 5 AVX2 type ports (IQ2_XS/IQ2_S/Q3_K/IQ3_S/IQ4_NL — host scalar references already exist in ik2_kernels.cu to copy) + hybrid miss-scheduler + **breaking the CUDA graph per token** (picks are device-side inside the graph; CPU needs host knowledge per visit). Highest-effort lever; park.

### E6 A/B round-1 (2026-11-22, same session, Q3 @131072, 6× --gen 256)

bench-cpufrac.ps1, env identical both arms incl. IK2_POP_OUT=1; only IK2_CPU_FRAC varies.
- ctrl-a median 22.96 (22.72/23.00/22.92/22.77/23.09/23.05); ctrl-b median 23.15 (23.16/23.16/22.87/23.18/23.13/23.05) — controls within ±0.8%.
- f0.15 → 25.63 (+11.6%); f0.25 → 25.86 (+12.6%); **f0.5 → 27.66 (+20.5%)** (26.95/27.36/27.48/27.83/28.87/29.51).
- CPU lane at f0.5: 20944 experts, 309.6 MiB/token moved off PCIe, 0.39 ms/visit avg; no slot deaths, no OOM, prefill unchanged 42-44.
- Monotonic rise through 0.5 → curve sweep dispatched (f0.5 ctrl / f0.75 / f0.9 / f0.5 ctrl) to find the peak before the live-config verdict.

## E6 RESULT (2026-11-22): hybrid CPU lane implemented — A/B in flight

### Implementation (fixer round-1, accepted after code review)
- `IK2_CPU_FRAC` (float 0..1, default 0): first ceil(frac×misses) experts per visit, ascending id, computed on CPU from pinned RAM via **ggml-cpu's own AVX2 vec_dot kernels** (the 9.78 GB/s probe measured exactly these; cover every expert type in both UD Q3_K_XL and Q4_K_XL — q3_K 5.9e-8, q8_0 8.3e-8, iq4_nl 1.3e-7, nvfp4 0 rel err; IQ2/IQ3 variants lack ggml quantizers for synthetic rows but run end-to-end). No hand port — cpu-probe file not needed.
- Mechanism: frac>0 disables the fast-token graph shortcut (verified :3447-3448 `g_cpu_frac <= 0` on the gate); deterministic split at :3869-3884 (misses insertion-sorted, first n_cpu to CPU, applied at :3901); activations D2H right after routing; results H2D into the combine buffer before moe_combine — visit waits for both lanes. Persistent worker pool, lazy init.
- Gates: default-off 40-token greedy **byte-identical to pre-df exe**; frac .25/.5 coherent English, deterministic across builds; `cpu lane: N experts, X MiB a token, Y ms a visit` stats line. Files: native_expert.{hpp,cpp} (shared kernels, additive), ik2_cpu.{hpp,cpp}, ik2_main.cu.
- Spot (single runs, host path): Q3 22.8 → 26.6 (.25) → **28.3 (.5)**; Q4 @8192 24.2 → 27.8 (.5).
- PRE-EXISTING BUG found (not ours): **Q4_K_XL at 131072 ctx dies `expert cache: no slot free` on the host path even with IK2_CPU_FRAC unset** (pristine pre-df exe reproduces: 394 tier slots / 3072 experts too few at that KV size). Q4 needs smaller ctx on this card until tier math changes.

### A/B (sh_2032f2f4e0014jlfrtFXGNE3tc, bench-cpufrac.ps1)
Arms: ctrl-a (frac unset) / 0.15 / 0.25 / 0.5 / ctrl-b, 6× --gen 256 each, Q3_K_XL 131072, env = live (IK2_POP_OUT=1, ROOM_MB=256, natural profile). Accept if median beats both controls beyond ±1.3%; watch no-slot-free, OOM, degenerate text. Logs: hill-climb\e6-*.log. Then: records + live-config decision (candidate env IK2_CPU_FRAC=<winner>).

### E6 A/B verdict (2026-11-22): ACCEPT — IK2_CPU_FRAC=0.5, live config updated

Curve (median of 6x --gen 256, Q3 @131072; env identical incl. IK2_POP_OUT=1, only IK2_CPU_FRAC varies):

| frac | 0 (ctrl) | 0.15 | 0.25 | 0.4 | 0.5 | 0.6 | 0.75 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| tok/s | 22.96 / 23.15 | 25.63 | 25.86 | 28.77 | **27.66 / 28.60 / 29.35 / 29.59 / 29.83** | 28.60 | 26.67 | 22.18 |

Peak at 0.5: beats 0.4 by +2-4% and 0.6 by +2.6-4.3% (beyond +/-1.3% noise), beats controls by ~+24-29%. At 0.75/0.9 CPU saturates (0.45/0.61 ms a visit, 546.8/911.0 MiB a token on lane). At 0.5: 20944 experts, 309.6 MiB/token off PCIe, 0.35-0.39 ms/visit, prefill unchanged 42-44, no slot deaths/OOM. Session drift visible across 0.5 arms (27.66 -> 29.83, same env) — take ~29.5 as representative; controls re-run in same arms confirm the delta is real.

Cumulative: 16.02 -> 19.05 (E1+E3) -> 23.06 (E7) -> ~29.6 (E6) = **+85%**. Live config `maya-inkling-2-mini.json` env now includes `"IK2_CPU_FRAC": "0.5"` (validated json.load). ON-mode accumulation order differs (not byte-identical); quality verified by coherent deterministic output on both quants + per-type selftests vs engine scalar (max rel err <=1.3e-7 on quantizable types). E6 opens a NEW two-bus ceiling (~36 tok/s ideal split; ~29.6 measured).

### E7 candidate (2026-11-22, zero-code): host-DMA path bench

- Round-1 spot run: IK2_POP_OUT=1 (forces host MV path, ik2_main.cu:3840; misses use copy-engine cudaMemcpyAsync, no graph) → 22.6 tok/s single-run vs 18.9 graph same session. Hypothesis: DMA (24.3 GB/s) beats the device copy kernel enough to pay for graph loss. A/B 6-sample bench queued after E5-B round-2 gates. Also fold lever A (nvidia-smi VRAM breakdown during decode) into same session.

### E5 round-1 result (2026-11-22): PREMISE WRONG for the default decode path — patch inert

The fixer implemented IK2_DF in the `exec` g_off branch, built, and passed all three byte-identical equivalence gates (DF=8, =4, unset; offload counters identical, 14976 picks = 39 tokens x 48 visits x 8). But it proved decode never runs that code:

- `forward()` :3452 short-circuits every T==1 token when `g_off` -> `fast_token` -> `fast_token_launches`, captured as ONE CUDA graph: device-side `k::route_token` -> `k::off_fetch_plan` -> `k::off_fetch_copy` (ik2_main.cu:2461-2477), all on the MAIN stream `g_s`. No host routing, no D2H sync, `exec` runs only for prefill (T=22). Temporary instrumentation (since removed) confirmed exec's g_off branch fires for the prompt only.
- The "host issue serialization" root cause above is therefore wrong for the bench config. Where the 17-vs-24.5 GiB/s gap actually lives (candidates): (1) per-byte efficiency of the SM copy kernel `off_fetch_copy` (device code reading pinned host memory over PCIe, typically below DMA copy-engine bandwidth), and/or (2) no overlap inside the graph — plan/copy/GEMM serialized on one stream, link idle during each visit's routing + expert GEMM window.
- Disposition of the inert patch: parked (default-off, byte-identical, zero risk), pending round-2 diagnosis of the REAL copy path. exe backup: build\Release\maya-ik2.pre-df.exe.
- Next: diagnosis-only round on `k::off_fetch_plan`/`k::off_fetch_copy` (src/ik2/ik2_off.cuh) — mechanism, access width, grid, tag semantics (hand-slot-only vs ring scan, ik2_off.cuh:97-99), whether the fast path maintains W.use; then ranked levers with gain math before any further edit.

### Lane-off control + VERDICT (2026-11-22, lab-bench-nolane.log)
- TRUE lane-off control (env -u IK2_CPU_FRAC, 6x gen 256): 9.89-9.90 tok/s all runs (graph path; cache stats print zeros).
- VERDICT: ACCEPT IK2_CPU_FRAC=0.5 for the lab. CPU lane = +86% on the lab (9.89 -> 18.40) vs +21% on the Windows box - Pascal emulated-mma GPU is compute-starved, CPU takes half the expert work.
- vs abandoned llama.cpp attempt (12.3 t/s @16K): +49%. Lab server live: http://192.168.1.82:8095 (LAN-only, ufw 192.168.1.0/24), config /home/lab/maya/maya-inkling-2-mini-lab.json, smoke 18.6 tok/s API / 29.7 prefill.
- 131K capacity arm (3x gen 256, live env): 17.49/17.50/17.51 -> 17.50 tok/s = -4.9% vs 16K baseline 18.40 (tier squeezed 1201/4.6 GiB -> 709/2.8 GiB by KV fp8 2.2 GiB; tier hit ~77 pct). Deep-context attention cost is separate and grows with actual prompt depth (llama.cpp data point: 10.5 @16K depth -> 7.9 @108K depth). Lab server restored and verified after bench.
### Lab config: 128K + low reasoning default (2026-11-22)
- maya-inkling-2-mini-lab.json: --max-context 16384->131072 + top-level "reasoning_effort": "low" (server.py:2842/:4014-4016; ik2 template effort_map low=0.2). Server restarted, smoke 17.7 tok/s (matches 131K capacity bench 17.50). Low-effort default config-verified (EFFORT_OK); output-verification inconclusive under max_tokens 200 cap. Requests override per-request via reasoning_effort/enable_thinking.
### Lab vision: CPU-only maya-vision (2026-11-22)
- Built /home/lab/maya/build-vision-cpu/bin/maya-vision (GGML_CUDA=OFF, gcc-14, patched pinned llama.cpp + inkling2v PROJECTOR_TYPE_INKLING2V); mmproj-BF16.gguf sha-verified on lab. Config vision entry added (gpu omitted = CPU encoder per request, zero VRAM). Smoke: 64x64 red PNG -> "A solid red square." correct, 18.2 tok/s text decode unaffected.
### Lab UI quant label fix (2026-11-22)
- Root cause: UI renders quant from the config's INSTALLER key (models.py:942 quant_label -> maya_quants.config_key; default IK2_QUANT=ud-q4_k_xl when absent). Lab config was hand-built without installer quant keys -> UI showed the family default "unsloth UD-Q4_K_XL" while the engine loads UD-Q3_K_XL. Fix: added installer family/quant/ik2_quant keys; server restarted and verified. Lesson: any hand-built config needs the installer quant keys or the UI label lies.
### Lab IQ3_XXS trial (2026-11-22): REJECTED - slower on Pascal
- UD-IQ3_XXS (11.66 GB, sha-verified) on lab @131072 live env: decode median 15.52 vs Q3_K_XL 17.50 (-11.3 pct). Tier hit better (82.5 pct, 912 slots/2.9 GiB) but IQ2_S/XS/XXS+IQ4_NL expert blocks have no MMA fast path -> GPU compute-bound despite fewer bytes. Quality OK (coherent numbered answer; minor sobes.tech boilerplate oddity in decode). Verdict: keep Q3_K_XL (NVFP4-dominated experts = MMA fast). Smaller-quant-is-faster assumption falsified on this engine: block-type kernel speed dominates byte savings.
### Lab Tier-A A/B (2026-11-22): ALL SEVEN LEVERS NEUTRAL on lab - plateau 17.5 confirmed
- Arms (6x --gen 256 @131072, live env): ctrl-a 17.50 / ctrl-b 17.49 (anchors +-0.5 pct). UP_TUNE 15.95 (-8.9 pct REJECT: measured t_up made the controller split less (0.28-0.39) and starved the lane; stock 0.36 default keeping split at 0.5 cap is better). SYNC_KILL 17.50 (0). STALE ~17.41 (-0.5 noise-negative). VWEIGHT ~17.52. CPU_ROWS ~17.52 (hand rows = ggml auto-vec, consistent with E9). CPU_MEAS ~17.56 (+0.3 noise). Lane telemetry: queue 0.3-0.66 vs link 0.35-1.03 ms - two buses balanced; eviction policy and per-visit overheads not binding at this tier/compute mix.
- KV audit finding: 48/48 planes written at 16K - NO dead KV slots, ExactLoop packing moot for this engine. i8r KV moot (our KV already fp8 ~400 B/token/visit < i8r 2112).
- Verdict: lab headroom is Tier B only (KV lending +15-30 pct depth-dependent; host-path fusion 2-5d) - user-gated 'later'. All levers stay default-off env-gated; live config unchanged. Windows deferred list: clock-lock check (3060 Ti util reads 20-40 pct on host path), HAGS A/B, UP_TUNE negative noted.
### Lab Tier-B P1: KV lending ACCEPTED (2026-11-22 overnight)
- IK2_KVLEND=1 (port of 381301c, adapted to resident-KV branch, host-path hit-via-lend at :4287-4290): 6x --gen 256 @131072 arms ctrl_a 17.43 / ctrl_b 17.54 / candidate 18.89 (+8.4 pct, beyond +-1.3 noise). 503 experts lent (1.85 GiB from position 8519 up), misses -32 pct (23.9k->16.2k). No slot deaths. ON-mode byte-identical (gates) - cache residency only.
- Applied to live lab config env (server picks up at next restart). Gain under source's +15-30 pct estimate because lab was already two-bus balanced; misses were only ~24 pct of picks.
### Lab Tier-B P2: IK2_FUSE NEUTRAL (2026-11-22 overnight)
- Arms 6x --gen 256 @131072 (no KVLEND isolate): ctrl_a 17.49 / FUSE 17.53 / ctrl_b 17.54 -> +0.2 pct inside noise. The byte-identical 3-launch->1 moe tail fusion is free but host-path launch overhead is not a lab bottleneck (consistent with Tier A). Stays default-off in tree, no config change.
- Stacked KVLEND+FUSE x3: 18.93 vs KVLEND-alone 18.89 -> stacking confirmed, FUSE adds nothing measurable. KV lending remains the whole Tier-B win so far.
### Lab Tier-B P3/P4 bench (2026-11-23 morning): elastic/decay/affinity all NEUTRAL at bench profile
- ctrl_a 17.49 / ctrl_b 17.54; KVLEND anchor 18.845 (consistent with accepted 18.89).
- IK2_ELASTIC=1: 18.855 (neutral vs anchor; controller worked as designed - hit rate 0.73 vs target 0.60, 4 rounds, 320 experts lent 1.17 GiB vs one-shot 503/1.85 GiB; extra 0.68 GiB one-shot lending bought ~nothing). Elastic value is at deeper/growing-KV workloads; stays default-off, KVLEND remains the live lever.
- KVLEND+IK2_DECAY=1: 18.845 (neutral - decayed ranking picks the same experts as raw use on short runs; value only on long mixed conversations). Default-off.
- KVLEND+IK2_LANE_AFFINITY=1: 18.87 (neutral - 8 workers on single-socket already fine). Default-off.
- Net live lab state: KVLEND=1 only; 18.9 tok/s @131K (+8.4 pct vs pre-TierB 17.5).
### Lab re-tune after KV lending (2026-11-23): frac 0.5 / room 64 STANDS
- Arms (6x --gen 256 @131072, all KVLEND=1): ctrl 18.84/18.88; f04 17.99 (-4.5 pct REJECT - less lane makes GPU compute-saturated despite fewer misses); f06 19.06 (+1.2 pct, inside +-1.3 noise, borderline - 5/6 runs at 19.05-19.11, tight; formally rejected, 12-sample confirm possible); r32 18.94 / r128 18.84 (noise). Config unchanged; lab best ~18.85-18.9.
- f04 mechanism: hits up 84.7k vs 81k, misses down 13.2k vs 16.7k, yet slower - the frac 0.5 point balances the two buses; less CPU lane shifts saturation to GPU compute.
### Lab DROUTE bench (2026-11-23): neutral on decode as designed; O1 baseline uncollected
- Arms (6x --gen 256 @131072, ALL KVLEND=1): ctrl_a 18.87 / IK2_DROUTE=1 18.89 / ctrl_b 18.87. DROUTE = +0.1 pct, neutral (expected - value is instrumentation, not speed).
- Gap: the per-visit three-timer stats line (:7548-7554 ON-only) did NOT print in lab runs - loose grep over the full log shows only kv lending lines. Need exact label from fixer to collect the lab-side routing/orchestration/lane ms baseline for O1 commissioning (Windows: 0.355 ms queue-drain / 0.061-0.075 ms orchestration / 0.112 ms lane wait).
- Cache consistent ~81.3k hits / ~16.4k misses per run; lending live in all arms (503 experts, 1.85 GiB).
### Lab O1/O2 bench (2026-11-23): DPACK REJECT -36 pct on lab; lab O-baseline collected
- Arms (6x --gen 256 @131072, all KVLEND=1): ctrl_a 18.76 / ctrl_b 18.74 clean. IK2_DPACK=1: ~11.94 (-36 pct REGRESSION). Trio DPACK+DCBATCH+DROUTE: ~12.06 same. Isolated cause = DPACK: mapped-host pack reads (cudaHostAllocMapped + plain loads ik2_kernels.cu:3699) are cheap on PCIe 4.0 Windows but ~0.65 ms/visit slower on Pascal PCIe 3.0 - zero-copy assumptions do not transfer across link speeds. Windows -10-13 us orchestration win does not transfer. If ever revisited: keep pack in device memory, overlap upload on copy stream. DPACK stays default-off everywhere; live config unaffected.
- Lab O-baseline (TRIO DROUTE timer line): orchestration 0.068-0.072 ms/visit [copies 0.046-0.050 dominant, launches 0.011 (cheap on Linux - no WDDM batch), lane jobs 0.008, classify 0.001]; lane wait 0.185-0.195; routing D2H 1.26-1.31 ms INFLATED by DPACK slow kernels (ctrl total is 1.10 ms/visit). Lab decomposition differs from Windows: launches cheap, copies expensive. DCBATCH unresolved (masked by DPACK; copies ceiling ~4 pct of visit, LOW priority).
- Portability notes for Windows tree (fold into next fixer report): ship ik2_cpu.hpp with O1b post_many; adopt CUDA_VERSION >= 13000 guard around cudaMemcpyBatchAsync (lab's clean cudaError_t batch form).
### Kernel-tuning round lab A/B (2026-11-23): change 3 REGRESSION, change 1 no-gain, 2+4 neutral
- Arms (6x --gen 256 @131072, KVLEND=1): ctrl_a (new default tiling) 15.75 / tune (IK2_TUNE=1 FMA) 15.73 / kt4 (IK2_KT_WARPS=4, old <4,2> tiling) 18.69 / ctrl_b 15.77. Old-tiling arm matches pre-kerntune anchor (18.87 +-1 pct drift).
- CHANGE 3 REJECT: the <8,1> occupancy chooser is a -16 pct regression on sm_61 (256-thread blocks lose to 128-thread; occupancy pct alone is not a predictor on this card). Falsified the fixer's 25-to-50 pct theory - ncu being blocked meant it shipped unvalidated; the A/B caught it (protocol works).
- CHANGE 1 (FMA contraction) REJECT: zero gain - compiler was already fusing. Keep IK2_TUNE off.
- CHANGES 2+4 neutral on decode (kt4 arm carries them and matches anchor). Keep (harmless; mv_blocks head path untested by decode bench).
- MITIGATION: IK2_KT_WARPS=4 added to lab config env; server restarted (PID 561698) - live server back at ~18.7-18.9 class speed.
- Lesson bank: on sm_61 shared-mem/latency profile favors 128-thread blocks; always A/B tiling theories, occupancy math alone lies; fixer caveat pattern (compiler-already-fusing) confirmed twice now.
### Kernel-round gate fix + END-STATE VERIFIED (2026-11-23)
- First revert attempt failed: arch_class() >= 70 gate wrong (arch_class() maps cc<75 -> 70 by design; Pascal passed and kept the losing tiling). Second revert attempt (MAYA_IK2_PASCAL ifdef) rejected pre-ship: define is PRIVATE to maya-ik2 target, invisible in ik2_kernels library TU.
- FINAL FIX (verified): kt_wide_blocks reads cudaDevAttrComputeCapabilityMajor (raw, pre-upclass), IK2_KT_WARPS pins checked first, unaskable -> 4 warps. IK2_KT_LOG=1 prints on lab: 'kernel tiling: 4 warps a block (device major 6, Pascal)'. No-env default bench: 18.68/18.71/18.70 = winning speed with NO config pin. Lab config cleaned (KT_WARPS removed), server PID 564378.
- LESSON BANK (3 entries this round): (1) occupancy pct is not a predictor on sm_61 - 128-thread blocks beat 256-thread by 16 pct; (2) arch_class() is a capability-class mapper that upclasses Pascal to 70 - never use it as a real-capability test; (3) target-PRIVATE defines are invisible to linked libraries - check CMake scope before ifdef-gating.
### O3 re-scope CLOSED per stop-rule (2026-11-23): split decision = 1 micros/visit
- Measured with DROUTE instrumentation (3 runs, 12288 visits): split 0.001 ms, lookups 0.001, copies 0.008, lane jobs 0.006, pack 0.012, launches 0.029-0.032; orchestration 0.057-0.060; visit 0.56-0.57 ms. Split is 1/57th of the orchestration the 4-5 pct estimate was attached to - a tenth of the 10 micros stop-rule.
- Ceiling re-based 0.2-0.4 pct (was 4-5). Negative banked for BOTH device-split variants (split AND whole-classification: lookups also 1 micros). Device form loses: host wakes for routing D2H anyway; post-split work unchanged; split needs residency publication (the O1 wall) for a 1 micros prize.
- Landed: inert pricing instrument - DROUTE stats line now decomposes [split/lookups/copies/lane jobs/tables/pack/launches] permanently (main.cu:316/:4443-4448/:4489/:7634-7641). Gates green, default-off byte-identical, pre-dsplit.exe backup.
- Remaining structural item: the launch block (29-32 micros x 48 = 2.7 pct on WINDOWS; lab launches 0.011 ms = ~0.5 pct, thin) - needs fewer-launches or captured-visit design, unapproved project.
- Push-to-20 now rides on the Phase-1 env stack (in flight).
### PUSH-TO-20 FINAL (2026-11-23): env space flat, plateau ~18.7 confirmed - goal not reachable by tuning
- Arms (6x each @131072 KVLEND=1; f06 12 samples): ctrl 18.72/18.69 / f06 18.76 (+0.2 pct - the 19.06 borderline DEAD, was session drift) / r16 18.65 / dcb 18.69 (batch-copy no-op on Linux too) / thr8 18.68 / thr6 18.56 (default 7 stays) / combo 18.76. ALL inside +-1.3 pct noise. No config change.
- Campaign accounting: Phase 1 env stack all-neutral; O3 closed by measurement (1 micros split, 0.2-0.4 pct ceiling); kernel round closed (tiling regression reverted, FMA already-fused, 2 neutrals); launch block = ~0.5 pct on lab, not commissionable.
- BINDING CONSTRAINTS (measured): PCIe 3.0 fetch bandwidth + Pascal emulated-mma compute; two-bus balanced (queue 0.33-0.62 vs link 0.49-0.96 ms/visit); GPU 92-98 pct busy with custom kernels instruction-bound.
- 20+ requires: launch-block/captured-visit project (+2-3 pct optimistic, weeks, low confidence) OR hardware (12 GB card: tier doubles, Strata-fork scaling evidence suggests 25+; the 3060 Ti Windows box does 30 for the same reason).
- Lab final standing: ~18.7-18.9 tok/s @131K (from 9.9 lane-off / 12.3 llama.cpp), server live with KVLEND+ELASTIC+DECAY config.
### Windows KV lending A/B (2026-11-23): ACCEPT +9.5 pct - the lab's winner transfers
- 12x ctrl / 12x IK2_KVLEND=1, --gen 256 @131072, standing env (incl. POP_OUT host path + CPU_FRAC 0.5), build\Release\maya-ik2.exe: ctrl 22.16-23.29 (median ~23.1, noisy box OpenCode/Discord) vs KVLEND 25.22-25.48 (median ~25.3) = +9.5 pct beyond noise.
- Mechanism: Windows tier at 131K is desktop-starved (332 slots/1.3 GiB) -> 1.84 GiB lent to ~481 experts triples effective coverage; cache misses 38.1k->27.6k (-27 pct). Same mechanics as lab (+8.4 pct at 709-slot base).
- APPLIED: IK2_KVLEND=1 added to maya-inkling-2-mini.json env (ReadAllLines method, json.load validated; env keys now CPU_FRAC/KV/KVLEND/POP_OUT/ROOM_MB/UNSUPPORTED/WARM). Server picks up at next restart.
- Windows standing decode: ~25.3 tok/s @131K CLI with lending (was ~23-24 in-equivalent-conditions without; pre-lending best across sessions ~29.7-30.1 was QUIET-box at 16K-era conditions - not comparable).
- Prior 263s startup pathology resolved: transient page-cache eviction (8.4s after warm load).
### Windows tiling A/B (2026-11-23): 8-warp default REJECTED on sm_86 too
- 12x default-8 / 11x kt4, --gen 256 @131072, standing env incl KVLEND=1: default-8 median 25.17 (one 21.16 outlier) vs KT_WARPS=4 median 27.10 (tight 26.94-27.45) = +7.67 pct for the OLD 4-warp tiling on sm_86. Occupancy theory falsified on BOTH arches (lab -16 pct, win -7.7 pct). The kernel-round default has been taxing the Windows box since it landed (the KVLEND A/B arms carried it).
- MITIGATION: IK2_KT_WARPS=4 added to Windows live config env. Windows standing with 4-warp + KVLEND ~27.1 tok/s noisy-box.
- HAGS: HwSchMode=2 = ALREADY ON; testing OFF needs admin toggle + reboot (disruptive, ambiguous prior) - handed to user, recommend skip.
### Windows queue closed (2026-11-23): universal 4-warp default; launch block banked
- Task A: kt_wide_blocks now `const bool w = kw == 8` - only the env override selects the 8-warp form; arch_class() out of the path; falsification comment (sm_61 -16 pct, sm_86 -7.7 pct); gates green, sm_86 default changed 8->4 warp with byte-identical output (tilings same-bytes, proven twice).
- Task B: launch block priced per-call (part0 gu_swiglu 14-15 micros incl ~10 first-call submit, part0 down 8-9, part1 pair 7-8; total 29-32 micros/visit): cheap fusions 4-7 micros = below the 10 micros bar; the only >=10 form needs per-type block-form expert kernels (only Q8_0 pair exists) = O3-scale kernel project for a Windows-only 2.1-2.8 pct; CUDA-graph visit infeasible (per-visit params; SetParams ~2-4 micros each on WDDM replaces the launches it saves). BANKED negative with the honest reopen condition.
- Queue final: (2) tiling resolved both arches; (3) HAGS already ON, OFF-test needs reboot - user decision, recommend skip; (4) launch block banked. Windows standing ~27.1 tok/s noisy-box (KVLEND + 4-warp tiling), best single run 26.75.
### Strata-fork borrows A/B (2026-11-23): POOL2 REJECT, CCACHE ACCEPT
- POOL2 (fork pool port, POD ring + host-drain + host-sibling pinning): 12x12 A/B ctrl median 26.815 vs pool2 25.675 = -4.3 pct REGRESSION, tight non-overlapping bands. The fixer's per-expert EWMA gain (-18 pct ms/expert) did not survive wall-clock: host-drain participation taxes the per-visit critical path (host orchestration is the serial spine on Windows; the fork's 1.35x was measured on their Linux/NUMA/native workload where the host is not the bottleneck). Stays default-off, byte-identical, zero risk. LESSON (2nd time): per-component metrics can improve while end-to-end regresses - wall-clock A/B is the arbiter.
- CCACHE rung 1a (token-id prefix match, snaps survive reset): ACCEPTED - byte-exact restore, two-conversation switch 0.000 s vs 0.672 s re-prefill baseline; IK2_CCACHE=1 applied to Windows live config. Decode unaffected by design. Rung 1b (ids cap) later if long conversations show memory pressure.
### Strata-fork borrows round 2 (2026-11-23): WATCH + E4LUT/XGLOBAL + PFSTG
- IK2_WATCH (new src/ik2/ik2_watch.h): named-stream TDR guard, report at IK2_WATCH_MS 1500, fatal exit 3 at 8000 naming the wait; wired routing+lane waits; byte-identical all modes; robustness tool, default-off, no speed claim.
- IK2_E4LUT (constant c_e4m3[256] LUT, bit-for-bit asserted vs ue4m3, 7 call sites) + IK2_XGLOBAL (x through global for NVFP4 pairs, byte-identical by construction): both gated default-off, byte-identical; A/B in flight - ALSO settles the type-mix conflict (docs say Q3_K_XL experts are IQ-only; E9 lane measured nvfp4 55.5 pct; if XGLOBAL moves the needle, NVFP4 runs on this box).
- IK2_PFSTG (contained prefill staging): per-prefill rings sized lazily from free VRAM, plan kernel untouched, tier intact (proven). At live config (room 64) free VRAM = 0 -> fallback always = INERT; demoed 41 slots/pool at room 2048 (trades tier). Two-half PIPELINING (the actual prompt win) not in this variant. Stays default-off; revisit on lab unfreeze (copy-limited 8192-token protocol is the real test).
- Big-chunk path stays dead on 8 GB (fr >= 20 GiB, hardware).
- NOISE ROOT CAUSE FOUND: two hung fixer gate processes held the GPU during earlier timing windows (killed) - explains the 15.8-27.7 bimodal spread; fixer timing anecdotes void; orchestrator 12x12 medians stand.
### NVFP4 micro-tests verdict + type-mix conflict settled (2026-11-23)
- DIRECT GGUF SCAN settles it: expert *exps* tensors = F32 40 + NVFP4 32 + IQ2_XS 1 + IQ2_S 3 + IQ3_XXS 5 + Q3_K 1 + IQ4_NL 4 + IQ3_S 2. NVFP4 IS in the experts (32 tensors); E9's 55.5 pct lane measurement VINDICATED; docs IQ-only claim and fixer no-NVFP4 caveat both WRONG.
- E4LUT/XGLOBAL A/B (12x sequential arms): ctrl 25.16 / e4lut 26.78 / xglobal 27.05 medians - INCONCLUSIVE (session warm-in confound in the sequential design; tail-to-tail inside drift). Both flags byte-identical by construction (LUT bit-for-bit asserted; same products same order) = zero-risk. Lesson: interleave arms for micro-levers.
- Disposition: both flags default-off in tree, zero-risk to enable (unproven speed). E4LUT+XGLOBAL enabled in Windows live config as proven-harmless freebies; removal = one env key each.
### Our own tree: branch ik2-tuned (2026-11-23, per user directive)
- Branch ik2-tuned created from ik2-ud-quants-offload; commit 868cbc9 (8 files +1607/-89) = the full accepted+default-off lever set; pushed to TokenTinkererAI fork. PR #1133 untouched (tracks the old branch). Upstream cherry-picks dropped per user. Windows repo working tree now clean (untracked: stray 1, START-INKLING.bat, data profiles/ids, hill-climb/ - intentional). This branch = the canonical OUR-ENGINE tree from now on.
### KV4 FINAL VERDICT (2026-11-23): quality gate FAILED decisively - BANKED default-off per the settled design
- Pre-registered gate: top-1 >= 98 pct vs fp8 over 5 natural 512-token generations (greedy, lane-off deterministic, no --seed - unknown argument, voided v1/v2). MEASURED: 0.4-1.6 pct agreement (P1 7/512, P2 8/512, P3 4/512, P4 2/512), first divergence at the second token every pair; P5 kv4 arm no output (moot). Both arms fluent - the 4-bit KV is a different distribution, not a small perturbation. Needle test moot (single-gate failure).
- THE SPLIT RECORD (the oracle's banking form): codec/pack/geometry PROVEN REUSABLE (round trip 0.11x fp8, rows rms 0.0476, Hadamard in-tree, selftests live); FOUR real bugs found+fixed along the way (test-harness host-pointer fault, missing kv4_pack_subblock call, signed-q-vs-unsigned-mag quantisation fabsf, device/host scale-encoder divergence, AND the final launch-argument misalignment kv4-into-q_ld at kernels.cu:7425 - prefill-harmless/decode-fatal, the true cause of the degenerate text); capacity PROVEN (56.2 pct of fp8, lending composes); speed benefit real at 131K (kv4 arms 28.5 vs 27.0 tok/s lane-off) but NOT banked; quality FAILED (top-1 ~1 pct vs 98 pct).
- Untried quality-preserving option on record: 8-bit K + 4-bit V split (storage parameterised by kv4_row_bytes) - earlier rejected as a blind hedge; only viable follow-up if ever wanted.
- Disposition: IK2_KV4 stays experimental default-off in tree (OFF path byte-identical, suite green); debug prints env-gated inert (IK2_KVDUMP/IK2_KV4STORE); backups pre-kv4..pre-kv4dec in build\Release.
### K8V4 quality verdict (2026-11-23): FAIL - same as K4V4; the 4-bit KV campaign closes BANKED
- The identical v3 protocol (5 natural prompts, --gen 512 @131072, greedy lane-off, IK2_KV4V=1 vs fp8): top-1 agreement 0.6-2 pct vs the 98 pct bar (P1 5/512, P2 10/512, P3 6/512, P4 3/512; first divergence at token index 1 on every pair; P5 kv4v generated line did not parse - moot at 4/4). Both arms fluent 511-token streams; kv4v at speed parity (27.3-28.7 vs fp8 27.3-28.9) with better cache stats (27-32k misses vs 29-36k); banner honest (K fp8 / V 4-bit, 1.75 GiB = 78.5 pct of fp8).
- CONCLUSION across all three 4-bit forms (K4V4 both, K8V4 V-only): 4-bit KV on this model = a DIFFERENT DISTRIBUTION at generation length - fluent but not the same text; token-exactness with fp8 unreachable at 4-bit. The best case was the short-prompt 40-gen (40/40 identical); 512-token generation diverges at the second token.
- SPLIT RECORD: codec/pack/geometry PROVEN REUSABLE (round trip 0.11x fp8, rows rms 0.0476, Hadamard in-tree, per-side storage, selftests live); SIX real engine bugs found+fixed during the campaign (harness host-pointer, missing pack call, signed-q fabsf, device/host encoder divergence, the q_ld launch misalignment at kernels.cu:7425, the forward_prefill_sup >64-token routing); capacity PROVEN (56.2/78.5 pct, lending composes mixed-size); speed real NOT banked; quality FAILED the pre-registered top-1 bar in all forms.
- OPEN PATH (user's call, not chosen): a QUALITY-metric acceptance (perplexity/eval/needle-recall) instead of token-exactness would reopen the 1.26-1.75 GiB capacity option. K4V4 re-measurement post-routing-fix available but unlikely to change the 512-gen character.
### CORRECTION (2026-11-24, oracle-verified): the KV4/K8V4 quality numbers above were VOID - a pairing artifact in the analysis
- The v3 top-1 computation paired the fp8/kv4 token streams across PROMPTS (off-diagonal); all reported numbers (top-1 0.4-2 pct, firstdiv 1) sit in the cross-prompt range. Log structure verified clean (10 ARM markers, correct order) - the bug was in the analysis script, not the runs.
- HONEST DIAGONAL NUMBERS: K8V4 firstdiv 121/47/30/426/195, agreement 130/65/37/428/212 of 512 (25/13/7/84/41 pct - P4 84 pct); K4V4 firstdiv 44/3/91/173/1, agreement 94/15/97/176/19 (2/5 prompts flip by token 3 - the honest argument against both-sides-4-bit: K-side score errors gate softmax directly).
- BOTH FORMS STILL FAIL the 98 pct top-1 bar at 512-gen - but the divergence is PROGRESSIVE (ordinary accumulated KV drift, K8V4 median ~121 tokens), not token-2 collapse. The format is far better than the void numbers claimed.
- GATE VERDICT (oracle): top-1-exactness at 512-gen is structurally the wrong instrument for ANY lossy KV INCLUDING fp8 - the engine's own docs (ik2_main.cu:3534) state attention is 'a few 1e-5 off the one-token path, so that drafting changes greedy answers' (knife-edge argmax, per-token flip prob 0.2-3 pct - 40-gen survives, 512-gen essentially never). No calibration arm (fp8-vs-fp16) was ever run; it would likely fail too.
- PROPER REGISTERED GATE (oracle, for any future re-judgment): harness self-check (pair by (prompt,arm), hard-fail parse misses, fp8-vs-fp8 stability) + teacher-forced KL (mean <0.02 nats, p99 <0.1) + needle recall at 131K (>=95 pct of fp8 own recall) + PPL delta (<+0.5 pct) + 64-token exact prefix (>=90 pct pairs).
- Standing verdict AMENDED: both forms stay BANKED default-off (the pre-registered bar failed), but the record now carries the honest progressive-divergence numbers + the measurement-error disclosure. K8V4 is the plausible-revisit candidate (median 121-token tracking); codec levers if ever: hot-row fp8 mix (sink rows), per-8 scales (5.5 bits/value).
### K8V4 FINAL DISPOSITION (2026-11-24, user decision)
- User: "keep the K8V4 as an option but KV8 stays the default" - IK2_KV4V=1 remains in-tree as a documented, env-gated option (requires IK2_KV=fp8, refuses IK2_CCACHE by name, banner K fp8 / V 4-bit E2M1 x ue4m3 ~1.75 GiB @131K = 78.5 pct of fp8); the default KV format stays fp8 (live config unchanged, carries no IK2_KV4V).
- The full re-judge table (the oracle-registered gate): 64-token exact prefix FAIL 60 pct (knife-edge sampling - even fp8-vs-fp16 would flip); KL PASS 4/4 (mean 0.002-0.007 nats, 3-9x margin); PPL FAIL 3/4 (+1.1-2.5 pct, self-text bias caveat); needle at 51K depth PASS 100 pct (byte-identical 47-token answers - depth recall untouched by the 4-bit V).
- Honest speed costs (measured at 102K depth): prefill 3.6x slower (215 vs 781 tok/s - the 4-bit V unpack in the chunked prefill), deep decode ~2x slower (5.83 vs 11.39 tok/s); parity at short/16K workloads. The option suits capacity-tight setups that accept long-prompt prefill cost.
- CAMPAIGN LEGACY: six real engine bugs fixed permanently (incl. the q_ld launch misalignment - the true degenerate-text cause - and the forward_prefill_sup >64-token routing); the --score teacher-forced instrument landed (default-off); the @tokens @file reader hardened (named errors, chunked read); two KV formats gated (K4V4/K8V4, both default-off, selftests live, lending composes with mixed row sizes).

### OPTIONS-REMAINING follow-through: A (room sweep) + B (Q4_K_M) â€” both NEUTRAL/REJECT
- A: Windows room/headroom sweep (IK2_ROOM_MB 64/128/256, 12x, lane OFF): medians 24.29/24.35/24.93, ranges overlap everywhere (+-4-5 pct); tier banners varied run-to-run (392-485 slots, 1.6-1.9 GiB) as desktop VRAM fluctuated at init, swamping the room effect. Verdict: keep ROOM 64; the headroom lever is exhausted (matches the lab sweep).
- B: UD-Q4_K_M vs Q3_K_XL (12x each, live env, lane ON): Q3 median 28.39 (27.66-28.83) vs Q4M median 27.07 (24.62-27.44) = -4.6 pct. Mechanism: +22 pct expert bytes (16.6 vs 13.6 GB) -> more bytes per miss; the native-block decode advantage does not overcome it (2 pools vs 4; load 3.8 vs 2.1 s). Q4M's best run trails Q3's median - robust. Verdict: REJECT swap per the established rule; Q3_K_XL stays default; Q4M stays on disk as the quality-first option (~5 pct decode cost).
- Remaining inventory (OPTIONS-REMAINING.md, unstarted): LM-head precision requant (fixer), int8/fp8 prefill GEMMs (labelled lossy), grouped 64-row M-block (bit-exact, +7 pct prefill prior), launch-fusion re-price, LM-head custom GEMV, hardware. Skill-checklist gaps: direct GEMM TFLOPS, needle at ~100 pct depth.
### Defaults flip + vision-on-CPU-by-default (2026-11-24, user directive: "commit everything to ik2-tuned with my settings at default ... vision on cpu on by default")
- Seven proven settings became CODE defaults (explicit env still wins): IK2_KV=fp8 (main.cu:449), IK2_KVLEND=on (:2993), IK2_CCACHE=on (:5841), IK2_E4LUT=on (:5878), IK2_XGLOBAL=on (:5883), IK2_CPU_FRAC=0.5 (:5819), IK2_ROOM_MB=64 (:6850). All gated experiments stay default-off. IK2_WARM/IK2_POP_OUT/IK2_UNSUPPORTED stay explicit.
- FINDING: IK2_KT_WARPS does not exist in this tree - the live config's key is inert decoration; the universal 4-warp tiling is the kernel chooser's default (kt_wide_blocks), not an env.
- Vision: serve/server.py DEFAULT_VISION_EXE/MMPROJ + default_ik2_vision (CPU, no gpu key); ik2 family gets it by default; absent binary/mmproj -> NOTE 'vision is off - no CPU encoder' and continue (Windows path); maya.py write_config_ik2 writes the same block for fresh configs. ds4/hy4/mm3 untouched.
- Gates: suite green; live config reproduces 27.66 tok/s (standing ~28); seven-unset == seven-explicit (14/14 lines, lane pinned); Windows serve vision NOTE clean + request served; KV4V+new-CCACHE -> NOTE + auto-disable, K8V4 runs.
- Commits: fa49ffb (WIP checkpoint: KV4/K8V4/score/reader/4-warp default) + 47dd834 (defaults + vision), pushed to TokenTinkererAI fork ik2-tuned (868cbc9..47dd834). fa49ffb subject carries a cosmetic BOM.
- Live config UNCHANGED (its explicit env keys now equal the code defaults except IK2_KT_WARPS=4 which is inert; could be dropped from the config - not done, harmless).
### Windows CPU vision live (2026-11-24, user directive "make the vision work on cpu on this box")

- Build: tools/vision/inkling2v_patch.py reported 'inkling2v vision added' but changed nothing â€” the vendored llama.cpp (pinned 3cf03257f) already carried the code (idempotent script, per-anchor skip confirmed via git status clean + inkling2v present in tools/mtmd/*). cmake -S tools/vision -B build-vision-cpu -DGGML_CUDA=OFF -DSTRATA_VISION_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -> CPU backend only (ggml-cpu, /arch:AVX2 + FMA + F16C). Build exit 0, 4,686,336 B at build-vision-cpu/bin/Release/maya-vision.exe, copied to build-vision-cpu/bin/maya-vision.exe (the default resolver's first candidate).
- mmproj: hardlink models/mmproj-BF16.gguf -> Maya-data/models/inkling-2-mini/mmproj-BF16.gguf; 641,805,760 B; destination sha256 2C165C78...1A666E7 EXACT vs the HF pin (orchestrator-verified too).
- End-to-end proof (server with the LIVE config, no vision block â€” the default resolves): banner 'images: the CPU encoder is on by default (...build-vision-cpu\bin\maya-vision.exe with mmproj-BF16.gguf); a "vision" block in the config overrides it' + 'images on'. 64x64 solid-red PNG -> 'A solid red square.' twice (3.4 s, 4.1 s), finish stop. VRAM flat (7802 -> 7816 MiB, 14 MiB allocator drift; encoder CUDA-free). Graceful CTRL_BREAK stop, exit 0, zero leftover processes.
- Standalone encode check: 'READY 2304 / OK 4 4 1 57', emb.bin 36,884 B = 20 B header + 16 x 2304 F32 (16 image tokens for the 64x64 test geometry).
- Artefacts are gitignored (models/ .gitignore:30, build-vision*/ :52) â€” the CODE default is what was committed/pushed (47dd834). No code change was needed for this box beyond 47dd834.
- Deviations: patch idempotency (nothing to change); the first e2e harness searched '^ready: http' without re.M and burned its own timeout before the request (timings above are real); one-sentence answers at reasoning_effort=high.
### P4/P5/P3/P2 outcome (2026-11-24, items 2-5 of the remaining-options inventory)

- **P4 launch-fusion re-price (DONE, priced)**: IK2_DROUTE=1 at the current binary: launches 0.029 ms/visit x 48 = 1.39 ms/token = 3.9 pct of the 35.7 ms token at 28 tok/s (up from the earlier 2.7 pct estimate). Orchestration total 0.061 ms/visit; launches are the largest single component. Ceiling for the captured-visit/fewer-launches project: ~+4 pct. Not started.
- **P5 LM-head GEMV (tiling half shipped, NEUTRAL)**: IK2_HEADGEMV=1 switches the 200k head's mv_blocks kernel to <8,4> (32 rows a block) for N>16384. Bit-exact: 40-token greedy identical, --score 499 positions hash-identical (both mean_nll 2.852284, ppl 17.327320). Speed A/B (12v12, --gen 256 @131072, standing env): off median 29.13, on median 29.14 = +0.03 pct, dead neutral. DIAGNOSIS: the head's ~4.05 ms (of ~35.7 ms/token) is instruction-bound per byte (72-94 GB/s effective on 290-380 MB/token of Q6_K head weights) - block/staging overhead was not the term. The deeper rewrite (vec4/uint4 unpack, two 64-value groups in flight, half2 dots + F32 group-end accumulation, launch_bounds(256,2), row prefetch; 2-3 days) targets the actual bottleneck but CHANGES accumulation order -> requires --score gates. Flag stays default-off (inert but harmless).
- **P3 grouped 64-row M-block (STOPPED at audit, wall)**: the prefill MoE GEMM's M-block is 1 by design (blockIdx.y = one (token,expert) slot; load_xs_f32 stages one token); the grouped kernels cap at kGroupMax=8 (run<1>/<2>/<4>/<8> on ng); the packer windows S=T*K <= 64 slots. Reaching 64 rows needs a K-tiled shared-staging kernel + a wider packer = 3-5 days; byte-identity would be free (M-blocking only regroups rows). Prior from the skill: +7 pct prefill (prefill-only). USER DECISION pending on commissioning.
- **P2 int8/fp8 prefill GEMMs (STOPPED at scope, platform wall)**: no fp8 GEMM exists in-tree (all tensor-core helpers are F16: mma16816, m16n8k8, wgmma, mma884, used only by attention) and compute_86 HAS NO fp8 tensor-core instruction (mma f32.e4m3 starts at Ada sm_89/Hopper). On this box an fp8 GEMM = fp8 storage decoded to f16 for f16 MMA = bandwidth-only, ~1.3-1.6x at best on the MoE GEMMs, for 2-4 weeks of work + a labelled quality trade. NOT WORTH COMMISSIONING on sm_86; would be a real lever only on Ada/Hopper.
- Net: of items 2-5, one measured neutral (P5 tiling), two real walls (P3 effort, P2 platform), one price confirmed (P4 3.9 pct). Decode standing unchanged ~28-29 tok/s @131K.

### Thinking level: low by default (2026-11-24, user directive "make reasonning low the default on both the lab and this machine")

- Code default: tools/maya_quants.py MODEL_DEFAULTS["ik2-mini"] "high" -> "low" (0.2; the model card's own is 0.9 "high"). This is what a setup writes into a new config, and what a config that names no level takes (serve/server.py checked_config_effort -> maya_quants.model_defaults). docs/INKLING2.md notes which default is Maya's.
- This machine: maya-inkling-2-mini.json reasoning_effort "high" -> "low" (gitignored live config).
- Lab (GTX 1070, ssh lab): ~/maya/maya-inkling-2-mini-lab.json "minimal" -> "low" (backup .bak-2026-11-24); the running server (pid 591972) ignored SIGINT because it was started detached (handler not installed when the disposition is SIG_IGN), stopped with SIGTERM in 52 s, clean; relaunched setsid/nohup from /home/lab/maya -> pid 739902, engine 739909, ready in 22 s. Log: "thinking level when a request names none: low"; a request with no reasoning_effort answered 391 for 17*23 (35 thinking tokens, 48 total, 3.8 s).
- Tests: tools/test_setup_ik2.py 29 tests OK (updated the two expectations this changes: the model's default level, and the vision block on a box where the CPU encoder is built). Commit ddd44b1 on ik2-tuned (not pushed).
### Lab Tier 0 attempt 1 (2026-11-24): WHOLESALE ik2-tuned MERGE FAILED GATES - rolled back

- Method: checkpointed the lab's dirty tree (`00d81ae`, 9 files +1668/-756), backed up the binary (`maya-ik2.pre-ik2tuned`, md5 1cdcfd01), fetched ik2-tuned (ddd44b1) as a git bundle, `merge --no-ff` -> `0e790980` on branch `ik2-tuned-merged`, 7 conflicted files / 26 hunks all resolved to incoming after per-file proof of superset + byte-identical Pascal guard lines.
- Baseline control (today, live env, --gen 256 @131072, comma-joined 65-id prompt): 18.50/18.37/18.45/18.46/18.43/18.37 -> median 18.44 (NOTE: below the documented 18.7-18.9; box state drift, use today's control for gates).
- MERGED BINARY: decode median 17.39 = **-5.7 pct**; interleaved 3-pair control 18.50 vs 17.37 = **-6.1 pct, real regression**. Not recovered by any default-on lever (17.34-17.37).
- DETERMINISM FAIL: 40-token greedy under live env = 72cf3116, 72cf3116, d2dfba4d (then again mixed) vs pre-merge 72cf3116 in 4/4. Stable only with IK2_CPU_FRAC=0 and levers off, and then still numerically different (56aacfc9 vs 294f0d33) -> the merged GPU kernel delta (attention/expert launch refactor + mv_blocks/head path) does not transfer to sm_61. Nondeterminism also suggests an uninitialised read in the incoming path on Pascal (compute-sanitizer not yet run).
- VERDICT: the ik2-tuned kernel delta is Ampere-shaped; wholesale sync is wrong. Rolled back (`git reset --hard 00d81ae`, binary restored, server back, pid 754709). Lab was down ~61 min.
- NEXT: surgical port of ONLY CCACHE + E4LUT + XGLOBAL onto `00d81ae` (branch `lab-tier0`), same gates (4/4 greedy determinism + median >= 18.3), then Tier 1 profiling.
- PROTOCOL TRAP (banked): `--tokens` splits on COMMAS only. A space-separated id file silently becomes a 1-token prompt (17.28-17.66 instead of 18.4) - join ids with commas for every lab bench, including the needle files.
- Build note: full `cmake --build build` exits 2 on the pre-existing, unrelated `ds4_gguf_test` (src/ds4/ds4_gguf.cu nvcuda::wmma at sm_61); build `maya-ik2 ik2_kernels_test` as targets instead.
### Lab Tier 0 attempt 2 (2026-11-24): surgical port - CCACHE + XGLOBAL SHIPPED, E4LUT REJECTED

- Branch `lab-tier0` from the lab's own `00d81ae`, 2 commits, 96 insertions / 5 deletions in 3 files (`454503db` port, `515e2d8b` drop E4LUT). Hunks taken from `git diff 00d81ae ik2-tuned-merged` and filtered; final diff greps clean for kv4|KV4|watch::|PFSTG|HEADGEMV|g_dr_split|score -> nothing unrelated rode along. No branch merge.
- Gates (server stopped; kernel suite OOMs while the engine holds VRAM), ~44 min downtime:
  - build + `ik2_kernels_test` exit 0, all passed.
  - 40-token greedy: levers OFF 72cf3116 4/4; CCACHE+XGLOBAL ON 72cf3116 4/4; both match the pre-merge reference -> the determinism gate the merge failed now passes.
  - decode 6x, levers forced off: 18.44/18.43/18.45/18.38/18.37/18.43 -> median 18.43 (gate >= 18.3 PASS; today's control 18.44). No regression.
- CCACHE (the win): cached_tokens 29 / 26 / 69 across three prefix-reuse requests (previously always 0); controlled TTFT cold 79-tok prompt 2.110 s vs prefix-hit (69/69 cached) 0.119 s = **1.99 s of prefill removed** (Windows equivalent was 0.672 s). Enabled IK2_CCACHE=1 in the live config.
- XGLOBAL: interleaved 6v6 ctrl 18.39 vs 18.35 = -0.2 pct, inside noise -> NEUTRAL on sm_61, enabled as specified but not a speed lever (its value is byte-identity by construction).
- E4LUT: REJECTED on measurement. Ported, built, gated: forced-off median 17.68 (17.66-17.77) vs pre-merge 18.41/18.36/18.45 = -4 pct even with the lever off (the per-scale `g_e4lut ? c_e4m3[x] : ue4m3(x)` select + constant-memory loads cost on Pascal); with IK2_E4LUT=1 17.57 = -5 pct. Reverted in 515e2d8b, no trace left. => its "byte-identical, zero risk" claim held on Windows but NOT on sm_61.
- Lab defaults are env-gated OFF in code (getenv && atoi) unlike ik2-tuned's on-by-default; the live config names CCACHE=1, XGLOBAL=1 explicitly.
- State: server pid 762649, engine 762667, /v1/models 200, ready 23 s. Nothing pushed to any remote (lab commits local).
### Lab Tier 1 profiling (2026-11-24) - the lab's real cost structure

Method: one gated instrument (IK2_PROF only) splitting the LM head out of `rest` in `run_head` (lab-tier0 @ 7087569c, 3 hunks, net +2/-1 in ik2_main.cu); zero effect when IK2_PROF is unset (greedy 72cf3116 unchanged). Profiles at 65-token and 102,020-token prompts, --gen 128, live env, levers on. Prof overhead measured at +29 pct (stream syncs), so long-context clean figures are estimates, flagged.

- SHORT context (per token): attention 15.6 ms (22 pct), router 2.1, uploads 18.0 (26 pct), experts 27.6 (39 pct), head 6.8 (10 pct); wall 70.2 ms profiled vs 54.3 clean (18.4 tok/s).
- 102K context (per token): attention 67.4 ms (49 pct, 4.3x the short figure), uploads 32.9 (24 pct), experts 27.8 (20 pct), head 6.8 (5 pct), router 2.1; wall 137.0 profiled, est. ~121 ms clean = ~8.3 tok/s.
- So context costs +66.8 ms/token of which ATTENTION is +51.8 ms (Pascal mma884 scales with keys as expected); experts +0.2, head -0.1 (context-independent). The head is a short-context term (10 pct), NOT the deep-context lever.
- DROUTE at long context: 0.635 ms/visit x 48 = 30.5 ms/token (~22 pct, inside the phase sums, not additive). Dominant host terms: tier tables 0.286 ms/visit (~13.7 ms/token), copies 0.205 (~9.8 ms/token); launches 0.004 ms/visit -> launch fusion is dead on the lab, as predicted.
- PREFILL is the lab's real long-context cliff: 102,020 tokens in 1232.5 s = 82.8 tok/s vs the Windows box's 130.6 s / 781.5 tok/s for the same doc = 9.4x slower. Engine counters: experts 282.9 s, 775.9 GiB copied, 202,299 misses. Prefill phases are NOT instrumented (forward_prefill_sup has only g_pf.start()).
- NEEDLE at 102K: exactly right - first 47 generated ids byte-identical to the Windows fp8 reference (win-kv4v-needle.log); depth 77.8 pct of the 131,072 window (deepest the box allows), needle at 100 pct of that material.
- Downtime ~58 min for the round (two 20.5-min long prefills dominate); server restarted pid 769050, engine 769065, /v1/models 200, cached_tokens 26->64 (CCACHE live).
- RANKED NEXT LEVERS (evidence-based, replaces the old tier list): (1) uploads 32.9 ms/token - staging copies are serialised with compute, the biggest tractable term, up to ~20 pct at long context if overlapped; (2) tier tables 13.7 ms/token host-side index work per visit - cheap to attack; (3) attention 67 ms - largest but Pascal-specific mma884 work, needs SASS (ncu blocked); (4) head 6.8 ms - uint4/fp32-FMA rewrite worth ~3 ms (5 pct long, 12 pct short); (5) instrument prefill phases (cheap) before commissioning any prefill work, since prefill is 9.4x behind Windows.
### Lab prefill attribution (2026-11-24) - the cliff is attention, not copies

Instrument: `PfProf g_pp` + `g_pprof` in ik2_main.cu, CUDA-event phases (no per-phase stream syncs), separate `profile (prefill):` line, gated by IK2_PROF only; commit ea70aa37 on lab-tier0 (1 file, 116+/17-), not pushed. Hooks: forward() prompt chunks (T>8) map embeds->uploads, per visit attention/router/uploads/experts/rest, run_head->head; offload expert H2D on g_cs bracketed per copy and summed separately. On this 8 GB box W.T_sup=0, so the 102K prompt runs the chunked forward() path (forward_prefill_sup instrumented for completeness, never entered).

Zero-effect gate (cross-binary, CPU_FRAC=0 for both): build/maya-ik2.pre-ik2tuned (md5 1cdcfd01) x4 and instrumented x4 -> all 8 `generated:` lines byte-identical; new written reference sha256(line)[:8] = 9cda1c2e (historical 72cf3116 not reproduced by this prompt; its recipe was a lost session one-liner). `profile` lines in unset runs: 0.

Prefill profiles (live env, --gen 1, event timing):
- 8,192 tokens: attention 8.88 s (33.1%, 1.084 ms/tok), experts 17.58 s (65.7%, 2.146), router 0.08, uploads 0.18, head 0.05; wall 26.47 s = 309.4 tok/s.
- 102,020 tokens: attention 947.63 s (80.8%, 9.289 ms/tok), experts 220.61 s (18.8%, 2.163), router 1.02, uploads 2.23, head 0.64; wall 1167.23 s = 87.4 tok/s.
- Attention per token rises 1.08 -> 9.29 ms (8.6x) while experts stay flat 2.15 -> 2.16: the key-axis compute, not bytes. Expert H2D 66.01 s at 11.76 GiB/s (775.9 GiB) is fully hidden behind 220.6 s of expert kernels.
- Instrument cost: none attributable (event records only, ~0.04 pct host; one g_cs sync a chunk; structurally zero when unset). Today's control 1172.35 s / 87.0 tok/s vs the earlier session's 1232.5 s / 82.8 = -4.9 pct wall, i.e. box-state variance; the same binary with the old sync-lap instrumentation ran 8K at 32.2 s vs 26.8 s here, confirming events avoid distortion.
- With decode numbers (attention 67.4 ms = 49 pct of the 102K token), ATTENTION IS THE SINGLE LEVER: 81 pct of prefill, 49 pct of deep decode. sm_61 has no real tensor-core MMA (emulated mma.m8n8k4) and the card sits at 32 pct memory util -> compute-bound.
- Side finding: under the live env (IK2_CPU_FRAC=0.5) the BASELINE binary was not deterministic on the 65-id gate prompt (4 distinct outputs); identity holds only with CPU_FRAC=0. Lab answers are therefore not bit-reproducible with the CPU lane on - relevant to any future quality gate.
- Downtime 44 min 04 s (server SIGTERM 16:54:27 -> ready 17:38:31); restart with the known recipe, /v1/models 200, cached_tokens served (0 then 25).
### Router knobs: cache bias and tail skip (2026-11-24, ported from PolyStrata's Inkling-2 small-card settings)

Two env-gated knobs on `ik2-tuned` (commits 813368c IK2_CACHE_BIAS, 38d7d5f IK2_TAIL_SKIP), both default-off and byte-identical at default (backup-vs-new 40-token greedy 8/8 identical; `--score` 511 positions identical, score-file SHA equal). Residency source: `ExpertCache::loc` (the table the "N of M in VRAM" banner counts), snapshotted per layer into `Layer::resid_h/resid_d` and rebuilt only when a slot moves.

- `IK2_CACHE_BIAS=F` adds F to the router score of experts that are resident, by rewriting the existing per-expert bias table (`exp_probs_b.bias` -> `probs_k`, re-uploaded only on residency change). No kernel signature change; decode and prefill both use it.
- `IK2_TAIL_SKIP=N` drops picks whose rank index >= N that are NOT resident and rescales the kept weights to sum to the original total; the freed slot becomes a duplicate at weight 0, so K shapes, group_route, expert kernels and the graph are untouched and nothing is fetched for it.
- Score scale measured (24576 router calls): mean 0.0729, std 0.1243; 8th-best 0.2267; top-8 cut gap (8th-9th) 0.0177. That is why the sweep moved from {0.05..0.5} to {0.01,0.02,0.05,0.1,0.25}: 0.05 is ~3x the cut gap.

SPEED (this box, 12 interleaved rounds per arm, --gen 256 @131072, standing env):
| arm | median tok/s | vs off | decode misses/run |
|---|---|---|---|
| off | 28.97 | - | 21520 |
| F=0.05 | 30.25 | +4.4 pct | 16196 |
| F=0.10 | 31.08 | +7.3 pct | 13402 |
| F=0.15 | 31.86 | +10.0 pct | 11037 |
| F=0.20 | 32.17 | +11.0 pct | 9903 |
| F=0.10 + tail 6 | 32.43 | +11.9 pct | 9281 |
| tail 6 alone | 30.40 | +4.9 pct | 15079 |

QUALITY (511-position natural text; 2047-position confirmation for F=0.15):
| setting | mean NLL | dNLL | top-1 vs off |
|---|---|---|---|
| off | 2.853005 | - | - |
| F=0.05 | 2.858306 | +0.005 | 511/511 same |
| F=0.10 | 2.860922 | +0.008 | 511/511 same |
| F=0.15 | 2.886183 | +0.033 | 511/511 and 2047/2047 same |
| F=0.20 | 2.953591 | +0.101 | 511/511 same |
| F=0.10 + tail 6 | 2.934433 | +0.081 | 511/511 same |
| tail 6 alone | 2.943-2.964 | +0.09..0.11 | not measured top-1 |
- 2047-position run: off 2.325408 vs F=0.15 2.353696 (+0.028), top-1 2047/2047 identical.
- Coherence at strongest settings (greedy 40 tokens, decoded): F=0.25 and tail-4 both stay on-prompt and fluent; text is the same "Expert offloading ..." answer, differing only in the following clause.
- CAVEAT: both knobs read live residency, so ON-mode outputs are not run-to-run reproducible (N=4 scored 3.126/3.185/3.205 across runs; F=0.1 had one outlier). Off mode stays fully deterministic. This also means an ON-mode config cannot use the byte-identity gate for future changes - the gate must be run with the knobs unset.
- Verdict: `IK2_CACHE_BIAS=0.15` is the value recommended for the live config (+10 pct decode, +0.028 NLL, top-1 unchanged on 2558 measured positions). Tail-skip adds little on top of bias and costs more quality, so it stays a knob, not a default. Residency effects are monotone in tier size; this box holds 485 of 3072 experts.
### Cache bias is a code default (2026-11-24, user chose 0.15 on the curve)

- `IK2_CACHE_BIAS` now defaults to 0.15 (was off): declaration ik2_main.cu:313 (0.15f, comment carries the measurements), parse :5917 (`bias_named` -> env wins, else 0.15), multi-GPU guard now refuses only a NAMED bias and silently drops the default (the tier is one GPU's), banner prints by default with the off switch: "cache bias: 0.15 to the experts in VRAM (IK2_CACHE_BIAS=0 turns it off)".
- Chosen on the measured curve (12 interleaved rounds a setting, --gen 256 @131072): off 28.97 tok/s / 21,520 misses -> 0.15 31.86 (+10.0 pct) / 11,037 misses, for +0.033 mean NLL and top-1 identical at 511/511 and 2,047/2,047 scored positions; 0.20 = 32.17 (+11.0) for +0.101 NLL. IK2_TAIL_SKIP stays off (+1.4 pct on top of the bias for +0.05 NLL more).
- Gates: Release build clean; ik2_kernels_test "all passed"; IK2_CACHE_BIAS=0 vs the pre-knob binary (maya-ik2.pre-routerknobs.exe) - 40-token greedy identical 2/2 and --score identical (mean NLL 2.853005, score-file SHA 05AB1AD2E9B07D2...); default run banner shows 0.15 and scores 2.886183 (the measured F=0.15 value); default greedy on the 22-token prompt is unchanged from off (short prompt, no eviction pressure).
- Docs: docs/INKLING2.md Decode paragraph now states the default, the measured numbers and IK2_TAIL_SKIP's status.
- Commit `eeccc1c` on ik2-tuned (unpushed: a749666, ddd44b1, 813368c, 38d7d5f, eeccc1c). The live Windows config names no IK2_CACHE_BIAS, so the default takes effect on its next start. Lab tree untouched (the knob is not ported there).
### Tail skip: T=1 fast path + default 4 (2026-11-25, user directive "stop A/B , keep the new with fast path and default value to 4 in code")

- A/B at IK2_TAIL_SKIP=6 (pre-t1fast vs fast-path binary, 10 interleaved rounds) was stopped by the user midway; log hill-climb/win-t1fast-direct.log is partial and was not read for a verdict.
- Fast path kept (commit f6f0f66): at T=1 tail_pass skips the need-set seed and the mark kernel and launches tail_apply alone with need=nullptr (resid read directly), because one token's K picks are distinct experts so the batch need-set reduces to the tier table. 3 launches -> 1 per routing call, x48 visits/token. Gates: suite all passed; default identity intact (see below); decisions unchanged (top-1 differences 52-63/511 both within and across binaries = residency jitter).
- Default IK2_TAIL_SKIP=4 (commit 8d483ba): named env still wins, IK2_TAIL_SKIP=0 disables; a NAMED knob on --gpus 2 still dies, the default is silently dropped there (same rule as the cache bias). Banner now always states it: "tail skip: ranks 5..8 left out when not in VRAM (IK2_TAIL_SKIP=0 turns it off)".
- Gates: build clean, ik2_kernels_test all passed; IK2_CACHE_BIAS=0 + IK2_TAIL_SKIP=0 reproduces the pre-knob reference exactly - mean NLL 2.853005, PPL 17.3398, score SHA 05AB1AD2...; NEW DEFAULT at --chunk 1 (decode-true): mean NLL 2.933697, PPL 18.7970, cache 186,828 hits / 9,744 misses, vs tail-off base 2.864606 / 17,732 misses - i.e. the default costs +0.069 NLL against bias-only and removes 45 pct of the misses.
- Docs: docs/INKLING2.md states the default and the trade.
- Speed basis for the default: round-2 three-arm on the fast-path binary, 12 interleaved rounds, --gen 256 @131072: off 31.56 / tail 4 32.51 / tail 6 32.09 tok/s medians; paired vs off +2.0 pct (sd 1.41) and +2.2 pct (sd 0.61); misses 10,871 / 4,787 / 7,837. Quality at T=1 earlier: +0.053 (tail 4), +0.031 (tail 6) mean NLL.
- Unpushed on ik2-tuned now: a749666, ddd44b1, 813368c, 38d7d5f, eeccc1c, 9682d36, f6f0f66, 8d483ba.
### Server-vs-bench gap closed: a stale exe in the live config (2026-11-25, user: "polystrata 30+ tok/s, our fork sub-20 through UI/endpoint")

- ROOT CAUSE, proven by size and hash: the live config's `exe` was `build\maya-ik2.exe` = 10,032,128 B, sha256[:16] 4721E5CA7D9CB111 â€” byte-identical to `maya-ik2.pre-routerknobs.exe`, i.e. the served engine had neither `IK2_CACHE_BIAS` (0.15) nor `IK2_TAIL_SKIP` (4). Its miss rate was ~2x (21,520 vs 11,037 a run). Repointed to `build\Release\maya-ik2.exe` (10,072,576 B, BDF19AE052946DB2); backup `maya-inkling-2-mini.json.bak-preflight`; every other config field untouched. Acceptance: the startup banner now prints `cache bias: 0.15 â€¦` and `tail skip: ranks 5..8 â€¦`.
- Effect: served decode went from 17.9-23.9 tok/s to **26.2-27.3** on the same box, and client wall-clock agrees with the engine's own `DONE â€¦ decode_ms` within 0.5 tok/s (so no client-side stall, no server withholding).
- Decomposition (all measured, `nvidia-smi -lms 200` shows GPU at 88-93 pct, SM 1935 MHz, i.e. busy, not host-stalled):

| step | tok/s | mechanism |
|---|---|---|
| campaign bench, 11/25 binary, 440-490 tier slots | 31.9 | reference |
| same script today | 29.7-30.1 | expert tier only 320-327 slots (1.3 GiB): budget = free VRAM - ROOM_MB, and the desktop holds ~6.9 GiB free vs the campaign's 8 |
| engine `--serve`, live env, warm | 28.6-29.2 | per-token `T` line + fflush, inherent to streaming (~3 pct) |
| + Python server | 26.2-27.3 | server per-token work (detokenise, loop guard, parser, status/telemetry) ~2 tok/s; GPU util 93 -> 88 |
| temp 0 vs 1.0 vs streamed | +-0.2 | no material effect |
| first request after start | -3 | cold graphs/kernels |

- Code-read facts (for any future work): the server adds ONLY `--serve` to argv (`serve/server.py:1124`); one `GEN` line per request, the engine emits one `T` per token and the server yields from it; sampling is in the engine's C++ host code (`pick`/`pick_top`, ik2_main.cu:7727-7763). With `IK2_CPU_FRAC>0` (live 0.5) `forward()` refuses the fast-token path (4231-4232) and maps `kCand -> kLogits` (4236), so **every sampled token copies all 200,058 logits to the host** â€” measured cost <=1 tok/s. Python's `thinking: N of max M, X tok/s` is `generated/(now-first_token)` (server.py:4343), prefill excluded.
- Latent bug hit twice (not the user's symptom): CLI under the FULL live env dies `expert cache: no slot free (slots 8, active 8 â€¦)` (ik2_main.cu:995) when the tier lands at 326 slots and pool 0 is the floor-8 pool. That is the known pool-floor-8 issue; it blocks a clean live-env CLI reference.
- Levers left for interactive speed, in order: (1) free VRAM before start (a bigger tier is worth ~2 tok/s, and it is the only way back toward 30); (2) decouple the Python server's per-token work (batched detokenise / lower priority) for ~2 tok/s; (3) GPU-side sampling to skip the 200K-logit host copy at temp>0 (~1 tok/s, needs a sampling kernel).
### Windows: speed that falls with context, and the cancel-then-resend stall (2026-11-25, user report)

USER REPORT (verbatim): "on this machine even with stable background load , speeds are all over the place and they start high but quickly fall , also sometimes a prompt won't process and cancelling it and sending it again solves it"

#### Symptom 1 - decode falls as the conversation grows: attention reads scale with position, nothing else
Measured (engine as configured, temp 0, max_tokens 256, one request at a time, nvidia-smi -lms 500 running through every arm):

| arm | what | decode tok/s |
|---|---|---|
| (b) fresh conversation each request, same 50-token prompt, 20 requests | flat: 27.7 .. 29.1, mean 28.66, sd 0.37 | slope ~0 |
| (a) one growing conversation, 20 turns, 50 -> 1,138 prompt tokens | 28.6 -> 26.7 | r(pt, decode) = -0.957 |
| (c) same growth with IK2_KVLEND=0 (copied config, port 8096) | 30.4-30.5 plateau -> 28.9 | same slope, shifted +1.6 |
| controlled size sweep, fresh request per size, no history | 89: 30.18 | 205: 30.28 | 379: 30.16 | 727: 29.36 | 1423: 29.30 | 2815: 28.32 | 5599: 28.35 | 11167: 26.37 | 22303: 24.58 |
| same sweep, KVLEND=0 | 89: 30.44 | 1423: 28.68 | 11167: 26.36 | 22303: 24.45 |

r(prompt_tokens, decode) = -0.968 with lending on, -0.965 with it off; the two columns agree within 0.3 tok/s at 11k and 22k. The box's own log: a 29,736-token answer at ~100k context did 17.7 tok/s. So 30.2 at 89 tokens -> 24.6 at 22k -> 17.7 at ~100k = the -41 pct the user describes.

Refuted by data: THERMAL/CLOCK (1935 MHz from the first request to the last, 61-63 C, 138 W of 225); VRAM TAKEN MID-RUN (used varies +-16 MiB across a whole 20-request arm, no trend; the KV pool is allocated whole at load so free VRAM is 192-211 MiB the whole time); CUDA-GRAPH RECAPTURE (arm b is 20 identically shaped requests at 28.66 +- 0.37 with no dip); LRU CHURN (the decay reproduces on the FIRST request a freshly started engine serves at 22k, 24.58, with no history); KV LENDING (decay identical with IK2_KVLEND=0, and 22k still leaves ~77 pct of the buffers never reached).

Mechanism (file:line): `src/ik2/ik2_kernels.cu:4744` `for (int j = j0 + warp; j < j1; j += 4)` reads one K row and one V row for EVERY past position, per layer per head per token; `:4804`/`:4810` set the row count from the position and split it across P blocks without shrinking it. Decode time = fixed expert cost + c * position, c ~ 0.25 tok/s per 1k tokens in the 1k-22k range, steeper over the first 1k - which is why it reads as "quickly falls, then settles".

Verdict: not a defect. The only in-engine lever is the KV read width (IK2_KV4V halves the V bytes) and it was already rejected for token-exactness; otherwise the price of context is real and belongs in a "a new chat is faster" note, not a bug hunt.

Side measurement worth its own A/B: IK2_KVLEND=0 was +1.6 tok/s in the multi-turn pattern at short context (28.66 -> 29.27 mean) and +-0.3 at 11k/22k - the campaign's +8.4 pct for lending does not reproduce on this card today. Not changed; it wants a 12-round interleaved A/B before any default moves.

#### Symptom 2 - "a prompt won't process, cancel and send again fixes it": two layers, both fixed
Scenarios that did NOT reproduce (six tried): cancel mid-decode then next request (TTFT 0.59 s); a second request during a stream then cancel the first (starts the moment the cancel lands, single FIFO, no hang); max_tokens larger than the remaining context (HTTP 400 in 3 ms, never reaches the engine - `serve/server.py:4224-4228`); two requests then a third; a queued request cancelled while queued. Ten cancels produced no stale DONE, no lost request, no engine restart.

Reproduces: a large UNCACHED prompt cancelled 2 s in. Pre-fix, the cancelled request ran to its end anyway (`done: 6 tokens in 27 s ... (disconnect, cancel=True)`) and the next request waited 24.8-26.6 s for its first byte, with `this request waited 24 s for the engine (0 request(s) behind it)` in the log - nothing behind it, the FIFO's holder was simply still working.

Layer 1, the engine: `ik2_main.cu:7648-7668` sized one `big_begin` pass to the WHOLE prompt and looked at `g_cancel` only between passes (`:7652`). A 28.6k prompt went in one pass => 26 s uninterruptible (131k would be ~45 s). FIX (commit 67e36fd): step the pass in `sub_chunk` pieces, capped at the `T_pass` the scratch was actually sized for (`std::max(1, std::min(sub_chunk, T_pass))`), and check `g_cancel` at each. Measured engine-side: STOP at 2 s -> `DONE 0 8192 ... cancel` at 2.22 s, i.e. 0.22 s after STOP (was: the whole prompt).

Layer 2, the server: `generate()` waited `self.lines.get(timeout=10)` and looked at the cancel only when that timed out - so after a client FIN the STOP could be up to 10 s late, and the next request sat behind it. FIX (commit 2480d5a): poll at 1.0 s; a quiet tick yields the same `None` keep-alive. The FIFO must still be held during the STOP+drain (`serve/server.py:1723-1770`, `5243-5246`) - one stdin/stdout pipe and one GEN slot is what makes it necessary; the engine now answers the STOP in 0.22 s, so the drain is short.

End-to-end after both (real socket close, not a pooled connection): next request's first byte 0.24-0.51 s, from 24.8-26.6 s. The resend works because the abandoned pass ends quickly now, not because the server was latched.

#### Gates
- Backup `build\Release\maya-ik2.pre-cancel.exe` (10,072,576 B, sha256[:16] BDF19AE052946DB2) taken before the build.
- Release build clean; `ik2_kernels_test` -> `all passed`.
- Default identity: `IK2_CACHE_BIAS=0 IK2_TAIL_SKIP=0` score over 511 positions -> mean NLL 2.853005, PPL 17.3398, file SHA `05AB1AD2E9B07D25CDB914D4814881510BCBFF869A8AC8F025A57C638CB64165` (exact match to the pre-routerknobs reference), after BOTH the first and the guarded edit.
- Prefill cost: 28,624-token prompt, --gen 1, interleaved: old 37.11 / 36.25 s, new 36.11 / 35.04 s (-2 to -3 pct, i.e. none).
- Commits: `67e36fd` (engine), `2480d5a` (server) on `ik2-tuned`, unpushed.

#### Honest notes
- The diag lane's own reproduction used `requests`' `r.close()`, which returns the connection to a pool instead of closing it: the server never saw a FIN, so its 26 s "next request waits" was partly harness artefact. A real socket close showed the same shape but with the 10 s queue wait as the dominant term; the fix closes both.
- Ordering confound: the multi-turn arms ran one after another on a warming box (46-49 C at start), so absolute levels differ between arms; clocks were pinned at 1935 MHz throughout, so the trend does not.
- ik2 prints no expert-cache hit/miss counters to the server (`_parse_stat` at `serve/server.py:1709-1710` only understands GLM `STAT` lines), so the tier was excluded indirectly, not measured.
### Lab moved to ik2-tuned @ a666e9d (2026-11-25, user: "replace the lab maya version with our latest push")

- Lab repo checkout: branch `lab-latest` at `a666e9da` (fetched from a bundle of the fork tip). Previous branch `lab-tier0` @ `ea70aa37` preserved as `lab-before-ik2tuned`; previous binary preserved as `build/maya-ik2.lab-tier0` (md5 4e30318b). Server stopped with SIGTERM (port freed, 0 engine procs).
- Build: `cmake --build build --target maya-ik2 ik2_kernels_test` exit 0 -> build/maya-ik2 25,306,888 B, md5 a6e2d666. `ik2_kernels_test`: all passed (incl. KV4 codec cases).
- Config `maya-inkling-2-mini-lab.json` gained `IK2_E4LUT=0` (backup `.bak-before-ik2tuned`): the new tree defaults E4LUT ON and that lever measured -4 pct on sm_61. Explicit env wins, so the card keeps its setting.
- PAIRED A/B (6 interleaved rounds, live env + E4LUT=0, 65-id prompt, --gen 256, 131072):
  - old (lab-tier0 binary): 18.09 / 18.03 / 18.14 / 18.09 / 18.27 / 18.00 -> median **18.09**
  - new (a666e9d): 17.46 / 17.48 / 17.52 / 17.54 / 17.53 / 17.54 -> median **17.53 = -3.1 pct**
  - Expert tier identical in both arms (709 vs 708 of 3072, 2.8 GiB). Misses collapse on the new binary (4,463 vs 20,305 per run: the cache-bias 0.15 default working), yet decode is SLOWER: the Ampere-shaped kernel delta costs more on sm_61 than the router gains save. Same shape as the tier-0 wholesale-merge attempt (-6.1 pct); the surgical port remains the fastest lab binary.
- Server restarted on the new binary (setsid/nohup, ready in 27 s, /v1/models 200): banner `thinking level when a request names none: low`, ready line `images on`; live request `17*23` -> '391', 60 completion tokens, thinking 26/47 of 256 at 17.9-19.0 tok/s served.
- Decision left to the user: keep the latest tree (-3.1 pct decode, plus cache-bias/tail-skip quality trade unmeasured on the lab) or restore `lab-tier0` (copy `build/maya-ik2.lab-tier0` back over `build/maya-ik2`, or `git checkout lab-before-ik2tuned` and rebuild).
- The read-only ISA audit lane was cancelled on the user's order before producing any SASS data.
### Lab cross-layer prefetch (IK2_XPF): built, gated, DROPPED (2026-11-25, user asked to spend the idle link)

- Attempt: 3 files, +412 lines (`src/ik2/ik2_off.cuh` +18, `ik2_off.cu` +110, `ik2_main.cu` +284) â€” profile-ranked plan kernel, staging-only copies on `g_cs`, producer/consumer hooks. Default-off byte-identical PASS (8/8 `generated:` lines identical; `--score` both 511 positions mean NLL 2.855532, PPL 17.3837). Reverted; saved on the lab as `/tmp/xpf-attempt.patch`; `lab-pf` has zero commits.
- ON mode crashed: `cudaStreamSynchronize(g_s): illegal memory access` â€” experts with `nopin[e]` have no bytes in the pinned base (they live only in the file mapping `Ly.fh`), so the copy dereferenced unregistered pages.
- Measured (graph token path only; POP_OUT unset, CPU_FRAC=0, because the live path cannot run XPF), one run each, gen 39:

| arm | tok/s | link traffic | prefetch copies | staging share of picks |
|---|---|---|---|---|
| control | 11.52 | 70.3 MiB/token | â€” | 4.2 pct |
| IK2_XPF=8 | 4.88 | 1534.6 MiB/token | 354.1/token | 1.7 pct |
| IK2_XPF=16 | 4.42 | 1789.1 MiB/token | 422.9/token | 1.8 pct |

- WHY: 48 visits x 8 experts x ~3.9 MiB = ~1.5 GiB/token, 22x the control's traffic; the static profile's top-non-resident ranking hit the actual picks only ~2 pct of the time. The idle-link budget is ~18 ms/token which at ~0.63 ms/expert is ~28 experts/token = **0.6 per visit** â€” N=8 is 13x over budget, so even a perfect predictor could not fit.
- STRUCTURAL FINDINGS worth keeping: (1) `IK2_DF` (decode prefetch, staging-only, non-blocking) exists and is DEAD under every live config â€” `forward()` routes T=1 to `fast_token()` only if `(g_fast||g_off) && T==1 && g_pop.empty() && g_cpu_frac <= 0`, and the live config sets both `IK2_POP_OUT` and `IK2_CPU_FRAC=0.5`, so decode runs the host token path which fetches each miss with `EC.get()` and uses no staging ring. (2) The profile format is usable: `int32 n_layer, int32 E, uint64 counts[n_layer*E]` (24 x 128 here), already read by `IK2_WARM`. (3) Reviving prefetch would need: hook the `g_mv` host token path, never copy `nopin[e]` experts through the pinned base, and cap by a bytes budget (<=1 expert/visit).
- Lab restored: `build/maya-ik2` rebuilt from `a666e9d`, kernels test all passed, server up, `/v1/models` 200.
### Lab CPU_FRAC sweep 0.5 -> 1.0 (2026-11-25, user directive "sweep cpu frac from 0.5 to 1")

Six arms (0.5/0.6/0.7/0.8/0.9/1.0), 6 rotated interleaved rounds, gen 256 at 131072, 208-token prompt, live env + IK2_E4LUT=0, ON the live binary (/home/lab/maya/build/maya-ik2, md5 f0543cbb, build-id 16d34d5b...; byte-twin of a6e2d666 apart from .strtab nvcc temp names).

| arm | median tok/s | paired d vs 0.5 | median misses |
|---|---|---|---|
| 0.5 | 17.480 | - | 3,970 |
| 0.6 | 17.530 | +0.48 pct | 4,124 |
| 0.7 | 17.535 | +0.60 pct | 4,198 |
| 0.8 | 17.520 | +0.50 pct | 4,210 |
| 0.9 | 17.595 | +0.97 pct | 4,754 |
| 1.0 | 18.335 | +5.21 pct | 13,883 |

Confirm (3 extra interleaved with 0.5): 1.0 = 18.34/18.33/18.30, 0.5 = 17.47/17.52/17.49 -> paired +4.98/+4.62/+4.63 pct. Combined n=9: 0.5 17.48, 1.0 18.33, +4.86 pct median (orchestrator-verified from the raw generation lines: 1.0 min 18.29 > 0.5 max 17.52, no overlap). 0 deaths, 0 reruns.

DROUTE (gen 32): at 0.5 the lane is cap-bound (share == cap 0.50) and CPU-queue-bound per visit (0.28 vs 0.03 ms); at 0.7 the link limits (0.30 vs 0.37); at 1.0 the D2H/link term collapses to 0.00 and the lane takes every miss (2.55 experts/visit) with lane wait 0.204 ms/visit (total 1.88 ms over 2.88 s - not wall-saturated, simply no longer waiting on the link). Mechanism: intermediate fractions leave the CPU idle half the time; only ownership of all misses pays.

Costs: cache misses triples (3.9k -> 13.9k) because the lane absorbs them - that is the mechanism, not a regression. Output is NOT bit-identical across fractions (different experts computed on CPU vs GPU, different accumulation order), so any change needs the --score gate, not a hash gate.

Downtime 29 m 07 s (SIGTERM 14:15:47, stopped 14:16:38, ready 14:45:45); config left at 0.5 for the caller to move; server healthy on the unchanged config afterwards.
### CPU-lane gate: the +4.9 pct is lossless, but the cache bias steers by a stale table (2026-11-25, lab)

Ladder on lab HEAD a666e9d, 512-id score file, --chunk 1, 511 positions, live env + IK2_E4LUT=0:

| knobs (bias/tail) | CPU_FRAC | mean NLL | PPL | misses | top-1 div vs same-knobs 0.5 |
|---|---|---|---|---|---|
| 0.15 / 4 | 0.5 | 2.884912 | 17.9020 | 6,089 | - |
| 0.15 / 4 | 1.0 | 3.029286 | 20.6825 | 32,858 | 107/511 (20.9 pct) |
| 0.15 / 4 | 0.5 (repeat) | 2.916454 | 18.4757 | 6,023 | - |
| 0.15 / 4 | 1.0 (repeat) | 3.029286 | 20.6825 | 32,858 | 108/511 (21.1 pct) |
| 0 / 0 | 0.5 | 2.844982 | 17.2013 | 34,738 | - |
| 0 / 0 | 1.0 | 2.847839 | 17.2505 | 80,705 | 63/511 (12.3 pct) |

- Routing-neutral gap: +0.002857 (n pair) and -0.009054 (r pair) => a tie. The CPU lane's arithmetic is NOT lossy at this resolution, and 1.0 is exactly reproducible (3.029286 twice, identical 32,858 misses) whereas 0.5 drifted (2.8849 vs 2.9165).
- Default-knob gap: +0.144374 then +0.112832 => systematic, not drift. Flipped tokens carry mean lp delta -0.193 (degradation of the true token) in the default arms vs symmetric +0.043/-0.042 noise in the neutral arms.
- **DEFECT FOUND**: at CPU_FRAC=1.0 a CPU-lane miss never takes a VRAM slot (ik2_main.cu:4799-4805: `++EC.misses; where[i]=nullptr`; only the urgent/lazy branches call EC.get/copy_in), so `IK2_CACHE_BIAS` reads the resident table as "cheap to compute" while it is really "resident, not necessarily cheap" - the bias then steers the router toward stale residents, and `IK2_TAIL_SKIP` drops non-resident picks that the lane would have computed for free. Same defect shape exists at Windows' 0.5, where the lane serves part of the misses.
- Accounting sanity: hits+misses = 196,608 (48 visits x 512 tokens x 8 experts) in every arm; `score cache: 0.00 GiB copied` at 1.0 is correct, not a broken counter - a lane-served miss never calls copy_in.
- DECISION (user, 2026-11-25): "Lab: 1.0 with the bias off today" - no code change; lab config gets IK2_CPU_FRAC=1.0 + IK2_CACHE_BIAS=0, tail chosen by a 1.0 bias-off tail-4-vs-0 score pair.
- The proper fix (not commissioned): bias/tail predicates should test "cheap to compute" (resident OR lane-served), not "resident".


### Lab: the four corners of CPU_FRAC x cache bias (2026-11-25, lab, binary a666e9d)

The shipped CPU_FRAC=1.0/bias0/tail0 config measured 16.2 tok/s against 18.33 for the arm it was derived from, so the four corners were run: 6 rotated interleaved rounds, `--gen 256`, prompt = first 32 ids of hill-climb/prefill-tokens-512.txt, `--max-context 131072`, live env + IK2_E4LUT=0, IK2_POP_OUT=data/expert-profile-live.bin. All 24 runs rc=0.

| arm | CPU_FRAC | IK2_CACHE_BIAS | IK2_TAIL_SKIP | median tok/s | median misses | vs A |
|---|---|---|---|---|---|---|
| A (old live) | 0.5 | 0.15 | 4 | 18.010 | 4,608 | - |
| B (shipped) | 1.0 | 0 | 0 | 16.275 | 38,478 | -9.6 pct |
| C | 1.0 | 0.15 | 0 | 17.780 | 26,600 | -1.3 pct |
| D | 1.0 | 0.15 | 4 | 18.785 | 16,618 | +4.3 pct |

- Absolute values sit ~+0.5 tok/s above the earlier 18.33/17.48 pair (shorter prompt), so all four are RELATIVE-only; D/A = +4.30 pct reproduces the reference +4.86 pct in direction and size. B and C are bit-identical across rounds (misses constant), A varies 2,190-5,005.
- MECHANISM (DROUTE, --gen 32, 1,536 visits): A lane 1.25 experts/visit, queue 0.32 ms vs link 0.41 ms (link-bound), lane wait 0.048 ms/visit. B lane 5.47 experts/visit, queue 0.93 ms vs link 0.00 ms, lane wait 0.537 ms/visit (11x A). So with the bias OFF the CPU lane becomes the per-visit critical path; the +4.9 pct of CPU_FRAC 1.0 only exists when a steering knob keeps the lane small (D/A), which is the same knob defect that costs quality (+0.14 NLL, 21 pct top-1 flips at 1.0).
- The lossless route (not taken): IK2_LANE_THREADS scaling, so the lane can carry the full miss volume without steering.
- DECISION (user): revert the lab to arm A and stop lab tuning. Config back to IK2_CPU_FRAC=0.5 with no IK2_CACHE_BIAS / IK2_TAIL_SKIP keys (code defaults 0.15 / 4 in force); restarted and verified.
- Downtime for the 4-arm sweep: ~18 m 36 s (SIGTERM 14:12:09Z -> /v1/models 200 14:30:45Z).
### PolyStrata on the lab: closed by inspection of its own decode (2026-11-26, user verdict)

- Fresh clone `/home/lab/poly-v2` (simplexai-labs/PolyStrata, 205 MB) with the Inkling docs read; the plan was setup.sh --family inkling reusing the local UD-Q3_K_XL GGUF (--custom-gguf), the CUDA 12 / sm_61 engine, --kv q8_0 (fp8 needs Ampere+), and a lossless-defaults vs lossy-small-card A/B against Maya arm A (18.01 tok/s CLI, 19.1 served, prefill 87.4 tok/s at 102K).
- The build/install lane was dispatched and then cancelled: PolyStrata's decode on this GTX 1070 with Inkling-2-Mini is much slower than Maya's. No numbers recorded beyond that verdict; nothing was installed into the lab config, and no PolyStrata work is commissioned.
- Lab state confirmed after the stop: no strata/poly processes, VRAM 9 MiB used / 8,096 free, port 8095 free, Maya server intentionally down. Only the pre-existing `/usr/local/bin/strata-oc-watch.sh` (pid 5244, memory-clock watchdog) and the socat bridge on 8081 remain. `/home/lab/maya` untouched (arm A: CPU_FRAC 0.5 + defaults bias 0.15 / tail 4, E4LUT=0, KV fp8, CCACHE/XGLOBAL/KVLEND/ELASTIC/DECAY on).

### Lab ISA audit of the attention loops (2026-11-26, read-only, sm_61 / GTX 1070)

Method: no build, no bench, no server change. `cuobjdump -sass` (CUDA 12.8 at /opt/cuda-12.8; the default /opt/cuda is 13.4 and cannot disassemble sm_61) over the existing `build/maya-ik2` of `lab-tier0` @ ea70aa37, landed with `nsys stats` on the existing 23 Nov capture, plus GGUF metadata and code reads.

Card measured (nsys TARGET_INFO_GPU, not spec sheets): **15 SMs** (not 20), 1.7845 GHz at capture, DRAM 256.256 GB/s, 65,536 regs/SM, 64 warps/SM, **48 KB max dynamic smem/block opt-in**. Peak FP32 6.85 TFLOPS, issue capacity 1.14e11 warp-instr/s.

**Finding 1 - the 947.6 s prefill attention runs no in-tree MMA kernel.** `attn_flash_f16_kernel` is gated off on Pascal (`if (cls < 80) g_flash = false`, main.cu:6060) AND cannot launch there anyway: kFaSmem = 88,064 B (kernels.cu:1227) against the 48 KB opt-in cap; it is compiled with REG:255 and STACK:1200 (spills) and never runs. The real path (main.cu:1850-1935) is: `unpack_kv_fp8_to_f16_vec4` -> cuBLAS `maxwell_sgemm_fp16_128x32_tn` for QK^T (m=nk, n=Tq, k=128, batch=3) -> a bias GEMM (`splitKreduce_kernel`) -> `attn_probs_bias_f16_kernel` softmax (global-memory pass) -> cuBLAS for PV. At 102K the query tile collapses: `Tq = lg_cap/(H*nk_ub) = 28.36e6/(24*102020) = 11` (144 at 8K), because the 256-query refill wants 3.2 GB and `lg_bytes >= avail` refuses it beside the 2.8 GiB expert pools (main.cu:4839-4856). So the dots are **n=11 skinny GEMMs, ~1.2 M launches**: 947.63 s / 102,020 tok = 296 GFLOPS = **4.3 pct of peak** (8K: 417 GFLOPS, 1.08 ms/token). The growth is ~T^2 work, not a cliff.
Dead on sm_61: `wgmma` (SM90-only, else __trap), `mma884` (only reached by attn_decode_v70 / attn_split), the flash kernel. Live: `mma16816`'s <700 branch (32 SHFL + 64 FFMA + 32 half2->float2 per m16n8k16) = the decode hot path, and `attn_probs_bias_f16` (prefill softmax).

**Finding 2 - decode attention IS an ISA problem, and it is issue-bound, not FMA-bound.** `attn_decode_mma_kernel<uint8_t>` (kernels.cu:1608-1798, launched grid (Hkv=8, n_split=7), block 128, smem 0, REG:128, launch_bounds(128,4) => 16 warps/SM). SASS of the per-step loop = 4,026 instructions per 16-key step: FFMA 1,140 (28.3 pct), F2F.F32.F16 800 (19.9 pct), int/addr/PRMT/LOP3 1,674 (41.6 pct), SHFL.IDX 404 (10.0 pct), LDG 77, misc 117 = **251.6 instructions per key per lane for 8 useful FMAs**. Work = 8 full visits x 8 KV heads x 102,020 keys = 6.53e6 key-units => 1.64e9 warp-instr => 14.4 ms at 100 pct issue; measured ~33 ms (44 pct of issue peak) vs a 6.5 ms DRAM floor. Waste source: m16n8k16's n=8 columns against R=3 heads, plus 50 F2F and 25 SHFL per key.

Geometry confirmed: head_count 24, head_count_kv 8 (GQA=3), head_dim 128, sliding_window 1024 every 6th layer => 24 layers, 20 sliding, 4 full, looped 6..17 x3 = 48 visits. KV = FP8 E4M3 with **one fp32 scale per row (per token per KV head)**, folded at store (store_k_warp, kernels.cu:327-378) and consumed via the e4m3x4_h4 bit trick (kernels.cu:1187-1202) with sc = qk_scale*log2e (main.cu:2664-2666) - not per-32, not per-tensor.

Priced (none commissioned; user deferred A2):
- A1 (1-2 d): hoist pointers/bias addressing, fold kv_cap wrap out of the step, keep uint4 fp8 loads -> ~1.1x.
- A2 (1-2 w, lossless): stop emulating m16n8k16 - pack 2 KV heads/warp so all 8 columns carry heads (3+3), or a 4-FFMA dot + 5-SHFL reduce per key-head. Per-key stream ~40 instr => issue floor 2.3 ms, DRAM 6.5 ms binds: attention 33 -> 7-9 ms, token 67.4 -> 42-45 ms (**1.5-1.6x**), same fp8 values with fp32 accumulation.
- A3 (1 d): regs 128 -> 96 for 24 warps/SM; only pays after A2.
- B (2-4 w, LOSSY, needs authorisation): int8 DP4A for QK+PV (~35-40 instr/key) => 3-5x on the stage, token ~30-35 ms; needs per-32 fp32 KV scales (+12.5 pct bytes) since the per-row fp8 scale is too coarse for int8 symmetry; QK error ~0.3-0.6 pct, PV on P in [0,1] is the sensitive part.
- Prefill structural (unpriced): sm_61-native fused kernel at <=48 KB smem, or a query-tiling scheme that fits the logits scratch beside the expert pools.
Not proposed by design: cp.async (absent on Pascal; the plain-copy emulation at kernels.cu:974 buys nothing) and half2/HFMA2 (1/64 rate on sm_61).
Missing measurement: a one-shot `ncu --set full` over a 102K prefill (none exists; attribution came from the code path plus 8K->102K scaling).

### Lab API "returns nothing": a 104k-token prompt against a 300 s client timeout (2026-11-26)

Symptom: API requests produced nothing while short chat worked. The server was healthy throughout (8095, `/v1/models` 200, engine pid 987148) - this was not a port or process fault.

Evidence from `/home/lab/maya/server-relaunch.log`:
- `reading the prompt: 104,358 tokens` started at 19:06:25, 19:11:30, 19:16:35, 19:21:41, 19:26:46, ... - six attempts, each exactly ~305 s apart, each ending `done: 0 tokens in 305 s (0.0 tok/s) (cancel, cancel=True)`. That is a 300 s client timeout plus automatic retry.
- Same window, sizes and prefill line counts (1 line/s): 76,521 tokens -> 281 s; 82,511 -> 320 s; 82,783 -> 773 s; 104,358 -> 1,743 s of logged progress before cancel. One 82,783 attempt was still reading at 186 s when `cancel=True` landed at 198 s.
- Every request that completed in the same window was short: `60 tokens in 4 s`, `16 in 2 s`, `32 in 2 s`, `106 tokens in 6 s (18.0 tok/s) (stop)`.

Arithmetic: measured prefill is 87.4 tok/s at 102K, so 104,358 tokens needs ~20 minutes (1,167 s measured for a 102,020-token document) against a 300 s budget. The call cannot succeed as configured.

Compounding factor: a cancelled prefill saves no prefix snapshot, so each retry repeats the whole read from zero - a livelock that also pins the GPU indefinitely (the six retries burned ~30 minutes of card time producing nothing).

Second trap found on Windows in the same investigation: `max_tokens` covers the thinking block too. An 18,930-token prompt with `max_tokens 48` used all 48 on reasoning and returned empty `content` while the engine logged a perfectly successful `done: 48 tokens in 27 s (18.4 tok/s)`.

Resolution (user's choice): raise the API client's timeout above ~25 minutes for this endpoint; do not cancel mid-read; give `max_tokens` room for thinking plus answer. The engine and config are unchanged.

### Lab VM headroom: IK2_ROOM_MB 64 -> 256 (2026-11-26, user request "let a bit more of vram headroom on the lab , like free 256mb")

- Edit: `maya-inkling-2-mini-lab.json` live env block `IK2_ROOM_MB` "64" -> "256" (backup `maya-inkling-2-mini-lab.json.bak-room256`); `installer.as_set_up.env` deliberately left at 64 because that block is the historical record of what setup.py wrote, not live config.
- Verified from `/proc/<engine>/environ`: `IK2_ROOM_MB=256`, `IK2_CPU_FRAC=0.5`.
- Effect: idle VRAM free 24 MiB -> 214 MiB (nvidia-smi used 7,891 of 8,192; the rest is driver overhead). Tier loses roughly 36 expert slots of ~709.
- No speed cost measured: quick prompt wall 1.234 s before and 1.234 s after; a 160-token answer served at 18.5 tok/s (inside the usual 17.5-19.5 band).
- Restart used the standard setsid/nohup recipe, ready in 24 s, server pid 1004969, engine pid 1004984.

### A2 decode rewrite, attempt 1: lossless, but aimed at a kernel this box never launches (2026-11-26, lab)

Built as specified on branch `lab-a2` (commit `b0c04bf`, off `lab-latest` @ a666e9d), +185/-2 in `src/ik2/ik2_kernels.cu`: `attn_decode_at2_kernel<KVT,R>` - 16 lanes a key, 8 dims a lane, K/V read and converted once and shared by all R heads, 4-step `shfl_xor` tree, bias element riding the same tree, a warp = 2 keys with independent halves merged at the end, fp32 accumulation over the same fp8 values. Gated `IK2_AT2=1`, default off.

Gates all passed: build + `ik2_kernels_test` (AT2 is MORE accurate than the emulation against the test's double reference: worst error 0.00 of tolerance vs 0.03-0.07); default-off identity byte-exact (greedy sha `9cda1c2eaccc3833`, score payload sha `42b67b08d405bd32`, NLL 2.863027); ON-mode identical to 0.000000 (0 of 511 top-1 divergences).

Speed: 17.49 -> 17.50 tok/s (+0.03 pct) on the live env, 11.54 -> 11.54 with CPU_FRAC=0, prefill untouched (314.9 vs 314.5).

WHY IT IS INERT (the finding that matters): nsys at 8192 ctx shows decode attention is `attn_split_kernel<u8>` - 2,256 instances x 382 us = 18.3 ms/token - and `attn_decode_mma`/`v70`/`at2` appear in NO trace. The fast token path (the only caller of `attn_decode_mma_fp8`) requires `g_pop.empty()`, i.e. `IK2_POP_OUT` unset, and without POP_OUT the model OOMs at load (11.4 GiB of experts on 8 GiB). The live config therefore always runs the host token path with the scalar split attention. The audit's 251.6 instr/key/lane was real (SASS reproduced) but described the wrong kernel: `mma<u8>` 251.6 instr/key (14.4 ms floor), `v70<u8>` 134.5 (7.7 ms), new `at2` 171.5 (9.8 ms).

Also found: any CLI prompt above 8192 illegal-accesses at `ik2_main.cu:4416` - pre-existing, reproduces on the pre-change binary at 10k/12k/16k/22k and regardless of ROOM 64/256, KVLEND, ELASTIC. The server's chunked path is unaffected (26,668-token requests have completed).

Next: the same rewrite aimed at `attn_split_kernel` behind `IK2_AT2S` (dispatch 2026-11-26), which is 49 pct of the deep decode token.
### Long-prompt illegal access: a tail-skip fill slot counted as a row (2026-11-26, lab, branch lab-a2)

NOT the cross-stream KV-lending hazard the earlier arms suggested. Fault: a plain out-of-bounds write in the MoE row bookkeeping.

Mechanism (proved): tail_apply_kernel gives a left-out slot, at weight 0, an expert the same token already picked ("lowest resident kept expert" = a popular one). run_prefill_moe counted an expert's rows by SLOT and documented cnt_e <= T, while the expert scratch ex16/h16/y32 is [T] (a row is a token). When the fill expert is picked by every token, cnt_e = T + #fills (measured 4329 picks over 4096 tokens, 225 zero-weight slots), so gather_f16_16_kernel writes cnt_e rows past the end; the sticky error surfaces at the next cudaStreamSynchronize(g_s) in the routing D2H (main.cu:4425), which read as idle-wedge. The duplicate row also raced its own token's row in scatter_add_rows (non-atomic +=).

Repro is PASS length, not prompt length: CLI --tokens 9000 faults at --chunk 4096 and 8192 (every time) but passes at 3072/3584/2048; 22067 @ chunk 2048 passed while 22067 @ 4096 faulted. CUDA_LAUNCH_BLOCKING=1 still faults (constant OOB, not a race); KVLEND/ELASTIC/OVERLAP/FLASH off all still fault. compute-sanitizer memcheck gave no kernel report (only the API illegal-address at the sync) - the signature of an OOB write into another allocation; a temporary per-phase sync trace then localised it (visit 29, MoE phase, T=4096).

Fix (commit f5d3f2b on lab-a2, +18/-3 in src/ik2/ik2_main.cu): a repeated slot shares its pick's row (inv[i]=inv[i-1]) and adds its 0 weight into that row's (wtr[row] += wt[i]), so cnt[e] counts one row per (expert, token) again and the [T] scratch invariant holds. No buffer, launch or VRAM change; nothing changes when nothing is left out. Pre-fix binary kept at build/maya-ik2.prekvfix.

Gates: build clean, ik2_kernels_test all passed. CLI post-fix rc=0 for 9000@4096, 9000@8192, 12000@8192, 22067@4096, 22067@2048, 9000@4096 with IK2_TAIL_SKIP=7; generated ids identical to the pre-fix chunk-2048 runs. Identity pre vs post binary: greedy 40 tok sha de1aa7dbdeb2be5a (3 reps each), --score --chunk 1 over 512 ids payload sha 1c874e32e7023ab4 / mean NLL 2.754963 (2 reps each) - byte-identical. Server (live env, IK2_AT2S=1, config md5 1c89a63d unchanged): 22,093 x3 clean (prefill 81.6/83.1/82.9 s, 64.3/64.4/70.4 ms/tok, 15.56/15.53/14.20 tok/s), 26,426 clean (104.9 s, 66.0 ms/tok, 15.16 tok/s), 20,633 clean (77.5 s, 64.8 ms/tok, 15.42 tok/s), /v1/models 200 after each, engine never restarted.

Deviations: no pre/post A/B of the SERVER failure - the OOB is proven real, deterministic and gone, but "the server's probabilistic 22k failure is this bug" is inference (the server runs --chunk 2048, where one fill on an expert already at cnt=T can just cross the boundary - which also explains why KVLEND/ELASTIC off changed residency and made it vanish). CUDA_LAUNCH_BLOCKING=1 did NOT silence the CLI fault (the earlier report said it did); the launch it names (swiglu_cap_f32) is one step after the culprit.

New observation, untouched: a short completion issued right after the 20.6k requests (cached_tokens = its whole prompt) answered with reasoning that quoted the PREVIOUS 20.6k prompt ("Chapter 5505, opening note 505531") - a possible snapshot-restore state leak; a fresh uncached short prompt answers correctly.\n\nDowntime 64 min 22 s (15:45:36 -> 16:49:58 local), server left UP on 8095 with the live config.\n
### Prefix cache restored state from the wrong conversation (2026-11-26, lab, branch lab-a2, commit 3846c36)

- Symptom (user-visible): a short prompt with `cached_tokens` equal to its whole self was answered with reasoning that quoted the PREVIOUS long prompt. Silent wrong output on the served path; a long prompt repeated after another long prompt could be answered as the other one (20,622-token prompt given back "Chapter 1101, opening" from the 22,082-token prompt).
- Reproduction 3/3, order-dependent: long -> short -> short (probe/B: short long short) and long-A -> long-B -> long-A all fail; long -> single short cannot (no snapshot yet). C2 proved caching itself works (repeated 20,622-token prompt: cached 20622 in 0.2 s). Corruption tracks the most recent prompt, not the requested one.
- Mechanism: `struct Snap` (ik2_main.cu ~7285-7295) holds the 4 convs (h, 48 visits) and the sliding layers' KV windows only - a FULL layer's K/V is never snapshotted ("positions rewrite cleanly", ~7284). A restore SKIPS the snapshot's whole prefix, so full-layer rows keep whatever prompt last wrote them. Rung 1a matched the snapshot by the snapshot's OWN ids and resumed at its `pos`; it never consulted `hist`, the engine's record of which tokens the state actually holds. Result: convs from the request's snapshot, full-layer K/V from another conversation. One 8-slot Snap ring is shared across requests by design (`reset()` keeps snapshots when `g_ccache`) and `pos` is reset per request (~7729-7737) - the ring is not the bug.
- Fix (3846c36, +15/-5, one file): the state match is hoisted into `state_match()` and the true common prefix `L` of `hist` and the request is computed BEFORE the CCACHE block; the rung-1a candidate filter is now `s2.pos > L` (:7632) instead of `s2.pos > ids.size()`, and `state_match()` is re-run after a restore so `L` is the restored prefix. Everything inside `[0, L)` is the request's own on every layer; the existing snapshot path already worked under exactly that invariant.
- Cost, stated plainly: returning to an OLDER conversation no longer resumes from its snapshot - it prefills (D3: 83 s instead of 5.9 s with a wrong answer). The agent-turn flow (same conversation, prompt extended) is untouched (it never needed a restore; `hist` covers it). The sound way back is snapshotting full-layer rows (~80 MB per 20k-token prefix here), a separate VRAM-budgeted change, not a guard.
- Gates: build clean, `ik2_kernels_test` all passed. Non-cache identity unchanged (greedy sha `de1aa7dbdeb2be5a`, score payload sha `1c874e32e7023ab4`, NLL 2.754963). Cache correctness after the fix: B3/B4, C3/C5 all `cached=0` -> `content='OK'`, no mention of the long prompt; D3 `cached=0` -> full 20,622-token prefill and reasons about A's own content. Engine-level (`--serve` GEN lines, CPU_FRAC=0, 200-token prompts): A,A,B,A,A -> restores logged only where sound (`restored the snapshot at 200 ... (state held 200)`), and A1==A2==A3==A4 token-for-token. Caching still works: repeated 20,622-token prompt `cached=20622`, HTTP prefill 0.0 ms, TTFT 0.2 s vs 77.5 s fresh. Served speed: fresh long prefill 76.6 s pre / 77.5 s post, decode 21.4 / 21.5 tok/s (noise).
- Downtime: three restart cycles, 9 m 07 s total (17:05:32->17:06:45, 17:12:07->17:13:21, 17:22:25->17:29:05 with the server down for the gate round). Server left UP on 8095, live config unchanged (md5 `1c89a63d`), `IK2_AT2S=1`, /v1/models 200.
- Caveat banked: with `IK2_CPU_FRAC=0.5` greedy decode is NOT bit-reproducible (same 400-token prompt uncached gave three different continuations - the lane split is timing-driven). "Same prompt, same answer" is only a valid gate with `IK2_CPU_FRAC=0`. Pre-existing, not the cache.
- Not done: full-layer K/V snapshots (the real repair for cross-conversation reuse); README/CHANGELOG entries; nothing pushed.
### A2S at 100K: the deep win is 2.05x, and it grows with depth (2026-11-26, lab, branch lab-a2 @ 3846c36, measurement only)

- Arms: 99,689-token prompt each (a 381,945-char harbour-operations log with a counter per block; needle paragraph at 77.9 pct depth), different opening line and needle code per arm so no prefix was shared, `cached_tokens: 0` on both. temp 0, max_tokens 128, non-stream, client timeout 3600 s, run detached.
| arm | prompt_tokens | prefill | decode | request wall |
|---|---|---|---|---|
| ON `IK2_AT2S=1` | 99,689 | 1,070.3 s, 93.1 tok/s | **83.02 ms/token, 12.0 tok/s** | 1,081.1 s |
| OFF `IK2_AT2S=0` | 99,689 | 1,070.4 s, 93.1 tok/s | **170.48 ms/token, 5.9 tok/s** | 1,092.3 s |
- Delta: **2.054x decode rate, 87.45 ms/token removed = 51.3 pct of the OFF token**. Prefill delta 0.009 pct (noise) - prefill is untouched by design.
- Orchestrator cross-check from the server log itself (`done:` lines): `17:53:23 done: 128 tokens in 1081 s (12.0 tok/s)` and `18:13:08 done: 128 tokens in 1092 s (5.9 tok/s)`; with the 1,070 s prefill on both sides the decode portions are ~10.7 s vs ~21.6 s for 128 tokens = 83 vs 169 ms/token. The fixer's ms/token figures are reproduced from the log alone.
- Depth trend measured so far: shallow **+3.23 pct** -> 20,753 tok 84.63 -> 62.14 ms (1.362x) -> 99,689 tok 170.48 -> 83.02 ms (**2.054x**). The win does not fade: the end-to-end ratio approaches the kernel's 3.26x as attention's share of the token goes to 1. Implied split at 100K, if the kernel ratio holds: attention ~126 ms of the OFF token (74 pct) and ~38.7 ms of the ON token, leaving a ~44 ms non-attention floor (expert streaming/MoE). That 74 pct is higher than the ISA audit's 49 pct - the two cannot be separated without a profiler run at depth (not done; no code change was allowed).
- Needle recall: both arms quoted their OWN code verbatim in the reasoning (`4815-ALPHA-77` ON, `9062-OMEGA-31` OFF); `content` was empty because the 128-token budget went to thinking. Incidental live confirmation that the 3846c36 cache fix holds at 100K (no cross-arm bleed).
- Serving fact of its own: deep prefill is **93.1 tok/s at 99.7K (17 min 50 s per request)** vs 297 tok/s at 20.6K - the prefill cliff is unchanged by AT2S and is now the dominant cost of a deep request.
- Downtime 3 m 44 s (three restart cycles: 74 s / 76 s / 74 s). Config restored byte-identical (md5 `1c89a63d1c22161c70e4b782cceeccf9`), `IK2_AT2S=1` in force, `/v1/models` 200, one short completion `'DONE'` with no long-prompt content, no stray processes.
- Deviations: prompt is 99,689 tokens (not exactly 100,000); needle answers landed in `reasoning_content`; the watchdog defaults to 90 s (not 180) and only fires on an engine that is silent AND idle, so it would not have tripped a working prefill - `MAYA_ENGINE_STALL_S=3600` and `MAYA_REQUEST_QUIET_S=3600` were exported on the arm command lines only and are unset in the final server.
### A0 fp16 scores: the deep prefill win is 1.50x (2026-11-26, lab, branch lab-a2 @ b7ddd79)

Gate `IK2_SCFP16=1` (default off) stores the attention score buffer as fp16 instead of fp32, doubling `lg_cap` slots at the same byte budget. Committed `b7ddd79` (4 files: main.cu lg_score_cap/sc_type/sc_ptr/dispatch, kernels templated probs, parity tests).

Depth trend (same binary, CLI, --gen 1, live knobs):
| prompt | OFF | ON | delta |
|---|---|---|---|
| 20,607 | 273.5 / 273.3 tok/s | 289.4 / 289.1 | +5.4 pct |
| 40,000 | 216.2 | 247.2 | +14.4 pct |
| 99,674 | 1064.98 s / 93.6 tok/s | 710.10 s / 140.4 tok/s | -33.3 pct wall, +50.0 pct |

Provenance (engine lines, back-to-back GPU-exclusive, 19:37-20:07): `prompt: 99674 tokens in 1064.98 s (93.6 tok/s)` vs `prompt: 99674 tokens in 710.10 s (140.4 tok/s)`. OFF reproduces the server-side 93.1.

MECHANISM (why super-linear): `Tq = lg_cap / (H * nk_ub)` so `Tq ∝ 1/nk`; blocks per pass `∝ nk`; the prefix fp8->f16 unpack runs ONCE PER BLOCK (`main.cu:1992-2008`) and is O(nk) per block => O(nk^2) unpack traffic per pass. Halving Tq halves that quadratic term as well as the score bytes and doubles tile fill. At 20.6k the GEMMs dominate; at 100k the unpack dominates. The gate therefore does part of A2's job (the per-block re-unpack) for free.

Tq proof (IK2_SCFP16_LOG): 8192 pass 341 -> 512 (25 -> 16 blocks); 40k deepest chunk 69 -> 139 (16 -> 8); 100k Tq 28 -> 56, blocks 49 -> 25.

AUDIT CORRECTIONS (authoritative over the earlier scoping report):
- `lg_cap` is floored by `64 << 20` => 67,108,864 score slots = 256 MiB, NOT 28,360,704 / 113.4 MB. So Tq at 102,020 keys is 27 -> 54, not 11 -> 22, and cuBLAS' 32-wide n-tile was already 84 pct full at 27 - the 'tile starvation' argument was void.
- `kRelbRows = 512` pins Tq in BOTH modes for nk <= ~10.9k, so 8k/20.6k arms are structurally blind to this gate.
- The 256-query refill path is DEAD on this model (W.T_sup = 0 because c.n_pass == 3, main.cu:6790; big_begin early-returns :5109).

QUALITY PRICE (labelled loss, prefill path only; `--score --chunk 512`, 511 positions): NLL 2.757829 -> 2.771214 (+0.0134, +0.49 pct), PPL 15.766 -> 15.978, 37/511 argmax flips (7.2 pct), mean |dlp| 0.101 (max 1.99), top-32 overlap 0.94 (min 0.72). Greedy 40 @65 changes de1aa7db -> ba9cde27. Note `--score --chunk 1` cannot see the gate (T=1 runs the decode path).

Gates: build clean; ik2_kernels_test 835 checks all passed (gate on and off) incl. new fp16-vs-fp32 parity cases; default-off identity exact (greedy de1aa7dbdeb2be5a, score payload 1c874e32e7023ab4, NLL 2.754963); decode flat (62.8 ms/token both arms at 20.6k).

Deviations: 100k arm ran as CLI ids (99,674) vs the earlier server prompt (99,689, +15 template tokens); IK2_SCFP16_LOG on for both arms (identical overhead); second arm had warm page cache (load outside prompt_s). A first cut using __half2float(uint16_t) converted the integer rather than the bits (NLL 12.65) - fixed to h2f(), parity test added.

Downtime 31 min 52 s. Server restored UP on 8095 from the untouched live config (md5 1c89a63d1c22161c70e4b782cceeccf9), IK2_AT2S=1, ROOM_MB 256, no IK2_SCFP16.
### Score budget sizing: 512 MiB buys +29.5 pct deep prefill at zero deep-decode cost (2026-11-26, lab, branch lab-a2 @ 0b2ee38)

- Mechanism recap: `Tq = lg_cap / (H * nk_ub)`. At 100k keys, every MiB of score budget is ~0.22 queries of Tq, and Tq sets how many query blocks re-run per prefill pass. cuBLAS runs the same QK/PV products about twice as fast per FLOP at n=56 as at n=28, so raising Tq is the deep-prefill lever (A0 proved it at 28 -> 56).
- Gate generalized in `0b2ee38` (src/ik2/ik2_main.cu +22/-13): `IK2_SC_MB=<MiB>` sizes `lg_cap = MiB*1048576/sizeof(score)`; `IK2_SC512=1` kept as an alias for 512; unset = the 64 Mi score floor (256 MiB). Banner names both buffers.
- VRAM accounting (measured, not assumed): `lg16` aliases `gu32`; `gu32_bytes = 4*max(..., (lg_cap+1)/2)` so the lg16 term is `2*lg_cap` bytes. **The MiB asked for is allocated twice** (lg + lg16), so 512 MiB costs ~1 GiB, and the tier pays for it.

| budget | prefill 99,674 | tok/s | Tq | blocks/pass | tier | expert bytes | gain | tier cost |
|---|---|---|---|---|---|---|---|---|
| unset (256 MiB) | 710.51 s | 140.3 | 56 | 25 | 631 slots / 2.5 GiB | 319.7 GiB | - | - |
| 384 MiB | 599.00 s | 166.4 | 84 | 17 | 564 / 2.2 GiB | 341.2 GiB | -15.7 pct / +18.6 pct | -10.6 pct |
| 512 MiB | 548.49 s | 181.7 | 112 | 13 | 501 / 2.0 GiB | 364.6 GiB | -22.8 pct / +29.5 pct | -20.6 pct |

- 384 MiB buys 69 pct of the gain for 51 pct of the tier. Not saturated at n=112: the deeper the prompt, the bigger the share of the win (40k was only +8.0 pct at 512).
- 40k prompt: off 161.67 s (247.4) / 512 MiB 149.69 s (267.2), Tq 139 -> 279.
- Caps verified (main.cu:2012-2013): `Tq = max(1, min(T-ts, lg_cap/(H*nk_ub)))`, then `min(Tq, kRelbRows=512)` only when `W.relb` is set, so it does not bind at depth; at 8,192 keys `kRelbRows` pins Tq=512 in both modes, which is why an 8k arm cannot see the change.

DECODE, the deciding measurement (fresh process, same 99,674-token prompt, --gen 128, IK2_AT2S=1):
| arm | prefill | generation | ms/token | decode misses |
|---|---|---|---|---|
| unset | 710.73 s | 127 tokens in 10.64 s (11.93 tok/s) | 83.78 | 2,649 |
| 512 MiB | 548.66 s | 127 tokens in 10.65 s (11.92 tok/s) | 83.86 | 3,397 |
- +0.10 pct = flat; the 128-token outputs are token-identical. The tier loss (+28 pct decode misses, 631 -> 501 slots) does not reach the deep decode critical path. Earlier shallow check: 62.83 -> 63.23 ms/token (+0.6 pct).
- Quality: the budget is numerically neutral. `--score --chunk 512` payload sha `9556e260ddaf880b`, NLL 2.771214, PPL 15.978023 identical at 256/384/512; first greedy token at depth `3256` in all three arms. It adds nothing to the fp16-score price (2.757829 -> 2.771214, 37/511 flips, top-32 0.94).

GATES: build clean; `ik2_kernels_test` all passed unset / IK2_SC512=1 / IK2_SC_MB=384; unset identity greedy `de1aa7dbdeb2be5a`, `--score --chunk 1` `1c874e32e7023ab4`, NLL 2.754963.

CAVEATS
- `--chunk 8192` with >=512 MiB OOMs at load (`cudaMalloc` in ik2_main.cu:452, before tier planning). The served config runs the default `--chunk 2048`, which loads comfortably.
- Back-to-back process loads briefly hit `expert cache: no slot free (slots 16, ...)` while the previous process's VRAM was not yet reclaimed: the tier is planned from free VRAM with little slack. Script artifact, but it is the same latent pool-floor-8 death we know.
- A0 record correction: the fp16-score ON-mode greedy sha is `5c2fb58926bb5035` (3 stable runs), not `ba9cde27`, which predated the fp16-load fix in that session.
- nvcc is not byte-reproducible on this lab: identical source rebuilt to md5 `d6bcac25` vs the serving `50179d9d`. Binary hashes are not an identity gate here; behavioural gates (greedy sha, score payload sha) are.

DECISION: enabled 512 MiB on the lab (user). Config env gains `IK2_SC_MB=512`, backup `maya-inkling-2-mini-lab.json.bak-pre-scmb512`, config md5 `1c89a63d...`/`1af46daf...` -> `9f7c57ef86bf236009331fff6ac52230`; server restarted (SIGTERM 51 s drain, ready 22 s), engine env verified `IK2_SC_MB=512, IK2_SCFP16=1, IK2_AT2S=1, IK2_ROOM_MB=256, IK2_CPU_FRAC=0.5`. Lab deep prefill is now ~9 m 09 s per 100K prompt (was 17 m 45 s before A0, 11 m 50 s after).
### Chunk sweep and the deep-token profile: 4096 is +3.7 pct prefill, and the expert floor is launch-bound (2026-11-26, lab, branch lab-a2 @ 0b2ee38)

Enablement proof for `IK2_SC_MB=512` (the `--serve` mode prints no budget banner): a live 99,700-token request prefilled in ~562.5 s (~177.3 tok/s, vs 181.7 CLI) and decoded 128 tokens in ~10.5 s (~82.0 ms/token). Both match the 512 MiB expectation, so the budget is in force. That request's answer was empty (`finish=length`, content '') because the 128-token cap went to the reasoning trace; the timing, not the content, is the proof.

CHUNK SWEEP, 99,674 tokens, shipped env (`IK2_SCFP16=1`, `IK2_SC_MB=512`), GPU-exclusive CLI:
| `--chunk` | prefill | tok/s | Tq | blocks/pass | tier | expert bytes | misses |
|---|---|---|---|---|---|---|---|
| 1024 | 605.45 s | 164.6 | 112 | 4 (last T=346) | 617 / 2.4 GiB | 600.5 GiB | 160,468 |
| 2048 (shipped) | 548.64 s | 181.7 | 112 | 13 | 501 / 2.0 GiB | 364.6 GiB | 96,686 |
| 3072 | 539.93 s | 184.6 | 112 | 13 | 419 / 1.7 GiB | 279.8 GiB | 74,009 |
| **4096** | **528.23 s** | **188.7** | 112 | 13 | 307 / 1.2 GiB | 246.2 GiB | 64,731 |
- 4096 is **+3.7 pct** over the shipped 2048; 1024 is âˆ’10.4 pct. Control reproduced 548.49 s â†’ 548.64 s (0.03 pct).
- Tq is identical (112) in every arm, so this isolates per-pass overhead, and the mechanism is expert traffic: larger chunks mean fewer passes and far fewer expert copies (364.6 â†’ 246.2 GiB), which outweighs the tier shrink 501 â†’ 307 (a chunk sets `Tm`, so the pass scratch eats the tier).
- 4096 loads fine at chunk 4096; the OOM boundary is 8192 with the 1 GiB of score buffers.

DEEP-TOKEN PROFILE (nsys `--trace=cuda --delay 600 --duration 60` over `--gen 1200`; the window was verified 100 pct decode via 706 LM-head instances; a first capture that mixed prefill tail was discarded):
| where | ms/token | pct | launches/token |
|---|---|---|---|
| attention (A2S split + combine + attn_res + prep_qkv) | 32.23 | 37.9 | 288 |
| expert kernels (gate/up/swiglu + down + weight decode) | 27.84 | 32.8 | 117 |
| LM head (`mv_blocks<8,2>`) | 6.82 | 8.0 | 1 |
| dense projections (qkv/wo/router/out) | 10.38 | 12.2 | 345 |
| routing/loop gates/tail/conv/rms/combine | 4.06 | 4.8 | 579 |
| kernels total | 81.33 | 95.7 | 1332 |
| gap (wall âˆ’ kernels) | 3.65 | 4.3 | â€” |
- Host launch API is 5.9 ms/token (940,439 `cudaLaunchKernel` in 60 s) but fully hidden behind 50.8 s of `cudaStreamSynchronize`: the host waits on the GPU.
- **Verdict: launch/occupancy-bound.** Expert picks run at 72.5 Âµs a pick for 11.3 MFLOP and ~2.9 MB of weights = 156 GFLOPS (2.3 pct of peak) and 40 GB/s (20 pct of HBM), across 117 launches of ~6.5 picks each, ~250 Âµs apiece. Attention A2S is ~20 GB/s and ~60 GFLOPS. Only the LM head approaches a roofline (36 GB/s, 18 pct of HBM). GPU busy 95.7 pct, so the 4.3 pct idle is launch ramp (1332 Ã— ~2.7 Âµs).
- Therefore slot-batching (32 slots a launch) should roughly halve the expert floor â‰ˆ âˆ’14 ms/token (âˆ’16 pct of the deep token) â€” the largest remaining item outside attention. Kernel-driven row tables alone are worth ~3-4 ms (the size of the gap) unless they also remove the per-visit routing D2H.

Deviations: the verification answer was empty (reasoning ate the 128-token budget); the first nsys window mixed 13.7 s of prefill tail and was re-run; no build this round (so the kernels-test gate is vacuous); tree clean at `0b2ee38`, server restored on the live config (md5 `9f7c57ef`), no `--chunk` in the config, downtime 63 m 43 s.
### Slot batching: not applicable, and chunk 4096 is free on decode (2026-11-27, lab, branch lab-a2 @ 0b2ee38)

SLOT BATCHING (`IK2_SLOTB`) WAS NOT BUILT â€” the premise was falsified before any code:
- The 117 expert launches are already per-visit batches: 48 visits/token, ~58 gate_up + ~59 down, ~1.2 launches a visit, 6.5 picks each. Grid is `(F/(WARPS*ROWS)) x picks` (kernels.cu:7867) = ~103 x 6.5 = **670 blocks per launch = 45 waves across the 15 SMs**. The launch is not starved; more rows cannot buy occupancy.
- The total prize for cutting launches is the measured inter-kernel gap: **3.65 ms/token = 4.3 pct**. GPU busy 95.7 pct.
- A 32-slot batch cannot be constructed at T=1: only a visit's K=8 picks share an input vector, and visits are strictly sequential (layer L+1's hidden is layer L's MoE output).
- Per-row cost is weight decode + dot, not input traffic: `IK2_XGLOBAL=0` (stages `x` in shared once per block) is 0.5 pct SLOWER (15.79/15.77/15.78 vs 15.86/15.87/15.85 tok/s, 3 rounds, bit-identical).
- The floor is a 1-token GEMV chain: 384 picks x ~2.9 MB = ~1.13 GB of expert weights per token in 27.84 ms = 41 GB/s (20 pct of HBM) at 2.3 pct of FLOP peak. PolyStrata's 32-slot trick pays where many rows are in flight; T=1 has no more rows to give.
- Next decode targets, in order: attention (32.2 ms) then the dense projections (10.4 ms across 345 launches of ~34 us GEMVs).

CHUNK 4096 IS FREE ON DEEP DECODE (99,674 tokens, shipped env, `IK2_AT2S=1`, `--gen 128`):
| `--chunk` | prefill | tok/s | decode | ms/token | tier | decode misses |
|---|---|---|---|---|---|---|
| 2048 | 548.42 s | 181.7 | 10.63 s | 83.70 | 501 / 2.0 GiB | 3,396 |
| **4096** | **527.90 s** | **188.8** | 10.65 s | 83.86 | 307 / 1.2 GiB | 5,634 |
- Prefill âˆ’20.5 s (âˆ’3.7 pct), decode +0.16 ms/token (+0.2 pct, noise), generated tokens identical. The tier falls 501 -> 307 and decode misses rise 66 pct, yet decode does not care â€” the extra misses overlap compute. Reproduced twice (528.23 s in the sweep, 527.90 s here).
- ENABLED: config `args` now `--model ... --max-context 131072 --drafting off --chunk 4096`, backup `maya-inkling-2-mini-lab.json.bak-pre-chunk4096`, config md5 `9f7c57ef...` -> **`43b9530064da3ed6384024acd328053d`**. Server stopped 51 s, ready 22 s, engine cmdline verified, `/v1/models` 200, live completion `'OK'`, GPU 7889/216 MiB.

Gates (third attempt, after two invalidated by stale exported env): greedy `de1aa7dbdeb2be5a`, `--score --chunk 1` payload `1c874e32e7023ab4`, NLL 2.754963. Tree clean at `0b2ee38`, no build this round, downtime 36 min 18 s.

Lab state now: 100K request = ~528 s prefill (188.8 tok/s) + 83.9 ms/token decode (11.9 tok/s), i.e. ~9 m 30 s to first useful token at 100K.
### Dense-projection batching: null result, the big matrices are already at 85 pct of HBM (2026-11-27, lab, branch lab-a2 @ 0b2ee38)

No code shipped. The row-pairing kernel (`mv_q8p2_kernel`, two rows a warp sharing one float4 `x` load, bit-exact) measured **0.2 pct SLOWER**, so it was reverted per the stop rule; the descriptor fusion was not built because its ceiling was measured at ~0.6 pct before any code.

Launch inventory (706-token pure-decode nsys window, per token):
| kernel | launches | ms | us each |
|---|---|---|---|
| `mv_q8p_kernel<4>` - q, k, v, r, gate, wo | 185.3 | 6.36 | 34.3 |
| `gemv2T` (cuBLAS; router, loop gates) | 112.2 | 1.28 | 11.4 |
| `mv_blocks_kernel<4,2>` | 24.0 | 2.15 | 89.4 |
| `mv_blocks_kernel<2,1>` | 14.0 | 0.29 | 20.9 |
| `mv_blocks_kernel<4,1>` | 9.0 | 0.20 | 23.0 |
| `mv_blocks_kernel<8,2>` (LM head, untouched) | 1.0 | 6.82 | 6817 |
| dense total excl. head | 344.5 | 10.28 | |

Per visit (48/token): ~3.9 `mv_q8p` + ~2.3 `gemv2T` + ~1 `mv_blocks` = ~7 launches / 214 us.

WHY IT CLOSED:
- Largest matrix (wq/wgate 2304x2304 Q8 = 5.6 MB) takes 34.3 us => **163 GB/s = 85 pct of HBM peak**. At the roofline; nothing to win.
- The same launch also re-reads `x` (21 MB vs 5.6 MB of weights) but `x` is 9.2 KB and L2-resident: the pairing kernel halved those reads and changed **nothing** (even 0.2 pct slower) â€” L2 was never the limiter.
- Only slack: small matrices (wk/wv 1.9 MB => 55 GB/s, 2 of 7 launches) and the launch ramp, whose entire budget is the measured 3.65 ms/token inter-kernel gap. A 5->1 descriptor fusion of the per-visit projections (192 of 345 launches) can reach only a fraction of that = **<= 0.6 pct**.
- `mv_blocks` (47 launches, 2.6 ms) already shares `x` across ROWS inside a warp (`row_dot_2`) â€” it is the already-done version of this idea, and runs at 25-40 GB/s only because its matrices are small.

GATES (on the corrected build; a first build had a template-placement error and ran a stale binary â€” nothing from it used):
- `ik2_kernels_test` all passed, gate unset and `IK2_DPB=1`.
- Identity OFF: greedy `de1aa7dbdeb2be5a`, score payload `1c874e32e7023ab4`, NLL 2.754963. Identity ON: identical, 511 positions, 0 top-1 divergences, max |dlp| exactly 0.
- Speed, prompt65 `--gen 256`, 6 rotated interleaved rounds: OFF median 14.105 s (18.08 tok/s) vs ON 14.135 s (18.05); five of six paired deltas negative. Secondary 20.6k arm `--gen 128`: 7.98 vs 8.01 s, prefill 75.27 vs 75.32 s â€” same -0.4 pct direction.

State: no commit, tree clean at `0b2ee38`, identity re-verified on the rebuilt binary; server UP on 8095 (ready 21 s, `/v1/models` 200, `content='OK'`), config md5 `43b9530064da3ed6384024acd328053d` unchanged, no `IK2_DPB`, `--chunk 4096` in force, GPU 7891 MiB / 0 pct. Downtime 35 min 16 s.

CONCLUSION FOR THE DECODE BUDGET: of the 83.9 ms deep token, attention 32.2 ms is the only item with real headroom; experts (27.8) are launch-saturated, dense (10.3) is at 85 pct of HBM on the matrices that matter, head (6.8) is one GEMV at 36 GB/s but only 8 pct of the token, and the gap floor is 3.65 ms.
### Decode attention round 2: parallelism shut, the 8 full-layer visits are a per-key serial chain (2026-11-27, lab, branch lab-a2 @ 0b2ee38)

DECOMPOSITION (40,000-token context, 872-token pure-decode window, 68.81 ms/token, live knobs, IK2_AT2S=1):
| family | launches/token | ms/token | us each |
|---|---|---|---|
| `attn_split_at2` total | 48.0 | 14.51 | 302 mixed |
| - 40 sliding visits (1024 keys) | 40.0 | 3.11 | 77.8 |
| - **8 full-attention visits** | **8.0** | **11.40** | **1425** |
| `attn_res_vec4<0,512>` + `<1,512>` | 143.9 | 1.32 | 9.2 |
| `attn_combine` | 48.0 | 0.38 | 7.9 |
| `prep_qkv_r_fp8` | 48.0 | 0.22 | 4.6 |
| family | 288 | 16.43 | |
Scaled to 99.7k (full visits x 2.5 keys): 28.5 + 3.11 + 1.32 + 0.38 + 0.22 = ~33.5 ms, which reproduces the 32.23 ms measured => **the 8 full-layer split instances are 87 pct of the family at depth**, and nothing else in it is worth 1 ms.

ROOFLINE VERDICT: a full instance at 40k moves 61.4 MB of fp8 K+V in 1425 us = 43 GB/s (17 pct of 256), 737 MFLOP = 517 GFLOPS (7.5 pct of 6.85 TF), ~11.5 M warp-instructions at 8.1e9/s (7 pct of 1.14e11). Neither roofline is close. It is a **per-key serial chain**: a 4-MAC lane dot, a 5-step warp reduction and two expf per key per head, R=3 heads a block = ~1.6 MAC per warp-instruction against 32 possible.

THE ONE PLAUSIBLE LEVER, TESTED AND SHUT: `P = min(128, nk/64)` is at its cap for any context above ~8,192 keys, so a deep launch is 8 kv-heads x 128 = 1024 blocks and, at `__launch_bounds__(128,4)`, only 16 of an SM's 64 warps. `IK2_AT3=1` raised the cap to 512 (4096 blocks), decode path only. Measured at 40k, `--gen 128`, 3 OFF / 3 ON interleaved: OFF median 8.66 s = **68.2 ms/token**, ON 8.73 s = **68.7 ms/token** => **-0.8 pct (ON slower)**. 4x the blocks moved nothing: it is bound INSIDE the warp, not by launch parallelism. Reverted, no commit, tree clean at `0b2ee38`.

A2S REFERENCE AT DEPTH: `IK2_AT2S=0` = 13.59 s = **107.0 ms/token** at 40k, so A2S is worth **-38.8 ms/token (-36 pct)** there â€” the family it replaced was far larger than what is left in it.

GATES: build clean; `ik2_kernels_test` all passed unset and with `IK2_AT3=1`; identity OFF and ON both greedy `de1aa7dbdeb2be5a`, score payload `1c874e32e7023ab4`, NLL 2.754963. NOTE the `--score --chunk 1` gate is blind to this knob by construction (nk <= 512 => P <= 8 in both modes), which is why both matched; the depth arms' greedy streams were identical (`e3830cb138` in 5 of 6 runs, one OFF run differed `1c15a89ca9` from the known CPU-lane nondeterminism, and all three ON runs matched OFF).

STATE: server UP on 8095, live config md5 `43b9530064da3ed6384024acd328053d` unchanged, no `IK2_AT3`, `--chunk 4096`, `/v1/models` 200, live completion `'OK'`, GPU 7891 MiB / 0 pct. Downtime 72 min 30 s.

WHAT REMAINS, PRICED: not parallelism, not retiling â€” the 32-lane per-key dot with a warp reduction IS the algorithm at ~1.6 MAC/instr. Reaching it needs a **prefill-style key x head tile kernel for the 8 full layers only** (fp8 K/V decoded into shared, matmul-shaped inner loop, changed online-softmax grouping) â€” a rewrite gated numerically like A2S, not a session's edit. Best case ~28.5 ms -> ~10 ms of an 83.9 ms deep token (~20 pct).
### Full-layer tiled decode attention, attempt 1 (AT4): 11 pct, not 25 â€” reverted, cause named (2026-11-27, lab, branch lab-a2 @ 0b2ee38)

Design (as validated by arithmetic in the prior session): kill the per-key cross-lane reduction â€” Phase A lane-owns-a-key for the score dot (no shuffles in the loop), Phase B one xor-max + one sum per 32 keys per head, Phase C lane-owns-dims PV with the 3 heads sharing V reads. Predicted 129 -> ~38 warp-instructions per (key, head) and ~96x fewer serialized reductions => 1425 -> 600-700 us per full visit.

Built: `attn_split_at4_kernel<KVT,R>` (+170 lines src/ik2/ik2_kernels.cu), gate `IK2_AT4=1` default OFF, dispatch `!kv4 && !kv4k && H%Hkv==0 && R<=4 && d_rel<=32 && window==(1<<30) && nk>=1024`; partials layout and `attn_combine_kernel` untouched; sliding visits and prefill excluded by construction. Tests +79 lines (`test_attn_decode_at4`, H=24/KV=8, nk 2500/5000, ring, f16+fp8).

MEASURED (40k, --gen 128, live knobs, 3 OFF / 3 ON interleaved):
| arm | median | ms/token |
|---|---|---|
| OFF | 8.64 s (14.69 tok/s) | 68.03 |
| ON | 8.48 s (14.97 tok/s) | 66.77 |
=> **-1.26 ms/token (-1.85 pct of the token) = -11.1 pct off the 11.40 ms full-visit total.** Prefill identical. Shallow control flat (14.16 vs 14.17 s; nk=65 < 1024 so the kernel never runs). In the 10-25 pct band => reverted, no commit, not enabled. Patch kept at `/tmp/at4-kernel.diff`.

CORRECTNESS: the double-precision arbiter gives worst 0.01 of tolerance for BOTH at2 and at4, at 2500/5000 keys, ring path, f16 and fp8 â€” the new kernel is numerically as good. Depth greedy streams identical (`e3830cb138`). NOTE the `--score --chunk 4096` arm is BLIND here: it runs the prefill attention path, so it proves nothing about a decode-only kernel.

WHY THE ARITHMETIC MISSED (named):
1. Phase A's scattered reads: a warp-instruction now has 32 lanes reading 16 B each from 32 different 128-B key rows (768-B stride) => 32 sectors per instruction, ~2x sector traffic for the same useful bytes, 32 outstanding requests. at2 read one key's row contiguously (perfect coalescing).
2. Phase C's serial tail: 32 sequential iterations per group, each with 3 shfl broadcasts per head and one 128-B V row â€” a dependency chain, where at2 processed 4 keys per warp iteration with K and V at the same offset and four warps interleaving.
Conclusion: the kernel was NOT instruction-issue-bound; it is request/latency-bound, which an instruction count cannot see.

NEXT ATTEMPT IF FUNDED (fixer's, agreed): same Phase A/B/C but stage the 32 keys through shared memory (32 x 128 B = 4 KB per warp, written scattered / read coalesced) and have Phase C take 2-4 keys per iteration with their V rows pre-staged â€” removes both new stalls while keeping the reduction win. Do NOT re-run the same kernel.

Gate-OFF identity intact: greedy `de1aa7dbdeb2be5a`, `--score --chunk 1` `1c874e32e7023ab4`, NLL 2.754963. Tree clean at `0b2ee38`; server UP on 8095, config md5 `43b9530064da3ed6384024acd328053d`, no `IK2_AT4`, downtime 42 min 00 s.
### Full-layer tiled decode attention, attempt 2: shared-memory staging is 9.7 pct SLOWER â€” the approach is closed (2026-11-27, lab, branch lab-a2 @ 0b2ee38)

The fp8 fault was a one-liner: `v_load4<uint8_t>`'s fast path is `__ldg` (global pointers only) and the staged branch handed it a SHARED pointer. Fixed with a shared-safe reader used in the staged path only (`k_load4_sh`, plain dereference + the same `e4m3fnx2_to_f2` decode); `v_load4` untouched. Tile, 144-B stride, 16-B cooperative store and the 4-key Phase C all kept.

Tests (both gate states): `ok at4: 2500 keys (f16/fp8) 0.01 | 5000 keys 0.01 | ring path 0.01` -> all passed. The fp8 fault is gone; the kernel is numerically correct in every case.

| 40k, --gen 128, live knobs | OFF (at2) | ON (staged at4) |
|---|---|---|
| rounds | 8.65 / 8.63 / 8.63 s | 9.47 / 9.50 / 9.45 s |
| median | **8.63 s = 67.95 ms/token** | **9.47 s = 74.57 ms/token** |
- ON is +6.6 ms/token = **-9.7 pct against at2**; the 8 full visits go 11.40 -> ~17.9 ms/token (+57 pct).
- Token streams identical `e3830cb138` in all six runs. Shallow control flat (nk=65 never reaches the kernel). Gate-OFF identity exact: greedy `de1aa7dbdeb2be5a`, score `1c874e32e7023ab4`, NLL 2.754963.

| variant | full-visit attention | vs at2 |
|---|---|---|
| at2 (live) | 11.40 ms | â€” |
| unstaged at4 | **10.14 ms** | **-11.1 pct** (best measured, never shipped) |
| staged at4 | ~17.9 ms | +57 pct |

WHY STAGING LOSES, and why this closes the approach rather than the implementation: the tile buys coalescing but costs **26,592 B of static shared per block = 3 blocks/SM = 12 warps of 64, against at2's 4 blocks/16 warps** â€” 25 pct of the resident warps â€” plus ~70 extra instructions per 32-key group (8 cooperative loads, 8 16-B shared stores, a `__syncwarp()`, 32 shared reads at a 4-way bank conflict). In a stall-bound kernel the warps ARE the resource: spending a quarter of them to fix a request pattern that the stall was not caused by cannot pay. Any future attempt at these 8 visits must not use shared staging.

State: no commit, tree clean at `0b2ee38`, kernel and test reverted and rebuilt, identity re-gated exact. Patch kept outside git at `/tmp/at4-smem.diff` (17,477 B) alongside `/tmp/at4-kernel.diff`. Server UP on 8095, `/v1/models` 200, `content='OK'`, env `IK2_AT2S=1 IK2_SCFP16=1 IK2_SC_MB=512`, no `IK2_AT4`, `--chunk 4096`, config md5 `43b9530064da3ed6384024acd328053d`, GPU 7891 MiB. Downtime 34 min 8 s.

DECODE ATTENTION IS CLOSED: 28.5 ms of the 83.9 ms deep token, warp/latency-bound, best available change = the unstaged at4 at -11.1 pct (-3.2 ms, ~3.8 pct of the token), not in the tree. Four attempts, three measured outcomes, one approach eliminated with a number.
### Knob predicate: corrected, gated `IK2_KNOB_CHEAP`, default OFF (2026-11-27, lab, branch lab-a2 @ d77e58b)

THE DEFECT (found in the CPU-lane quality gate, 2026-11-25): `IK2_CACHE_BIAS` and `IK2_TAIL_SKIP` test RESIDENCY as a proxy for "cheap to compute". With a CPU lane present a lane-served miss is never resident yet costs no fetch, so the bias steered toward stale residents and tail-skip dropped picks the lane would have computed free. At `IK2_CPU_FRAC=1.0` that was +0.144 NLL and 21 pct top-1 flips with `score cache: 0.00 GiB copied`.

THE FIX (commit `8a53ba7`, then gated by `d77e58b`): `Layer` gains `cheap_d/cheap_h`; `route_tables()` builds `cheap = resident OR lane-served`, mirroring the splitter exactly (misses sorted by expert id, first `ceil(share x n_miss)` go to the lane, using the splitter's own moving `g_cpu_share`). The bias rewrites `probs_k` from it; `tail_seed`, `tail_apply`'s keep test and `route_token_kernel`'s inline tail use `resid | cheap` (the weight-0 placeholder still prefers a genuinely resident expert). `IK2_CPU_FRAC=0` or no pool => `cheap == residency` => old behaviour exactly.

GATE `IK2_KNOB_CHEAP` (default OFF, commit `d77e58b`): `cheap_d/cheap_h` are ALLOCATED ONLY UNDER THE GATE, so when off the call sites get `nullptr` and the kernels fall back to `resid_d` alone â€” no residency-shaped stand-in, no leftover cost. Banner only when on. Live config does not name it (md5 `43b9530064da3ed6384024acd328053d` unchanged).

MEASURED (prompt65 `--gen 256`, live env, 6 rotated rounds; quality `--score prefill-tokens-512.txt --chunk 1`, 511 positions, lane-off reference 2.853193):

| gate | tok/s median | mean NLL | top-1 flips vs lane-off |
|---|---|---|---|
| OFF (pre-fix predicate) | **18.15** (18.08-18.18) | 2.954453 | 96/511 = 18.8 pct |
| ON (corrected) | 16.56 (16.44-16.58) | **2.900099** | 76/511 = 14.9 pct |

ON is **8.76 pct SLOWER** and buys **-0.054 NLL** (previous round: -0.037, flips 19.8 -> 15.7 pct; the split feedback is timing-driven so the digit does not reproduce, and the doc records both rounds).

WHY IT CANNOT PAY: at `CPU_FRAC=1.0` the lane serves nearly every miss, so under the corrected predicate the bias correctly goes to ~0 and tail-skip stops dropping â€” that is the arm that measured 16.3-16.6 tok/s because the lane saturates. And the lane cannot be widened: **`IK2_LANE_THREADS` is NOT the CPU_FRAC lane's knob** (it sizes the `IK2_LANE` parity server, which is off here â€” a probe at `IK2_LANE_THREADS=8` still shows `CPU lane: 7 threads`); the real pool is `--threads - 1` = 7 workers on an 8-thread/4-core box, already the ceiling.

VERDICT: the fix is correct and shipped behind a default-off gate; the live arm (CPU_FRAC 0.5 + code defaults bias 0.15 / tail 4) stays because it is 8.8 pct faster with 0.05 worse NLL. The gate is for anyone who wants quality-first with a heavy CPU lane.

Gates: build clean; `ik2_kernels_test` all passed both gate states; identity with gate OFF exact (greedy `de1aa7dbdeb2be5a`, score `1c874e32e7023ab4`, NLL 2.754963). Downtime 14 min 37 s; server UP on 8095, `/v1/models` 200, completion `content='OK'`, GPU 7891 MiB.
### Prefill profile: the PV GEMM is the anomaly, and the budget lever has not flattened (2026-11-27, lab, branch lab-a2 @ d77e58b)

PROFILE (nsys `/opt/cuda-12.8/bin/nsys`, 99,673 tokens, `--gen 1`, chunk 4096, `IK2_SCFP16=1`, `IK2_SC_MB=512`; profiled run 530.52 s = +0.6 pct over 527.47 s so it is usable; 1,075,362 launches):

| phase | total s | share |
|---|---|---|
| attention | **308.0** | **58.3 pct** |
| experts | 149.8 | 28.3 pct |
| dense (QKV/wo/router/loop gates) | 70.9 | 13.4 pct |

Attention breakdown: PV GEMM 181.3 s (kernel `maxwell_sgemm_fp16_128x128_nn`, 171.74 s of it), QK GEMM 70.2 s (`maxwell_fp16_sgemm_fp16_128x128_tn`), `attn_probs_bias_f16_kernel` 38.36 s, `unpack_kv_fp8_to_f16_vec4` 13.86 s, bias GEMM 2.1, prep 1.1.

THE ANOMALY: QK and PV do the SAME 269.1 TFLOP. QK runs at **3,833 GFLOPS (56 pct of peak)** with grid (4,8,3) = 96 blocks = 51 warps/SM; PV runs at **1,484 GFLOPS (21.7 pct)** with grid (1,4,3)-(1,8,3) = 12-24 blocks = 6-13 warps/SM. PV has N pinned at D = 128 (one N-tile), M = Tq = 112 tiled to 1-4 blocks, and K = nk serial and huge â€” a long-K/few-tile GEMM, which is the split-K case. Not bandwidth (score traffic 12 GB/s of 256). **Giving PV QK's geometry saves ~108 s = 20 pct of the run.** Experts already run at 4,735 GFLOPS (69 pct) and softmax at 110 GB/s (43 pct), so they have little headroom.

ARITHMETIC CORRECTION: the model's work is ~1,334.9 TFLOP (attention QK+PV 2 x 269.1, bias 3.8, experts 609.5, dense 183.4) = **2.52 TFLOPS achieved = 36.9 pct of the 6.85 TFLOPS peak**. The earlier "563.6 TFLOP / 165 s at 50 pct" floor was wrong (it forgot the 48 layer VISITS vs 24 layers, i.e. the loop runs each token through 48 visits). True floor at 50 pct is **~390 s**; the 528 s run is at 74 pct of that, and the upside is ~1.35x overall, concentrated in PV.

SCORE BUDGET SWEEP (same prompt, back-to-back):

| `IK2_SC_MB` | prefill s | tok/s | Tq at 99.7k | tier | copied |
|---|---|---|---|---|---|
| 512 (control) | 527.47 | 189.0 | 112 | 307 | 246.2 GiB |
| 640 | 496.07 | 200.9 | 140 | 245 | 265.0 GiB |
| **768** | **466.50** | **213.7** | 168 | 182 | 283.3 GiB |

Each 128 MiB buys 5.95 pct (0.0465 pct a MiB), identical at both steps â€” the lever has NOT flattened in this range. Decode check at 20,607 tokens, `--gen 128`: 63.07 -> 63.07 -> 63.23 ms/token, so 768 costs **+0.25 pct decode** at that depth (tier 307 -> 182 slots). Caveat: the decode check is at 20.6k; at >=40k the lost tier could bite and must be re-checked before 768 becomes permanent.

VERDICT: `IK2_SC_MB=768` recommended (exact change: the env block, `"IK2_SC_MB": "768"`), and PV split-K is the remaining structural target (~108 s). Not enabled at capture time; config md5 unchanged. Downtime 48 min 41 s. Evidence at /tmp/ik2p-512.nsys-rep, /tmp/pref-{512,640,768}.log, /tmp/pref-dec-{512,640,768}.log.
### Split-K PV GEMM: -112 s off every 100K prefill (2026-11-27, lab, branch lab-a2 @ c10bcb7)

- The profile named PV as the anomaly: QK and PV move the same 269.1 TFLOP, but `maxwell_fp16_sgemm_fp16_128x128_tn` ran at 3,833 GFLOPS (56 pct) while `maxwell_sgemm_fp16_128x128_nn` ran at 1,484 GFLOPS (21.7 pct) for 171.74 s, because at Tq=112 the PV GEMM is M=128 (D), N=112 (Tq), K=nk (<=32,768 a chunk), batch 3 = **3 blocks** with a 100k-long K.
- Attempt 1 (hand-written split-K GEMM, 256 threads, 64x128 tile, S in blockIdx.z) was numerically exact (6.2e-7 of full scale vs a host double reference) but **1.08 TFLOPS = 2.7x slower than cuBLAS**; reverted, not left as dead code.
- Shipped design `IK2_PVSK=S` (commit c10bcb7, 5 files +202/-9, patch /tmp/pv-sk.patch, gate default off): keep cuBLAS, split K over S slices each on its own stream (one stream a (slice, KV group)), FP32 partials `S x kRelbRows(512) x HD(3072) x 4 B` = 6.29 MiB a slice (S=4 => 25.2 MiB, tier unchanged at 307 slots), then one reducer sums them into `W.o`. S=1 writes straight to the output (no partials).
- Speed, 99,673 tokens, chunk 4096, SCFP16, SC_MB 512: off **527.96 s / 188.8 tok/s**; S=2 419.03 (237.9); **S=4 415.71 / 239.8 (-112.25 s, -21.3 pct)**; S=8 415.93 (no better than 4). PV implied 171.74 -> **59.49 s = 4,524 GFLOPS = 66 pct of peak** (65.4 pct of the GEMM removed).
- Quality: `--score` over 4,096 tokens through the real prefill path, off 2.232597 / PPL 9.3241 vs S=4 2.235130 / 9.3477 = **+0.002533 NLL (+0.11 pct), 0/4095 next-token flips**, max |dlp| 3.839 (median 0.05, p99 0.83) from router near-ties across the 48 visits. 40k greedy stream identical `e3830cb138` both ways. Not covered: end-to-end quality at nk~100k (no score at that size).
- Gates: build clean; `ik2_kernels_test` all passed with the gate unset and at IK2_PVSK=4, including a new geometry case (H=24, KV=8, D=128, Tq=112, nk=2,500) within 2e-3 of the double reference (worst 6.2e-7) and S=1 vs S=4 agreement (5.5e-4); identity exact (greedy `de1aa7dbdeb2be5a`, `--score --chunk 1` payload `1c874e32e7023ab4`, NLL 2.754963).
- Second job, the 768 MiB decode re-check at 40k: SC_MB 512 prefill 134.44 s / decode 67.95 ms a token / tier 307; 768 prefill 130.22 s / **68.19 ms a token (+0.35 pct)** / tier 182, tokens identical. So 768 is safe on decode at both 20.6k (+0.25 pct) and 40k; NOT enabled (user chose PVSK only).
- Enabled on the lab: `IK2_PVSK=4` added to the config env (backup .bak-pre-pvsk), config md5 `43b9530064da3ed6384024acd328053d` -> **`44ed858625eccbbca48ca61d3c38d053`** (15 env keys), server restarted, engine env verified.
- Served verification through the API: 33,637-token prompt 97 s prefill (347 tok/s); **100,837-token prompt, 427.1 s wall, 102 tokens, answered `HARBOUR-CHECK-41` verbatim** => prefill ~418.6 s = **241 tok/s** (was 528 s / 189 before PVSK, 1,065 s / 93.6 at the sprint's start). Decode ~83 ms a token, unchanged.
- Deviations: `ncu` cannot profile this Pascal card, so the split was sized from nsys grid/occupancy plus wall clock; `IK2_PVSK` covers the fp8-KV prefill path only; downtime 74 min 26 s for the build/test window plus ~1 min for the enable restart.
