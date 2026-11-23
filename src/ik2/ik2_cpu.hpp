// src/ik2/ik2_cpu.hpp - maya-ik2's CPU lane: a routed expert that is not in VRAM computed where it is, in host
// memory, while the GPU runs the experts it holds - instead of waiting for the expert's 18 MB over PCIe.
// NVFP4 experts only (AVX-512 F / BW; the weights read as they are stored), a few worker threads sharing an expert's
// rows.  The arithmetic is ref/inkling2.py's in F32 (the activation not rounded to FP16 as the device's is).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "strata/kernels/cpu/native_expert.hpp"

namespace ik2::cpu {

/// Whether this CPU (and OS) runs the AVX-512 kernels.
bool available();
/// The AVX-512 gate of the NVFP4 row kernels alone (the GGUF forms take whatever ggml-cpu picked).
bool nvfp4_ok();
/// Whether this build takes GGUF-form experts at all (ggml-cpu's row dots: AVX2 or AVX-512, chosen at run time).
bool native_ok();

struct NativeJob;   // (defined below: IK2_POOL2's job records point at one)

/// Worker threads taking tasks in order; wait(): until every task posted so far (and the ones they posted) is done.
class Pool {
public:
    explicit Pool(int n);
    ~Pool();
    void post(std::function<void()> f);
    /// The same, n at once: one lock and one wake for a block of tasks (a visit posts 3n a CPU expert, and every
    /// post is a lock + a wake on the routing round trip's critical path).
    void post_many(std::function<void()>* f, int n);
    void wait();
    int size() const { return (int) th_.size(); }
    /// IK2_WATCH's report: the tasks still outstanding (the lane's stalls are CPU work, not a stream's).
    long long pending() const { return p2_on() ? (long long) p2_pending_.load() : (long long) q_.size(); }

    /// IK2_POOL2: one expert's dots as POD tasks on a lock-free ring (no std::function, no heap, one claim each), the
    /// host working the ring while it waits.  False when the gate is off: the caller keeps the closure path then.
    bool p2_native_expert(NativeJob* job, int C, int F, float gcap, float ucap);
    bool p2_on() const { return p2_; }

private:
    void worker(int i);
    // ---- IK2_POOL2 (see ik2_cpu.cpp's block)
    struct P2Task {
        int kind;      // 0: gate/up rows, 1: down rows
        int job;       // the P2Job slot
        int gen;       // that slot's generation when the task was pushed (a late task of a reused slot is dropped)
        int part;      // gate (0) or up (1) for kind 0
        int r0, r1;    // the rows of this task
    };
    struct P2Job {
        const void* fp = nullptr;   // strata::kernels::cpu::NativeFmt*
        NativeJob* job = nullptr;
        int n = 1, nC = 0, nF = 0;
        float gcap = 0, ucap = 0;
        std::atomic<int> left{0};
        int gen = 0;
    };
    struct P2Slot {
        std::atomic<uint32_t> seq{0};
        P2Task t{};
    };
    static constexpr int kP2Cap = 256, kP2Mask = kP2Cap - 1, kP2Jobs = 64;
    bool p2_ = false;
    std::unique_ptr<P2Job[]> p2jobs_;
    std::unique_ptr<P2Slot[]> p2ring_;
    alignas(64) std::atomic<uint64_t> p2_enq_{0};
    alignas(64) std::atomic<uint64_t> p2_deq_{0};
    alignas(64) std::atomic<int> p2_pending_{0};
    alignas(64) std::atomic<uint32_t> p2_park_{0};
    int p2_next_job_ = 0;
    bool p2_push(const P2Task& t);
    bool p2_pop(P2Task& t);
    void p2_run(const P2Task& t);
    void p2_drain();               // the host works the ring until nothing is pending
    void p2_worker(int i);         // a worker's park/claim loop on the ring
    int p2_job_slot();             // the next job record, retired by wait()
    void p2_host_pin();            // the calling (host) thread, once, to the first core's sibling

    std::vector<std::thread> th_;
    std::deque<std::function<void()>> q_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    int pending_ = 0;
    bool stop_ = false;
};

/// out[r - r0] = W[r] . x for rows r0 .. r1 - 1 of an NVFP4 matrix (rows of K values, K % 64 == 0).
void nvfp4_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out);

/// One NVFP4 expert: gate, up [F][C] and down [C][F] blocks, x [C]; y [C] = down . h, h = gcap tanh(g / gcap)
/// sigmoid(g) ucap tanh(u / ucap), g = (gate . x) sg, u = (up . x) su - the raw sums of down (its scale is the
/// caller's).  The rows shared by the pool's threads; done when pool.wait() returns.  job must live until then.
struct ExpertJob {
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    const float* x = nullptr;
    float sg = 1.0f, su = 1.0f;
    float* y = nullptr;
    std::vector<float> gu, h;
    std::atomic<int> left{0};
};
void post_expert(Pool& pool, ExpertJob* job, int C, int F, float gcap, float ucap);

// ---- IK2_CPU_FRAC: the experts in their GGUF forms (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, the IQ ones), ggml-cpu's own
// quantized dots - AVX2 on a plain PC, AVX-512 where there is one - and the layer's geometry worked out once.

/// One layer's expert geometry: the three tensors' types, their row bytes, the offsets inside [gate|up|down].
struct NativeLayer {
    strata::kernels::cpu::NativeFmt f;
    bool ok = false;   ///< ggml-cpu has a dot for the three types and the packed size is the engine's slot
};
/// Fills l for the layer's gate / up / down types; slot_bytes: the engine's bytes an expert (the packed form must be
/// exactly that, else the CPU cannot read the same bytes the GPU would).
bool native_layer(int ty0, int ty1, int ty2, int64_t C, int64_t F, size_t slot_bytes, NativeLayer& l);

/// One expert of a GGUF-form layer: y [C] = down . h, h the engine's capped activation of (gate . x, up . x).
struct NativeJob {
    const NativeLayer* lay = nullptr;
    const uint8_t* blob = nullptr;   ///< the expert's [gate rows | up rows | down rows] (pinned host memory)
    const float* x = nullptr;        ///< [C] host
    float sg = 1.0f, su = 1.0f;
    float* y = nullptr;              ///< [C] host: the down sums' rows
    std::vector<float> gu, h;
    std::vector<uint8_t> act, hac;   ///< the quantized activations (ggml-cpu's vec_dot_type)
    std::atomic<int> left{0};
};
/// gate/up rows and down rows shared by the pool's threads; done when pool.wait() returns.  job lives until then.
void post_expert_native(Pool& pool, NativeJob* job, int C, int F, float gcap, float ucap);

/// The dots against a host scalar dequant: one line a type into `out`, false when one is off by more than 1e-3.
bool selftest(const std::vector<int>& types, std::string& out);
/// ggml's name of a type ("q3_K", "nvfp4", ...).
const char* type_name(int type);

/// A synthetic expert row of `type` in `row` (8 x 256 values from known floats): the micro-bench's and the selftest's
/// input, so a hand-tuned kernel can be checked against the same bytes.
bool make_test_row(int type, std::vector<uint8_t>& row, std::vector<float>& ref, size_t* row_bytes = nullptr);
/// GB/s of the row dots over `rows` rows split across the pool (all cores), `reps` times: hand-tuned where this build
/// has one, else ggml-cpu's.  One line into `out`.  `hand`: only the hand kernel (false: only ggml's).
void bench(Pool& pool, const std::vector<int>& types, int rows, int reps, const char* which, std::string& out);

}  // namespace ik2::cpu
