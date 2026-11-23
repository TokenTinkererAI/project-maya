// src/ik2/ik2_cpu.cpp - maya-ik2's CPU lane (the interface: ik2_cpu.hpp).
#include "ik2_cpu.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX   // (the affinity topology's header: its min/max macros would eat std::min/std::max below)
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif

namespace ik2::cpu {

bool available() { return nvfp4_ok(); }

bool nvfp4_ok() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
    return false;
#endif
}

bool native_ok() { return strata::kernels::cpu::native_experts_available(); }

// (IK2_LANE_AFFINITY: the coarsest half of Strata's NUMA binding, and all a single-socket box has - one node, so
//  first-touch placement has nothing to choose between; what is left is keeping each worker on a core of its own.  The
//  OS's reported cores, a worker round-robin over them - the count stays the operator's (--threads).)
namespace {

bool affinity_wanted() {
    static const bool on = [] {
        const char* v = std::getenv("IK2_LANE_AFFINITY");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

std::vector<int> core_lps() {
    std::vector<int> lps;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len ? len : 1);
    if (len && GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        for (DWORD off = 0; off < len;) {
            const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) (buf.data() + off);
            if (e->Relationship == RelationProcessorCore) {
                const ULONG_PTR m = e->Processor.GroupMask[0].Mask;
                for (int b = 0; b < 64; ++b)
                    if (m & ((ULONG_PTR) 1 << b)) {
                        lps.push_back(b);   // (each core's first logical processor: no SMT sibling)
                        break;
                    }
            }
            off += e->Size;
        }
    }
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0)   // (a single socket's CPUs; the OS's order is the topology's)
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) lps.push_back(i);
#endif
    return lps;
}

void pin_worker(int i) {
    if (!affinity_wanted()) return;
    static const std::vector<int> lps = core_lps();
    if (lps.empty()) return;
    const int lp = lps[(size_t) i % lps.size()];
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << lp);
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(lp, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#else
    (void) lp;
#endif
}

}  // namespace

Pool::Pool(int n) {
    // (IK2_POOL2: the POD ring - see the block below.  Read here so the workers start in the right loop.)
    p2_ = getenv("IK2_POOL2") && std::atoi(getenv("IK2_POOL2")) != 0;
    if (p2_) {
        p2jobs_ = std::make_unique<P2Job[]>(kP2Jobs);
        p2ring_ = std::make_unique<P2Slot[]>(kP2Cap);
        for (int i = 0; i < kP2Cap; ++i) p2ring_[i].seq.store((uint32_t) i, std::memory_order_relaxed);
    }
    for (int i = 0; i < n; ++i) th_.emplace_back([this, i]() { worker(i); });
}
void Pool::worker(int i) {
    pin_worker(i);
    if (p2_) {
        p2_worker(i);
        return;
    }
    for (;;) {
        std::function<void()> f;
        {
            // spin a while before sleeping: an expert's chunks come ~1 ms apart, a wake-up costs tens of us
            std::unique_lock<std::mutex> lk(m_, std::defer_lock);
            for (int spin = 0;; ++spin) {
                lk.lock();
                if (stop_ || !q_.empty()) break;
                if (spin > 20000) {
                    cv_.wait(lk, [this]() { return stop_ || !q_.empty(); });
                    break;
                }
                lk.unlock();
                _mm_pause();
            }
            if (stop_ && q_.empty()) return;
            f = std::move(q_.front());
            q_.pop_front();
        }
        f();
        {
            std::lock_guard<std::mutex> lk(m_);
            if (--pending_ == 0) done_.notify_all();
        }
    }
}
Pool::~Pool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
}
void Pool::post(std::function<void()> f) {
    {
        std::lock_guard<std::mutex> lk(m_);
        q_.push_back(std::move(f));
        ++pending_;
    }
    cv_.notify_one();
}
void Pool::post_many(std::function<void()>* f, int n) {
    if (n <= 0) return;
    {
        std::lock_guard<std::mutex> lk(m_);
        for (int i = 0; i < n; ++i) q_.push_back(std::move(f[i]));
        pending_ += n;
    }
    if (n == 1) cv_.notify_one();
    else cv_.notify_all();
}
void Pool::wait() {
    if (p2_ && p2_pending_.load(std::memory_order_acquire) > 0) {   // (IK2_POOL2: the caller works the ring, not spins)
        p2_host_pin();
        p2_drain();
        return;
    }
    std::unique_lock<std::mutex> lk(m_, std::defer_lock);
    for (int spin = 0; spin < 200000; ++spin) {   // (the caller spins too: the work is short)
        lk.lock();
        if (pending_ == 0) return;
        lk.unlock();
        _mm_pause();
    }
    lk.lock();
    done_.wait(lk, [this]() { return pending_ == 0; });
}

// ---- IK2_POOL2 (the strata-fork pool's design, ported to this workload: strata-fork engine/src/kernels/cpu/pool.cpp)
//
// What the old path paid per task, and this one does not: a std::function for every row-chunk (a heap allocation and a
// lock pair on each post), a mutex lock on every pop AND on every completion, and a host that sat on that same mutex
// in its wait loop.  A visit posts 3n tasks a CPU expert (n = the workers), so a token paid thousands of mallocs and
// thousands of lock pairs for work whose whole point is to keep the cores reading bytes.
//
// The port's pieces, and where each comes from:
//  * POD tasks on a bounded MPMC ring, claimed one per thread (the fork's `head`/`claim`, pool.hpp:342, pool.cpp:1062).
//    Their claim is a CAS on one packed `epoch | njobs | index` word, so a worker that woke for an older batch cannot
//    claim from the next one - issue #29's double-run race.  Here the same invariant is carried by the ring itself: a
//    slot's sequence tag is only re-armed by the pop that took it, so a late worker's stale position can never claim a
//    live task; the *job record* also carries a generation (`P2Task::gen`), and a task whose generation is stale is
//    dropped rather than run, which is the epoch tag's guard over the part of the state that outlives one batch.
//  * The host works the ring while it waits (`p2_drain`), the fork's measured ~9% (pool.hpp:141-149: the unpinned
//    spinning host costs it a third of the pool's bandwidth).
//  * The host thread pinned to the first core's SMT sibling, so a GPU interrupt on logical processor 0 does not land on
//    the thread issuing the copies and launches (the fork's 26.9 -> 36.32 GB/s, and their `HostCore::Sibling`).
//  * alignas(64) on the counters workers and host contend on (their review finding C3).
// NOT ported: their GCC-only target("") ISA dispatch and their int8/VNNI activations - our lane keeps ggml-cpu's dots,
// untouched, and MSVC has no function multiversioning.

// (the host's own park: pin the *calling* thread, once, to the first physical core's second logical processor - the
// sibling, not the LP the GPU's interrupts land on.  No sibling (a 1-thread-per-core machine): the first core's LP.)
static void p2_pin_host_once() {
#if defined(_WIN32)
    static const int lp = [] {
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        std::vector<char> buf(len ? len : 1);
        if (!len || !GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                     (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len))
            return -1;
        for (DWORD off = 0; off < len;) {
            const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) (buf.data() + off);
            if (e->Relationship == RelationProcessorCore) {
                const ULONG_PTR m = e->Processor.GroupMask[0].Mask;
                int first = -1;
                for (int b = 0; b < 64; ++b)
                    if (m & ((ULONG_PTR) 1 << b)) {
                        if (first < 0) first = b;
                        else return b;   // (the sibling)
                    }
                return first;
            }
            off += e->Size;
        }
        return -1;
    }();
    if (lp >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << lp);
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    CPU_SET(1, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}
void Pool::p2_host_pin() {
    static std::atomic<bool> once{false};
    bool expected = false;
    if (once.compare_exchange_strong(expected, true)) p2_pin_host_once();
}

int Pool::p2_job_slot() {
    const int j = p2_next_job_++ % kP2Jobs;
    P2Job& r = p2jobs_[j];
    // (a slot is free once its last down task ran; wait() per visit means this never spins in practice)
    for (uint32_t spins = 0; r.left.load(std::memory_order_acquire) != 0; ++spins) {
        if (spins > 4000000u) {
            std::fprintf(stderr, "ik2: cpu pool2 stalled: job slot %d still running\n", j);
            std::abort();
        }
        _mm_pause();
    }
    ++r.gen;
    return j;
}

bool Pool::p2_push(const P2Task& t) {
    uint32_t spins = 0;
    for (;;) {
        uint64_t pos = p2_enq_.load(std::memory_order_relaxed);
        P2Slot& s = p2ring_[(size_t) pos & kP2Mask];
        const int32_t d = (int32_t) (s.seq.load(std::memory_order_acquire) - (uint32_t) pos);
        if (d == 0 && p2_enq_.compare_exchange_weak(pos, pos + 1, std::memory_order_acq_rel)) {
            s.t = t;
            s.seq.store((uint32_t) (pos + 1), std::memory_order_release);
            return true;
        }
        _mm_pause();
        if ((++spins & 0xfffffu) == 0) return false;   // (a full ring: the caller's visit is stuck; see p2_native_expert)
    }
}

bool Pool::p2_pop(P2Task& t) {
    uint64_t pos = p2_deq_.load(std::memory_order_relaxed);
    P2Slot& s = p2ring_[(size_t) pos & kP2Mask];
    if ((int32_t) (s.seq.load(std::memory_order_acquire) - (uint32_t) (pos + 1)) != 0) return false;   // empty
    if (!p2_deq_.compare_exchange_weak(pos, pos + 1, std::memory_order_acq_rel)) return false;         // (retry next)
    t = s.t;
    s.seq.store((uint32_t) (pos + (uint64_t) kP2Cap), std::memory_order_release);
    return true;
}

void Pool::p2_run(const P2Task& t) {
    namespace kn = strata::kernels::cpu;
    P2Job& r = p2jobs_[t.job];
    if (t.gen == r.gen) {   // (a task of a reused slot's earlier generation: dropped, not run twice)
        const kn::NativeFmt* fp = (const kn::NativeFmt*) r.fp;
        NativeJob* j = r.job;
        if (t.kind == 0) {
            const uint8_t* gate = j->blob;
            const uint8_t* up = j->blob + fp->up_off;
            const int up_ty = fp->mixed() ? fp->u_type : fp->gu_type;
            const size_t up_row = fp->mixed() ? fp->u_row : fp->gu_row;
            kn::native_dot_rows(t.part ? up_ty : fp->gu_type, t.part ? up_row : fp->gu_row, t.part ? up : gate,
                                j->act.data(), r.nC, t.r0, t.r1, j->gu.data() + (size_t) t.part * r.nF + t.r0);
            if (r.left.fetch_sub(1, std::memory_order_acq_rel) == 1) {   // (this chunk was the last: activation, down)
                const int nF = r.nF, n = r.n;
                for (int i = 0; i < nF; ++i) {
                    const float g = j->gu[i] * j->sg, u = j->gu[(size_t) nF + i] * j->su;
                    j->h[i] = r.gcap * std::tanh(g / r.gcap) * (1.0f / (1.0f + std::exp(-g))) * r.ucap *
                              std::tanh(u / r.ucap);
                }
                kn::native_quant_h(*fp, j->h.data(), j->hac.data());
                p2_pending_.fetch_add(n, std::memory_order_acq_rel);   // (before the pushes: the count never dips to 0)
                for (int c = 0; c < n; ++c) {
                    P2Task d{};
                    d.kind = 1, d.job = t.job, d.gen = t.gen, d.part = 0;
                    d.r0 = (int) ((long) r.nC * c / n), d.r1 = (int) ((long) r.nC * (c + 1) / n);
                    if (!p2_push(d)) {
                        std::fprintf(stderr, "ik2: cpu pool2: the ring is full pushing down rows\n");
                        std::abort();
                    }
                }
            }
        } else {
            kn::native_dot_rows(fp->d_type, fp->d_row, j->blob + fp->down_off, j->hac.data(), r.nF, t.r0, t.r1,
                                j->y + t.r0);
        }
    }
    p2_pending_.fetch_sub(1, std::memory_order_acq_rel);
}

void Pool::p2_drain() {
    uint32_t spins = 0;
    for (;;) {
        if (p2_pending_.load(std::memory_order_acquire) == 0) return;
        P2Task t;
        if (p2_pop(t)) {
            p2_run(t);
            spins = 0;
            continue;
        }
        _mm_pause();
        if ((++spins & 0xffffu) == 0) {   // (a whole pause budget with nothing to claim: sleep on the counter)
            std::unique_lock<std::mutex> lk(m_);
            done_.wait_for(lk, std::chrono::milliseconds(2), [this]() {
                return p2_pending_.load(std::memory_order_acquire) == 0;
            });
        }
    }
}

void Pool::p2_worker(int i) {
    for (;;) {
        if (stop_) return;
        P2Task t;
        if (p2_pop(t)) {
            p2_run(t);
            continue;
        }
        // (the park: spin on the pending count, then sleep - the fork's kSpinBeforeSleep, without their WAITPKG path)
        uint32_t spins = 0;
        while (p2_pending_.load(std::memory_order_acquire) == 0) {
            if (stop_) return;
            _mm_pause();
            if ((++spins & 0xffffu) == 0) {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait_for(lk, std::chrono::milliseconds(2), [this]() {
                    return stop_ || p2_pending_.load(std::memory_order_acquire) != 0;
                });
            }
        }
    }
}

bool Pool::p2_native_expert(NativeJob* job, int C, int F, float gcap, float ucap) {
    if (!p2_) return false;
    namespace kn = strata::kernels::cpu;
    const kn::NativeFmt* fp = &job->lay->f;
    job->gu.resize((size_t) 2 * F);
    job->h.resize(F);
    job->act.resize(std::max<size_t>(fp->act_bytes, 16));
    job->hac.resize(std::max<size_t>(fp->h_bytes, 16));
    kn::native_quant_act(*fp, job->x, job->act.data());
    const int n = std::max(1, (int) th_.size());
    const int js = p2_job_slot();
    P2Job& r = p2jobs_[js];
    r.fp = fp, r.job = job, r.n = n, r.nC = (int) fp->n_embd, r.nF = (int) fp->n_ff;
    r.gcap = gcap, r.ucap = ucap;
    r.left.store(2 * n, std::memory_order_release);
    p2_pending_.fetch_add(2 * n, std::memory_order_acq_rel);
    const int gen = r.gen;
    for (int part = 0; part < 2; ++part)
        for (int c = 0; c < n; ++c) {
            P2Task t{};
            t.kind = 0, t.job = js, t.gen = gen, t.part = part;
            t.r0 = (int) ((long) F * c / n), t.r1 = (int) ((long) F * (c + 1) / n);
            if (!p2_push(t)) {
                std::fprintf(stderr, "ik2: cpu pool2: the ring is full pushing gate/up rows\n");
                std::abort();
            }
        }
    cv_.notify_all();   // (one wake a batch: a spinning worker sees the count without it)
    return true;
}

namespace {

// ggml_ue4m3_to_fp32 of every byte
struct ScaleTable {
    float v[256];
    ScaleTable() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) {
                v[x] = 0.0f;
                continue;
            }
            const int e = (x >> 3) & 0xF, m = x & 7;
            v[x] = (e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7)) * 0.5f;
        }
    }
};
const ScaleTable g_scale;

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// a 36-byte block: four UE4M3 scales, then 8 bytes a 16-value sub-block (value j < 8 the low nibble of byte j)
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
void nvfp4_rows_avx512(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    const __m128i tab = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const float* xp = x;
        for (int i = 0; i < nb; ++i, b += 36, xp += 64) {
#define IK2_SUB(sb, acc)                                                                       \
    {                                                                                          \
        const __m128i raw = _mm_loadl_epi64((const __m128i*) (b + 4 + (sb) * 8));             \
        const __m128i lo = _mm_and_si128(raw, m4);                                             \
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);                          \
        const __m128i val = _mm_shuffle_epi8(tab, _mm_unpacklo_epi64(lo, hi));                 \
        const __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(val));                        \
        const __m512 xs = _mm512_mul_ps(_mm512_loadu_ps(xp + (sb) * 16), _mm512_set1_ps(g_scale.v[b[sb]])); \
        acc = _mm512_fmadd_ps(v, xs, acc);                                                     \
    }
            IK2_SUB(0, acc0)
            IK2_SUB(1, acc1)
            IK2_SUB(2, acc0)
            IK2_SUB(3, acc1)
#undef IK2_SUB
        }
        out[r - r0] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}
#endif

void nvfp4_rows_scalar(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int nb = K / 64;
    const size_t rb = (size_t) nb * 36;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * rb;
        double acc = 0.0;
        for (int i = 0; i < nb; ++i, b += 36)
            for (int sb = 0; sb < 4; ++sb) {
                float a = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t by = b[4 + sb * 8 + j];
                    a += kv[by & 0xF] * x[i * 64 + sb * 16 + j] + kv[by >> 4] * x[i * 64 + sb * 16 + j + 8];
                }
                acc += (double) g_scale.v[b[sb]] * a;
            }
        out[r - r0] = (float) acc;
    }
}

}  // namespace

void nvfp4_rows(const uint8_t* W, int K, int r0, int r1, const float* x, float* out) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool fast = available();
    if (fast) {
        nvfp4_rows_avx512(W, K, r0, r1, x, out);
        return;
    }
#endif
    nvfp4_rows_scalar(W, K, r0, r1, x, out);
}

// ---- IK2_CPU_FRAC: the GGUF forms through ggml-cpu's quantized dots (see the header)

bool native_layer(int ty0, int ty1, int ty2, int64_t C, int64_t F, size_t slot_bytes, NativeLayer& l) {
    std::string err;
    l.ok = strata::kernels::cpu::native_fmt_mixed(ty0, ty1, ty2, C, F, l.f, err) && l.f.bytes == slot_bytes;
    return l.ok;
}

void post_expert_native(Pool& pool, NativeJob* job, int C, int F, float gcap, float ucap) {
    namespace kn = strata::kernels::cpu;
    if (pool.p2_on() && pool.p2_native_expert(job, C, F, gcap, ucap)) return;   // (IK2_POOL2: POD tasks, no closures)
    const kn::NativeFmt* fp = &job->lay->f;
    const uint8_t* gate = job->blob;
    const uint8_t* up = job->blob + fp->up_off;
    const uint8_t* down = job->blob + fp->down_off;
    const int up_ty = fp->mixed() ? fp->u_type : fp->gu_type;
    const size_t up_row = fp->mixed() ? fp->u_row : fp->gu_row;
    const int nF = (int) fp->n_ff, nC = (int) fp->n_embd;
    job->gu.resize((size_t) 2 * F);
    job->h.resize(F);
    job->act.resize(std::max<size_t>(fp->act_bytes, 16));
    job->hac.resize(std::max<size_t>(fp->h_bytes, 16));
    kn::native_quant_act(*fp, job->x, job->act.data());
    const int n = std::max(1, pool.size());
    job->left = 2 * n;   // gate's chunks and up's
    // (the chunks go in as one block: a lock and a wake a block rather than a chunk - 3n posts of this expert alone,
    //  and this runs on the routing round trip's critical path; the work and its order are the same)
    static thread_local std::vector<std::function<void()>> tq;
    auto dchi = [&pool, job, fp, C, nC, nF, n, down]() {
        tq.clear();
        tq.reserve((size_t) n);
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) C * c / n), r1 = (int) ((long) C * (c + 1) / n);
            tq.emplace_back([job, fp, r0, r1, nF, down]() {
                strata::kernels::cpu::native_dot_rows(fp->d_type, fp->d_row, down, job->hac.data(), nF, r0, r1,
                                                      job->y + r0);
            });
        }
        pool.post_many(tq.data(), (int) tq.size());
    };
    tq.clear();
    tq.reserve((size_t) 2 * n);
    for (int part = 0; part < 2; ++part)
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) F * c / n), r1 = (int) ((long) F * (c + 1) / n);
            tq.emplace_back([job, fp, part, r0, r1, nF, nC, up_ty, up_row, gate, up, gcap, ucap, dchi]() {
                strata::kernels::cpu::native_dot_rows(part ? up_ty : fp->gu_type, part ? up_row : fp->gu_row,
                                                      part ? up : gate, job->act.data(), nC, r0, r1,
                                                      job->gu.data() + (size_t) part * nF + r0);
                if (--job->left == 0) {   // (the last chunk: the activation, then down's rows)
                    for (int i = 0; i < nF; ++i) {
                        const float g = job->gu[i] * job->sg, u = job->gu[(size_t) nF + i] * job->su;
                        job->h[i] = gcap * std::tanh(g / gcap) * (1.0f / (1.0f + std::exp(-g))) * ucap *
                                    std::tanh(u / ucap);
                    }
                    strata::kernels::cpu::native_quant_h(*fp, job->h.data(), job->hac.data());
                    dchi();
                }
            });
        }
    pool.post_many(tq.data(), (int) tq.size());
}

bool selftest(const std::vector<int>& types, std::string& out) { return strata::kernels::cpu::native_selftest(types, out); }

const char* type_name(int type) { return strata::kernels::cpu::native_type_name(type); }

bool make_test_row(int type, std::vector<uint8_t>& row, std::vector<float>& ref, size_t* row_bytes) {
    return strata::kernels::cpu::native_make_test_row(type, row, ref, row_bytes);
}

static double now_s() {
    using clk = std::chrono::steady_clock;
    static const clk::time_point t0 = clk::now();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

void bench(Pool& pool, const std::vector<int>& types, int rows, int reps, const char* which, std::string& out) {
    char line[256];
    for (int ty : types) {
        std::vector<uint8_t> one;
        std::vector<float> ref;
        size_t rb = 0;
        if (!make_test_row(ty, one, ref, &rb) || rb == 0) {
            std::snprintf(line, sizeof line, "  bench %-8s n/a\n", strata::kernels::cpu::native_type_name(ty));
            out += line;
            continue;
        }
        const int n = 256;
        std::vector<uint8_t> w((size_t) rb * rows);
        for (int r = 0; r < rows; ++r) std::memcpy(w.data() + (size_t) r * rb, one.data(), rb);
        std::vector<uint8_t> act(strata::kernels::cpu::native_act_bytes_for(ty), 0);
        std::vector<float> xf(256), af(256);
        std::memcpy(xf.data(), ref.data(), 256 * 4);
        for (int i = 0; i < 256; ++i) af[i] = std::sin(0.11f * (float) i);
        strata::kernels::cpu::native_quant_act_test(ty, af.data(), act.data());
        std::vector<float> out_v(rows);
        const int nw = std::max(1, pool.size());
        const bool hand = std::strcmp(which, "ggml") != 0;
        const double t0 = now_s();
        for (int rep = 0; rep < reps; ++rep) {
            for (int c = 0; c < nw; ++c) {
                const int r0 = rows * c / nw, r1 = rows * (c + 1) / nw;
                pool.post([&w, &act, &out_v, rb, r0, r1, ty, hand]() {
                    strata::kernels::cpu::native_dot_rows_ex(ty, rb, w.data(), act.data(), 256, r0, r1, out_v.data(),
                                                             hand);
                });
            }
            pool.wait();
        }
        const double dt = now_s() - t0;
        const double gb = (double) rows * rb * reps / dt / 1e9;
        std::snprintf(line, sizeof line, "  bench %-8s %6.2f GB/s  (%s, %d rows x %zu B, %d threads)\n",
                      strata::kernels::cpu::native_type_name(ty), gb, hand ? "hand" : "ggml", rows, rb, nw);
        out += line;
    }
}

void post_expert(Pool& pool, ExpertJob* job, int C, int F, float gcap, float ucap) {
    job->gu.resize((size_t) 2 * F);
    job->h.resize(F);
    const int n = std::max(1, pool.size());
    job->left = 2 * n;   // gate's chunks and up's
    auto down = [&pool, job, C, F, n]() {
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) C * c / n), r1 = (int) ((long) C * (c + 1) / n);
            pool.post([job, F, r0, r1]() { nvfp4_rows(job->down, F, r0, r1, job->h.data(), job->y + r0); });
        }
    };
    for (int part = 0; part < 2; ++part)
        for (int c = 0; c < n; ++c) {
            const int r0 = (int) ((long) F * c / n), r1 = (int) ((long) F * (c + 1) / n);
            pool.post([job, C, F, gcap, ucap, part, r0, r1, down]() {
                nvfp4_rows(part ? job->up : job->gate, C, r0, r1, job->x, job->gu.data() + (size_t) part * F + r0);
                if (--job->left == 0) {   // the last chunk: the activation, then down's rows
                    for (int i = 0; i < F; ++i) {
                        const float g = job->gu[i] * job->sg, u = job->gu[(size_t) F + i] * job->su;
                        job->h[i] = gcap * std::tanh(g / gcap) * (1.0f / (1.0f + std::exp(-g))) * ucap * std::tanh(u / ucap);
                    }
                    down();
                }
            });
        }
}

}  // namespace ik2::cpu

