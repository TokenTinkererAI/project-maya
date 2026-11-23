// ik2_watch.h - IK2_WATCH: the host's two waits get a name when they stop returning (the strata-fork pattern, borrowed:
// engine/include/strata/core/stream_watch.hpp for the named-stream registry, engine/src/program/generate.cpp's
// stall_report for the report itself).
//
// WHY. A kernel or a transfer that outlives the Windows TDR window (~2 s by default) is killed by the driver, and what
// the engine sees is a context loss with no clue which visit it was on: the process dies inside a driver call, or a
// stream reports an error the counters cannot explain.  This tree's host-MV path has exactly two blocking waits - the
// routing D2H's (main.cu's sync/spin) and the CPU lane's (Pool::wait) - and both can sit for seconds under extreme load
// (a paging driver, a busy desktop, a 1M-context chunk).  Naming the wait, the visit and the streams turns that into one
// line a reader can act on.
//
// On (IK2_WATCH=1): each wait is timed; past IK2_WATCH_MS (1500 default, under TDR's 2 s) the report prints, and past
// IK2_WATCH_FATAL_MS (8000 default) the engine stops with a named message instead of outliving the driver's reset.
// Off: one timestamp read a wait (the same clock IK2_DROUTE already takes) and nothing else, so ON is byte-identical:
// this is a diagnostic, it changes no arithmetic and no scheduling.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace ik2::watch {

// ---- the named streams (the fork's StreamWatch without its mutex: this registry is written once at init and read by
// the waiting host thread only, so there is nothing to guard)
inline void* g_stream[8] = {};
inline const char* g_stream_what[8] = {};
inline int g_nstream = 0;
inline const char* g_stage = "nothing";   // the stage the host last launched: what a stall names as in flight

inline void add_stream(cudaStream_t s, const char* what) {
    if (g_nstream >= 8) return;
    g_stream[g_nstream] = (void*) s, g_stream_what[g_nstream] = what, ++g_nstream;
}
inline bool on() {
    static const bool v = [] {
        const char* e = std::getenv("IK2_WATCH");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v;
}
inline void note(const char* stage) {
    if (on()) g_stage = stage;
}

inline double threshold_s() {
    static const double v = [] {
        const char* e = std::getenv("IK2_WATCH_MS");
        return (e ? std::atoi(e) : 1500) / 1000.0;
    }();
    return v;
}
inline double fatal_s() {
    static const double v = [] {
        const char* e = std::getenv("IK2_WATCH_FATAL_MS");
        return (e ? std::atoi(e) : 8000) / 1000.0;
    }();
    return v;
}
inline double now() {
    using clk = std::chrono::steady_clock;
    static const clk::time_point t0 = clk::now();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

/// The report: the wait, the visit, the last stage launched, then every named stream asked WITHOUT blocking - the
/// fork's distinction matters here: "had finished" is a lost wake-up, "still running" is a kernel that will be reset.
inline void report(std::FILE* f, const char* where, double secs, int layer, int visit, int pos, int T,
                   long long lane_pending, double lane_ms) {
    std::fprintf(f, "ik2: stall %.2f s in %s (visit %d, layer %d, position %d, %d token%s); the last stage launched "
                    "was \"%s\"\n", secs, where, visit, layer, pos, T, T == 1 ? "" : "s", g_stage);
    for (int i = 0; i < g_nstream; ++i) {
        const cudaError_t e = cudaStreamQuery((cudaStream_t) g_stream[i]);
        if (e == cudaSuccess)
            std::fprintf(f, "  %s had finished, but the host was still waiting for it\n", g_stream_what[i]);
        else if (e == cudaErrorNotReady)
            std::fprintf(f, "  %s was still running\n", g_stream_what[i]);
        else {
            std::fprintf(f, "  %s could not be asked: %s\n", g_stream_what[i], cudaGetErrorString(e));
            (void) cudaGetLastError();
        }
    }
    if (lane_pending > 0 || lane_ms > 0)
        std::fprintf(f, "  the CPU lane: %lld task%s outstanding, this wait %.1f ms\n", lane_pending,
                     lane_pending == 1 ? "" : "s", lane_ms);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
        std::fprintf(f, "  VRAM: %.0f MiB free of %.0f MiB\n", (double) free_b / 1048576.0,
                     (double) total_b / 1048576.0);
    else
        (void) cudaGetLastError();
    std::fflush(f);
}

/// One wait's guard: construct with where it is (and the visit it is on), then call done() when it returns - the
/// destructor cannot report, the streams' state is gone by then.
struct Wait {
    Wait(const char* where, int layer, int visit, int pos, int T)
        : where_(where), layer_(layer), visit_(visit), pos_(pos), T_(T), t0_(on() ? now() : 0.0) {}
    void done(long long lane_pending = 0, double lane_ms = 0) const {
        if (!on()) return;
        const double s = now() - t0_;
        if (s < threshold_s()) return;
        report(stderr, where_, s, layer_, visit_, pos_, T_, lane_pending, lane_ms);
        if (s > fatal_s()) {
            std::fprintf(stderr, "ik2: giving up after %.1f s in %s - a wait this long is past the driver's own "
                                 "patience (its report is above), so the engine stops here, by name\n", s, where_);
            std::fflush(stderr);
            std::exit(3);
        }
    }
    const char* where_;
    int layer_, visit_, pos_, T_;
    double t0_;
};

}  // namespace ik2::watch
