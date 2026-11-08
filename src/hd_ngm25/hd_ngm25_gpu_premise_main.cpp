// GI0 — ngm2.5 GPU premise-check micro-benchmark (design/ngm2_5_gpu.md §2/§8).
//
// THROWAWAY. The whole GPU port rests on one unmeasured claim: a realized-sparse
// arena on device beats the dense ngm3-GPU compute floor because it moves ~4.6x
// less data through a bandwidth-bound response sweep. This harness measures that
// claim BEFORE any production kernel is written, so a negative result kills the
// port cheaply (a valid outcome, not a failure).
//
// Experimental design (documented decision, this session): instead of comparing
// against a separately-measured ngm3 number with a different measurement method,
// run BOTH arenas through the SAME harness on the SAME adapter:
//   * SPARSE  — realized-sparse arena (matched-config sizes: ~342k synapses over
//               256 dendrites, input-major). This is what ngm2.5-GPU would run.
//   * DENSE   — the ngm3 rep-inflated equivalent (rep_cnt = 2^depth = 8x the same
//               structure, ~1.57M synapses). This models ngm3-GPU's response
//               traffic under an identical kernel/measurement, the cleanest
//               control for "traffic proportional to realized structure".
// The response sweep is the ONLY pass measured (R2 accumulate + penalty + reduce);
// winner/adapt/inhibition are out of scope for the premise (GI6 measures the whole
// step). The kernel uses ngm2.5's structural-identity syn_key (constants
// 3266489917/2246822519/668265263/374761393) and shared_prng's pcg_uniform, so it
// is representative of the real per-synapse draw cost.
//
// Faithful-traffic note: the kernel reads perm first and skips input_idx/seg for
// sub-threshold synapses (exactly as the CPU response_dendrite does), so the 12
// B/syn device read set is modeled correctly. It accumulates into a single
// per-thread scalar and workgroup-reduces to one dendrite activity, OMITTING the
// per-seg binning + root->leaf tree-push (O(seg_cnt=128) per dendrite, negligible
// vs the O(syn/dendrite) sweep). That omission does not touch the dominant global
// read traffic the premise turns on. It is a perf proxy, not a semantic kernel.
//
// Runs the REAL coast::gpu::gpu_device facade (register_buffer / register_kernel /
// register_plan / build_plan, then begin/enqueue/submit/rendezvous in a tight
// loop) so the measured path is the production replay path. Run UNSANDBOXED — the
// sandbox blocks /dev/dri so Vulkan enumerates zero adapters.
//
// Output: adapter, per-step wall + device-wait time for SPARSE and DENSE, and the
// dense/sparse ratio (the premise number). Verdict text is printed; the go/no-go
// DECISION is the user's (design §8), recorded in benchmarks/perf_ngm25_gpu_premise.md.

#include "core/gpu/gpu_device.h"
#include "core/gpu/shared_prng.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace std::chrono;
namespace g = coast::gpu;

// Matched-config geometry (mirrors the ngm3 GPU probe: 64 neurons x 4 proximal
// dendrites = 256 dendrites, 784-wide mnist proximal input, branch depth 3).
constexpr std::uint32_t kInputSize = 784;
constexpr std::uint32_t kDendCnt   = 256;
constexpr std::uint32_t kRepCnt     = 8;   // 2^depth, depth=3 -> dense inflation
constexpr std::uint32_t kMaxSegId   = 15;  // 2^(depth+1)-1
constexpr float         kPermThres  = 0.3f;

// Realized-sparse synapse count. Init is 256*784=200,704 (all seg=1); after
// branching the realized arena grows. The design cites ~342k realized synapses on
// the matched config — use it as the steady-state sparse size.
constexpr std::uint32_t kSparseSyn = 342000;
// Dense ngm3-equivalent: rep-inflated, 256 * 8 * 784 = 1,605,632.
constexpr std::uint32_t kDenseSyn  = kDendCnt * kRepCnt * kInputSize;

// Host-side uniform, std140-compatible (8 scalars, 32 B, multiple of 16).
struct Params {
    std::uint32_t step;
    std::uint32_t seed;
    std::uint32_t dend_cnt;
    float         perm_thres;
    float         inv_perm_range;
    float         atten;
    float         inv_inp_sum;
    float         _pad;
};
static_assert(sizeof(Params) == 32, "Params must match the WGSL std140 layout");

// The response-sweep kernel. shared_prng_wgsl() (pcg_hash/pcg_uniform) is
// prepended at registration; syn_key is transcribed line-for-line from
// include/hd_ngm25/hd_ngm25_prng.h (the RNG-parity requirement, design §7).
const char* kResponseWgsl = R"WGSL(
struct Params {
    step: u32,
    seed: u32,
    dend_cnt: u32,
    perm_thres: f32,
    inv_perm_range: f32,
    atten: f32,
    inv_inp_sum: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read>       syn_perm:      array<f32>;
@group(0) @binding(1) var<storage, read>       syn_input_idx: array<u32>;
@group(0) @binding(2) var<storage, read>       syn_seg:       array<u32>;
@group(0) @binding(3) var<storage, read>       dend_offset:   array<u32>;
@group(0) @binding(4) var<storage, read>       input_concat:  array<f32>;
@group(0) @binding(5) var<storage, read>       half_max:      array<f32>;
@group(0) @binding(6) var<storage, read>       inv_sum:       array<f32>;
@group(0) @binding(7) var<storage, read_write> dend_act:      array<f32>;
@group(0) @binding(8) var<uniform>             rc:            Params;

const WG: u32 = 64u;
var<workgroup> partial: array<f32, WG>;

// ngm2.5 structural-identity key (hd_ngm25_prng.h:38-47). WGSL u32 multiply wraps
// mod 2^32, matching C++ unsigned overflow -> bit-identical draws to the CPU path.
fn syn_key(seed: u32, step: u32, dend: u32, seg: u32, input_idx: u32) -> u32 {
    return seed
         ^ (step      * 3266489917u)
         ^ (dend      * 2246822519u)
         ^ (seg       * 668265263u)
         ^ (input_idx * 374761393u);
}

@compute @workgroup_size(WG)
fn response_main(@builtin(workgroup_id) wg: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = wg.x;
    let t = lid.x;
    var acc = 0.0;
    if (d < rc.dend_cnt) {
        let begin = dend_offset[d];
        let end   = dend_offset[d + 1u];
        for (var s = begin + t; s < end; s = s + WG) {
            let perm = syn_perm[s];
            if (perm <= rc.perm_thres) { continue; }   // sub-threshold: perm read only
            let pos = syn_input_idx[s];
            let seg = syn_seg[s];
            let inp = input_concat[pos];
            var contrib = inp;
            let draw = pcg_uniform(syn_key(rc.seed, rc.step, d, seg, pos)) * half_max[pos];
            if (draw > inp) {
                let strength = (perm - rc.perm_thres) * rc.inv_perm_range;
                contrib = contrib - strength * (1.0 - inp * inv_sum[pos]);
                if (contrib < 0.0) { contrib = 0.0; }
            }
            acc = acc + contrib;
        }
    }
    partial[t] = acc;
    workgroupBarrier();
    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u && d < rc.dend_cnt) {
        dend_act[d] = clamp(partial[0] * rc.atten * rc.inv_inp_sum, 0.0, 1.0);
    }
}
)WGSL";

template <class T>
std::span<const std::byte> as_bytes_vec(const std::vector<T>& v) {
    return std::as_bytes(std::span<const T>(v));
}

// A self-contained arena + plan + buffers on the device, and the timed loop.
struct sweep {
    g::buffer_id perm, iidx, seg, doff, inp, hmax, isum, dact, params;
    g::kernel_id kernel;
    g::plan_id   plan;
    std::uint32_t dend_cnt;
    std::uint64_t total_syn;
};

// Build a synthetic realized arena of `total_syn` synapses over `dend_cnt`
// dendrites, input-major, and register everything on the device. `dense` toggles
// the seg-id distribution (leaves for the rep-inflated arena vs a realized mix).
sweep make_sweep(g::gpu_device& dev, std::uint32_t dend_cnt,
                 std::uint64_t total_syn, bool dense) {
    // Even-ish distribution across dendrites (remainder on the first few).
    std::vector<std::uint32_t> off(dend_cnt + 1, 0);
    const std::uint64_t base = total_syn / dend_cnt;
    const std::uint64_t rem  = total_syn % dend_cnt;
    for (std::uint32_t d = 0; d < dend_cnt; ++d)
        off[d + 1] = off[d] + static_cast<std::uint32_t>(base + (d < rem ? 1 : 0));

    std::vector<float>         perm(total_syn);
    std::vector<std::uint32_t> iidx(total_syn);
    std::vector<std::uint32_t> seg(total_syn);
    // Input-major fill: input_idx cycles 0..kInputSize-1; perm spread so ~half
    // exceed the 0.3 threshold (deterministic, no RNG needed for a perf proxy).
    for (std::uint64_t s = 0; s < total_syn; ++s) {
        const std::uint32_t pos = static_cast<std::uint32_t>(s % kInputSize);
        iidx[s] = pos;
        perm[s] = 0.05f + 0.5f * static_cast<float>((s * 2654435761u) & 0xffffu) / 65535.0f;
        if (dense) {
            // rep-inflated: leaves of a full depth-3 tree, seg in [8,15].
            seg[s] = 8u + static_cast<std::uint32_t>((s / kInputSize) % kRepCnt);
        } else {
            // realized mix across the tree [1,15], skewed toward shallow nodes.
            const std::uint32_t r = static_cast<std::uint32_t>((s * 40503u) & 0xfu);
            seg[s] = 1u + (r % kMaxSegId);
        }
    }

    std::vector<float> input(kInputSize), hmax(kInputSize), isum(kInputSize);
    float sum = 0.0f;
    for (std::uint32_t i = 0; i < kInputSize; ++i) {
        // mnist-like: many low, some high, all in [0,1].
        input[i] = ((i * 2246822519u) & 0x3u) == 0u
                 ? 0.4f + 0.6f * static_cast<float>((i * 668265263u) & 0xffu) / 255.0f
                 : 0.02f;
        sum += input[i];
    }
    for (std::uint32_t i = 0; i < kInputSize; ++i) {
        hmax[i] = 0.5f;                                  // half of a unit max
        isum[i] = sum > 0.0f ? 1.0f / sum : 0.0f;
    }

    sweep sw{};
    sw.dend_cnt  = dend_cnt;
    sw.total_syn = total_syn;
    auto reg = [&](std::size_t bytes) {
        return dev.register_buffer(g::topology::gpu_resident, bytes, /*ping_pong=*/false);
    };
    sw.perm = reg(perm.size() * sizeof(float));
    sw.iidx = reg(iidx.size() * sizeof(std::uint32_t));
    sw.seg  = reg(seg.size()  * sizeof(std::uint32_t));
    sw.doff = reg(off.size()  * sizeof(std::uint32_t));
    sw.inp  = reg(input.size() * sizeof(float));
    sw.hmax = reg(hmax.size() * sizeof(float));
    sw.isum = reg(isum.size() * sizeof(float));
    sw.dact = reg(std::size_t{dend_cnt} * sizeof(float));
    sw.params = dev.register_buffer(g::topology::uniform, sizeof(Params), false);

    dev.upload(sw.perm, as_bytes_vec(perm));
    dev.upload(sw.iidx, as_bytes_vec(iidx));
    dev.upload(sw.seg,  as_bytes_vec(seg));
    dev.upload(sw.doff, as_bytes_vec(off));
    dev.upload(sw.inp,  as_bytes_vec(input));
    dev.upload(sw.hmax, as_bytes_vec(hmax));
    dev.upload(sw.isum, as_bytes_vec(isum));

    Params p{};
    p.step = 0; p.seed = 12345u; p.dend_cnt = dend_cnt;
    p.perm_thres = kPermThres; p.inv_perm_range = 1.0f / (1.0f - kPermThres);
    p.atten = 0.8f; p.inv_inp_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    dev.upload(sw.params, std::as_bytes(std::span<const Params>(&p, 1)));

    const std::string src = std::string(g::shared_prng_wgsl()) + kResponseWgsl;
    sw.kernel = dev.register_kernel(src, "response_main");

    g::step_plan sp;
    g::plan_op op{};
    op.kind   = g::op_kind::compute;
    op.kernel = sw.kernel;
    g::plan_dispatch disp{};
    disp.bindings = {
        {sw.perm,   g::access::read,  0},
        {sw.iidx,   g::access::read,  1},
        {sw.seg,    g::access::read,  2},
        {sw.doff,   g::access::read,  3},
        {sw.inp,    g::access::read,  4},
        {sw.hmax,   g::access::read,  5},
        {sw.isum,   g::access::read,  6},
        {sw.dact,   g::access::write, 7},
        {sw.params, g::access::read,  8},
    };
    disp.dispatch = { dend_cnt, 1, 1 };
    op.dispatches = { std::move(disp) };
    sp.ops = { std::move(op) };

    sw.plan = dev.register_plan();
    dev.build_plan(sw.plan, sp);
    return sw;
}

struct timing { double step_us; double wait_us; };

timing run_loop(g::gpu_device& dev, const sweep& sw, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        dev.begin_step();
        dev.enqueue(sw.plan);
        dev.submit_step();
        dev.rendezvous_step();
    }
    double step_ns = 0.0, wait_ns = 0.0;
    for (int i = 0; i < iters; ++i) {
        const auto a = steady_clock::now();
        dev.begin_step();
        dev.enqueue(sw.plan);
        const auto b = steady_clock::now();
        dev.submit_step();
        dev.rendezvous_step();               // BARRIER 1 = device wait_idle
        const auto c = steady_clock::now();
        step_ns += duration_cast<nanoseconds>(c - a).count();
        wait_ns += duration_cast<nanoseconds>(c - b).count();
    }
    return { step_ns / iters / 1000.0, wait_ns / iters / 1000.0 };
}

} // namespace

int main(int argc, char** argv) {
    const int iters  = argc > 1 ? std::atoi(argv[1]) : 3000;
    const int warmup = 300;
    // Size sweep: fixed dendrite count, growing realized synapse count. Two anchor
    // points matter — kSparseSyn (realized-sparse, what ngm2.5-GPU runs) and
    // kDenseSyn (ngm3 rep-inflated model). The extra points let a fixed+linear fit
    // separate per-dispatch overhead from per-synapse (bandwidth) cost, so the
    // sub-linear ratio between the two anchors is explained, not guessed.
    const std::vector<std::uint32_t> sizes = {
        kInputSize * kDendCnt,    // 200,704 — init (all seg=1)
        kSparseSyn,               // 342,000 — realized-sparse anchor
        2 * kSparseSyn,           // 684,000
        kDenseSyn                 // 1,605,632 — dense ngm3 anchor
    };
    try {
        auto dev = g::make_gpu_device();
        std::printf("adapter: %s\n", dev->adapter_name().c_str());
        std::printf("iters=%d warmup=%d dendrites=%u\n\n", iters, warmup, kDendCnt);

        struct point { std::uint32_t syn; timing t; };
        std::vector<point> pts;
        for (std::uint32_t n : sizes) {
            sweep sw = make_sweep(*dev, kDendCnt, n, /*dense=*/false);
            if (auto ck = dev->check(); !ck) {
                std::fprintf(stderr, "device fault building %u-syn arena\n", n);
                return 1;
            }
            pts.push_back({ n, run_loop(*dev, sw, warmup, iters) });

            std::vector<float> out(kDendCnt);
            const bool rb = dev->readback(sw.dact, std::as_writable_bytes(std::span<float>(out)));
            float omax = 0.0f, omean = 0.0f;
            if (rb) { for (float x : out) { omax = std::max(omax, x); omean += x; } omean /= out.size(); }
            std::printf("  %8u syn (%5.0f/dend, %5.1f MB): %7.2f us/step wall, "
                        "%7.2f us device-wait  [sanity mean=%.3f max=%.3f]\n",
                        n, double(n) / kDendCnt, n * 12.0 / 1e6,
                        pts.back().t.step_us, pts.back().t.wait_us, omean, omax);
        }

        // Least-squares fixed + per-synapse fit of the device-wait times.
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        const double m = static_cast<double>(pts.size());
        for (auto& p : pts) {
            const double x = p.syn, y = p.t.wait_us;
            sx += x; sy += y; sxx += x * x; sxy += x * y;
        }
        const double slope = (m * sxy - sx * sy) / (m * sxx - sx * sx); // us/syn
        const double fixed = (sy - slope * sx) / m;                     // us

        auto wait_at = [&](std::uint32_t n) {
            for (auto& p : pts) if (p.syn == n) return p.t.wait_us;
            return 0.0;
        };
        const double sparse_us = wait_at(kSparseSyn);
        const double dense_us  = wait_at(kDenseSyn);

        std::printf("\n=== GI0 premise result (adapter above) ===\n");
        std::printf("  fit: device-wait ~= %.1f us fixed  +  %.3f ns/synapse\n",
                    fixed, slope * 1000.0);
        std::printf("       (~%.1f us per-dispatch floor; traffic cost linear in syn count)\n",
                    fixed);
        std::printf("  realized-sparse (%u syn): %.1f us device-wait\n", kSparseSyn, sparse_us);
        std::printf("  dense ngm3 model (%u syn): %.1f us device-wait\n", kDenseSyn, dense_us);
        std::printf("  sparse/dense: %.2fx faster on the sweep (traffic ratio %.2fx; "
                    "gap = the fixed floor)\n",
                    dense_us > 0 ? dense_us / sparse_us : 0.0,
                    double(kDenseSyn) / kSparseSyn);
        std::printf("\n  reference floors (design/ngm2_5_gpu.md): ngm3-GPU ~508 us/step "
                    "device (WAIT_IDLE, whole step), ngm2-CPU ~193 us/step (5195 SPS).\n");
        std::printf("  NOTE: this measures the RESPONSE sweep ONLY. adapt + the\n"
                    "  single-workgroup winner/inhibition passes are NOT here (GI6).\n"
                    "  go/no-go on these numbers is the user's call (design §8).\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ngm25-gpu-premise] FAILED: %s\n", e.what());
        return 1;
    }
}
