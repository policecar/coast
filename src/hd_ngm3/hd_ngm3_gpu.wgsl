// hd_ngm3_gpu compute kernels (S6). Three entry points, one module. Ported from
// the CPU read/output path in hd_ngm3_neuron_group.cpp::compute_step (stats +
// fused response/combine + local inhibition). Adapt (S7) and branch (S8) are NOT
// here. shared_prng.wgsl (pcg_hash / pcg_uniform) is concatenated ahead of this
// source at module load, and so are the noise-gate window constants
// (NOISE_RESP_* / NOISE_LEARN_*), generated from ngm2::response_noise_window and
// ngm2::learning_noise_window in hd_ngm3_gpu_neuron_group.cpp, and the stats
// record width STATS_W (k_stats_width there).
//
//   stats_main             — per-channel reduction of input_buf -> stats[chan][STATS_W]
//                            {sum, avg, true-max, noise, count}. One workgroup per
//                            channel. noise replicates ngm2::hoyer_noise.
//   response_combine_main  — one workgroup PER NEURON. Per (dendrite,rep) cell:
//                            transform-reduce over input_size synapses with a
//                            stateless pcg draw standing in for the CPU
//                            low_dis(nrng); rep_act = clamp(acc*atten*inv_sum).
//                            Then combine both banks' per-neuron max -> neuron_act,
//                            update the decaying mx_act.
//   (local inhibition is no longer a kernel here — it is injected at build time
//    by the swappable inhibition::gpu_inhibitor_t; see src/inhibition/.)
//
// The RNG is intentionally NOT bit-equivalent to the CPU mt19937 stream (MH3).

const WG:   u32 = 256u;   // stats / inhibition reduction width
const WG_N: u32 = 64u;    // response_combine threads per neuron-workgroup

// ngm2::sigmoid verbatim: inv_step = 1/(1 - steepness^0.1); start = -trans*inv_step.
fn ng_sigmoid(x: f32, steepness: f32, transition: f32) -> f32 {
    let inv_step = 1.0 / (1.0 - pow(steepness, 0.1));
    let start    = -transition * inv_step;
    return 1.0 / (1.0 + exp(-(x * inv_step + start)));
}

struct Params {
    step:                 u32,
    seed:                 u32,
    neuron_cnt:           u32,
    rep_cnt:              u32,
    a_d_cnt:              u32,
    p_d_cnt:              u32,
    a_input_size:         u32,
    p_input_size:         u32,
    perm_thres:           f32,
    inv_perm_range:       f32,
    _pad2:                f32,   // was local_inhib_strength; the injected inhibitor has its own uniform
    noise_lo:             f32,   // combine-noise low bound (configurable; default 0.01)
    noise_range:          f32,   // combine-noise span = hi - lo (default 0.04)
    frozen:               u32,   // unread by the kernels (a frozen plan omits the learning
                                 // passes); kept so the uniform's layout is unchanged
    _pad3: f32, _pad4: f32,
};

// ---------------------------------------------------------------------------
// S7 (adapt) uniforms. AdaptParams is the bank-agnostic scalar set; BankDims
// carries the per-bank {d_cnt, input_size} so one generic adapt_main entry
// point can be dispatched twice (apical, proximal) with two bind groups.
// ---------------------------------------------------------------------------

const SEG_MAX: u32 = 32u;   // covers branch_depth up to 4 (max_seg_id = 31)

struct AdaptParams {
    step:                 u32,
    seed:                 u32,
    neuron_cnt:           u32,
    rep_cnt:              u32,
    max_seg_id:           u32,
    perm_thres:           f32,
    inv_perm_range:       f32,
    primary_lr:           f32,
    secondary_lr:         f32,
    common_lr:            f32,
    stochastic_win_thres: f32,
    mismatch_act_thres:   f32,
    mismatch_smoothing:   f32,
    wf_steep:             f32,   // weight_filter shape
    wf_trans:             f32,
    // Formerly activity_learning_window; the window is tracked now (edges from
    // adapt_scalars[4..7], steepness WT_STEEP_* from the window-tracking prelude).
    _pad_w0:              f32,
    _pad_w1:              f32,
    _pad_w2:              f32,
    _pad_w3:              f32,
    // ---- S8 branch scalars (also fed via this bank-agnostic uniform) ----
    branch_depth:             u32,   // split_size = 1 << (branch_depth - bit_width(old_sid))
    accumulated_theta_thres:  f32,
    min_mismatch_deviation:   f32,
    min_mismatch_percentage:  f32,
    branch_interval:          u32,   // learning steps between branch checks; keeps 24 scalars / 96 B
};

struct BankDims {
    d_cnt:       u32,
    input_size:  u32,
    jitter_salt: u32,   // per-bank branch-jitter salt (0 for proximal)
    _p1:         u32,
};

// ===========================================================================
// stats_main — input_buf -> stats {sum, avg, true-max, noise, count}
// ===========================================================================

@group(0) @binding(0) var<storage, read>       s_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_stats: array<f32>;   // [chan_cnt][STATS_W]
@group(0) @binding(2) var<storage, read>       s_chof:  array<u32>;   // channel per input position

var<workgroup> s_sum: array<f32, WG>;
var<workgroup> s_max: array<f32, WG>;
var<workgroup> s_sq:  array<f32, WG>;
var<workgroup> s_cnt: array<u32, WG>;

@compute @workgroup_size(WG)
fn stats_main(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    // One workgroup per input channel: reduce the positions that belong to
    // channel `ch` (SPEC.md M.3.2: input statistics are per channel). With a
    // single channel this visits every position in the same order as before.
    let ch = wid.x;
    let t  = lid.x;
    let n  = arrayLength(&s_input);

    // one pass — grid-stride local sum + sum of squares + true-max + count (max
    // reduced from 0.0, matching ngm2::compute_input_stats), then tree-reduce.
    var sm = 0.0;
    var sq = 0.0;
    var mx = 0.0;
    var cn = 0u;
    for (var i = t; i < n; i = i + WG) {
        if (s_chof[i] != ch) { continue; }
        let v = s_input[i];
        sm = sm + v;
        sq = sq + v * v;
        mx = max(mx, v);
        cn = cn + 1u;
    }
    s_sum[t] = sm;
    s_sq[t]  = sq;
    s_max[t] = mx;
    s_cnt[t] = cn;
    workgroupBarrier();

    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            s_sum[t] = s_sum[t] + s_sum[t + stride];
            s_sq[t]  = s_sq[t]  + s_sq[t + stride];
            s_max[t] = max(s_max[t], s_max[t + stride]);
            s_cnt[t] = s_cnt[t] + s_cnt[t + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let total = s_sum[0];
        let count = s_cnt[0];
        // hoyer_noise: 0 unless the sum is normal, 0 for a single element,
        // else 1 - (sqrt(n) - L1/L2) / (sqrt(n) - 1), clamped.
        var noise = 0.0;
        if (is_normal_f32(total)) {
            if (count == 1u) {
                noise = 0.0;
            } else {
                let sqrt_n = sqrt(f32(count));
                let hoyer  = (sqrt_n - total / sqrt(s_sq[0])) / (sqrt_n - 1.0);
                noise = clamp(1.0 - hoyer, 0.0, 1.0);
            }
        }
        let o = ch * STATS_W;
        s_stats[o + 0u] = total;                                      // sum
        s_stats[o + 1u] = select(0.0, total / f32(count), count > 0u); // avg
        s_stats[o + 2u] = s_max[0];                                   // true-max
        s_stats[o + 3u] = noise;                                      // 1 - Hoyer
        s_stats[o + 4u] = f32(count);                                 // count
    }
}

// ===========================================================================
// response_combine_main — one workgroup per neuron
// ===========================================================================

@group(0) @binding(0)  var<storage, read>       a_perm:   array<f32>;
@group(0) @binding(1)  var<storage, read>       a_input:  array<f32>;
@group(0) @binding(2)  var<storage, read>       a_stats:  array<f32>;   // [chan_cnt][STATS_W]
@group(0) @binding(3)  var<storage, read_write> a_repact: array<f32>;
@group(0) @binding(4)  var<storage, read>       p_perm:   array<f32>;
@group(0) @binding(5)  var<storage, read>       p_input:  array<f32>;
@group(0) @binding(6)  var<storage, read>       p_stats:  array<f32>;   // [chan_cnt][STATS_W]
@group(0) @binding(7)  var<storage, read_write> p_repact: array<f32>;
@group(0) @binding(8)  var<storage, read_write> neuron_apical_max:   array<f32>;
@group(0) @binding(9)  var<storage, read_write> neuron_proximal_max: array<f32>;
@group(0) @binding(10) var<storage, read_write> neuron_act:          array<f32>;
@group(0) @binding(12) var<uniform>             rc: Params;
@group(0) @binding(13) var<storage, read>       a_seg:    array<u32>;
@group(0) @binding(14) var<storage, read>       p_seg:    array<u32>;
@group(0) @binding(15) var<storage, read>       a_chof:   array<u32>;
@group(0) @binding(16) var<storage, read>       p_chof:   array<u32>;

var<workgroup> wamax: array<f32, WG_N>;
var<workgroup> wpmax: array<f32, WG_N>;

// std::isnormal from the exponent bits: 0 is zero or subnormal, all ones is inf
// or NaN. Bit tests, because WGSL lets an implementation treat inf and NaN
// arithmetic and comparisons as indeterminate.
fn is_normal_f32(x: f32) -> bool {
    let e = bitcast<u32>(x) & 0x7f800000u;
    return e != 0u && e != 0x7f800000u;
}

// Bank attenuation, response window — ngm2::noise_attenuation with
// ngm2::response_noise_window, as compute_step's resp_atten.
fn bank_atten(noise: f32, avg: f32) -> f32 {
    return 1.0 - ng_sigmoid((noise - NOISE_RESP_START) / NOISE_RESP_WIDTH,
                            NOISE_RESP_STEEPNESS, NOISE_RESP_TRANSITION)
               * (1.0 - ng_sigmoid(avg, 0.5, 0.5));
}

// Apical bank response for neuron `nrn`, thread-strided over its cells. Writes
// a_repact to global; returns this thread's partial max (so the combine max
// needs no storage round-trip). Mirrors respond_proximal below.
fn respond_apical(nrn: u32, tid: u32) -> f32 {
    if (rc.a_d_cnt == 0u || rc.a_input_size == 0u) { return 0.0; }
    // Bank-wide: attenuation from the min noise and min average over the
    // channels with input and >= 2 entries, leaf normaliser 1 / total sum
    // (SPEC.md M.3.2, M.7.32).
    let chans = arrayLength(&a_stats) / STATS_W;
    var tot = 0.0;
    var noise_min = 3.4e38;
    var avg_min = 3.4e38;
    var gated = false;   // ngm2::noise_gate_min_t: only channels with input and >= 2 entries
    for (var c = 0u; c < chans; c = c + 1u) {
        let c_sum = a_stats[c * STATS_W];
        tot = tot + c_sum;
        if (is_normal_f32(c_sum) && a_stats[c * STATS_W + 4u] >= 2.0) {
            avg_min = min(avg_min, a_stats[c * STATS_W + 1u]);
            noise_min = min(noise_min, a_stats[c * STATS_W + 3u]);
            gated = true;
        }
    }
    let inv_total = select(0.0, 1.0 / tot, tot > 0.0);
    let atten     = select(1.0, bank_atten(noise_min, avg_min), gated);
    let cells_per_n = rc.a_d_cnt * rc.rep_cnt;
    var part_max = 0.0;
    for (var local = tid; local < cells_per_n; local = local + WG_N) {
        let cell = nrn * cells_per_n + local;
        let base = cell * rc.a_input_size;
        // Per-segment sums with a per-segment floor, then leaf = path sum from
        // the root down (ngm2's order; SPEC.md M.7.31). Matches the CPU group.
        var seg_acc: array<f32, SEG_MAX>;
        var deepest = 1u;
        for (var i = 0u; i < rc.a_input_size; i = i + 1u) {
            let p_val = a_perm[base + i];
            if (p_val <= rc.perm_thres) { continue; }   // connected iff p > tau
            let sid = a_seg[base + i];
            deepest = max(deepest, sid);
            let inp_val = a_input[i];
            // This synapse's channel: its own 1/sum and draw range.
            let co = a_chof[i] * STATS_W;
            let inv_sum     = select(0.0, 1.0 / a_stats[co], a_stats[co] != 0.0);
            let low_dis_max = a_stats[co + 2u] / 2.0;
            var acc = seg_acc[sid] + inp_val;
            // Keyed on the LOGICAL synapse (dendrite, segment, input), not the
            // replica row: every copy of a synapse gets the same draw, i.e. one
            // draw per synapse per step (SPEC.md M.0.6 b).
            let syn = ((cell / rc.rep_cnt) * 32u + sid) * rc.a_input_size + i;
            let key = (syn * 2654435761u) ^ (rc.step * 747796405u) ^ rc.seed ^ 0x9e3779b9u;
            if (pcg_uniform(key) * low_dis_max > inp_val) {
                acc = acc - (p_val - rc.perm_thres) * rc.inv_perm_range * (1.0 - inp_val * inv_sum);
                if (acc < 0.0) { acc = 0.0; }
            }
            seg_acc[sid] = acc;
        }
        var leaf = 0.0;
        var shift = i32(firstLeadingBit(deepest));
        loop {
            if (shift < 0) { break; }
            leaf = seg_acc[deepest >> u32(shift)] + leaf;
            shift = shift - 1;
        }
        let ra = clamp(leaf * atten * inv_total, 0.0, 1.0);
        a_repact[cell] = ra;
        part_max = max(part_max, ra);
    }
    return part_max;
}

fn respond_proximal(nrn: u32, tid: u32) -> f32 {
    if (rc.p_d_cnt == 0u || rc.p_input_size == 0u) { return 0.0; }
    // Bank-wide: attenuation from the min noise and min average over the
    // channels with input and >= 2 entries, leaf normaliser 1 / total sum
    // (SPEC.md M.3.2, M.7.32).
    let chans = arrayLength(&p_stats) / STATS_W;
    var tot = 0.0;
    var noise_min = 3.4e38;
    var avg_min = 3.4e38;
    var gated = false;   // ngm2::noise_gate_min_t: only channels with input and >= 2 entries
    for (var c = 0u; c < chans; c = c + 1u) {
        let c_sum = p_stats[c * STATS_W];
        tot = tot + c_sum;
        if (is_normal_f32(c_sum) && p_stats[c * STATS_W + 4u] >= 2.0) {
            avg_min = min(avg_min, p_stats[c * STATS_W + 1u]);
            noise_min = min(noise_min, p_stats[c * STATS_W + 3u]);
            gated = true;
        }
    }
    let inv_total = select(0.0, 1.0 / tot, tot > 0.0);
    let atten     = select(1.0, bank_atten(noise_min, avg_min), gated);
    let cells_per_n = rc.p_d_cnt * rc.rep_cnt;
    var part_max = 0.0;
    for (var local = tid; local < cells_per_n; local = local + WG_N) {
        let cell = nrn * cells_per_n + local;
        let base = cell * rc.p_input_size;
        // Per-segment sums, per-segment floor (see respond_apical).
        var seg_acc: array<f32, SEG_MAX>;
        var deepest = 1u;
        for (var i = 0u; i < rc.p_input_size; i = i + 1u) {
            let p_val = p_perm[base + i];
            if (p_val <= rc.perm_thres) { continue; }   // connected iff p > tau
            let sid = p_seg[base + i];
            deepest = max(deepest, sid);
            let inp_val = p_input[i];
            // This synapse's channel: its own 1/sum and draw range.
            let co = p_chof[i] * STATS_W;
            let inv_sum     = select(0.0, 1.0 / p_stats[co], p_stats[co] != 0.0);
            let low_dis_max = p_stats[co + 2u] / 2.0;
            var acc = seg_acc[sid] + inp_val;
            // Keyed on the LOGICAL synapse (dendrite, segment, input), not the
            // replica row: every copy of a synapse gets the same draw, i.e. one
            // draw per synapse per step (SPEC.md M.0.6 b).
            let syn = ((cell / rc.rep_cnt) * 32u + sid) * rc.p_input_size + i;
            let key = (syn * 2654435761u) ^ (rc.step * 747796405u) ^ rc.seed ^ 0x85ebca6bu;
            if (pcg_uniform(key) * low_dis_max > inp_val) {
                acc = acc - (p_val - rc.perm_thres) * rc.inv_perm_range * (1.0 - inp_val * inv_sum);
                if (acc < 0.0) { acc = 0.0; }
            }
            seg_acc[sid] = acc;
        }
        var leaf = 0.0;
        var shift = i32(firstLeadingBit(deepest));
        loop {
            if (shift < 0) { break; }
            leaf = seg_acc[deepest >> u32(shift)] + leaf;
            shift = shift - 1;
        }
        let ra = clamp(leaf * atten * inv_total, 0.0, 1.0);
        p_repact[cell] = ra;
        part_max = max(part_max, ra);
    }
    return part_max;
}

@compute @workgroup_size(WG_N)
fn response_combine_main(@builtin(workgroup_id) wid: vec3<u32>,
                         @builtin(local_invocation_id) lid: vec3<u32>) {
    let nrn = wid.x;
    let t   = lid.x;
    if (nrn >= rc.neuron_cnt) { return; }

    let a_part = respond_apical(nrn, t);
    let p_part = respond_proximal(nrn, t);

    wamax[t] = a_part;
    wpmax[t] = p_part;
    workgroupBarrier();

    var stride = WG_N / 2u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            wamax[t] = max(wamax[t], wamax[t + stride]);
            wpmax[t] = max(wpmax[t], wpmax[t + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (t == 0u) {
        // apical_max defaults to 1.0 when the apical bank is absent (CPU parity).
        let amax = select(1.0, wamax[0], rc.a_d_cnt > 0u);
        let pmax = wpmax[0];
        neuron_apical_max[nrn]   = amax;
        neuron_proximal_max[nrn] = pmax;

        // Combine noise: uniform[noise_lo, noise_lo+noise_range] — configurable
        // (default 1%-5%, matching ngm2/ngm3 CPU), NOT the dataset noise_range.
        // Stateless draw keyed on (neuron, step).
        let nz = rc.noise_lo + rc.noise_range * pcg_uniform((nrn * 2246822519u) ^ (rc.step * 3266489917u) ^ rc.seed);
        let act = clamp(amax * pmax + nz, 0.0, 1.0);
        neuron_act[nrn] = act;
        // (The decaying running max is learning state; winner_main updates it.)
    }
}

// ===========================================================================
// winner_main — single workgroup. Reads the post-inhibition group output
// (group_outp_, published by record() after inhibition) and produces the
// group-level scalars the adapt pass needs: {winner_idx, act_sum, win_act} and
// the activity learning window edges for this step (adapt_scalars[4..7]), and
// advances the window edges (ngm/window_tracking.h) on learning steps.
//
// CPU reference (compute_step lines 469-486):
//   mx_act  = max(group_outp)
//   win_act = uniform(mx_act * stochastic_win_thres, mx_act)   <- group rgen
//   act_sum = sum(group_outp)
//   winner  = uniform among the n with group_outp[n] + eps >= win_act (M.4.4),
//             else no_winner.
// Both draws (win_act, and which qualifying neuron wins) are stateless PCG
// stand-ins (MH3): keyed on (step, seed), deterministic per (seed, step) but NOT
// the CPU mt19937 stream.
// ===========================================================================

@group(0) @binding(0) var<storage, read>       w_outp: array<f32>;   // group_outp_
@group(0) @binding(1) var<storage, read_write> w_scal: array<u32>;   // adapt_scalars[8]
@group(0) @binding(2) var<uniform>             wp: AdaptParams;
@group(0) @binding(3) var<storage, read>       w_act:   array<f32>;  // neuron_act (pre-inhibition)
@group(0) @binding(4) var<storage, read_write> w_mxact: array<f32>;  // neuron_mx_act
// Learning-step counter, device-resident so no step needs a host readback:
// [count lo, count hi, count % branch_interval, branch due this step].
@group(0) @binding(5) var<storage, read_write> w_cnt:   array<u32>;
// Activity window edges [winner lower, winner upper, other lower, other upper]:
// learning state, the edges the NEXT learning step adapts with.
@group(0) @binding(6) var<storage, read_write> w_track: array<f32>;

var<workgroup> w_max: array<f32, WG>;
var<workgroup> w_sum: array<f32, WG>;
var<workgroup> w_mx_b:     f32;
var<workgroup> w_sum_b:    f32;
var<workgroup> w_winact_b: f32;

@compute @workgroup_size(WG)
fn winner_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let n = wp.neuron_cnt;

    // phase A — max + sum over group_outp.
    var mx = 0.0;
    var sm = 0.0;
    for (var i = t; i < n; i = i + WG) {
        let v = w_outp[i];
        mx = max(mx, v);
        sm = sm + v;
    }
    w_max[t] = mx;
    w_sum[t] = sm;
    workgroupBarrier();

    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            w_max[t] = max(w_max[t], w_max[t + stride]);
            w_sum[t] = w_sum[t] + w_sum[t + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        w_mx_b  = w_max[0];
        w_sum_b = w_sum[0];
        // win_act = uniform(mx*thres, mx), stateless draw keyed on (step, seed).
        let lo  = w_mx_b * wp.stochastic_win_thres;
        let key = (wp.step * 2654435761u) ^ wp.seed ^ 0xb5297a4du;
        w_winact_b = lo + (w_mx_b - lo) * pcg_uniform(key);
    }
    workgroupBarrier();
    let win_act = w_winact_b;

    // phase B — the winner: uniform among the k neurons with group_outp[i] + eps
    // >= win_act (SPEC.md M.4.4), the floor(u·k)-th in index order, u clamped as
    // in the CPU's ngm2::pick_winner (a [0,1) draw can round to 1); sentinel n if
    // none. Serial on one thread: the selection is order-sensitive and neuron_cnt
    // is small. Its own stateless draw, keyed like win_act with another salt.
    if (t == 0u) {
        let pick = pcg_uniform((wp.step * 2654435761u) ^ wp.seed ^ 0x3c6ef372u);
        var k = 0u;
        for (var i = 0u; i < n; i = i + 1u) {
            if (w_outp[i] + 1.1920929e-7 >= win_act) { k = k + 1u; }
        }
        var best = n;
        if (k > 0u) {
            var j = min(u32(pick * f32(k)), k - 1u);
            for (var i = 0u; i < n; i = i + 1u) {
                if (w_outp[i] + 1.1920929e-7 >= win_act) {
                    if (j == 0u) { best = i; break; }
                    j = j - 1u;
                }
            }
        }
        w_scal[0] = best;                      // winner_idx (== n if none)
        w_scal[1] = bitcast<u32>(w_sum_b);     // act_sum
        w_scal[2] = bitcast<u32>(win_act);     // win_act
        w_scal[3] = bitcast<u32>(w_mx_b);      // mx_act (diagnostic)
        // This step's window edges for adapt_main: the ones the previous learning
        // step left.
        w_scal[4] = bitcast<u32>(w_track[0]);
        w_scal[5] = bitcast<u32>(w_track[1]);
        w_scal[6] = bitcast<u32>(w_track[2]);
        w_scal[7] = bitcast<u32>(w_track[3]);

        // Advance the edges with this step's raw activities (learning steps only;
        // mirrors ngm::window_tracking::update). Serial: neuron_cnt is small.
        if (w_sum_b != 0.0) {
            if (best < n) {
                let a = w_act[best];
                w_track[0] = wt_step(w_track[0], WT_Q_LOWER, select(0.0, 1.0, a < w_track[0]));
                w_track[1] = wt_step(w_track[1], WT_Q_UPPER, select(0.0, 1.0, a < w_track[1]));
            }
            var below_lo = 0u;
            var below_hi = 0u;
            var cnt      = 0u;
            for (var i = 0u; i < n; i = i + 1u) {
                if (i == best) { continue; }
                let a = w_act[i];
                if (a < w_track[2]) { below_lo = below_lo + 1u; }
                if (a < w_track[3]) { below_hi = below_hi + 1u; }
                cnt = cnt + 1u;
            }
            if (cnt > 0u) {
                let fc = f32(cnt);
                w_track[2] = wt_step(w_track[2], WT_Q_LOWER, f32(below_lo) / fc);
                w_track[3] = wt_step(w_track[3], WT_Q_UPPER, f32(below_hi) / fc);
            }
        }
    }

    // A zero group output is not a learning step (SPEC.md M.4.15): adapt_main and
    // branch_main skip (act_sum == 0, due == 0), and neither the running max nor
    // the learning-step counter moves. On a learning step the running max decays
    // and folds in this step's activity before adapt_main reads it, and the
    // counter advances; a branch is due when it reaches a multiple of
    // branch_interval — the CPU group's (++input_cnt) % branch_interval == 0.
    let learned = w_sum_b != 0.0;
    if (learned) {
        for (var i = t; i < n; i = i + WG) {
            w_mxact[i] = max(w_mxact[i] * 0.99999, w_act[i]);
        }
    }
    if (t == 0u) {
        var due = 0u;
        if (learned) {
            let lo = w_cnt[0] + 1u;
            w_cnt[0] = lo;
            if (lo == 0u) { w_cnt[1] = w_cnt[1] + 1u; }
            var phase = w_cnt[2] + 1u;
            if (phase >= wp.branch_interval) { phase = 0u; }
            w_cnt[2] = phase;
            due = select(0u, 1u, phase == 0u);
        }
        w_cnt[3] = due;
    }
}

// ===========================================================================
// adapt_main — one workgroup PER NEURON. Dendrites are processed serially; for
// each dendrite thread 0 builds the segment-tree fold into workgroup-shared
// memory (the leaf "first to reach max_activity" rule and the max-fold are
// order-dependent, so a single thread does them), then ALL WG_N threads sweep
// that dendrite's (rep,syn) cells in parallel. This replaces the original
// thread-per-dendrite map, which left WG_N - d_cnt threads idle (only ~4 of 64
// active at proximal_cnt=4) with each busy thread grinding rep_cnt*input_size
// cells serially. The sweep is order-independent: adapt_main has no PRNG draws
// and each cell (perm_base+syn) is written by exactly one thread, so the result
// is bit-identical to the serial version — only occupancy changes.
// Generic over bank: the bound buffers + BankDims select apical vs proximal,
// dispatched twice from record(). Port of adapt_bank_neuron (CPU lines 544-657)
// plus the per-neuron adapt_weight / synapse_weight (lines 491-516).
// ===========================================================================

// std::isnormal for the non-negative maxima here: true iff finite, normal,
// non-zero. Clamped values are <= 1 so no inf; NaN fails the comparison.
fn is_normal_pos(x: f32) -> bool { return x >= 1.1754944e-38; }

// adapt-loop attenuation — learning window (ngm2::learning_noise_window, as the
// CPU adapt_atten), distinct from the response loop's bank_atten.
fn adapt_atten(noise: f32, avg: f32) -> f32 {
    return 1.0 - ng_sigmoid((noise - NOISE_LEARN_START) / NOISE_LEARN_WIDTH,
                            NOISE_LEARN_STEEPNESS, NOISE_LEARN_TRANSITION)
               * (1.0 - ng_sigmoid(avg, 0.5, 0.5));
}

@group(0) @binding(0)  var<storage, read_write> ad_perm:    array<f32>;
@group(0) @binding(1)  var<storage, read>       ad_seg:     array<u32>;
@group(0) @binding(2)  var<storage, read_write> ad_hist:    array<f32>;
@group(0) @binding(3)  var<storage, read_write> ad_mm:      array<f32>;
@group(0) @binding(4)  var<storage, read>       ad_repact:  array<f32>;
@group(0) @binding(5)  var<storage, read>       ad_input:   array<f32>;
@group(0) @binding(6)  var<storage, read>       ad_bankmax: array<f32>;   // neuron_*_max
@group(0) @binding(7)  var<storage, read>       ad_stats:   array<f32>;   // [chan_cnt][STATS_W]
@group(0) @binding(8)  var<storage, read>       ad_act:     array<f32>;   // neuron_act
@group(0) @binding(9)  var<storage, read>       ad_mxact:   array<f32>;   // neuron_mx_act
@group(0) @binding(10) var<storage, read>       ad_outp:    array<f32>;   // group_outp_
@group(0) @binding(11) var<storage, read>       ad_scal:    array<u32>;   // [8]
@group(0) @binding(12) var<uniform>             ap: AdaptParams;
@group(0) @binding(13) var<uniform>             bd: BankDims;
@group(0) @binding(14) var<storage, read>       ad_chof:    array<u32>;   // channel per input position

// Per-dendrite segment-tree fold, shared across the workgroup (was private per
// thread — ~16 KB aggregate private pressure at WG_N=64; now 256 B total).
// Rewritten by thread 0 for each dendrite, read by all threads in the sweep.
var<workgroup> as_segw: array<f32, SEG_MAX>;
var<workgroup> as_sega: array<f32, SEG_MAX>;

@compute @workgroup_size(WG_N)
fn adapt_main(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let nrn = wid.x;
    let t   = lid.x;
    if (nrn >= ap.neuron_cnt) { return; }
    if (bitcast<f32>(ad_scal[1]) == 0.0) { return; }   // zero output: not a learning step (M.4.15)

    let d_cnt      = bd.d_cnt;
    let input_size = bd.input_size;
    if (d_cnt == 0u || input_size == 0u) { return; }

    // adapt_bank_neuron early-out: degenerate bank max -> no adaptation.
    let max_activity = ad_bankmax[nrn];
    if (!is_normal_pos(max_activity)) { return; }
    let inv_max_activity = 1.0 / max_activity;

    // ---- per-neuron synapse_weight (every thread computes it; trivial cost) ----
    let outp_n  = ad_outp[nrn];
    let winner  = ad_scal[0];
    let act_sum = bitcast<f32>(ad_scal[1]);
    var adapt_weight: f32;
    if (nrn == winner) {
        adapt_weight = ng_sigmoid(1.0 - outp_n, ap.wf_steep, ap.wf_trans);
    } else {
        let ratio = select(0.0, outp_n / act_sum, act_sum > 0.0);
        adapt_weight = ng_sigmoid(1.0 - ratio, ap.wf_steep, ap.wf_trans) * ap.common_lr;
    }
    let mxact_n = ad_mxact[nrn];
    let act_n   = ad_act[nrn];
    // Tracked activity learning window: the winner's or the others' edges.
    let we      = select(6u, 4u, nrn == winner);
    let w_lower = bitcast<f32>(ad_scal[we]);
    let w_upper = bitcast<f32>(ad_scal[we + 1u]);
    let env = min(
        max(1.0 - ng_sigmoid(mxact_n, 0.5, 0.5),
            ng_sigmoid(act_n, WT_STEEP_LOWER, w_lower)),
        1.0 - ng_sigmoid(act_n, WT_STEEP_UPPER, w_upper));
    let synapse_weight = adapt_weight * env;

    // ---- mismatch normaliser: the max over all channels. Attenuation and the
    //      high/low threshold are per channel, looked up per synapse below
    //      (SPEC.md M.3.2). ----
    var s_max = 0.0;
    for (var c = 0u; c < arrayLength(&ad_stats) / STATS_W; c = c + 1u) { s_max = max(s_max, ad_stats[c * STATS_W + 2u]); }
    let inv_max_inp    = select(0.0, 1.0 / s_max, s_max > 0.0);

    let leaf_begin     = (ap.max_seg_id + 1u) / 2u;   // == rep_cnt
    let rep_cnt        = ap.rep_cnt;
    let dendrite_block = rep_cnt * input_size;

    for (var d = 0u; d < d_cnt; d = d + 1u) {
        // ---- fold: thread 0 only (sequential leaf rule + tree max-fold) ----
        if (t == 0u) {
            let act_base = nrn * d_cnt * rep_cnt + d * rep_cnt;

            for (var k = 0u; k < SEG_MAX; k = k + 1u) { as_segw[k] = 0.0; as_sega[k] = 0.0; }

            // leaves: first leaf reaching max_activity gets primary rate, rest secondary.
            var max_response_seen = false;
            for (var si = leaf_begin; si <= ap.max_seg_id; si = si + 1u) {
                let a = ad_repact[act_base + (si - leaf_begin)];
                as_sega[si] = a;
                if (!max_response_seen && a + 1.1920929e-7 >= max_activity) {
                    max_response_seen = true;
                    as_segw[si] = synapse_weight * ap.primary_lr;
                } else {
                    as_segw[si] = a * synapse_weight * ap.secondary_lr * inv_max_activity;
                }
            }
            // max-fold pairs up the tree.
            var level_start = leaf_begin;
            loop {
                if (level_start <= 1u) { break; }
                var si = level_start;
                loop {
                    if (si >= level_start * 2u) { break; }
                    let parent = si / 2u;
                    as_segw[parent] = max(as_segw[si], as_segw[si + 1u]);
                    as_sega[parent] = max(as_sega[si], as_sega[si + 1u]);
                    si = si + 2u;
                }
                level_start = level_start / 2u;
            }
        }
        workgroupBarrier();

        // ---- sweep all (rep, syn) cells, grid-strided across the workgroup.
        //      syn in [0, dendrite_block); cur_inp = input_buf[syn % input_size]
        //      (invariant over rep). Each cell written by exactly one thread. ----
        let perm_base = nrn * d_cnt * dendrite_block + d * dendrite_block;
        for (var syn = t; syn < dendrite_block; syn = syn + WG_N) {
            let cur_inp = ad_input[syn % input_size];
            let sid     = ad_seg[perm_base + syn];
            let co      = ad_chof[syn % input_size] * STATS_W;
            let s_avg   = ad_stats[co + 1u];
            let attenuation    = adapt_atten(ad_stats[co + 3u], s_avg);
            let high_thres     = s_avg / 2.0 + 1.1920929e-7;
            // guarded like the CPU context (identical for inputs in [0, 1])
            let inv_high_range = select(0.0, 1.0 / (1.0 - high_thres), high_thres < 1.0);
            let inv_high_thres = select(0.0, 1.0 / high_thres, high_thres != 0.0);

            let xform = select((high_thres - cur_inp) * inv_high_thres,
                               (cur_inp - high_thres) * inv_high_range,
                               cur_inp > high_thres);
            let theta = clamp(as_segw[sid] * xform * attenuation, 0.0, 1.0);

            let pold = ad_perm[perm_base + syn];
            let add  = select(0.0, theta, cur_inp > high_thres);
            let pnew = clamp(pold * (1.0 - theta) + add, 0.0, 1.0);
            ad_perm[perm_base + syn] = pnew;

            ad_hist[perm_base + syn] = ad_hist[perm_base + syn] + theta;

            let act_ratio = as_sega[sid] * inv_max_activity;
            if (act_ratio >= ap.mismatch_act_thres) {
                let inp_ratio = cur_inp * inv_max_inp;
                var mm = select(inp_ratio, 1.0 - inp_ratio, pnew > ap.perm_thres);
                mm = mm * act_ratio;
                ad_mm[perm_base + syn] =
                    ad_mm[perm_base + syn] * (1.0 - ap.mismatch_smoothing)
                    + mm * ap.mismatch_smoothing;
            }
        }
        workgroupBarrier();   // before next d's thread-0 fold overwrites as_seg*
    }
}

// ===========================================================================
// branch_main — M1.7 branch_bank: split mature, high-mismatch synapses into two
//   deeper segments. One workgroup per neuron, thread-per-dendrite grid-stride
//   (same map as adapt_main). Recorded only on branch-interval steps (host-gated
//   in record()), as the last pass of the step's encoder so it sees this step's
//   adapt writes to adapt_history / mismatch. Generic over bank; dispatched
//   ×{proximal, apical?}. Verbatim port of neuron_group_t::branch_bank.
//
//   segment_ids are READ-WRITE here (read-only in adapt) -> distinct bind group.
//   Permanence jitter (CPU uniform(-0.1,0.1) from rgen) -> stateless PCG draw
//   keyed on (synapse coord, draw slot) ^ step ^ seed ^ salt (MH3 divergence,
//   consistent with respond/adapt). No workgroup-shared memory, scalar only.
// ===========================================================================

@group(0) @binding(0) var<storage, read_write> br_perm: array<f32>;
@group(0) @binding(1) var<storage, read_write> br_seg:  array<u32>;
@group(0) @binding(2) var<storage, read_write> br_hist: array<f32>;
@group(0) @binding(3) var<storage, read_write> br_mm:   array<f32>;
@group(0) @binding(4) var<uniform>             bp: AdaptParams;
@group(0) @binding(5) var<uniform>             bb: BankDims;
@group(0) @binding(6) var<storage, read>       br_cnt: array<u32>;   // winner_main's learning-step counter

// bit_width(x) for x >= 1: position of the highest set bit, +1. Equivalent to
// std::bit_width on the always-positive seg_ids reaching split (root=1, splits
// yield >= 2). firstLeadingBit returns the 0-indexed top bit (0xFFFFFFFF for 0).
fn bit_width_pos(x: u32) -> u32 { return firstLeadingBit(x) + 1u; }

// Permanence jitter: uniform[-0.1, 0.1), keyed per (synapse coord, draw slot).
fn perm_jitter(coord: u32, slot: u32) -> f32 {
    // bb.jitter_salt separates the banks: coord is a bank-local flat index, so
    // without it apical and proximal cells at the same index drew identically.
    let key = (coord * 2654435761u) ^ (bp.step * 747796405u) ^ bp.seed
            ^ (slot * 0x9e3779b9u) ^ 0xa1b2c3d4u ^ bb.jitter_salt;
    return pcg_uniform(key) * 0.2 - 0.1;
}

// Split predicate (CPU `qualifies` lambda): mature (adapt_history >= theta thres),
// high-mismatch (>= mm_thres), and not already a leaf (child id fits the tree).
fn qualifies(base: u32, idx: u32, mm_thres: f32, max_seg_id: u32) -> bool {
    return br_hist[base + idx] >= bp.accumulated_theta_thres
        && br_mm  [base + idx] >= mm_thres
        && (br_seg[base + idx] * 2u + 1u) <= max_seg_id;
}

@compute @workgroup_size(WG_N)
fn branch_main(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    // C1 GPU-side gate (numerically inert on non-branch steps). winner_main sets
    // br_cnt[3] when this step's learning-step count reached a multiple of
    // branch_interval — the CPU group's (++input_cnt) % branch_interval == 0 —
    // and never on a step that did not learn (M.4.15). The count lives on the
    // device: only the device knows, mid-step, whether the output summed to 0.
    // Early-out precedes every buffer write below.
    if (br_cnt[3] == 0u) { return; }

    let nrn = wid.x;
    let t   = lid.x;
    if (nrn >= bp.neuron_cnt) { return; }

    let d_cnt      = bb.d_cnt;
    let input_size = bb.input_size;
    if (d_cnt == 0u || input_size == 0u) { return; }

    let rep_cnt        = bp.rep_cnt;
    let dendrite_block = rep_cnt * input_size;
    let max_seg_id     = bp.max_seg_id;

    for (var d = t; d < d_cnt; d = d + WG_N) {
        let base = nrn * d_cnt * dendrite_block + d * dendrite_block;

        // ---- pass 1: gather mm_avg with skip-if-same-seg-id (rep dedup) ----
        var mm_avg           = 0.0;
        var real_synapse_cnt = 0.0;
        for (var inp = 0u; inp < input_size; inp = inp + 1u) {
            mm_avg = mm_avg + br_mm[base + inp];
            real_synapse_cnt = real_synapse_cnt + 1.0;
            for (var r = 1u; r < rep_cnt; r = r + 1u) {
                let prev = inp + (r - 1u) * input_size;
                let cur  = prev + input_size;
                if (br_seg[base + prev] == br_seg[base + cur]) { continue; }
                mm_avg = mm_avg + br_mm[base + cur];
                real_synapse_cnt = real_synapse_cnt + 1.0;
            }
        }
        if (real_synapse_cnt <= 0.0) { continue; }
        mm_avg = mm_avg / real_synapse_cnt;

        // ---- pass 2: gather mm_std (variance) with same dedup walk ----
        var mm_std = 0.0;
        for (var inp = 0u; inp < input_size; inp = inp + 1u) {
            let d0 = mm_avg - br_mm[base + inp];
            mm_std = mm_std + d0 * d0;
            for (var r = 1u; r < rep_cnt; r = r + 1u) {
                let prev = inp + (r - 1u) * input_size;
                let cur  = prev + input_size;
                if (br_seg[base + prev] == br_seg[base + cur]) { continue; }
                let dv = mm_avg - br_mm[base + cur];
                mm_std = mm_std + dv * dv;
            }
        }
        mm_std = mm_std / real_synapse_cnt;

        let mm_thres = mm_avg
                     + mm_std * bp.min_mismatch_deviation
                     + 1.0 / f32(input_size);

        // ---- pass 3: count qualifying synapses (predicate triplet) ----
        var mm_cnt = 0.0;
        for (var inp = 0u; inp < input_size; inp = inp + 1u) {
            if (qualifies(base, inp, mm_thres, max_seg_id)) { mm_cnt = mm_cnt + 1.0; }
            for (var r = 1u; r < rep_cnt; r = r + 1u) {
                let prev = inp + (r - 1u) * input_size;
                let cur  = prev + input_size;
                if (br_seg[base + prev] == br_seg[base + cur]) { continue; }
                if (qualifies(base, cur, mm_thres, max_seg_id)) { mm_cnt = mm_cnt + 1.0; }
            }
        }
        if (mm_cnt < f32(input_size) * bp.min_mismatch_percentage) { continue; }

        // ---- pass 4: split-write ----
        for (var inp = 0u; inp < input_size; inp = inp + 1u) {
            var r = 0u;
            loop {
                if (r >= rep_cnt) { break; }
                let idx = inp + r * input_size;
                if (qualifies(base, idx, mm_thres, max_seg_id)) {
                    let old_sid    = br_seg[base + idx];
                    let split_size = 1u << (bp.branch_depth - bit_width_pos(old_sid));
                    let new_sid_a  = old_sid * 2u;
                    let new_sid_b  = new_sid_a + 1u;
                    let old_perm   = br_perm[base + idx];
                    let new_perm_a = clamp(old_perm + perm_jitter(base + idx, 0u), 0.0, 1.0);
                    let new_perm_b = clamp(old_perm + perm_jitter(base + idx, 1u), 0.0, 1.0);

                    for (var s = r; s < r + split_size; s = s + 1u) {
                        let a_idx = inp + s                  * input_size;
                        let b_idx = inp + (s + split_size)   * input_size;
                        br_seg [base + a_idx] = new_sid_a;
                        br_seg [base + b_idx] = new_sid_b;
                        br_hist[base + a_idx] = 0.0;
                        br_hist[base + b_idx] = 0.0;
                        br_mm  [base + a_idx] = 0.0;
                        br_mm  [base + b_idx] = 0.0;
                        br_perm[base + a_idx] = new_perm_a;
                        br_perm[base + b_idx] = new_perm_b;
                    }
                    r = r + split_size * 2u;
                } else {
                    r = r + 1u;
                }
            }
        }
    }
}
