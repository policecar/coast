// ngm2.5 GPU kernels (design/ngm2_5_gpu.md). shared_prng.wgsl (pcg_hash /
// pcg_uniform) is concatenated ahead of this at registration; syn_key below is a
// LINE-FOR-LINE transcription of include/hd_ngm25/hd_ngm25_prng.h (design §7 hard
// requirement) so CPU and GPU draw bit-identical values for the same synapse at
// the same step. WGSL u32 multiply wraps mod 2^32, matching C++ unsigned overflow.
//
// GI1 scope: the forward path only — response (R2, per dendrite) + combine (R3,
// per neuron). Winner/adapt (R5/R6/R7) and branch (R8) arrive in GI3/GI4.
//
// P1 CLAMP ORDER (SPEC.md M.7.31): like the CPU response_dendrite, every segment
// sum is floored at 0 immediately after each penalised synapse, in arena slot
// order. That makes the sum order-dependent, so it is not a plain parallel sum:
// each synapse acts on its segment's sum as x -> max(x + a, b) (unpenalised:
// a = input, b = -inf; penalised: a = input - penalty, b = 0), and such maps
// compose associatively, (a1,b1) then (a2,b2) = (a1 + a2, max(b1 + a2, b2)). The
// kernel composes them per segment over contiguous synapse runs and combines the
// runs in order, which equals the serial CPU result up to floating-point
// summation order. (It used to floor each bin once after the sweep — a different
// function whenever a sum dipped below 0 mid-sweep; M.7.31.)
//
// Storage widths: syn_input_idx (CPU u16) and syn_seg (CPU u8) are u32 here —
// WGSL has no sub-word storage (design §4, G2).

// Structural-identity key — hd_ngm25_prng.h:38-47, constants verbatim.
fn syn_key(seed: u32, step: u32, dend: u32, seg: u32, input_idx: u32) -> u32 {
    return seed
         ^ (step      * 3266489917u)
         ^ (dend      * 2246822519u)
         ^ (seg       * 668265263u)
         ^ (input_idx * 374761393u);
}

// Per-step uniforms. vec2 fields carry the per-TYPE (apical=x, proximal=y) R1
// scalars; std140-clean (vec2 aligned to 8). Total 80 B (see host static_assert).
struct Params {
    seed:         u32,
    step:         u32,
    neuron_cnt:   u32,
    dend_cnt:     u32,
    apical_cnt:   u32,       // dendrites per neuron
    proximal_cnt: u32,       // dendrites per neuron
    max_seg_id:   u32,
    frozen:       u32,       // unread by the kernels (a frozen plan omits the learning
                             // passes); kept so the uniform's layout is unchanged
    perm_thres:     f32,
    inv_perm_range: f32,
    noise_lo:       f32,
    noise_range:    f32,
    inp_sum:      vec2<f32>,  // R1 per-type: is the type live? (isnormal check)
    inv_inp_sum:  vec2<f32>,  // R1 per-type normalization
    atten:        vec2<f32>,  // R1 per-type min-combined attenuation
    _pad:         vec2<f32>,
};

const WG:      u32 = 64u;
const SEG_MAX: u32 = 32u;      // 2^(4+1); covers max_branch_level <= 6? NO — <= 4.
                              // GPU caps depth at 4 (as ngm3-GPU does); the ctor
                              // rejects deeper configs. Keeps the WG*SEG_MAX shared
                              // reduction buffer at 64*32*4 = 8 KB (LDS-friendly).

// -------------------------------- response (R2) ----------------------------
// One workgroup per dendrite. Each thread takes a CONTIGUOUS run of the
// dendrite's synapses (in arena slot order) and composes, per segment, the
// floored-sum maps of P1 above into a private (a, b) pair. An order-preserving
// tree then combines adjacent runs (thread t with t + stride, stride = 1, 2, 4,
// ...), so thread 0 ends with the whole dendrite's maps and the segment sum is
// max(a, b) (the sum starts at 0). Run-to-run DETERMINISTIC (GI2c): same runs,
// same combine order every run. Thread 0 then tree-pushes root->leaf, attenuates
// + normalizes leaves, and writes the leaf max to dend_act[d]. dend_type selects
// the per-type R1 scalars.

@group(0) @binding(0) var<storage, read>       r_syn_perm:      array<f32>;
@group(0) @binding(1) var<storage, read>       r_syn_input_idx: array<u32>;
@group(0) @binding(2) var<storage, read>       r_syn_seg:       array<u32>;
@group(0) @binding(3) var<storage, read>       r_dend_offset:   array<u32>;
@group(0) @binding(4) var<storage, read>       r_dend_type:     array<u32>;   // 0 apical, 1 proximal
@group(0) @binding(5) var<storage, read>       r_input_concat:  array<f32>;
@group(0) @binding(6) var<storage, read>       r_half_max:      array<f32>;   // per position
@group(0) @binding(7) var<storage, read>       r_inv_sum:       array<f32>;   // per position
@group(0) @binding(8)  var<storage, read_write> r_dend_act:     array<f32>;
@group(0) @binding(9)  var<uniform>             rp:             Params;
// Per-dendrite seg activity [dend * seg_cnt + si] the adapt pass reads and folds:
// leaves hold the attenuated/normalized activity, inner nodes the raw pushed sums
// (matches the CPU response_dendrite output that adapt_dendrite consumes).
@group(0) @binding(10) var<storage, read_write> r_seg_activity: array<f32>;

// Combine scratch: at tree level `stride` the right-hand runs publish their
// (a, b) pairs here, at most WG/2 of them: (WG/2) * 2 * SEG_MAX floats = 8 KB for
// WG=64, SEG_MAX=32. Thread 0 reuses the start as its final segment sums.
var<workgroup> seg_rows: array<f32, WG * SEG_MAX>;

// "b = -inf" of an unpenalised map; finite so b + a stays well-defined.
const NO_FLOOR: f32 = -3.0e38;

@compute @workgroup_size(WG)
fn response_main(@builtin(workgroup_id) wg: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = wg.x;
    let t = lid.x;
    if (d >= rp.dend_cnt) { return; }

    let seg_cnt = rp.max_seg_id + 1u;
    // This thread's composed maps, one per segment: x -> max(x + pa, pb).
    var pa: array<f32, SEG_MAX>;
    var pb: array<f32, SEG_MAX>;
    for (var b = 0u; b < seg_cnt; b = b + 1u) { pa[b] = 0.0; pb[b] = NO_FLOOR; }

    let ti = r_dend_type[d];               // 0 apical, 1 proximal
    let live = select(rp.inp_sum.y, rp.inp_sum.x, ti == 0u);
    // CPU response_dendrite skips a dead type via !std::isnormal(inp_sum). WGSL has
    // no isNormal builtin, so `live > 0.0` stands in: it rejects 0, negatives and
    // NaN (NaN > 0.0 is false) like !isnormal, but — unlike !isnormal — it lets a
    // subnormal-positive value THROUGH. inp_sum here is a sum of clamped [0,1]
    // activities, so a subnormal is unreachable in practice; that is the only case
    // where the two guards differ, and it cannot arise from valid input.
    if (live > 0.0) {
        let begin = r_dend_offset[d];
        let end   = r_dend_offset[d + 1u];
        let run   = (end - begin + WG - 1u) / WG;
        let s0    = min(begin + t * run, end);
        let s1    = min(s0 + run, end);
        for (var s = s0; s < s1; s = s + 1u) {
            let perm = r_syn_perm[s];
            if (perm <= rp.perm_thres) { continue; }  // sub-threshold: perm read only
            let pos = r_syn_input_idx[s];
            let seg = r_syn_seg[s];
            let inp = r_input_concat[pos];
            // The CPU's operations, in its order: sum += inp; on a penalty
            // sum -= strength * (1 - inp/sum_i) and floor at 0 — applied to both
            // halves of the map (the floor only to b: max(x + a, b) >= b).
            var a  = pa[seg] + inp;
            var fb = pb[seg] + inp;
            let draw = pcg_uniform(syn_key(rp.seed, rp.step, d, seg, pos)) * r_half_max[pos];
            if (draw > inp) {
                let strength = (perm - rp.perm_thres) * rp.inv_perm_range;
                let pen = strength * (1.0 - inp * r_inv_sum[pos]);
                a  = a - pen;
                fb = max(fb - pen, 0.0);
            }
            pa[seg] = a;
            pb[seg] = fb;
        }
    }

    // Order-preserving tree: run t (left) absorbs run t + stride (right).
    var stride = 1u;
    loop {
        if (stride >= WG) { break; }
        let slot = (t / (2u * stride)) * 2u * SEG_MAX;
        if (t % (2u * stride) == stride) {
            for (var b = 0u; b < seg_cnt; b = b + 1u) {
                seg_rows[slot + b]           = pa[b];
                seg_rows[slot + SEG_MAX + b] = pb[b];
            }
        }
        workgroupBarrier();
        if (t % (2u * stride) == 0u) {
            for (var b = 0u; b < seg_cnt; b = b + 1u) {
                let ra = seg_rows[slot + b];
                let rb = seg_rows[slot + SEG_MAX + b];
                pb[b] = max(pb[b] + ra, rb);
                pa[b] = pa[b] + ra;
            }
        }
        workgroupBarrier();
        stride = stride * 2u;
    }

    if (t == 0u) {
        let inv_inp_sum = select(rp.inv_inp_sum.y, rp.inv_inp_sum.x, ti == 0u);
        let atten       = select(rp.atten.y,       rp.atten.x,       ti == 0u);
        // Each segment's floored sum: its composed map applied to the start value 0.
        for (var si = 0u; si < seg_cnt; si = si + 1u) {
            seg_rows[si] = max(pa[si], pb[si]);
        }
        // push root -> leaf: parent adds into both children
        let leaf_begin = (rp.max_seg_id + 1u) / 2u;
        for (var si = 1u; si < leaf_begin; si = si + 1u) {
            seg_rows[2u * si]      = seg_rows[2u * si]      + seg_rows[si];
            seg_rows[2u * si + 1u] = seg_rows[2u * si + 1u] + seg_rows[si];
        }
        // attenuate + normalize leaves (write back so adapt reads them), take max
        var max_act = 0.0;
        for (var si = leaf_begin; si <= rp.max_seg_id; si = si + 1u) {
            let a = clamp(seg_rows[si] * atten * inv_inp_sum, 0.0, 1.0);
            seg_rows[si] = a;
            max_act = max(max_act, a);
        }
        r_dend_act[d] = max_act;
        // publish this dendrite's seg activity (attenuated leaves, raw inner) for adapt.
        let seg_cnt = rp.max_seg_id + 1u;
        let base = d * seg_cnt;
        for (var si = 0u; si < seg_cnt; si = si + 1u) { r_seg_activity[base + si] = seg_rows[si]; }
    }
}

// -------------------------------- combine (R3) -----------------------------
// One thread per neuron. Type-max over the neuron's dendrites (apical -1.0
// absent-sentinel), apical*proximal + D2 noise, clamp -> neuron_act. (The
// decaying running max is learning state; winner_main updates it.)

@group(0) @binding(0) var<storage, read>       c_dend_act:     array<f32>;
@group(0) @binding(1) var<storage, read>       c_dend_type:    array<u32>;
@group(0) @binding(2) var<storage, read_write> c_neuron_act:   array<f32>;
@group(0) @binding(4) var<uniform>             cp:             Params;

@compute @workgroup_size(WG)
fn combine_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= cp.neuron_cnt) { return; }

    let per_neuron = cp.apical_cnt + cp.proximal_cnt;
    let d0 = n * per_neuron;
    var apical   = -1.0;                    // absent sentinel (CPU neuron_type_max)
    var proximal =  0.0;
    for (var k = 0u; k < per_neuron; k = k + 1u) {
        let d = d0 + k;
        let a = c_dend_act[d];
        if (c_dend_type[d] == 0u) { apical = max(apical, a); }
        else                      { proximal = max(proximal, a); }
    }
    if (apical < 0.0) { apical = 1.0; }     // no apical dendrites -> identity
    apical   = clamp(apical,   0.0, 1.0);
    proximal = clamp(proximal, 0.0, 1.0);

    // D2 noise: syn_key(seed, step, neuron, 0, 0) — seg=0 sentinel (never a real
    // synapse), so no collision with response draws (CPU combine_neuron).
    let noise = cp.noise_lo + cp.noise_range * pcg_uniform(syn_key(cp.seed, cp.step, n, 0u, 0u));
    let act = clamp(apical * proximal + noise, 0.0, 1.0);
    c_neuron_act[n] = act;
}

// ================================ WINNER + ADAPT ===========================
// R5 (winner + two adapt passes), R6 (envelope), R7 (dendrite sweep). Semantic
// reference: src/hd_ngm25/hd_ngm25_adapt.cpp (the map is law). Statement order in
// the R7 sweep is load-bearing (theta -> perm IIR -> adapt_hist -> mismatch).
//
// Single adaptation (SPEC.md M.4.14): the winner's PRIMARY adapt runs as its own
// dispatch (adapt_primary), every other neuron's SECONDARY adapt in
// adapt_secondary, which skips the winner — each neuron is adapted exactly once
// per step, so the two passes touch disjoint dendrites.

const WG_A:    u32 = 64u;
const EPS_A:   f32 = 1.1920929e-7;   // FLT_EPSILON

// ngm2::sigmoid with precomputed params (sigmoid_params_t): 1/(1+exp(-(x*inv+start)))
fn ng_sigmoid(x: f32, inv_step: f32, start: f32) -> f32 {
    return 1.0 / (1.0 + exp(-(x * inv_step + start)));
}

// Adapt uniforms. vec2 fields are 8-B aligned; 112 B total (host static_assert).
struct AdaptParams {
    seed:         u32,
    step:         u32,
    neuron_cnt:   u32,
    max_seg_id:   u32,
    apical_cnt:   u32,
    proximal_cnt: u32,
    dend_cnt:     u32,
    _pu:          u32,
    primary_lr:         f32,
    secondary_lr:       f32,
    common_lr:          f32,
    perm_thres:         f32,
    win_thres:          f32,
    mismatch_act_thres: f32,
    mismatch_smoothing: f32,
    _pf:                f32,
    sig_default: vec2<f32>,     // 1 - sigmoid(mx_act) uses DEFAULT shape
    sig_wf:      vec2<f32>,     // weight_filter
    wt_inv_step: vec2<f32>,     // tracked window: inv_step of the lower / upper edge's fixed steepness
    _pad_w:      vec2<f32>,     // (formerly the second activity_learning_window shape)
    inv_last_max_inp: vec2<f32>,  // per type
    _pv:              vec2<f32>,
};

// -------------------------------- winner (R5) ------------------------------
@group(0) @binding(0) var<storage, read>       w_outp: array<f32>;   // group_outp (post-inhib)
@group(0) @binding(1) var<storage, read_write> w_scal: array<u32>;   // [idx, act_sum, win_act, mx_act, window edges x4]
@group(0) @binding(2) var<uniform>             wp:     AdaptParams;
@group(0) @binding(3) var<storage, read>       w_act:  array<f32>;   // neuron_act (pre-inhibition)
@group(0) @binding(4) var<storage, read_write> w_mxact: array<f32>;  // neuron_mx_act
// Learning steps so far (u32, wraps). The host reads it back only when a branch
// boundary could have been reached (grow_if_branched), not every step.
@group(0) @binding(5) var<storage, read_write> w_lcnt:  array<u32>;
// Activity window edges [winner lower, winner upper, other lower, other upper]:
// learning state, the edges the NEXT learning step adapts with.
@group(0) @binding(6) var<storage, read_write> w_track: array<f32>;

@compute @workgroup_size(WG_A)
fn winner_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    // Single-thread serial: neuron_cnt is small and choosing the j-th neuron above
    // the threshold (loop 3) is order-sensitive. Deterministic.
    if (lid.x != 0u) { return; }
    var mx = 0.0;
    var sum = 0.0;
    for (var i = 0u; i < wp.neuron_cnt; i = i + 1u) { let o = w_outp[i]; mx = max(mx, o); sum = sum + o; }
    let lo = mx * wp.win_thres;
    let win_act = lo + pcg_uniform(syn_key(wp.seed, wp.step, 0xFFFFFFFFu, 0u, 0u)) * (mx - lo);
    // D6: the winner is uniform among the k neurons clearing win_act (SPEC.md
    // M.4.4) — the floor(u·k)-th in index order, u clamped as in the CPU's
    // ngm2::pick_winner (a [0,1) draw can round to 1). Same key as the CPU.
    let pick = pcg_uniform(syn_key(wp.seed, wp.step, 0xFFFFFFFFu, 1u, 0u));
    var k = 0u;
    for (var i = 0u; i < wp.neuron_cnt; i = i + 1u) {
        if (w_outp[i] + EPS_A >= win_act) { k = k + 1u; }
    }
    var widx = 0xFFFFFFFFu;
    if (k > 0u) {
        var j = min(u32(pick * f32(k)), k - 1u);
        for (var i = 0u; i < wp.neuron_cnt; i = i + 1u) {
            if (w_outp[i] + EPS_A >= win_act) {
                if (j == 0u) { widx = i; break; }
                j = j - 1u;
            }
        }
    }
    w_scal[0] = widx;
    w_scal[1] = bitcast<u32>(sum);
    w_scal[2] = bitcast<u32>(win_act);
    w_scal[3] = bitcast<u32>(mx);
    // This step's window edges for the adapt passes: the ones the previous
    // learning step left.
    w_scal[4] = bitcast<u32>(w_track[0]);
    w_scal[5] = bitcast<u32>(w_track[1]);
    w_scal[6] = bitcast<u32>(w_track[2]);
    w_scal[7] = bitcast<u32>(w_track[3]);

    // A zero group output is not a learning step (SPEC.md M.4.15): the adapt
    // passes read act_sum == 0 from w_scal[1] and skip, and neither the running max
    // nor the learning-step count moves. On a learning step the running max decays
    // and folds in this step's activity before the adapt passes read it.
    if (sum != 0.0) {
        for (var n = 0u; n < wp.neuron_cnt; n = n + 1u) {
            w_mxact[n] = max(w_mxact[n] * 0.99999, w_act[n]);
        }
        w_lcnt[0] = w_lcnt[0] + 1u;

        // Advance the window edges with this step's raw activities (mirrors
        // ngm::window_tracking::update).
        let n_cnt = wp.neuron_cnt;
        if (widx < n_cnt) {
            let a = w_act[widx];
            w_track[0] = wt_step(w_track[0], WT_Q_LOWER, select(0.0, 1.0, a < w_track[0]));
            w_track[1] = wt_step(w_track[1], WT_Q_UPPER, select(0.0, 1.0, a < w_track[1]));
        }
        var below_lo = 0u;
        var below_hi = 0u;
        var cnt      = 0u;
        for (var i = 0u; i < n_cnt; i = i + 1u) {
            if (i == widx) { continue; }
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

// -------------------------------- adapt (R5/R6/R7) -------------------------
@group(0) @binding(0)  var<storage, read_write> a_perm:        array<f32>;
@group(0) @binding(1)  var<storage, read>       a_input_idx:   array<u32>;
@group(0) @binding(2)  var<storage, read>       a_seg:         array<u32>;
@group(0) @binding(3)  var<storage, read_write> a_adapt_hist:  array<f32>;
@group(0) @binding(4)  var<storage, read_write> a_mismatch:    array<f32>;
@group(0) @binding(5)  var<storage, read>       a_dend_offset: array<u32>;
@group(0) @binding(6)  var<storage, read>       a_dend_type:   array<u32>;
@group(0) @binding(7)  var<storage, read>       a_dend_act:    array<f32>;
@group(0) @binding(8)  var<storage, read>       a_seg_activity:array<f32>;
@group(0) @binding(9)  var<storage, read>       a_neuron_act:  array<f32>;
@group(0) @binding(10) var<storage, read>       a_neuron_mxact:array<f32>;
@group(0) @binding(11) var<storage, read>       a_group_outp:  array<f32>;
@group(0) @binding(12) var<storage, read>       a_input_concat:array<f32>;
// packed per-position ctx: (high_thres, inv_high_range, inv_high_thres, adapt_atten)
@group(0) @binding(13) var<storage, read>       a_ctx4:        array<vec4<f32>>;
@group(0) @binding(14) var<storage, read>       a_wscal:       array<u32>;
@group(0) @binding(15) var<uniform>             ap:            AdaptParams;

var<workgroup> aw_segw: array<f32, SEG_MAX>;   // folded seg weights
var<workgroup> aw_sega: array<f32, SEG_MAX>;   // folded seg activity

// Adapt one neuron's dendrites with the given R6 envelope input weight. Called by
// ALL threads of the workgroup with the same (n, weight) -> uniform control flow,
// so the workgroupBarriers are legal.
fn adapt_neuron_work(n: u32, weight: f32, t: u32) {
    let per_neuron = ap.apical_cnt + ap.proximal_cnt;
    let d0 = n * per_neuron;

    // type-max over the neuron's dendrites (apical -1 absent sentinel).
    var ap_max = -1.0;
    var px_max = 0.0;
    for (var k = 0u; k < per_neuron; k = k + 1u) {
        let d = d0 + k; let a = a_dend_act[d];
        if (a_dend_type[d] == 0u) { ap_max = max(ap_max, a); } else { px_max = max(px_max, a); }
    }
    if (ap_max < 0.0) { ap_max = 1.0; }
    ap_max = clamp(ap_max, 0.0, 1.0);
    px_max = clamp(px_max, 0.0, 1.0);

    // R6 envelope (three sigmoid shapes).
    let nact = a_neuron_act[n];
    let nmx  = a_neuron_mxact[n];
    // Tracked activity learning window: the winner's or the others' edges.
    let we      = select(6u, 4u, n == a_wscal[0]);
    let w_lower = bitcast<f32>(a_wscal[we]);
    let w_upper = bitcast<f32>(a_wscal[we + 1u]);
    let env = min(max(1.0 - ng_sigmoid(nmx, ap.sig_default.x, ap.sig_default.y),
                      ng_sigmoid(nact, ap.wt_inv_step.x, -w_lower * ap.wt_inv_step.x)),
                  1.0 - ng_sigmoid(nact, ap.wt_inv_step.y, -w_upper * ap.wt_inv_step.y));
    let synapse_weight = weight * env;

    let seg_cnt    = ap.max_seg_id + 1u;
    let leaf_begin = (ap.max_seg_id + 1u) / 2u;

    for (var k = 0u; k < per_neuron; k = k + 1u) {
        let d = d0 + k;
        let ti = a_dend_type[d];
        let type_max = select(px_max, ap_max, ti == 0u);
        // CPU skips via !std::isnormal(type_max). WGSL has no isNormal builtin, so
        // `type_max > 0.0` stands in: it rejects 0/negative/NaN like !isnormal but
        // lets a subnormal-positive through. type_max is a max of clamped [0,1]
        // activities, so a subnormal is unreachable; the guards differ only there.
        if (type_max > 0.0) {
            let inv_max = 1.0 / type_max;
            let base = d * seg_cnt;
            if (t == 0u) {
                // leaf weights + copy seg activity
                for (var si = 0u; si < seg_cnt; si = si + 1u) { aw_segw[si] = 0.0; aw_sega[si] = a_seg_activity[base + si]; }
                var primary_seen = false;
                for (var si = leaf_begin; si <= ap.max_seg_id; si = si + 1u) {
                    if (!primary_seen && aw_sega[si] + EPS_A >= type_max) {
                        primary_seen = true;
                        aw_segw[si] = synapse_weight * ap.primary_lr;
                    } else {
                        aw_segw[si] = aw_sega[si] * synapse_weight * ap.secondary_lr * inv_max;
                    }
                }
                // fold leaves -> root, parent = max(children), both arrays
                var level_start = leaf_begin;
                loop {
                    if (level_start <= 1u) { break; }
                    var si = level_start;
                    loop {
                        if (si >= level_start * 2u) { break; }
                        aw_segw[si / 2u] = max(aw_segw[si], aw_segw[si + 1u]);
                        aw_sega[si / 2u] = max(aw_sega[si], aw_sega[si + 1u]);
                        si = si + 2u;
                    }
                    level_start = level_start / 2u;
                }
            }
            workgroupBarrier();

            // sweep synapses (grid-strided; each synapse handled by one thread).
            let begin = a_dend_offset[d];
            let end   = a_dend_offset[d + 1u];
            for (var s = begin + t; s < end; s = s + WG_A) {
                let pos = a_input_idx[s];
                let seg = a_seg[s];
                let inp = a_input_concat[pos];
                let c = a_ctx4[pos];   // high_thres, inv_high_range, inv_high_thres, adapt_atten
                let high_thres = c.x;
                let ramp = select((high_thres - inp) * c.z, (inp - high_thres) * c.y, inp > high_thres);
                let theta = clamp(aw_segw[seg] * ramp * c.w, 0.0, 1.0);
                let pnew = clamp(a_perm[s] * (1.0 - theta) + select(0.0, theta, inp > high_thres), 0.0, 1.0);
                a_perm[s] = pnew;
                a_adapt_hist[s] = a_adapt_hist[s] + theta;
                let act_ratio = aw_sega[seg] * inv_max;
                if (act_ratio >= ap.mismatch_act_thres) {
                    let inv_lmi = select(ap.inv_last_max_inp.y, ap.inv_last_max_inp.x, ti == 0u);
                    let inp_ratio = inp * inv_lmi;
                    var mm = select(inp_ratio, 1.0 - inp_ratio, pnew > ap.perm_thres);
                    mm = mm * act_ratio;
                    a_mismatch[s] = a_mismatch[s] * (1.0 - ap.mismatch_smoothing) + mm * ap.mismatch_smoothing;
                }
            }
            workgroupBarrier();   // before the next dendrite reuses aw_seg*
        }
    }
}

// Loop 3 (primary): the single winner neuron, weight = sigmoid(1 - out[winner]).
// Distinct dispatch (1 workgroup) that completes before adapt_secondary.
@compute @workgroup_size(WG_A)
fn adapt_primary(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (bitcast<f32>(a_wscal[1]) == 0.0) { return; }   // zero output: not a learning step (M.4.15)
    let widx = a_wscal[0];
    if (widx == 0xFFFFFFFFu) { return; }   // no neuron reached the winner threshold
    let weight = ng_sigmoid(1.0 - a_group_outp[widx], ap.sig_wf.x, ap.sig_wf.y);
    adapt_neuron_work(widx, weight, lid.x);
}

// Loop 4 (secondary): every neuron except the winner, weight =
// sigmoid(1 - out/act_sum) * common_lr. One workgroup per neuron.
@compute @workgroup_size(WG_A)
fn adapt_secondary(@builtin(workgroup_id) wg: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wg.x;
    if (n >= ap.neuron_cnt) { return; }
    let act_sum = bitcast<f32>(a_wscal[1]);
    if (act_sum == 0.0) { return; }    // zero output: not a learning step (M.4.15)
    if (n == a_wscal[0]) { return; }   // the winner adapted once, in adapt_primary
    // Guard 0/0 -> NaN when every post-inhibition output is 0 (see the CPU adapt_all
    // note). select discards the NaN in the unused operand; `!= 0` (not `> 0`) keeps
    // a negative act_sum matching the CPU/ngm2 path exactly.
    let sec = select(0.0, ng_sigmoid(1.0 - a_group_outp[n] / act_sum, ap.sig_wf.x, ap.sig_wf.y),
                     act_sum != 0.0);
    adapt_neuron_work(n, sec * ap.common_lr, lid.x);
}
