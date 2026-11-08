// PCG-1D32 stateless hash PRNG (Jarzynski & Olano, "Hash Functions for GPU
// Rendering"). Stateless: each draw is a pure function of a u32 key, so no
// per-neuron RNG state lives on the GPU. Callers form the key from the entity
// coordinates and time, e.g. pcg_hash(neuron ^ dendrite ^ rep ^ step ^ slot),
// giving a stream that is deterministic per (seed, step) yet decorrelated across
// cells. ~6 ALU ops, no platform-specific intrinsics.
//
// This snippet is concatenated ahead of an entity's WGSL at module-load time
// (see coast::gpu::shared_prng_wgsl()). It defines functions only — no bindings,
// no entry points — so it composes cleanly with any kernel.

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uniform f32 in [0, 1). 2^-32 scaling of the full 32-bit hash output.
fn pcg_uniform(input: u32) -> f32 {
    return f32(pcg_hash(input)) * (1.0 / 4294967296.0);
}
