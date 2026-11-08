// mnist_io_gpu compute kernel (S5):
//
//   img_main    — read the current image by index, add per-pixel uniform noise,
//                 clamp to [0,1], write into the entity's output buffer. This is
//                 the GPU analogue of mnist_io::process()'s copy+noise loop. The
//                 noise stream is the stateless PCG hash (shared_prng.wgsl,
//                 concatenated ahead of this source at module load), so it is NOT
//                 bit-identical to the CPU mt19937 stream — S5's bar is "sane,
//                 in-range", not parity (parity is pinned at S12).
//
// IMAGE_SIZE is fixed at 784 (28×28); the host asserts get_image_size() == 784.

const IMAGE_SIZE: u32 = 784u;

// ---- read-image + noise + clamp -------------------------------------------

struct Params {
    idx:       u32,   // current image index into the dataset
    key:       u32,   // per-step PRNG key component (step ^ seed)
    noise_lo:  f32,
    noise_hi:  f32,
};

@group(0) @binding(0) var<storage, read>       dataset: array<f32>;
@group(0) @binding(1) var<storage, read_write> output:  array<f32>;
@group(0) @binding(2) var<uniform>             params:  Params;

@compute @workgroup_size(64)
fn img_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= IMAGE_SIZE) { return; }
    let pix   = dataset[params.idx * IMAGE_SIZE + i];
    let r     = pcg_uniform((params.key * 2654435761u) + i);
    let noise = params.noise_lo + (params.noise_hi - params.noise_lo) * r;
    output[i] = clamp(pix + noise, 0.0, 1.0);
}
