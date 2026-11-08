+++
title = "retina"
weight = 41
+++

# retina — Centre-Surround ON/OFF Front End

A **retina node** sits between a dataset node and the neuron groups. It reads one
image buffer and emits **one polarity** of its centre-surround response: an ON
node passes increments, an OFF node decrements. Two nodes reading the same image
model the ON and OFF pathways. The dataset nodes stay pure stimulus providers; the
retina is the first entity that only transforms its input.

```text
mnist_input ──┬──> retina (on)  ──┐
              └──> retina (off) ──┴──> ngm2_neuron_group  (input channels "on", "off")
```

## Response

Per output plane `y` (one per image channel, or one luminance plane):

```text
d   = gain * (G(center_sigma) * y  -  k * G(surround_sigma) * y  -  (1 - k) * mean(y))
ON  = clamp( d, 0, 1) + U(noise_range),  clamped to <= 1
OFF = clamp(-d, 0, 1) + U(noise_range),  clamped to <= 1
```

- `k = surround_weight`. With `k = 1` the node reports pure local contrast: a
  uniform field gives no response at any level. With `k = 0` it reports
  luminance relative to the mean of the current image (adaptation to the mean
  light level). In between, a uniform region below the image mean drives the OFF
  node weakly (with mnist at `k = 0.7`, the black background sits at 0.1–0.3 OFF).
- The Gaussians are separable, truncated at 3σ per axis (a square support) and
  renormalised, with nearest-pixel borders. Both sigmas must lie in
  `[0.1, 100]` pixels (`min_sigma`, `max_sigma`): at 0.1 the kernel is already a
  delta, and far below it `σ²` underflows and the kernel turns NaN; 100 exceeds
  every supported image and bounds the kernel size and blur cost.
  `surround_sigma >= 1.2 * center_sigma` is required (`min_surround_ratio`): a
  centre wider than its surround inverts the polarity (a bright spot on black
  drives OFF), and as the sigmas approach each other the response at `k = 1`
  cancels towards the noise floor. At `k = 0` the surround term vanishes and its
  blur is skipped.
- `gain` and `surround_weight` must not exceed `FLT_MAX`: the filter runs in
  `float`, where a larger value becomes `inf` and `inf · 0` in a uniform region
  becomes NaN. `width * height * channels` must fit in an `int`, the type the
  blur indexes with.
- `normalize: true` (off by default) adds **divisive normalisation** after the
  centre-surround stage. Each output plane's gain-free response `x` (the `d` above
  with `gain = 1`) is divided by `norm_sigma` plus its root mean square (RMS) over
  the whole plane:

  ```text
  d = gain * x / (norm_sigma + RMS(x))
  ```

  Where `RMS(x) ≫ norm_sigma` the response is contrast-invariant: an image at twice
  the contrast gives the same `d`, so the group sees every image at about the same
  contrast. Where `RMS(x) ≪ norm_sigma` it stays near-linear (`d ≈ gain · x /
  norm_sigma`), so a faint image is not blown up to full contrast. `gain` still sets
  where the clamp at 1 starts clipping: the normalised plane has RMS
  `gain · RMS(x) / (norm_sigma + RMS(x)) < gain`. `norm_sigma` (default 0.05, same
  units as `x`) must be finite and at least `min_norm_sigma` (1e-12) when
  `normalize` is on, which keeps a blank plane (`RMS = 0`) at 0: a zero
  `norm_sigma` would make it 0/0, and a much smaller one could overflow
  `gain / norm_sigma` at extreme gain, so the plane would come out `inf · 0` = NaN.
  The floor is a safety margin, not the overflow threshold, which lies below
  `gain / DBL_MAX` (about 1.9e-270 even at `gain = FLT_MAX`). It is ignored when off. The noise floor is
  added after polarisation, so it is never normalised. With `normalize: false` the
  output is bit-identical to a build without the option. The RMS is taken per
  output plane, so with `luminance: false` each colour channel is normalised on its
  own, which equalises the channels' contrast and discards their contrast ratio.
- **Lower `gain` when you turn `normalize` on.** With `normalize` on, the output
  RMS approaches `gain` from below; on MNIST it is about 0.73 · `gain`, and peaks
  are about 5× the RMS, so on MNIST they clip at 1 unless `gain` is about 0.25.
  Other datasets have a different peak-to-RMS ratio and need their own value. `gain` means the same with and without the option, so the default stays 1,
  but at that default the ON channel saturates. Measured on the first 10,000 MNIST
  training images with the default parameters (`surround_weight = 1.0`; at the shipped
  MNIST config's 0.7, `RMS(x)` (first row) is about 16 % higher, but with `normalize` on
  `RMS(d)` rises only about 4 % and the clipped shares move by at most 4 points);
  *clipped* is the share of pixels
  with `d > 0` (`d < 0` for OFF) that reach the clamp:

  | gain | normalize | RMS(d) p5 / p50 / p95 | max \|d\| | ON clipped | OFF clipped |
  |---|---|---|---|---|---|
  | 1.0 | off | 0.099 / 0.138 / 0.177 | 0.59 | 0.0 % | 0.0 % |
  | 1.0 | on | 0.665 / 0.734 / 0.780 | 3.89 | 61.1 % | 3.6 % |
  | 0.5 | on | 0.332 / 0.367 / 0.390 | 1.94 | 14.8 % | 0.0 % |
  | 0.25 | on | 0.166 / 0.184 / 0.195 | 0.97 | 0.0 % | 0.0 % |

  At `gain = 1`, turning `normalize` on amplifies the MNIST response about 5× and
  makes most of the ON channel binary. The table also shows how far the default
  `norm_sigma = 0.05` normalises MNIST: `RMS(x) / (norm_sigma + RMS(x))` is about
  0.73, so the response is only partly contrast-invariant, and the p5–p95 spread of
  `RMS(d)` narrows from about ±28 % to about ±8 %.
- **Biological basis.** The stage has a biological counterpart: the retina
  adjusts its gain to input contrast (Shapley & Victor 1978; the fast component
  acts within 0.1 s, Baccus & Meister 2002), and divisive normalisation is the
  standard model of that gain change (Carandini & Heeger 2012). The closest
  published model is the LGN model of Bonin, Mante & Carandini (2005):
  centre-surround response, divided by `c50 + local contrast`, then rectified,
  which is this chain with `norm_sigma` as `c50`. Two differences: the pool here
  is the RMS of the centre-surround output over the whole image, where theirs is
  the stimulus contrast under a surround-sized field; and the division is
  stateless, so the slow (~10 s) adaptation component is not modelled.
- `luminance: true` converts RGB to `Y = 0.299 R + 0.587 G + 0.114 B` before
  filtering (needs `channels: 3`); otherwise each channel is filtered on its own.
- The output noise is always on (default `[0, 0.05]`): it is the "null" signal,
  and it keeps every output buffer's sum above zero, which the dendrite's
  per-input normalisation divides by. `validate()` therefore requires
  `noise_range[1] > 0`; `[0, 0]` is refused. The upper end is at most 0.1
  (`max_noise`): the floor is a null signal, and a high one would bury the
  response (a low end near 1 saturates every output at 1).

## Wiring and sizes

The node's output size is `width * height * out_channels` and is fixed by the
config, because `sim_builder` may initialise a consumer before the retina. At
`init_io()` the node checks that it has exactly one input and that the input's
size equals `width * height * channels`; a mismatch throws. The input's channel
name is free (the network editor shows whatever name the config uses, `in` for an
unwired node); `sim_builder` refuses a node that repeats a channel name, so two
inputs cannot collapse into one unnoticed (see
[tools]({{% relref "/architecture/modules/tools" %}})). Each hop through a
double-buffered `io_buffer` adds one step of latency, so a group behind a retina
sees the image one step later than it would reading the dataset directly.

The node is stateless apart from its noise RNG, which is saved and restored with
the simulation state. The retina format has always carried the `rng` group, so a
checkpoint where it is missing is corrupt and `load_state` throws, rather than
silently restarting the noise stream from the seed. An incomplete one (`mt` or
`tempered` short, `mti` missing or past the batch) throws for every entity, not
only the retina (see `read_rng` in
[logging]({{% relref "/architecture/modules/logging" %}})). Next to it the checkpoint carries a configuration record
of the node's parameters (SPEC.md M.7.8, see
[logging]({{% relref "/architecture/modules/logging" %}})): on load the
configuration wins, and every parameter that differs from the record is reported
in one warning, e.g. `gain 2 -> 3`. `unit.retina` checks that the record covers
every config key, so a parameter added later cannot drop out of the check unnoticed.
The headless run manifest lists the configs in use by retina nodes under
`retina`, keyed by config name, with the config's seed (before any offset); the seed each node
actually uses is in `node_seeds` (see
[sim_core]({{% relref "/architecture/modules/sim_core" %}})). The node is
CPU-only; feeding a GPU neuron group from it has not been tested yet.

## Configuration

Retina configs live in a top-level `"retina"` section, keyed by name, and nodes
use `"entity_type": "retina"`:

```json
"retina": {
  "on":  { "width": 28, "height": 28, "channels": 1, "polarity": "on",
           "center_sigma": 1.0, "surround_sigma": 3.0, "surround_weight": 0.7,
           "gain": 2.0, "noise_range": [0.0, 0.05], "random_seed": 1 },
  "off": { "...": "same, with", "polarity": "off", "random_seed": 2 }
}
```

Give the two nodes different seeds (here, or with each node's
`random_seed_offset`): with equal effective seeds both draw the same noise
sequence, and the "null" signal of the ON and OFF channels is perfectly
correlated. The shipped configs use 1 and 2, apart from the dataset's 0. The
effective seed (config seed plus `--seed-offset` plus the node's offset) is
composed modulo 2³² like every seed (see *Random seeds* in
[core]({{% relref "/architecture/modules/core" %}})), so the seed the RNG uses is
the one `node_seeds` in the manifest and the checkpoint's configuration record
report. (The manifest's `retina` object holds the config's seed, before offsets.)

`polarity` must be `"on"` or `"off"`; any other string throws at load, as for
every config enum (`COAST_JSON_STRICT_ENUM`). A neuron
group reads the two nodes as two input channels, which its channel mappings route
to two neuron channels and on to dendrites listing both (see
`configs/mnist_retina.json` and `configs/chairs3d_retina.json`).

## Files

| File | Contents |
|------|----------|
| `retina/retina_cfg.h` | `retina_param_t`, `polarity_t`, `cfg_store` |
| `retina/retina.h` / `.cpp` | `validate()`, `contrast_filter_t` (the filter), `polarize()` |
| `retina/retina_io.h` / `.cpp` | `retina_io` entity |
| `json_cfg/retina_json.h` | JSON adaptors (polarity via `COAST_JSON_STRICT_ENUM`) |
