# Colour NR: experimental joint guide

The `nr-color-v2` worktree contains an opt-in alternative to the per-channel linear V2.
Enable it with `NRV2_JOINT=1` (linear mode must also be enabled). It is a CPU reference
experiment, not a release default. GPU renders use the existing linear-mode CPU fallback.

## Method

- Filter signed, linear `R-Y` and `B-Y`, with original Rec.2020 luminance restored at each
  pixel. No channel clipping, root transform or Jensen correction is used in this filter.
- Estimate Y/chroma noise from median squared high-pass residuals at spacings 1, 2 and 4.
  Larger spacings account for positive correlations from demosaicing. Fit equal-count
  luminance bands with nonnegative coefficients, reject texture-dominated outliers and
  stop extrapolating beyond the bands supported by usable noise measurements.
- Make a fixed, lightly denoised guide. At a clear directional edge, average along the
  smoother axis; this keeps thin isoluminant lines visible to the guide.
- Apply conservative, symmetric colour exchanges at increasing spatial scales. The two
  chroma components have identical neighbour weights. Subtract the expected noise from
  guide distances; luminance texture cannot simply increase the raw-pixel retention factor.
  Long taps also check the guide edges between their endpoints.
- Symmetric exchanges preserve channel sums on the full-resolution grid, including its
  borders. This does not promise identical means inside arbitrary cropped regions.
- Colour amount controls the radius and ramps the filtered contribution continuously from
  zero to the normal default of 25. Colour Detail changes edge tolerance; Smoothness changes
  radius. `NRV2_K`, `NRV2_SHARED`, `NRV2_LUMGUIDE` and `NRV2_LUMG_OFF` belong to the preceding
  V2 experiment and do not tune this guide.
- Windowed 8-bit renders request extra surrounding pixels for the diffusion's spatial
  dependencies. The existing window/full-frame comparison retains its original tolerance.

## Local checks

```sh
NRV2_JOINT=1 cargo test --release -p lightcraft-pipeline --lib
NRV2_JOINT=1 cargo test --release -p lightcraft-pipeline --lib probe_nrc -- --ignored --nocapture
NRV2_JOINT=1 LIGHTCRAFT_GPU=0 cargo run --release -p lightcraft-cli -- render INPUT.ARW -o OUTPUT.png --set detail.nrColor=100
```

`NRV2_DEBUG=1` prints the fitted Y/R-Y/B-Y model and the supported luminance range.
`LIGHTCRAFT_PROFILE=1` prints the filter time. `NRV2_JOINT=0` selects the preceding V2;
`NRV2_LINEAR=0` selects the older root-based V2.

`NRV2_JOINT_STEP=2` is a separate experimental memory-saving grid. The full-resolution
default is 1. The reduced grid loses more thin colour detail and its window alignment has
not been verified; use the default for quality comparisons.

Five ordinary regression tests cover signed dark means, luminance, chroma-noise reduction,
leaf/black and isoluminant edges, a two-pixel isoluminant line, structured highlights and
spatially correlated noise. The existing ignored probes also cover three noise seeds and
two strengths. The full pipeline suite was executed with joint mode enabled.

## Comparing real exports

```sh
python3 tools/nrc-metrics.py REF.png VARIANT.png --crop 1325 2085 360 240
```

The script uses four actual spatial quadrants for flat-tile selection, one reference mask
for all variants, and reports both mean a*/b* colour and average per-pixel C*. The latter
alone can reward residual noise. PNG measurements describe the displayed result relative
to a clipped, noisy reference; they are not float ground truth. The candle reference yields
few genuinely flat dark tiles, so its noise scores have limited coverage.

The camera-look near-black guard from the user's test package `40b889f3` has also been
brought into this worktree. Without it, source decoding already creates isolated white
pixels, invalidating comparisons with that package. Its existing regression tests are
retained alongside the guard.

## Remaining work

See [`nr-color-v2-history.md`](nr-color-v2-history.md) for everything tried, measured and dropped, the speed
numbers and the known limits (no outlier brake for isolated colour specks yet; flat dark areas smoothed a bit
less than the root-based V2; little slider effect above 25; prototype switches instead of a process version).
