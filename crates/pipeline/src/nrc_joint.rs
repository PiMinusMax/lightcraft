//! Linear colour NR with a fixed, denoised joint guide. Strength controls the spatial scale;
//! noise-normalized guide distances control which neighbours can exchange colour. Both chroma
//! components use the same weights. Symmetric fluxes preserve the grid's channel sums, including
//! at its boundaries; no nonnegative clamp or square-root bias correction is involved.

use lightcraft_color::luminance_2020;
use lightcraft_raster::{Rgb32f, blur::gaussian, resample::Filter};

use crate::{
    for_rows,
    local::{NrcNoise, NrcParams, fit_nonneg_line},
};

#[derive(Debug)]
struct JointNoise {
    model: NrcNoise,
    /// Do not extrapolate beyond bands that contain usable noise measurements.
    ymax: [f32; 3],
    tolerance: f32,
}

impl JointNoise {
    fn var(&self, i: usize, y: f32) -> f32 {
        self.model.var(i, y.min(self.ymax[i]))
    }
}

fn yuv(c: [f32; 3]) -> [f32; 3] {
    let y = luminance_2020(c);
    [y, c[0] - y, c[2] - y]
}

/// Noise of Y, R-Y and B-Y measured from small high-pass residuals, not the colour variance
/// of a wide smoothing window. The median-square calibration is for Gaussian residuals.
/// Luminance bands with predominantly texture may only lower their weight in the fit: they
/// cannot force a shadow scene's extrapolated highlight noise arbitrarily high.
fn noise_model(img: &Rgb32f) -> JointNoise {
    const BANDS: usize = 16;
    let mut samples = Vec::new();
    let stride = img.len().div_ceil(crate::local::NRC_SAMPLES).max(1) | 1;
    if img.width < 3 || img.height < 3 {
        return JointNoise { model: NrcNoise { alpha: [0.0; 3], beta: [1e-14; 3] }, ymax: [0.0; 3], tolerance: 1.0 };
    }
    for index in (0..img.len()).step_by(stride) {
        let (x, y) = (index % img.width, index / img.width);
        if x == 0 || y == 0 || x + 1 == img.width || y + 1 == img.height {
            continue;
        }
        let Some(c) = img.data.get(index) else { continue };
        let mut residual = [[0.0f32; 3]; 3];
        let mut brightness = 0.0;
        let mut valid = true;
        for (scale, delta) in [1usize, 2, 4].into_iter().enumerate() {
            // Tiny images use the nearest supported scale rather than extending an edge
            // into the noise samples.
            let delta = delta.min(x).min(y).min(img.width - x - 1).min(img.height - y - 1);
            let neighbours = [index - delta, index + delta, index - delta * img.width, index + delta * img.width];
            let mut m = [0.0f32; 3];
            for n in neighbours {
                if let Some(p) = img.data.get(n) {
                    m = std::array::from_fn(|i| m[i] + 0.25 * p[i]);
                }
            }
            residual[scale] = std::array::from_fn(|i| (c[i] - m[i]).powi(2));
            brightness += m[0] / 3.0;
            valid &= c.iter().chain(m.iter()).all(|v| v.is_finite());
        }
        if valid {
            samples.push((brightness.max(0.0), residual));
        }
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let per = samples.len().div_ceil(BANDS).max(32);
    let mut pts: [Vec<(f64, f64, f64)>; 3] = std::array::from_fn(|_| Vec::new());
    for band in samples.chunks(per).filter(|b| b.len() >= 32) {
        let y = band.iter().map(|p| p.0 as f64).sum::<f64>() / band.len() as f64;
        for (i, pts) in pts.iter_mut().enumerate() {
            let mut variance = 1e-14f64;
            for scale in 0..3 {
                let mut r: Vec<f32> = band.iter().map(|p| p.1[scale][i]).collect();
                let middle = r.len() / 2;
                let (_, median, _) = r.select_nth_unstable_by(middle, f32::total_cmp);
                // Each kernel has norm squared 1.25. Larger spacings recover noise
                // hidden by the positive spatial correlations of demosaiced RGB.
                variance = variance.max(*median as f64 / (0.454_936_4 * 1.25));
            }
            pts.push((y, variance, band.len() as f64));
        }
    }
    let mut model = NrcNoise { alpha: [0.0; 3], beta: [1e-14; 3] };
    let mut ymax = [0.0; 3];
    for (i, pts) in pts.iter().enumerate() {
        if pts.is_empty() {
            continue;
        }
        // Seed from the darker half, then include all bands with a bounded influence for
        // unexpectedly high residuals (foliage etc.). Refit rather than clamp the intercept.
        let (mut a, mut b) = fit_nonneg_line(&pts[..pts.len().div_ceil(2)]);
        for _ in 0..4 {
            let weighted: Vec<_> = pts
                .iter()
                .filter_map(|&(y, v, w)| {
                    let excess = v / (a * y + b).max(1e-14);
                    (excess <= 4.0).then_some((y, v, w * (2.0 / excess.max(2.0)).powi(4)))
                })
                .collect();
            if !weighted.is_empty() {
                (a, b) = fit_nonneg_line(&weighted);
                ymax[i] = weighted.iter().map(|p| p.0 as f32).fold(0.0, f32::max);
            }
        }
        model.alpha[i] = a as f32;
        model.beta[i] = (b as f32).max(1e-14);
    }
    JointNoise { model, ymax, tolerance: 1.0 }
}

/// Squared distance beyond the expected noise in a difference of two guide samples. A small
/// relative luminance allowance lets a smooth surface exchange colour without crossing a
/// significant luminance or isoluminant colour boundary. The guide is fixed across all scales.
fn distance(a: [f32; 3], b: [f32; 3], noise: &JointNoise) -> f32 {
    let y = 0.5 * (a[0] + b[0]);
    let allowance = (0.025 * y.max(0.0)).powi(2);
    let terms: [f32; 3] = std::array::from_fn(|i| {
        let v = noise.var(i, y);
        let excess = ((a[i] - b[i]).powi(2) - v).max(0.0);
        // The 3x3 pilot blur reduces uncorrelated noise to 1/9. Keep a conservative
        // allowance for correlations left by demosaicing and grid resampling.
        // (A difference within the noise adds exactly 0: no division needed.)
        if excess == 0.0 { 0.0 } else { excess / (3.0 * noise.tolerance * v + allowance) }
    });
    terms.into_iter().sum::<f32>()
}

/// The exchange weight `exp(−max(distance, path))`; exactly 1 when both are 0 (no `exp` needed).
#[inline]
fn weight(distance: f32, path: f32) -> f32 {
    let m = distance.max(path);
    if m == 0.0 { 1.0 } else { (-m).exp() }
}

/// Denoise the pilot along a clearly smoother axis at strong edges. An isotropic blur
/// dilutes a two-pixel isoluminant line before the edge decision even sees it. Weak or
/// ambiguous structure uses the isotropic pilot, so noise alone does not protect itself.
fn pilot(original: &Rgb32f, noise: &JointNoise) -> Rgb32f {
    let mut guide = gaussian(original, 0.8);
    let (w, h) = (original.width, original.height);
    for_rows(&mut guide.data, w, |y, row| {
        for (x, out) in row.iter_mut().enumerate() {
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                continue;
            }
            let index = y * w + x;
            let Some(&p) = original.data.get(index) else { continue };
            let mut means = [[0.0f32; 3]; 2];
            let mut variances = [0.0; 2];
            for (axis, offset) in [1, w].into_iter().enumerate() {
                let (Some(&a), Some(&b)) = (original.data.get(index - offset), original.data.get(index + offset)) else { continue };
                means[axis] = std::array::from_fn(|i| (a[i] + p[i] + b[i]) / 3.0);
                variances[axis] = (0..3)
                    .map(|i| {
                        let m = means[axis][i];
                        ((a[i] - m).powi(2) + (p[i] - m).powi(2) + (b[i] - m).powi(2)) / (9.0 * noise.var(i, p[0]))
                    })
                    .sum::<f32>();
            }
            if variances[0] > 4.0f32.max(4.0 * variances[1]) {
                *out = means[1];
            } else if variances[1] > 4.0f32.max(4.0 * variances[0]) {
                *out = means[0];
            }
        }
    });
    guide
}

/// The two chroma components `R − Y`, `B − Y`, or a pair of per-axis values (horizontal, vertical).
type Pair = [f32; 2];
type PairImage = lightcraft_raster::Image<Pair>;

fn rgb2(y: f32, uv: Pair) -> [f32; 3] {
    let r = y + uv[0];
    let b = y + uv[1];
    [r, (y - 0.2627 * r - 0.0593 * b) / 0.6780, b]
}

/// One conservative horizontal/vertical diffusion step. Pair weights are symmetric; the
/// guide is never replaced by the filtered result. Long taps also account for the intervening
/// guide edges, so a wide pass cannot jump over a thin differently coloured object.
///
/// Per pixel: `out = p`, then the lower neighbour's exchange, then the upper one's, in that order
/// (the order the result depends on, bit for bit).
fn diffuse(src: &PairImage, dst: &mut PairImage, links: &PairImage, delta: usize, vertical: bool, rate: f32) {
    if vertical { diffuse_v(src, dst, links, delta, rate) } else { diffuse_h(src, dst, links, delta, rate) }
}

#[inline(always)]
fn exchange(out: &mut Pair, p: Pair, q: Pair, weight: f32) {
    out[0] += weight * (q[0] - p[0]);
    out[1] += weight * (q[1] - p[1]);
}

fn diffuse_h(src: &PairImage, dst: &mut PairImage, links: &PairImage, delta: usize, rate: f32) {
    let w = src.width;
    for_rows(&mut dst.data, w, |y, out| {
        let (Some(s), Some(l)) = (src.data.get(y * w..(y + 1) * w), links.data.get(y * w..(y + 1) * w)) else { return };
        for (x, o) in out.iter_mut().enumerate() {
            let p = s[x];
            *o = p;
            if x >= delta {
                exchange(o, p, s[x - delta], rate * l[x - delta][0]);
            }
            if x + delta < w {
                exchange(o, p, s[x + delta], rate * l[x][0]);
            }
        }
    });
}

fn diffuse_v(src: &PairImage, dst: &mut PairImage, links: &PairImage, delta: usize, rate: f32) {
    let (w, h) = (src.width, src.height);
    for_rows(&mut dst.data, w, |y, out| {
        let Some(s) = src.data.get(y * w..(y + 1) * w) else { return };
        out.copy_from_slice(s);
        if y >= delta
            && let (Some(up), Some(lu)) = (src.data.get((y - delta) * w..(y - delta + 1) * w), links.data.get((y - delta) * w..(y - delta + 1) * w))
        {
            for (((o, &p), &q), lk) in out.iter_mut().zip(s).zip(up).zip(lu) {
                exchange(o, p, q, rate * lk[1]);
            }
        }
        if y + delta < h
            && let (Some(down), Some(ld)) = (src.data.get((y + delta) * w..(y + delta + 1) * w), links.data.get(y * w..(y + 1) * w))
        {
            for (((o, &p), &q), lk) in out.iter_mut().zip(s).zip(down).zip(ld) {
                exchange(o, p, q, rate * lk[1]);
            }
        }
    });
}

pub(crate) fn denoise(img: &mut Rgb32f, params: &NrcParams, detail: f32, amount: f32) {
    let step = std::env::var("NRV2_JOINT_STEP").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1).clamp(1, 2);
    // Continuous ramp from zero; from the ordinary default Color=25 onward the spatial
    // scale controls the extra smoothing. Noise calibration and edge decisions stay fixed.
    denoise_grid(img, params, step, detail, (4.0 * amount).clamp(0.0, 1.0));
}

fn denoise_grid(img: &mut Rgb32f, params: &NrcParams, step: usize, detail: f32, strength: f32) {
    if img.is_empty() || strength <= 0.0 {
        return;
    }
    let start = crate::profiling().then(std::time::Instant::now);
    // A maximum 2x grid keeps colour boundaries visible even at high strength. The old grid
    // grew to 16x with the radius, making the colour loss inseparable from slider strength.
    let step = if img.width.is_multiple_of(step) && img.height.is_multiple_of(step) { step } else { 1 };
    let (w, h) = (img.width.div_ceil(step), img.height.div_ceil(step));
    // Transform before resampling; negative linear differences are valid throughout.
    let mut original = Rgb32f::new(w, h);
    if step == 1 {
        // the generic loop below with one sample per pixel, same operations (0 + c, / 1)
        for_rows(&mut original.data, w, |y, row| {
            for (x, p) in row.iter_mut().enumerate() {
                if let Some(&c) = img.data.get(y * w + x) {
                    let c = yuv(c);
                    *p = c.map(|v| (0.0f32 + v) / 1.0f32);
                }
            }
        });
    } else {
        for_rows(&mut original.data, w, |y, row| {
            for (x, p) in row.iter_mut().enumerate() {
                let mut count = 0.0f32;
                for sy in y * step..((y + 1) * step).min(img.height) {
                    for sx in x * step..((x + 1) * step).min(img.width) {
                        if let Some(&c) = img.data.get(sy * img.width + sx) {
                            let c = yuv(c);
                            *p = std::array::from_fn(|i| p[i] + c[i]);
                            count += 1.0;
                        }
                    }
                }
                *p = p.map(|v| v / count.max(1.0));
            }
        });
    }
    let lap = |what: &str, t: &mut Option<std::time::Instant>| {
        if let Some(t0) = t {
            eprintln!("      joint {what}: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
            *t = Some(std::time::Instant::now());
        }
    };
    let mut tt = start;
    lap("transform", &mut tt);
    let mut noise = noise_model(&original);
    lap("noise model", &mut tt);
    noise.tolerance = (1.5 - detail).clamp(0.5, 1.5);
    let guide = pilot(&original, &noise);
    lap("pilot", &mut tt);
    if std::env::var_os("NRV2_DEBUG").is_some() {
        eprintln!("    nr joint: {w}x{h}, Y/R-Y/B-Y alpha {:?} beta {:?}, supported Y {:?}", noise.model.alpha, noise.model.beta, noise.ymax);
    }
    // Working buffers carry the two chroma components only (Y is restored from the image at the end).
    let mut filtered: PairImage = original.map(|p| [p[1], p[2]]);
    // the full-resolution reconstruction reads Y from `img`: the transformed copy is no longer needed
    let original = (step != 1).then_some(original);
    let mut scratch = PairImage::new(w, h);
    let mut paths = PairImage::new(w, h);
    let mut links = PairImage::new(w, h);
    // Direct neighbours: the path over one step is the pair's own distance, and its weight is exp(−distance).
    {
        use rayon::prelude::*;
        paths.data.par_chunks_mut(w).zip(links.data.par_chunks_mut(w)).enumerate().for_each(|(y, (prow, lrow))| {
            for (x, (out, link)) in prow.iter_mut().zip(lrow.iter_mut()).enumerate() {
                let index = y * w + x;
                let Some(&g) = guide.data.get(index) else { continue };
                *link = [0.0; 2];
                if x + 1 < w
                    && let Some(&b) = guide.data.get(index + 1)
                {
                    out[0] = distance(g, b, &noise);
                    link[0] = weight(out[0], out[0]);
                }
                if y + 1 < h
                    && let Some(&b) = guide.data.get(index + w)
                {
                    out[1] = distance(g, b, &noise);
                    link[1] = weight(out[1], out[1]);
                }
            }
        });
    }
    lap("links 1", &mut tt);
    // Scale is independent of the edge decision. K has no role in deciding whether a leaf
    // is noise. Two dyadic sweeps provide broad smoothing with a fixed number of scales.
    let sigma = params.passes.last().map_or(0.0, |p| p.sigma / step as f32);
    let mut remaining = sigma * sigma;
    let mut delta = 1usize;
    while remaining > 0.01 && delta <= w.max(h) {
        let rate = (remaining / (4.0 * (delta as f32).powi(2))).min(0.25);
        for _ in 0..2 {
            diffuse(&filtered, &mut scratch, &links, delta, false, rate);
            std::mem::swap(&mut filtered, &mut scratch);
            diffuse(&filtered, &mut scratch, &links, delta, true, rate);
            std::mem::swap(&mut filtered, &mut scratch);
        }
        remaining -= 4.0 * rate * (delta as f32).powi(2);
        let next = delta.saturating_mul(2);
        lap(&format!("scale {delta}"), &mut tt);
        if !(remaining > 0.01 && next <= w.max(h)) {
            break;
        }
        // In one pass: maxima over paths of length 2·delta (from the two adjacent delta-length paths,
        // read from `paths`, written to `scratch`) and the next scale's pair weights from them.
        {
            use rayon::prelude::*;
            let old = &paths;
            scratch.data.par_chunks_mut(w).zip(links.data.par_chunks_mut(w)).enumerate().for_each(|(y, (prow, lrow))| {
                for (x, (out, link)) in prow.iter_mut().zip(lrow.iter_mut()).enumerate() {
                    let index = y * w + x;
                    let Some(&p) = old.data.get(index) else { continue };
                    *out = p;
                    if x + delta < w
                        && let Some(q) = old.data.get(index + delta)
                    {
                        out[0] = out[0].max(q[0]);
                    }
                    if y + delta < h
                        && let Some(q) = old.data.get(index + delta * w)
                    {
                        out[1] = out[1].max(q[1]);
                    }
                    *link = [0.0; 2];
                    let Some(&g) = guide.data.get(index) else { continue };
                    for (axis, offset, valid) in [(0, next, x + next < w), (1, next * w, y + next < h)] {
                        if valid && let Some(&other) = guide.data.get(index + offset) {
                            link[axis] = weight(distance(g, other, &noise), out[axis]);
                        }
                    }
                }
            });
        }
        std::mem::swap(&mut paths, &mut scratch);
        delta = next;
    }
    drop(scratch);
    drop(paths);
    drop(links);
    match &original {
        None => {
            let fw = img.width;
            for_rows(&mut img.data, fw, |y, row| {
                for (x, p) in row.iter_mut().enumerate() {
                    let Some(&q) = filtered.data.get(y * w + x) else { continue };
                    let result = rgb2(luminance_2020(*p), q);
                    *p = std::array::from_fn(|i| p[i] + strength * (result[i] - p[i]));
                }
            });
        }
        Some(original) => {
            // Reconstruct only rows. Keep the fine residual with a shared, noise-subtracted
            // Wiener share; a flat noisy patch receives no automatic protection from noisy Y.
            let mean = gaussian(original, 1.0);
            let corr = gaussian(&original.map(|p| p.map(|v| v * v)), 1.0);
            let share = corr.zip_map(&mean, |c, m| {
                let snr = (1..3).map(|i| ((c[i] - m[i] * m[i]).max(0.0) / noise.var(i, m[0]) - 1.0).max(0.0)).sum::<f32>() * 0.5;
                [snr / (snr + 8.0); 3]
            });
            let share = gaussian(&share, 0.8);
            // Resize one output row at a time using the raster resampler's exact pixel centres.
            use lightcraft_raster::resample::weights;
            let ux = weights(w, img.width, Filter::Bilinear);
            let uy = weights(h, img.height, Filter::Bilinear);
            let ow = img.width;
            for_rows(&mut img.data, ow, |y, row| {
                let Some((y0, wy)) = uy.get(y) else { return };
                for (x, p) in row.iter_mut().enumerate() {
                    let Some((x0, wx)) = ux.get(x) else { continue };
                    let (mut f, mut o, mut a) = ([0.0; 2], [0.0; 3], 0.0);
                    for (j, &dy) in wy.iter().enumerate() {
                        for (i, &dx) in wx.iter().enumerate() {
                            let index = (y0 + j) * w + x0 + i;
                            if let (Some(q), Some(r), Some(s)) = (filtered.data.get(index), original.data.get(index), share.data.get(index)) {
                                let weight = dx * dy;
                                f = std::array::from_fn(|c| f[c] + weight * q[c]);
                                o = std::array::from_fn(|c| o[c] + weight * r[c]);
                                a += weight * s[0];
                            }
                        }
                    }
                    let c = yuv(*p);
                    let q = [f[0] + a * (c[1] - o[1]), f[1] + a * (c[2] - o[2])];
                    let result = rgb2(c[0], q);
                    *p = std::array::from_fn(|i| p[i] + strength * (result[i] - p[i]));
                }
            });
        }
    }
    if let Some(start) = start {
        eprintln!("    nr joint: {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::NrcPass;

    fn params(step: usize) -> NrcParams {
        NrcParams { off: 0.003, passes: vec![NrcPass { sigma: 4.0, step, k: 64.0, pct: 0.2 }, NrcPass { sigma: 12.0, step, k: 64.0, pct: 0.2 }] }
    }

    fn noisy(clean: &Rgb32f, seed: u64, sigma: f32) -> Rgb32f {
        let mut result = clean.clone();
        let mut seed = seed;
        for p in &mut result.data {
            for v in p {
                let mut n = 0.0;
                for _ in 0..4 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    n += (seed >> 11) as f32 / (1u64 << 53) as f32 - 0.5;
                }
                *v += sigma * 1.732 * n;
            }
        }
        result
    }

    /// The optimised filter reproduces the reference implementation (8b89526d) bit for bit: same scales,
    /// weights, paths and reconstruction, only fewer passes and smaller buffers.
    #[test]
    fn joint_matches_the_reference_bit_for_bit() {
        let mut scene = Rgb32f::new(97, 83);
        for (i, p) in scene.data.iter_mut().enumerate() {
            let (x, y) = (i % 97, i / 97);
            // edges, a thin line, a dark patch and a gradient
            *p = if x < 30 {
                [0.08, 0.016, 0.02]
            } else if x < 32 {
                [0.01, 0.05, 0.01]
            } else if y < 40 {
                [0.0015, 0.004, 0.009]
            } else {
                [0.02 + 0.001 * x as f32, 0.03, 0.01 + 0.0005 * y as f32]
            };
        }
        for (w, h) in [(97usize, 83usize), (96, 82), (5, 3), (1, 7)] {
            let mut clean = Rgb32f::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    clean.data[y * w + x] = scene.data[y * 97 + x];
                }
            }
            for seed in [0x2545_f491_4f6c_dd1d, 0x9e37_79b9_7f4a_7c15] {
                let input = noisy(&clean, seed, 0.004);
                for step in [1, 2] {
                    for (detail, strength, sigma) in [(0.5, 1.0, 12.0), (0.25, 0.6, 4.0), (0.9, 1.0, 27.0)] {
                        let p = NrcParams { off: 0.003, passes: vec![NrcPass { sigma, step, k: 64.0, pct: 0.2 }] };
                        let (mut a, mut b) = (input.clone(), input.clone());
                        denoise_grid(&mut a, &p, step, detail, strength);
                        crate::nrc_joint_reference::denoise_grid(&mut b, &p, step, detail, strength);
                        for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
                            for c in 0..3 {
                                assert!(
                                    x[c].to_bits() == y[c].to_bits(),
                                    "{w}x{h} step {step} sigma {sigma}: pixel {i} channel {c}: {} vs {}",
                                    x[c],
                                    y[c]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// BENCH (ignored, prints): the optimised filter against the reference on the same 24 MP image,
    /// interleaved so both see the same machine load. Median of 3 each.
    #[test]
    #[ignore]
    fn bench_joint_against_reference() {
        let (w, h) = (6000usize, 4000usize);
        let mut clean = Rgb32f::new(w, h);
        for (i, p) in clean.data.iter_mut().enumerate() {
            let (x, y) = (i % w, i / w);
            *p = if (x / 300 + y / 300) % 2 == 0 { [0.03, 0.02, 0.01] } else { [0.004, 0.006, 0.012] };
        }
        let input = noisy(&clean, 0x2545_f491_4f6c_dd1d, 0.004);
        let p = NrcParams { off: 0.003, passes: vec![NrcPass { sigma: 27.0, step: 1, k: 64.0, pct: 0.2 }] };
        let (mut new, mut old) = (Vec::new(), Vec::new());
        for _ in 0..3 {
            let mut a = input.clone();
            let t = std::time::Instant::now();
            denoise_grid(&mut a, &p, 1, 0.5, 1.0);
            new.push(t.elapsed().as_secs_f64());
            let mut b = input.clone();
            let t = std::time::Instant::now();
            crate::nrc_joint_reference::denoise_grid(&mut b, &p, 1, 0.5, 1.0);
            old.push(t.elapsed().as_secs_f64());
        }
        new.sort_by(f64::total_cmp);
        old.sort_by(f64::total_cmp);
        println!("joint 24 MP: optimised {:.2} s, reference {:.2} s (medians of {new:.2?} / {old:.2?})", new[1], old[1]);
    }

    #[test]
    fn joint_keeps_negative_means_luminance_and_reduces_chroma_noise() {
        for (w, h) in [(128, 96), (127, 95), (1, 19), (19, 1)] {
            let clean = Rgb32f::filled(w, h, [0.006, 0.004, 0.0008]);
            for seed in [0x2545_f491_4f6c_dd1d, 0x1234_5678_9abc_def1] {
                let input = noisy(&clean, seed, 0.003);
                for step in [1, 2] {
                    let mut output = input.clone();
                    denoise_grid(&mut output, &params(step), step, 0.5, 1.0);
                    assert!(output.data.iter().flatten().all(|v| v.is_finite()));
                    for (a, b) in input.data.iter().zip(&output.data) {
                        assert!((luminance_2020(*a) - luminance_2020(*b)).abs() < 1e-7);
                    }
                    // Thin inputs cannot support a spatial noise estimate; finite/luminance
                    // checks above still apply, but their sample mean is too noisy for this bound.
                    if w < 3 || h < 3 {
                        continue;
                    }
                    for c in 0..3 {
                        let difference = input.data.iter().zip(&output.data).map(|(a, b)| (b[c] - a[c]) as f64).sum::<f64>() / input.len() as f64;
                        assert!(difference.abs() < 2e-6, "step {step}, {w}x{h}, mean error {difference}");
                    }
                    let energy = |im: &Rgb32f| {
                        im.data
                            .iter()
                            .map(|&p| {
                                let p = yuv(p);
                                let c = yuv(clean.data[0]);
                                (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)
                            })
                            .sum::<f32>()
                            / im.len() as f32
                    };
                    assert!(energy(&output) < 0.08 * energy(&input));
                    assert!(output.data.iter().any(|p| p.iter().any(|&v| v < 0.0)), "negative values were clipped");
                }
            }
        }
    }

    #[test]
    fn joint_protects_leaf_and_isoluminant_edges_without_chroma_halos() {
        let leaf = [0.02, 0.12, 0.01];
        let black = [0.0005; 3];
        let red = [0.08, 0.0155, 0.02];
        let mut green = [0.012, 0.04, 0.015];
        let gain = luminance_2020(red) / luminance_2020(green);
        green = green.map(|v| v * gain);
        for (left, right) in [(leaf, black), (red, green)] {
            let clean = Rgb32f::from_fn(192, 128, |x, _| if x < 96 { left } else { right });
            for seed in [0x2545_f491_4f6c_dd1d, 0x1234_5678_9abc_def1] {
                let input = noisy(&clean, seed, 0.003);
                for step in [1, 2] {
                    let mut output = input.clone();
                    denoise_grid(&mut output, &params(step), step, 0.5, 1.0);
                    let step_uv = [yuv(left)[1] - yuv(right)[1], yuv(left)[2] - yuv(right)[2]];
                    let length = step_uv.iter().map(|v| v * v).sum::<f32>().sqrt();
                    for x in [92, 93, 98, 99] {
                        let mut mean = [0.0; 3];
                        for y in 16..112 {
                            let p = yuv(output.get(x, y));
                            mean = std::array::from_fn(|i| mean[i] + p[i] / 96.0);
                        }
                        let expected = yuv(if x < 96 { left } else { right });
                        let error = ((mean[1] - expected[1]).powi(2) + (mean[2] - expected[2]).powi(2)).sqrt() / length;
                        assert!(error < 0.08, "step {step}, x {x}, colour error {}%", 100.0 * error);
                    }
                }
            }
        }
    }

    #[test]
    fn joint_noise_fit_does_not_treat_bright_texture_as_shot_noise() {
        let clean = Rgb32f::from_fn(256, 192, |x, y| {
            let l = if x < 192 { 0.003 + x as f32 * 0.00002 } else { 0.15 + if (x + y) % 2 == 0 { 0.08 } else { -0.08 } };
            [l; 3]
        });
        let input = noisy(&clean, 0x2545_f491_4f6c_dd1d, 0.002).map(yuv);
        let model = noise_model(&input);
        for i in 0..3 {
            assert!(model.var(i, 0.15) < 5.0 * model.var(i, 0.005), "texture dominated noise fit: {model:?}");
        }
    }

    #[test]
    fn joint_keeps_a_two_pixel_isoluminant_line() {
        let grey = [0.03; 3];
        let mut red = [0.08, 0.0155, 0.02];
        let gain = luminance_2020(grey) / luminance_2020(red);
        red = red.map(|v| v * gain);
        let clean = Rgb32f::from_fn(192, 128, |x, _| if (94..96).contains(&x) { red } else { grey });
        let mut output = noisy(&clean, 0x2545_f491_4f6c_dd1d, 0.006);
        denoise_grid(&mut output, &params(1), 1, 0.5, 1.0);
        let reference = yuv(red);
        let mut mean = [0.0; 3];
        for y in 16..112 {
            for x in 94..96 {
                let p = yuv(output.get(x, y));
                mean = std::array::from_fn(|i| mean[i] + p[i] / 192.0);
            }
        }
        let kept = (mean[1] * reference[1] + mean[2] * reference[2]) / (reference[1].powi(2) + reference[2].powi(2));
        assert!(kept > 0.85, "thin line keeps only {}%", 100.0 * kept);
    }

    #[test]
    fn joint_estimates_correlated_noise_at_a_matching_scale() {
        let clean = Rgb32f::filled(256, 192, [0.03; 3]);
        let correlated = gaussian(&noisy(&clean, 0x2545_f491_4f6c_dd1d, 0.012), 0.8).map(yuv);
        let model = noise_model(&correlated);
        for i in 0..3 {
            let mean = correlated.data.iter().map(|p| p[i] as f64).sum::<f64>() / correlated.len() as f64;
            let variance = correlated.data.iter().map(|p| (p[i] as f64 - mean).powi(2)).sum::<f64>() / correlated.len() as f64;
            let predicted = model.var(i, 0.03) as f64;
            assert!(predicted > 0.6 * variance && predicted < 1.6 * variance, "channel {i}: estimated {predicted}, actual {variance}");
        }
    }
}
