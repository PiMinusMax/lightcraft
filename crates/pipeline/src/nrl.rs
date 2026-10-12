//! Luminance noise reduction (prototype for #1129). An à trous wavelet of linear Y; each detail level
//! is shrunk against the brightness-dependent noise model of the colour NR (the same samples and fit,
//! channel Y), and the change of Y is added to R, G and B alike. Adding instead of scaling keeps the
//! chroma and lets black or negative pixels move; a wide blur of the removed part puts back the
//! local mean that shrinking skewed noise tails would move.
//!
//! Amount sets the threshold (in noise sigmas of the level), Detail lowers it on the two fine levels,
//! Contrast on the three coarse ones.

use lightcraft_color::luminance_2020;
use lightcraft_raster::{Plane, Rgb32f, blur::gaussian};

use crate::{
    for_rows,
    nrc_joint::{JointNoise, noise_from_samples, noise_samples_of, yuv},
};

/// Detail levels (holes 1, 2, 4, 8, 16 px).
pub const LEVELS: usize = 5;
/// Levels with a robust (median) local energy; coarser levels average.
const ROBUST_LEVELS: usize = 2;
/// Blur of the removed part whose mean is put back (px).
const BIAS_SIGMA: f32 = 12.0;

/// The sliders, 0–1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub amount: f32,
    pub detail: f32,
    pub contrast: f32,
}

/// The new luminance noise reduction is on unless `LIGHTCRAFT_NRL=0` (prototype switch).
pub fn enabled() -> bool {
    std::env::var("LIGHTCRAFT_NRL").map_or(true, |v| v != "0")
}

/// Gaussian window of the local energy of coarse level `j` (px).
fn energy_sigma(j: usize) -> f32 {
    (2.0 * (1u32 << j) as f32 + 1.0) / 12f32.sqrt()
}

/// How far an output pixel reads its neighbours (px): the coarsest detail level, its energy window
/// and the bias blur. A window rendered on its own needs this much context around it.
pub fn halo() -> usize {
    let reach = |sigma: f32| lightcraft_raster::blur::box_radii(sigma).iter().sum::<usize>();
    // the à trous steps: 2 · 2^j on each side
    let wavelet: usize = (0..LEVELS).map(|j| 2usize << j).sum();
    wavelet + reach(energy_sigma(LEVELS - 1)) + reach(BIAS_SIGMA) + 4
}

/// Threshold of level `j` in noise sigmas of that level.
pub fn threshold(p: &Params, j: usize) -> f32 {
    let t = (2.0 * p.amount.clamp(0.0, 1.0)).sqrt();
    let f = if j < ROBUST_LEVELS { 1.25 - 0.5 * p.detail.clamp(0.0, 1.0) } else { 1.0 - 0.8 * p.contrast.clamp(0.0, 1.0) };
    t * f
}

const B3: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Mirror `i` into `0..n` (without repeating the edge sample).
#[inline]
fn mirror(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    if i < 0 {
        i = -i;
    }
    if i >= n {
        i = 2 * n - 2 - i;
    }
    i.clamp(0, n - 1) as usize
}

/// `line` with `m` mirrored samples on each side, into `pad` (len + 2m).
fn pad_row(line: &[f32], m: usize, pad: &mut Vec<f32>) {
    let n = line.len();
    pad.clear();
    pad.extend((0..m).map(|k| line[mirror(k as isize - m as isize, n)]));
    pad.extend_from_slice(line);
    pad.extend((0..m).map(|k| line[mirror((n + k) as isize, n)]));
}

/// One à trous step: separable B3 spline with holes of `2^j` px, mirrored at the borders.
fn smooth(src: &Plane, j: usize) -> Plane {
    let (w, h) = (src.width, src.height);
    let s = 1usize << j;
    let mut tmp = Plane::new(w, h);
    for_rows(&mut tmp.data, w, |y, row| {
        let mut pad = Vec::with_capacity(w + 4 * s);
        pad_row(&src.data[y * w..(y + 1) * w], 2 * s, &mut pad);
        let taps: [&[f32]; 5] = std::array::from_fn(|k| &pad[k * s..k * s + w]);
        for (x, o) in row.iter_mut().enumerate() {
            *o = B3[0] * taps[0][x] + B3[1] * taps[1][x] + B3[2] * taps[2][x] + B3[3] * taps[3][x] + B3[4] * taps[4][x];
        }
    });
    let mut out = Plane::new(w, h);
    for_rows(&mut out.data, w, |y, row| {
        let rows: [&[f32]; 5] = std::array::from_fn(|k| {
            let r = mirror(y as isize + (k as isize - 2) * s as isize, h);
            &tmp.data[r * w..(r + 1) * w]
        });
        for (x, o) in row.iter_mut().enumerate() {
            *o = B3[0] * rows[0][x] + B3[1] * rows[1][x] + B3[2] * rows[2][x] + B3[3] * rows[3][x] + B3[4] * rows[4][x];
        }
    });
    out
}

/// Median of nine (Devillard's 19-exchange network).
#[inline]
fn med9(mut p: [f32; 9]) -> f32 {
    let mut s = |a: usize, b: usize| {
        let (lo, hi) = (p[a].min(p[b]), p[a].max(p[b]));
        p[a] = lo;
        p[b] = hi;
    };
    for (a, b) in [
        (1, 2),
        (4, 5),
        (7, 8),
        (0, 1),
        (3, 4),
        (6, 7),
        (1, 2),
        (4, 5),
        (7, 8),
        (0, 3),
        (5, 8),
        (4, 7),
        (3, 6),
        (1, 4),
        (2, 5),
        (4, 7),
        (4, 2),
        (6, 4),
        (4, 2),
    ] {
        s(a, b);
    }
    p[4]
}

/// Local energy of the detail plane `d` (level `j`) as an estimate of its mean square: on the fine levels the
/// median of `d²` over 9 samples `2^(j+1)` px apart (an isolated outlier does not protect itself), scaled by
/// the median of χ²(1); on the coarse levels a Gaussian mean.
fn energy(d: &Plane, j: usize) -> Plane {
    let (w, h) = (d.width, d.height);
    if j < ROBUST_LEVELS {
        // median(d²) = median(|d|)²
        let s = 2usize << j;
        let mut e = Plane::new(w, h);
        for_rows(&mut e.data, w, |y, row| {
            let mut pads: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(w + 2 * s));
            for (k, pad) in pads.iter_mut().enumerate() {
                let r = mirror(y as isize + (k as isize - 1) * s as isize, h);
                pad_row(&d.data[r * w..(r + 1) * w], s, pad);
                pad.iter_mut().for_each(|v| *v = v.abs());
            }
            let t: [&[f32]; 9] = std::array::from_fn(|k| &pads[k / 3][(k % 3) * s..(k % 3) * s + w]);
            for (x, o) in row.iter_mut().enumerate() {
                let m = med9([t[0][x], t[1][x], t[2][x], t[3][x], t[4][x], t[5][x], t[6][x], t[7][x], t[8][x]]);
                *o = m * m / 0.454_936_4;
            }
        });
        e
    } else {
        let sq = d.map(|v| v * v);
        gaussian(&sq, energy_sigma(j))
    }
}

/// What the luminance NR measures before it filters: the noise model of Y and the noise gain of each
/// level. A windowed render sees only part of the frame, so both come from fixed tiles of the whole
/// frame ([`tiles`]), the same for every window.
#[derive(Clone, Debug)]
pub struct Stats {
    pub noise: JointNoise,
    /// `E[d_j²] / var(Y)` of the noise on level `j`.
    pub gains: [f32; LEVELS],
}

/// Side of the tiles the statistics are measured on (px).
const TILE: usize = 128;
/// Tiles per row and column: many small tiles follow a mostly dark frame's few bright parts better
/// than a few large ones.
const GRID: usize = 12;

/// The windows of a `full_w × full_h` frame the statistics are measured on: the whole frame up to the
/// area of `GRID²` tiles, else `GRID × GRID` tiles spread evenly over it.
pub fn tiles(full_w: usize, full_h: usize) -> Vec<crate::PixelWindow> {
    if full_w * full_h <= GRID * GRID * TILE * TILE || full_w < 2 * TILE || full_h < 2 * TILE {
        return vec![crate::PixelWindow { x: 0, y: 0, w: full_w, h: full_h }];
    }
    let at = |n: usize, k: usize| ((n - TILE) as f64 * (k as f64 + 0.5) / GRID as f64).round() as usize;
    (0..GRID * GRID).map(|k| crate::PixelWindow { x: at(full_w, k % GRID), y: at(full_h, k / GRID), w: TILE, h: TILE }).collect()
}

/// Brightness for the noise level: the second smoothing (a few px), so each side of an edge keeps
/// its own level.
fn levels_of(img: &Rgb32f) -> (Plane, Plane, Plane) {
    let y = Plane { width: img.width, height: img.height, data: img.data.iter().map(|&c| luminance_2020(c)).collect() };
    let c1 = smooth(&y, 0);
    let c2 = smooth(&c1, 1);
    (y, c1, c2)
}

/// The noise model and level gains of `tiles` (white-balanced linear RGB at output scale).
///
/// Level gain: `d_j² / var(c2)` at the noise samples, median per brightness band (16 bands), the
/// 30 % quantile of the bands. Texture only raises some bands; the quantile follows the flat ones.
pub fn stats(tiles: &[Rgb32f]) -> Stats {
    let tiles: Vec<&Rgb32f> = tiles.iter().filter(|t| t.width >= 8 && t.height >= 8).collect();
    let stride = crate::nrc_joint::noise_sample_stride(tiles.iter().map(|t| t.len()).sum());
    let samples = tiles.iter().flat_map(|t| noise_samples_of(t, stride, yuv)).collect();
    let noise = noise_from_samples(samples);
    // brightness and the detail of each level at every stride-th pixel of each tile
    let mut bright: Vec<f32> = Vec::new();
    let mut detail: [Vec<f32>; LEVELS] = std::array::from_fn(|_| Vec::new());
    for t in &tiles {
        let (y, c1, c2) = levels_of(t);
        let idx: Vec<usize> = (0..t.len()).step_by(stride.max(1)).collect();
        bright.extend(idx.iter().map(|&i| c2.data[i]));
        let mut levels = vec![y, c1, c2];
        for (j, d) in detail.iter_mut().enumerate() {
            if levels.len() < j + 2 {
                let next = smooth(&levels[j], j);
                levels.push(next);
            }
            d.extend(idx.iter().map(|&i| levels[j].data[i] - levels[j + 1].data[i]));
        }
    }
    let mut order: Vec<usize> = (0..bright.len()).collect();
    order.sort_by(|&a, &b| bright[a].total_cmp(&bright[b]).then(a.cmp(&b)));
    let var: Vec<f32> = order.iter().map(|&i| noise.var(0, bright[i])).collect();
    const BANDS: usize = 16;
    let per = order.len().div_ceil(BANDS).max(32);
    let gains = detail.map(|d| {
        let mut med: Vec<f32> = Vec::new();
        let mut r: Vec<f32> = Vec::with_capacity(per);
        for (band, var) in order.chunks(per).zip(var.chunks(per)) {
            if band.len() < 32 {
                continue;
            }
            r.clear();
            r.extend(band.iter().zip(var).map(|(&i, &v)| d[i] * d[i] / v));
            let m = r.len() / 2;
            let (_, v, _) = r.select_nth_unstable_by(m, f32::total_cmp);
            med.push(*v / 0.454_936_4);
        }
        med.sort_by(f32::total_cmp);
        med.get(((med.len().max(1) - 1) as f32 * 0.3).round() as usize).copied().unwrap_or(0.0)
    });
    Stats { noise, gains }
}

/// Luminance noise reduction of linear RGB in place, with `stats` of the whole frame (`None`:
/// measured on `img`).
pub fn denoise(img: &mut Rgb32f, p: &Params, stats: Option<&Stats>) {
    if img.width < 8 || img.height < 8 || p.amount <= 0.0 {
        return;
    }
    let start = crate::profiling().then(std::time::Instant::now);
    let mut tt = start;
    let lap = |what: &str, t: &mut Option<std::time::Instant>| {
        if let Some(t0) = t {
            eprintln!("      nrl {what}: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
            *t = Some(std::time::Instant::now());
        }
    };
    let own;
    let Stats { noise, gains } = match stats {
        Some(s) => s,
        None => {
            own = self::stats(std::slice::from_ref(&*img));
            &own
        }
    };
    lap("stats", &mut tt);
    let (w, h) = (img.width, img.height);
    let (y, c1, c2) = levels_of(img);
    let var = c2.map(|v| noise.var(0, v));
    lap("y, c1, c2, var", &mut tt);
    let mut out = Plane::new(w, h);
    let mut levels: Vec<Plane> = vec![y.clone(), c1, c2];
    for (j, gain) in gains.iter().enumerate() {
        if levels.len() < j + 2 {
            let next = smooth(&levels[j], j);
            levels.push(next);
        }
        lap("smooth", &mut tt);
        let d = levels[j].zip_map(&levels[j + 1], |a, b| a - b);
        lap("detail", &mut tt);
        let e = energy(&d, j);
        lap("energy", &mut tt);
        let t = threshold(p, j);
        let k = t * t * gain;
        for_rows(&mut out.data, w, |row, o| {
            let i0 = row * w;
            for (x, o) in o.iter_mut().enumerate() {
                let i = i0 + x;
                let g = (1.0 - k * var.data[i] / e.data[i].max(1e-30)).max(0.0);
                *o += d.data[i] * g;
            }
        });
        // the finer approximation is no longer needed (y is kept for the bias correction)
        if j > 0 {
            levels[j] = Plane::new(0, 0);
        }
    }
    let coarse = levels.pop().unwrap_or_else(|| Plane::new(w, h));
    for (o, c) in out.data.iter_mut().zip(&coarse.data) {
        *o += c;
    }
    lap("shrink", &mut tt);
    let removed = y.zip_map(&out, |a, b| a - b);
    let back = gaussian(&removed, BIAS_SIGMA);
    for_rows(&mut img.data, w, |row, px| {
        let i0 = row * w;
        for (x, c) in px.iter_mut().enumerate() {
            let i = i0 + x;
            let dy = out.data[i] + back.data[i] - y.data[i];
            // no new negative values: the undershoot next to edges would reach the later stages' floors
            *c = c.map(|v| (v + dy).max(v.min(0.0)));
        }
    });
    lap("bias + apply", &mut tt);
    if let Some(t) = start {
        eprintln!(
            "    nr luminance (wavelet): {:.0} ms, Y noise alpha {:.3e} beta {:.3e}, level gains {:?}",
            t.elapsed().as_secs_f64() * 1e3,
            noise.model.alpha[0],
            noise.model.beta[0],
            gains
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noisy(w: usize, h: usize, level: impl Fn(usize, usize) -> f32, sigma: impl Fn(f32) -> f32) -> (Rgb32f, Rgb32f) {
        let mut seed = 0x9e37_79b9u32;
        let mut normal = || {
            let mut u = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed as f64 + 0.5) / (u32::MAX as f64 + 1.0)
            };
            ((-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()) as f32
        };
        let clean = Rgb32f::from_fn(w, h, |x, y| [level(x, y); 3]);
        let mut img = clean.clone();
        for p in &mut img.data {
            let n = sigma(p[0]) * normal();
            *p = p.map(|v| v + n);
        }
        (clean, img)
    }

    fn stats(img: &Rgb32f, xs: std::ops::Range<usize>, ys: std::ops::Range<usize>) -> (f32, f32) {
        let v: Vec<f32> = ys.flat_map(|y| xs.clone().map(move |x| (x, y))).map(|(x, y)| luminance_2020(img.data[y * img.width + x])).collect();
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (m, (v.iter().map(|a| (a - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt())
    }

    /// Shot plus read noise on a dark and a bright half: the noise falls on both, the mean of the dark
    /// half stays (the old filter darkened it and left most of its noise).
    #[test]
    fn reduces_noise_in_shadows_and_keeps_the_mean() {
        let (_, mut img) = noisy(256, 128, |x, _| if x < 128 { 0.004 } else { 0.2 }, |y| (0.002 * y + 4e-6).sqrt());
        let before = [stats(&img, 16..112, 16..112), stats(&img, 144..240, 16..112)];
        denoise(&mut img, &Params { amount: 0.5, detail: 0.5, contrast: 0.0 }, None);
        let after = [stats(&img, 16..112, 16..112), stats(&img, 144..240, 16..112)];
        for (b, a) in before.iter().zip(&after) {
            assert!(a.1 < 0.5 * b.1, "noise {} -> {}", b.1, a.1);
            assert!((a.0 - b.0).abs() < 0.01 * b.0 + 2e-5, "mean {} -> {}", b.0, a.0);
        }
    }

    /// A pixel that is black before the NR moves with its surroundings.
    #[test]
    fn black_pixels_are_not_kept_black() {
        let (_, mut img) = noisy(64, 64, |_, _| 0.05, |y| (0.002 * y + 4e-6).sqrt());
        img.data[32 * 64 + 32] = [0.0; 3];
        denoise(&mut img, &Params { amount: 1.0, detail: 0.5, contrast: 0.0 }, None);
        assert!(luminance_2020(img.data[32 * 64 + 32]) > 0.01);
    }

    /// Contrast and Detail act on different levels.
    #[test]
    fn detail_and_contrast_change_the_result() {
        let (_, img) = noisy(128, 128, |x, y| 0.1 + 0.02 * ((x / 16 + y / 16) % 2) as f32, |y| (0.002 * y + 4e-6).sqrt());
        let run = |detail, contrast| {
            let mut a = img.clone();
            denoise(&mut a, &Params { amount: 0.5, detail, contrast }, None);
            a
        };
        let base = run(0.5, 0.0);
        assert_ne!(base.data, run(0.5, 1.0).data);
        assert_ne!(base.data, run(1.0, 0.0).data);
    }

    /// Scaling the input scales the output (exposure is applied after the filter).
    #[test]
    fn scaling_the_input_scales_the_output() {
        let src = lightcraft_scenes::demo_library()[0].render(200, 133);
        let g = 2f32.powf(1.3);
        let p = Params { amount: 0.3, detail: 0.5, contrast: 0.0 };
        let mut a = src.clone();
        denoise(&mut a, &p, None);
        let mut b = src.map(|c| c.map(|v| v * g));
        denoise(&mut b, &p, None);
        for (x, y) in a.data.iter().zip(&b.data) {
            for k in 0..3 {
                assert!((x[k] - y[k] / g).abs() <= 1e-3 * x[k].abs().max(1e-3), "{x:?} vs {:?}", y.map(|v| v / g));
            }
        }
    }

    #[test]
    fn med9_is_the_median() {
        let v = [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0];
        assert_eq!(med9(v), 5.0);
    }
}

