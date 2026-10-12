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
    nrc_joint::{JointNoise, noise_model_of, yuv},
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
        gaussian(&sq, (2.0 * (1u32 << j) as f32 + 1.0) / 12f32.sqrt())
    }
}

/// Noise gain of each level: `d_j² / var(y)` at the noise samples, median per brightness band
/// (16 bands of the local brightness `bright`), the 30 % quantile of the bands. Texture only
/// raises some bands; the quantile follows the flat ones.
struct Bands {
    /// sample indices in brightness order
    idx: Vec<usize>,
    var: Vec<f32>,
}

impl Bands {
    fn new(bright: &Plane, noise: &JointNoise) -> Bands {
        let stride = crate::nrc_joint::noise_sample_stride(bright.len());
        let mut idx: Vec<usize> = (0..bright.len()).step_by(stride).collect();
        idx.sort_by(|&a, &b| bright.data[a].total_cmp(&bright.data[b]).then(a.cmp(&b)));
        let var = idx.iter().map(|&i| noise.var(0, bright.data[i])).collect();
        Bands { idx, var }
    }

    fn gain(&self, d: &Plane) -> f32 {
        const BANDS: usize = 16;
        let per = self.idx.len().div_ceil(BANDS).max(32);
        let mut med: Vec<f32> = Vec::new();
        let mut r: Vec<f32> = Vec::with_capacity(per);
        for (band, var) in self.idx.chunks(per).zip(self.var.chunks(per)) {
            if band.len() < 32 {
                continue;
            }
            r.clear();
            r.extend(band.iter().zip(var).map(|(&i, &v)| d.data[i] * d.data[i] / v));
            let m = r.len() / 2;
            let (_, v, _) = r.select_nth_unstable_by(m, f32::total_cmp);
            med.push(*v / 0.454_936_4);
        }
        if med.is_empty() {
            return 0.0;
        }
        med.sort_by(f32::total_cmp);
        med[((med.len() - 1) as f32 * 0.3).round() as usize]
    }
}

/// Luminance noise reduction of linear RGB in place.
pub fn denoise(img: &mut Rgb32f, p: &Params) {
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
    let (w, h) = (img.width, img.height);
    let noise = noise_model_of(img, yuv);
    lap("noise model", &mut tt);
    let y = Plane { width: w, height: h, data: img.data.iter().map(|&c| luminance_2020(c)).collect() };
    let c1 = smooth(&y, 0);
    let c2 = smooth(&c1, 1);
    // brightness for the noise level: a few px, so each side of an edge keeps its own level
    let bands = Bands::new(&c2, &noise);
    let var = c2.map(|v| noise.var(0, v));
    lap("y, c1, c2, bands, var", &mut tt);
    let mut out = Plane::new(w, h);
    let mut gains = [0.0f32; LEVELS];
    let mut levels: Vec<Plane> = vec![y.clone(), c1, c2];
    for (j, gain) in gains.iter_mut().enumerate() {
        if levels.len() < j + 2 {
            let next = smooth(&levels[j], j);
            levels.push(next);
        }
        lap("smooth", &mut tt);
        let d = levels[j].zip_map(&levels[j + 1], |a, b| a - b);
        *gain = bands.gain(&d);
        lap("detail + gain", &mut tt);
        let e = energy(&d, j);
        lap("energy", &mut tt);
        let t = threshold(p, j);
        let k = t * t * *gain;
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
            *c = c.map(|v| v + dy);
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
        denoise(&mut img, &Params { amount: 0.5, detail: 0.5, contrast: 0.0 });
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
        denoise(&mut img, &Params { amount: 1.0, detail: 0.5, contrast: 0.0 });
        assert!(luminance_2020(img.data[32 * 64 + 32]) > 0.01);
    }

    /// Contrast and Detail act on different levels.
    #[test]
    fn detail_and_contrast_change_the_result() {
        let (_, img) = noisy(128, 128, |x, y| 0.1 + 0.02 * ((x / 16 + y / 16) % 2) as f32, |y| (0.002 * y + 4e-6).sqrt());
        let run = |detail, contrast| {
            let mut a = img.clone();
            denoise(&mut a, &Params { amount: 0.5, detail, contrast });
            a
        };
        let base = run(0.5, 0.0);
        assert_ne!(base.data, run(0.5, 1.0).data);
        assert_ne!(base.data, run(1.0, 0.0).data);
    }

    #[test]
    fn med9_is_the_median() {
        let v = [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0];
        assert_eq!(med9(v), 5.0);
    }
}
