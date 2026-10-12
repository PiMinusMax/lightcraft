//! Scene-linear preparation: white balance + exposure, and the spatial planes the per-pixel stage
//! needs (edge-aware base layer for highlights/shadows, clarity/texture bands, dehaze veil).

use lightcraft_color::cct::{temp_tint_to_xy, wb_matrix, xy_to_temp_tint};
use lightcraft_color::{REC2020, luminance_2020};
use lightcraft_develop::{DevelopSettings, Process};
use lightcraft_raster::blur::gaussian;
use std::sync::Arc;

use lightcraft_raster::{Plane, Rgb32f, par_join};

use crate::geometry::Frame;
use crate::{Prepared, Quality, SourceInfo, for_rows, masks, timed};

/// White balance (relative to the source's as-shot white) and exposure, in place.
pub fn scene_linear_pre(img: &mut Rgb32f, info: &SourceInfo, s: &DevelopSettings) {
    wb_gain(img, info, s, 2f32.powf(s.light.exposure as f32));
}

/// White balance only (exposure is applied by the per-pixel stage, see [`crate::Prepared`]).
pub fn white_balance(img: &mut Rgb32f, info: &SourceInfo, s: &DevelopSettings) {
    wb_gain(img, info, s, 1.0);
}

/// The white-balance matrix (linear Rec.2020) for the settings, or `None` when the as-shot white
/// is kept. A raw source with its camera colour model ([`crate::CameraColor`]) is re-developed for
/// the new white in camera space, as Lightroom does (so a neutral under the chosen white renders
/// neutral and saturated colours move as the camera's matrices say); other sources are adapted
/// with Bradford, luminance-preserving.
pub fn wb_matrix_for(info: &SourceInfo, s: &DevelopSettings) -> Option<[[f32; 3]; 3]> {
    let (t, tint) = effective_wb(info, s);
    if (t - info.as_shot_temp).abs() < 1e-6 && (tint - info.as_shot_tint).abs() < 1e-6 {
        return None;
    }
    if let Some(cc) = info.camera_color.as_deref().filter(|_| info.raw && !info.relative_wb)
        && let Some(m) = lightcraft_raw::color::rebalance(&cc.tags, cc.developed_for, temp_tint_to_xy(t, tint))
    {
        return Some(m.to_f32());
    }
    let set = wb_matrix(&REC2020, effective_white(info, s));
    let shot = wb_matrix(&REC2020, temp_tint_to_xy(info.as_shot_temp, info.as_shot_tint));
    let m = set.mul(&shot.inverse().unwrap_or(lightcraft_color::Mat3::IDENTITY));
    // Normalize so neutral luminance is preserved (WB shouldn't change exposure).
    let g = m.apply([1.0, 1.0, 1.0]);
    let y = g[0] * 0.2627 + g[1] * 0.6780 + g[2] * 0.0593;
    Some(m.mul(&lightcraft_color::Mat3::diag(1.0 / y, 1.0 / y, 1.0 / y)).to_f32())
}

fn wb_gain(img: &mut Rgb32f, info: &SourceInfo, s: &DevelopSettings, gain: f32) {
    let m = wb_matrix_for(info, s);
    let w = img.width;
    for_rows(&mut img.data, w, |_, row| {
        for p in row.iter_mut() {
            let mut c = *p;
            if let Some(m) = &m {
                c = [
                    m[0][0] * c[0] + m[0][1] * c[1] + m[0][2] * c[2],
                    m[1][0] * c[0] + m[1][1] * c[1] + m[1][2] * c[2],
                    m[2][0] * c[0] + m[2][1] * c[1] + m[2][2] * c[2],
                ];
            }
            *p = [(c[0] * gain).max(0.0), (c[1] * gain).max(0.0), (c[2] * gain).max(0.0)];
        }
    });
}

/// The white balance actually in effect (presets resolve to their Kelvin values for raw files; settings written on
/// the relative scale before a raw's camera matrices reached its white balance give the same colour change on the
/// Kelvin scale, see [`legacy_to_kelvin`]).
pub fn effective_wb(info: &SourceInfo, s: &DevelopSettings) -> (f64, f64) {
    use lightcraft_develop::WbMode;
    match s.wb.mode {
        WbMode::AsShot => (info.as_shot_temp, info.as_shot_tint),
        _ if legacy_wb(info, s) => legacy_to_kelvin(s.wb.temp, s.wb.tint, (info.as_shot_temp, info.as_shot_tint)),
        m if info.raw && !info.relative_wb => m.preset().unwrap_or((s.wb.temp, s.wb.tint)),
        _ => (s.wb.temp, s.wb.tint),
    }
}

/// The source white the render adapts from the as-shot white to: [`effective_wb`] as a chromaticity, except that a
/// white balance still on the relative scale is composed exactly ([`legacy_white`]) rather than read back through
/// its rounded Kelvin / tint.
fn effective_white(info: &SourceInfo, s: &DevelopSettings) -> lightcraft_color::Xy {
    if legacy_wb(info, s)
        && s.wb.mode != lightcraft_develop::WbMode::AsShot
        && let Some(xy) = legacy_white(s.wb.temp, s.wb.tint, (info.as_shot_temp, info.as_shot_tint))
    {
        return xy;
    }
    let (t, tint) = effective_wb(info, s);
    temp_tint_to_xy(t, tint)
}

/// Whether `s` holds a white balance written on the relative scale for a source that is edited in Kelvin today.
fn legacy_wb(info: &SourceInfo, s: &DevelopSettings) -> bool {
    info.raw && !info.relative_wb && info.legacy_relative_wb && s.wb.scale == lightcraft_develop::WbScale::Legacy
}

/// The white that renders exactly as `(temp, tint)` did on the relative scale, where 6500 K / 0 meant *as shot* and
/// the source's real as-shot white is `as_shot`.
///
/// The relative render adapted the picture from 6500 K / 0 to `(temp, tint)` (Bradford, [`wb_matrix_for`]); the
/// Kelvin render adapts it from `as_shot` to the result. Bradford adaptations are von Kries scalings in one cone
/// space, so they compose exactly: the new white's cone response is the old target's times the as-shot white's
/// over the old reference's, and the two renders give the same matrix (up to the scale the luminance
/// normalisation removes). `None` when the arithmetic has nowhere to go (a non-finite input, a white outside the
/// chromaticity plane).
pub fn legacy_white(temp: f64, tint: f64, as_shot: (f64, f64)) -> Option<lightcraft_color::Xy> {
    use lightcraft_color::{BRADFORD, Xy};
    if !(temp.is_finite() && tint.is_finite() && as_shot.0.is_finite() && as_shot.1.is_finite()) {
        return None;
    }
    let cones = |t: f64, tint: f64| BRADFORD.apply(temp_tint_to_xy(t, tint).to_xyz());
    let (old, reference, shot) = (cones(temp, tint), cones(6500.0, 0.0), cones(as_shot.0, as_shot.1));
    let inverse = BRADFORD.inverse()?;
    let mut new = [0.0; 3];
    for i in 0..3 {
        if reference[i].abs() < 1e-12 {
            return None;
        }
        new[i] = old[i] * shot[i] / reference[i];
    }
    let xy = Xy::from_xyz(inverse.apply(new));
    (xy.x.is_finite() && xy.y.is_finite() && xy.y > 0.0).then_some(xy)
}

/// The Kelvin / tint of [`legacy_white`]: what a white balance written on the relative scale reads as, and becomes
/// when it is next edited ([`normalize_wb`]). Exact wherever the white has a Kelvin value (1000–50000 K after the
/// composition; the relative slider reached 2000–13550 K around 6500 K, so only a far-blue as-shot white pushed
/// to the slider's blue end lands beyond, and clamps), to the precision of the Kelvin / tint parametrisation
/// (better than 0.01 % away from its 4000 K seam). Unchanged when the arithmetic has nowhere to go.
pub fn legacy_to_kelvin(temp: f64, tint: f64, as_shot: (f64, f64)) -> (f64, f64) {
    if temp == 6500.0 && tint == 0.0 && as_shot.0.is_finite() && as_shot.1.is_finite() {
        // the relative scale's as-shot reference, exactly (a Custom value left where As Shot was stays as shot)
        return as_shot;
    }
    match legacy_white(temp, tint, as_shot).map(xy_to_temp_tint) {
        Some((t, tint)) if t.is_finite() && tint.is_finite() => (t, tint),
        _ => (temp, tint),
    }
}

/// Bring the white balance of `d` onto the scale `info` is edited on, before a new value is written into it: on a
/// raw with a Kelvin scale the values are marked Kelvin, and ones still on the relative scale become the Custom
/// Kelvin / tint that renders the same ([`legacy_to_kelvin`]; a preset mode's look is kept as that custom
/// value, since the preset's Kelvin value would be a different look). Elsewhere the scale is cleared, so values
/// edited on a rendered photo or a raw without matrices never pass for Kelvin when pasted onto one that has them.
pub fn normalize_wb(d: &mut DevelopSettings, info: &SourceInfo) {
    use lightcraft_develop::{WbMode, WbScale};
    if !(info.raw && !info.relative_wb) {
        d.wb.scale = WbScale::Legacy;
        return;
    }
    if legacy_wb(info, d) && d.wb.mode != WbMode::AsShot {
        let (t, tint) = legacy_to_kelvin(d.wb.temp, d.wb.tint, (info.as_shot_temp, info.as_shot_tint));
        d.wb.temp = t;
        d.wb.tint = tint;
        d.wb.mode = WbMode::Custom;
    }
    d.wb.scale = WbScale::Kelvin;
}

/// Self-guided filter (He et al.) on a single plane with Gaussian windows of `sigma` px.
pub fn guided(p: &Plane, sigma: f32, eps: f32) -> Plane {
    let (ma, mb) = guided_coeffs(p, sigma, eps);
    guided_apply(p, &ma, &mb)
}

/// Blurred guided-filter coefficients (mean a, mean b) of `p`; every pass row-parallel, and the
/// two independent blurs of each step run side by side.
fn guided_coeffs(p: &Plane, sigma: f32, eps: f32) -> (Plane, Plane) {
    let (mean, corr) = par_join(|| gaussian(p, sigma), || gaussian(&p.map(|v| v * v), sigma));
    let a = corr.zip_map(&mean, |c, m| {
        let var = (c - m * m).max(0.0);
        var / (var + eps)
    });
    let b = mean.zip_map(&a, |m, a| m - a * m);
    par_join(|| gaussian(&a, sigma), || gaussian(&b, sigma))
}

/// `q = a·p + b` per pixel.
fn guided_apply(p: &Plane, a: &Plane, b: &Plane) -> Plane {
    let mut q = Plane::new(p.width, p.height);
    let w = p.width;
    for_rows(&mut q.data, w, |y, row| {
        for ((v, pv), (av, bv)) in row.iter_mut().zip(p.row(y)).zip(a.row(y).iter().zip(b.row(y))) {
            *v = av * pv + bv;
        }
    });
    q
}

/// Fast guided filter (He & Sun 2015): coefficients computed on a `s×` subsampled plane and
/// bilinearly upsampled; the output keeps full-resolution edges. Equivalent to [`guided`] for
/// large windows at a fraction of the cost.
pub fn guided_fast(p: &Plane, sigma: f32, eps: f32) -> Plane {
    use lightcraft_raster::resample::{Filter, resize};
    let s = guided_fast_step(sigma);
    if s <= 1 {
        return guided(p, sigma, eps);
    }
    let (lw, lh) = (p.width.div_ceil(s).max(1), p.height.div_ceil(s).max(1));
    let lo = resize(p, lw, lh, Filter::Box);
    let (ma, mb) = guided_coeffs(&lo, sigma / s as f32, eps);
    let up = |c: &Plane| resize(c, p.width, p.height, Filter::Bilinear);
    guided_apply(p, &up(&ma), &up(&mb))
}

/// Separable median (`2r+1` px along each row, then each column; clamped edges). It keeps the
/// steps between regions where they are and removes features thinner than about `r` px.
pub fn median_hv(p: &Plane, r: usize) -> Plane {
    if r == 0 || p.data.is_empty() {
        return p.clone();
    }
    let rows = |src: &Plane| {
        let mut out = Plane::new(src.width, src.height);
        for_rows(&mut out.data, src.width, |y, row| median_row(src.row(y), row, r));
        out
    };
    transpose(&rows(&transpose(&rows(p))))
}

/// Running median of `src` over `2r+1` taps (clamped edges) into `out`: a sorted window that
/// drops the leaving value and inserts the entering one.
fn median_row(src: &[f32], out: &mut [f32], r: usize) {
    let n = src.len();
    if n == 0 {
        return;
    }
    let at = |i: isize| src[i.clamp(0, n as isize - 1) as usize];
    let r_ = r as isize;
    let mut win: Vec<f32> = (-r_..=r_).map(at).collect();
    win.sort_unstable_by(f32::total_cmp);
    for (x, o) in out.iter_mut().enumerate() {
        *o = win[r];
        let (old, new) = (at(x as isize - r_), at(x as isize + r_ + 1));
        if old.to_bits() != new.to_bits() {
            let i = win.binary_search_by(|v| v.total_cmp(&old)).unwrap_or_else(|i| i.min(win.len() - 1));
            win.remove(i);
            let j = win.binary_search_by(|v| v.total_cmp(&new)).unwrap_or_else(|j| j);
            win.insert(j, new);
        }
    }
}

fn transpose(p: &Plane) -> Plane {
    let (w, h) = (p.width, p.height);
    let mut t = Plane::new(h, w);
    for_rows(&mut t.data, h, |x, row| {
        for (y, v) in row.iter_mut().enumerate() {
            *v = p.data[y * w + x];
        }
    });
    t
}

/// Detail scale of the highlights/shadows base from process V2, as a fraction of the long edge:
/// the base is the edge-aware filter of the [`median_hv`] of log luminance at this radius, so
/// twigs, lines and fine texture brighter or darker than their surroundings take their
/// surroundings' shift (and keep their contrast, as with Exposure) while region boundaries stay
/// sharp (no halos). V1's base is the edge-aware filter of log luminance itself (issue #632).
pub const BASE_DETAIL: f64 = 0.0015;
/// Largest [`base_detail_radius`] (px).
pub const BASE_DETAIL_MAX: usize = 24;

/// Median radius (px) of the highlights/shadows base at `px_per_long` (see [`BASE_DETAIL`]).
pub fn base_detail_radius(px_per_long: f64) -> usize {
    ((BASE_DETAIL * px_per_long).round().max(0.0) as usize).min(BASE_DETAIL_MAX)
}

/// Median radius (px) of the highlights/shadows base of `s`'s rendering process at `px_per_long`:
/// 0 (no median) under V1, [`base_detail_radius`] from V2.
pub fn tone_base_detail(s: &DevelopSettings, px_per_long: f64) -> usize {
    match s.process.process() {
        Process::V1 => 0,
        Process::V2 => base_detail_radius(px_per_long),
    }
}

/// The highlights/shadows base layer of log luminance `l`: the fast guided filter ([`BASE_EPS`])
/// of its [`median_hv`] at `detail` px ([`tone_base_detail`]); at 0, of `l` itself (V1).
pub fn tone_base(l: &Plane, sigma: f32, detail: usize) -> Plane {
    if detail == 0 {
        return guided_fast(l, sigma, BASE_EPS);
    }
    guided_fast(&median_hv(l, detail), sigma, BASE_EPS)
}

/// Subsampling step of [`guided_fast`] for `sigma` (1 = no subsampling: the plain guided filter).
pub fn guided_fast_step(sigma: f32) -> usize {
    (sigma / 3.0).floor().clamp(1.0, 16.0) as usize
}

/// Luminance noise reduction: guided filter of log luminance (`sigma` px, `eps`), blended by `k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NrLum {
    pub sigma: f32,
    pub eps: f32,
    pub k: f32,
}

/// Colour noise reduction: chromaticity blurred by `sigma` px, mixed in by `t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NrColor {
    pub sigma: f32,
    pub t: f32,
}

/// Noise-reduction parameters at an output long edge of `out_long` px (see [`denoise`]).
pub fn nr_params(s: &DevelopSettings, src_long: usize, out_long: usize) -> (Option<NrLum>, Option<NrColor>) {
    let lum = (s.detail.nr_luminance / 100.0) as f32;
    let col = (s.detail.nr_color / 100.0) as f32;
    let scale = (out_long as f32 / src_long.max(1) as f32).clamp(0.05, 1.0);
    let l = (lum > 0.0).then(|| {
        let detail = (s.detail.nr_detail / 100.0) as f32;
        NrLum { sigma: (1.0 + 2.5 * lum) * scale.max(0.4), eps: 0.002 + lum * lum * 0.25 * (1.0 - 0.7 * detail), k: lum.sqrt() }
    });
    let c = (col > 0.0).then(|| {
        let sigma = (1.5 + 6.0 * col) * scale.max(0.35) * (1.0 + (s.detail.nr_color_smoothness / 100.0) as f32);
        let keep = (s.detail.nr_color_detail / 100.0) as f32 * 0.5;
        NrColor { sigma, t: col * (1.0 - keep) }
    });
    (l, c)
}

/// Noise reduction at output resolution: luminance via an edge-aware self-guided filter on
/// log-luminance, colour by blurring chromaticity (rgb / Y) and re-applying the original luminance.
/// Radii scale with how much the source was downsampled (preview noise is already averaged out).
pub fn denoise(img: &mut Rgb32f, s: &DevelopSettings, src_long: usize, out_long: usize) {
    let (lum, col) = nr_params(s, src_long, out_long);
    let w = img.width;
    if let Some(nr) = lum {
        let _t = crate::profiling().then(std::time::Instant::now);
        let l = img.map(log_lum);
        let f = guided(&l, nr.sigma, nr.eps);
        for_rows(&mut img.data, w, |y, row| {
            for (x, p) in row.iter_mut().enumerate() {
                let i = y * w + x;
                let d = (f.data[i] - l.data[i]) * nr.k;
                let g = d.exp2();
                *p = p.map(|v| v * g);
            }
        });
        if let Some(t) = _t {
            eprintln!("    nr luminance: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
    }
    if col.is_some() && nr_v2_enabled() {
        color_nr_v2(img, s, src_long, out_long);
        return;
    }
    if let Some(nr) = col {
        let _t = crate::profiling().then(std::time::Instant::now);
        let chroma = img.map(|c| {
            let y = luminance_2020(c).max(1e-6);
            [c[0] / y, c[1] / y, c[2] / y]
        });
        let b = gaussian(&chroma, nr.sigma);
        for_rows(&mut img.data, w, |y, row| {
            for (x, p) in row.iter_mut().enumerate() {
                let i = y * w + x;
                let yl = luminance_2020(*p);
                let c0 = chroma.data[i];
                let cb = b.data[i];
                *p = [0, 1, 2].map(|k| ((c0[k] + (cb[k] - c0[k]) * nr.t) * yl).max(0.0));
            }
        });
        if let Some(t) = _t {
            eprintln!("    nr colour: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
    }
}

// PROTOTYPE (tuning knobs via env while the method is worked out)
/// The new colour noise reduction is on unless `LIGHTCRAFT_NR_V2=0`.
pub fn nr_v2_enabled() -> bool {
    std::env::var("LIGHTCRAFT_NR_V2").map_or(true, |v| v != "0")
}

/// A prototype tuning knob from the environment, read once per process (they are read in per-pixel code).
fn knob(name: &'static str, default: f32) -> f32 {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock, PoisonError};
    static READ: OnceLock<Mutex<HashMap<&'static str, Option<f32>>>> = OnceLock::new();
    let map = READ.get_or_init(|| Mutex::new(HashMap::new()));
    let mut m = map.lock().unwrap_or_else(PoisonError::into_inner);
    m.entry(name).or_insert_with(|| std::env::var(name).ok().and_then(|v| v.parse().ok())).unwrap_or(default)
}

/// [`nrc_linear`], read once (it is asked per pixel).
fn nrc_linear_cached() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| knob("NRV2_LINEAR", 1.0) > 0.5)
}

/// Most samples the noise level of the new colour noise reduction is measured from. The CPU and
/// the GPU sample the same pixels (every [`nrc_sample_stride`]-th of the plane).
pub const NRC_SAMPLES: usize = 131_072;

/// Distance between the measured samples of a plane of `len` pixels.
pub fn nrc_sample_stride(len: usize) -> usize {
    len.div_ceil(NRC_SAMPLES).max(7)
}

/// One self-guided pass of the new colour noise reduction: Gaussian windows of `sigma` px, the
/// coefficients computed on a plane subsampled by `step`, `eps = k ×` the `pct`-quantile of the
/// local variance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NrcPass {
    pub sigma: f32,
    pub step: usize,
    pub k: f32,
    pub pct: f32,
}

/// Parameters of the new colour noise reduction: the offset of the square roots and the passes
/// (fine, then coarse).
#[derive(Clone, Debug, PartialEq)]
pub struct NrcParams {
    pub off: f32,
    pub passes: Vec<NrcPass>,
}

/// The new colour noise reduction's parameters at an output long edge of `out_long` px, `None`
/// when Color is 0.
pub fn nrc_params(s: &DevelopSettings, src_long: usize, out_long: usize) -> Option<NrcParams> {
    let amount = (s.detail.nr_color / 100.0) as f32;
    if amount <= 0.0 {
        return None;
    }
    let detail = (s.detail.nr_color_detail / 100.0) as f32;
    let smooth = 1.0 + (s.detail.nr_color_smoothness / 100.0) as f32;
    let scale = (out_long as f32 / src_long.max(1) as f32).clamp(0.05, 1.0);
    let sigma = (knob("NRV2_S0", 2.0) + knob("NRV2_S1", 10.0) * amount) * smooth * scale;
    let k = knob("NRV2_K", if nrc_linear() { 64.0 } else { 16.0 }) * amount * (1.5 - detail);
    let pct = knob("NRV2_PCT", 0.2);
    let coarse = knob("NRV2_COARSE", 3.0);
    // the coefficients vary over about sigma: a grid of sigma / 2 holds them
    let div = knob("NRV2_STEPDIV", 2.0).max(1.0);
    let pass = |sigma: f32, k: f32| NrcPass { sigma, step: (sigma / div).floor().clamp(1.0, 16.0) as usize, k, pct };
    let mut passes = vec![pass(sigma, k)];
    if coarse > 1.0 {
        // the coarse pass sees thin colour detail only diluted over its wide window: its own, gentler threshold
        passes.push(pass(sigma * coarse, k * knob("NRV2_COARSE_K", 1.0)));
    }
    Some(NrcParams { off: knob("NRV2_OFF", 0.003), passes })
}

/// Chroma noise variance per channel as a function of the local luminance: `α · Y + β` (shot plus read noise).
#[derive(Clone, Copy, Debug)]
pub struct NrcNoise {
    pub alpha: [f32; 3],
    pub beta: [f32; 3],
}

impl NrcNoise {
    #[inline]
    pub fn var(&self, i: usize, y: f32) -> f32 {
        (self.alpha[i] * y.max(0.0) + self.beta[i]).max(1e-14)
    }
}

/// Fit [`NrcNoise`] to the local variances `var` against the local luminance `ym` (channel 0): the pixels are
/// split into 16 luminance bands (square-root spaced), each band's `pct`-quantile is its noise floor, and
/// `α`, `β ≥ 0` are fitted to the bands by weighted least squares.
pub fn nrc_noise_model(var: &Rgb32f, ym: &Rgb32f, pct: f32) -> NrcNoise {
    const BANDS: usize = 16;
    let stride = nrc_sample_stride(var.data.len()).min(4);
    let mut samples: Vec<(f32, [f32; 3])> = var
        .data
        .iter()
        .zip(&ym.data)
        .step_by(stride)
        .filter(|(v, y)| y[0].is_finite() && v.iter().all(|x| x.is_finite()))
        .map(|(v, y)| (y[0].max(0.0), *v))
        .collect();
    // equal-count bands by luminance: shadows keep their own bands however bright the highlights are
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let per = (samples.len() / BANDS).max(1);
    let bands: Vec<&[(f32, [f32; 3])]> = samples.chunks(per).filter(|c| c.len() >= 32).collect();
    let mut alpha = [0.0f32; 3];
    let mut beta = [1e-12f32; 3];
    for i in 0..3 {
        // per band: mean luminance, `pct`-quantile of the variance (its noise floor), weight
        let pts: Vec<(f64, f64, f64)> = bands
            .iter()
            .map(|b| {
                let mut c: Vec<f32> = b.iter().map(|s| s.1[i]).collect();
                let k = ((c.len() - 1) as f32 * pct.clamp(0.0, 1.0)) as usize;
                let (_, q, _) = c.select_nth_unstable_by(k, f32::total_cmp);
                (b.iter().map(|s| s.0 as f64).sum::<f64>() / b.len() as f64, *q as f64, b.len() as f64)
            })
            .collect();
        if pts.is_empty() {
            continue;
        }
        let (a, b) = fit_nonneg_line(&pts);
        alpha[i] = a as f32;
        beta[i] = (b as f32).max(1e-12);
    }
    if std::env::var_os("NRV2_DEBUG").is_some() {
        eprintln!("    nr colour v2 noise model: alpha {alpha:?} beta {beta:?} ({} bands)", bands.len());
    }
    NrcNoise { alpha, beta }
}

/// Weighted least squares `v = a · l + b` over `(l, v, weight)` with `a, b ≥ 0`: the free fit if both come out
/// non-negative, else the better of the line through the origin (`b = 0`) and the constant (`a = 0`).
pub fn fit_nonneg_line(pts: &[(f64, f64, f64)]) -> (f64, f64) {
    let sw: f64 = pts.iter().map(|p| p.2).sum::<f64>().max(1e-300);
    let (ml, mv) = (pts.iter().map(|p| p.2 * p.0).sum::<f64>() / sw, pts.iter().map(|p| p.2 * p.1).sum::<f64>() / sw);
    let (cov, vl) = pts.iter().fold((0.0, 0.0), |(c, v), p| (c + p.2 * (p.0 - ml) * (p.1 - mv), v + p.2 * (p.0 - ml) * (p.0 - ml)));
    if vl > 0.0 {
        let a = cov / vl;
        let b = mv - a * ml;
        if a >= 0.0 && b >= 0.0 {
            return (a, b);
        }
    }
    let sse = |a: f64, b: f64| pts.iter().map(|p| p.2 * (p.1 - a * p.0 - b).powi(2)).sum::<f64>();
    let ll: f64 = pts.iter().map(|p| p.2 * p.0 * p.0).sum();
    let origin = if ll > 0.0 { (pts.iter().map(|p| p.2 * p.0 * p.1).sum::<f64>() / ll).max(0.0) } else { 0.0 };
    let constant = mv.max(0.0);
    if sse(origin, 0.0) < sse(0.0, constant) { (origin, 0.0) } else { (0.0, constant) }
}

/// `eps` per channel from the sampled local variances.
pub fn nrc_eps(samples: &[[f32; 3]], k: f32, pct: f32) -> [f32; 3] {
    [0, 1, 2].map(|c| {
        let mut v: Vec<f32> = samples.iter().map(|s| s[c]).filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return 1e-12;
        }
        let i = ((v.len() - 1) as f32 * pct.clamp(0.0, 1.0)) as usize;
        let (_, m, _) = v.select_nth_unstable_by(i, f32::total_cmp);
        k * m.max(1e-12)
    })
}

/// PROTOTYPE: `NRV2_LINEAR=1` filters linear colour differences `c − Y` (nothing clipped, the local mean
/// kept exactly) with a threshold from a noise model `σ²(Y) = α·Y + β`, instead of square-root differences
/// with one threshold for the whole image.
pub fn nrc_linear() -> bool {
    nrc_linear_cached()
}

/// The joint guide (`nrc_joint`), on unless `NRV2_JOINT=0` (PROTOTYPE switch).
pub fn nrc_joint_enabled() -> bool {
    nr_v2_enabled() && nrc_linear() && knob("NRV2_JOINT", 1.0) > 0.5
}

/// Chroma of a pixel: linear differences `c − Y` (see [`nrc_linear`]), else differences of square roots
/// (`√(c + off) − √(Y + off)`).
#[inline]
pub fn nrc_chroma(c: [f32; 3], off: f32) -> [f32; 3] {
    if nrc_linear() {
        let y = luminance_2020(c);
        return c.map(|v| v - y);
    }
    let y = (luminance_2020(c).max(0.0) + off).sqrt();
    c.map(|v| (v.max(0.0) + off).sqrt() - y)
}

/// A pixel with its chroma replaced by `q`, its luminance kept. `removed` is the variance the passes
/// took out of each square-root channel: smoothing averages square roots, whose square is below the mean
/// of the squares (Jensen) by that variance, so adding it back keeps the local mean colour.
#[inline]
pub fn nrc_recombine(p: [f32; 3], q: [f32; 3], removed: [f32; 3], off: f32) -> [f32; 3] {
    if nrc_linear() {
        // additive: channel = Y + filtered difference, then Y pulled back exactly (each channel has its own a)
        let y = luminance_2020(p);
        let c = q.map(|d| y + d);
        let dy = y - luminance_2020(c);
        return c.map(|v| v + dy);
    }
    let yl = luminance_2020(p);
    let ys = (yl.max(0.0) + off).sqrt();
    let c: [f32; 3] = std::array::from_fn(|i| ((ys + q[i]).max(0.0).powi(2) + removed[i] - off).max(0.0));
    let y1 = luminance_2020(c);
    if y1 > 1e-7 && yl > 0.0 { c.map(|v| v * (yl / y1)) } else { [yl.max(0.0); 3] }
}

/// Self-guided coefficients of one pass at its subsampled size (see [`NrcPass`]): `a` and `b`
/// per pixel of `lo`, blurred.
fn nrc_coeffs(lo: &Rgb32f, ylo: &Rgb32f, pass: &NrcPass) -> (Rgb32f, Rgb32f, Rgb32f) {
    let sg = pass.sigma / pass.step.max(1) as f32;
    let (mean, corr) = par_join(|| gaussian(lo, sg), || gaussian(&lo.map(|c| c.map(|v| v * v)), sg));
    let var = corr.zip_map(&mean, |c, m| [0, 1, 2].map(|i| (c[i] - m[i] * m[i]).max(0.0)));
    let a = if nrc_linear() {
        let ym = gaussian(ylo, sg);
        let model = nrc_noise_model(&var, &ym, pass.pct);
        if knob("NRV2_SHARED", 0.0) > 0.5 {
            // one coefficient for all channels: each channel's variance in units of its noise, averaged, so
            // every channel gets the same transition width at an edge (no hue change across it). With
            // `NRV2_LUMGUIDE > 0`, luminance structure (in units of its own noise) counts too.
            let lg = knob("NRV2_LUMGUIDE", 0.0);
            let lum = (lg > 0.0).then(|| {
                let m2 = gaussian(&ylo.map(|c| [c[0] * c[0], 0.0, 0.0]), sg);
                let vy = m2.zip_map(&ym, |m2, m| {
                    let v = (m2[0] - m[0] * m[0]).max(0.0);
                    [v, v, v]
                });
                let ly = nrc_noise_model(&vy, &ym, pass.pct);
                (vy, ly)
            });
            let mut a = var.zip_map(&ym, |v, y| {
                let s = (0..3).map(|i| v[i] / model.var(i, y[0])).sum::<f32>() / 3.0;
                [s; 3]
            });
            if let Some((vy, ly)) = &lum {
                for ((s, v), y) in a.data.iter_mut().zip(&vy.data).zip(&ym.data) {
                    let add = lg * v[0] / ly.var(0, y[0]);
                    *s = s.map(|x| x + add);
                }
            }
            a.map(|s| s.map(|s| s / (s + pass.k)))
        } else {
            // per channel; with `NRV2_LUMGUIDE > 0` luminance structure (in units of its own noise) raises every
            // channel's share, so colour on textured or edged luminance (leaves, fur) is kept
            let lg = knob("NRV2_LUMGUIDE", 0.0);
            if lg > 0.0 {
                let m2 = gaussian(&ylo.map(|c| [c[0] * c[0], 0.0, 0.0]), sg);
                let vy = m2.zip_map(&ym, |m2, m| {
                    let v = (m2[0] - m[0] * m[0]).max(0.0);
                    [v, v, v]
                });
                let ly = nrc_noise_model(&vy, &ym, pass.pct);
                let s = var.zip_map(&ym, |v, y| [0, 1, 2].map(|i| v[i] / model.var(i, y[0])));
                // only luminance variance well above its noise counts as structure (flat noisy areas stay smoothed)
                let off = knob("NRV2_LUMG_OFF", 3.0);
                let g = vy.zip_map(&ym, |v, y| [lg * (v[0] / ly.var(0, y[0]) - off).max(0.0); 3]);
                s.zip_map(&g, |s, g| [0, 1, 2].map(|i| (s[i] + g[i]) / (s[i] + g[i] + pass.k)))
            } else {
                var.zip_map(&ym, |v, y| [0, 1, 2].map(|i| v[i] / (v[i] + pass.k * model.var(i, y[0]))))
            }
        }
    } else {
        let stride = nrc_sample_stride(var.data.len());
        let samples: Vec<[f32; 3]> = var.data.iter().step_by(stride).copied().collect();
        let eps = nrc_eps(&samples, pass.k, pass.pct);
        var.map(|v| [0, 1, 2].map(|i| v[i] / (v[i] + eps[i])))
    };
    let b = mean.zip_map(&a, |m, a| [0, 1, 2].map(|i| m[i] - a[i] * m[i]));
    // what `a · d + b` takes out of the variance: var − a² · var
    let bias = if nrc_linear() { 0.0 } else { bias_knob() };
    let removed = var.zip_map(&a, |v, a| [0, 1, 2].map(|i| (1.0 - a[i] * a[i]) * v[i] * bias));
    let (ab, rb) = par_join(|| par_join(|| gaussian(&a, sg), || gaussian(&b, sg)), || gaussian(&removed, sg));
    (ab.0, ab.1, rb)
}

/// PROTOTYPE: `NRV2_BIAS=0` leaves the removed variance out (the mean colour of noisy areas then drifts).
fn bias_knob() -> f32 {
    knob("NRV2_BIAS", 1.0)
}

/// One pass's coefficients and the taps that upsample them to the full size
/// (`resize(…, Filter::Bilinear)`, evaluated a row at a time).
struct NrcUp {
    a: Rgb32f,
    b: Rgb32f,
    r: Rgb32f,
    /// For a pass that smooths only the coarse part (`NRV2_COARSE_DETAIL`): its subsampled input. The pass then
    /// replaces `lo` by `a · lo + b` and keeps `d − lo` (finer detail, such as thin coloured lines) as it is.
    lo: Option<Rgb32f>,
    ux: Vec<(usize, Vec<f32>)>,
    uy: Vec<(usize, Vec<f32>)>,
}

#[inline]
fn madd3(acc: [f32; 3], v: [f32; 3], f: f32) -> [f32; 3] {
    [acc[0] + f * v[0], acc[1] + f * v[1], acc[2] + f * v[2]]
}

impl NrcUp {
    /// Apply this pass to `d`, row `y` of the full-size chroma, in place, and add the variance it
    /// removes to `removed`.
    fn apply_row(
        &self,
        y: usize,
        d: &mut [[f32; 3]],
        removed: &mut [[f32; 3]],
        va: &mut Vec<[f32; 3]>,
        vb: &mut Vec<[f32; 3]>,
        vr: &mut Vec<[f32; 3]>,
    ) {
        let lw = self.a.width;
        for v in [&mut *va, &mut *vb, &mut *vr] {
            v.clear();
            v.resize(lw, [0.0; 3]);
        }
        if let Some((y0, wy)) = self.uy.get(y) {
            for (j, &f) in wy.iter().enumerate() {
                let rows = (y0 + j) * lw..(y0 + j + 1) * lw;
                let (Some(ra), Some(rb), Some(rr)) = (self.a.data.get(rows.clone()), self.b.data.get(rows.clone()), self.r.data.get(rows)) else {
                    continue;
                };
                for x in 0..lw {
                    va[x] = madd3(va[x], ra[x], f);
                    vb[x] = madd3(vb[x], rb[x], f);
                    vr[x] = madd3(vr[x], rr[x], f);
                }
            }
        }
        // the subsampled input upsampled to this row (only for a coarse-only pass)
        let vl: Vec<[f32; 3]> = match &self.lo {
            None => Vec::new(),
            Some(lo) => {
                let mut row = vec![[0.0f32; 3]; lw];
                if let Some((y0, wy)) = self.uy.get(y) {
                    for (j, &f) in wy.iter().enumerate() {
                        if let Some(r) = lo.data.get((y0 + j) * lw..(y0 + j + 1) * lw) {
                            for x in 0..lw {
                                row[x] = madd3(row[x], r[x], f);
                            }
                        }
                    }
                }
                (0..d.len())
                    .map(|x| {
                        let Some((x0, wx)) = self.ux.get(x) else { return [0.0; 3] };
                        wx.iter().enumerate().fold([0.0; 3], |acc, (k, &f)| row.get(x0 + k).map_or(acc, |v| madd3(acc, *v, f)))
                    })
                    .collect()
            }
        };
        for (x, (p, rem)) in d.iter_mut().zip(removed.iter_mut()).enumerate() {
            let Some((x0, wx)) = self.ux.get(x) else { continue };
            let (mut ca, mut cb, mut cr) = ([0.0; 3], [0.0; 3], [0.0; 3]);
            for (k, &f) in wx.iter().enumerate() {
                if let (Some(a), Some(b), Some(r)) = (va.get(x0 + k), vb.get(x0 + k), vr.get(x0 + k)) {
                    ca = madd3(ca, *a, f);
                    cb = madd3(cb, *b, f);
                    cr = madd3(cr, *r, f);
                }
            }
            match &self.lo {
                None => *p = [0, 1, 2].map(|i| ca[i] * p[i] + cb[i]),
                Some(_) => {
                    let cl = vl.get(x).copied().unwrap_or([0.0; 3]);
                    *p = [0, 1, 2].map(|i| p[i] + (ca[i] - 1.0) * cl[i] + cb[i]);
                }
            }
            *rem = [0, 1, 2].map(|i| rem[i] + cr[i].max(0.0));
        }
    }
}

/// Scratch rows of [`NrcUp::apply_row`].
#[derive(Default)]
struct NrcScratch {
    va: Vec<[f32; 3]>,
    vb: Vec<[f32; 3]>,
    vr: Vec<[f32; 3]>,
}

/// Row `y` of the chroma after the passes in `ups`, computed from the image row `src`, and the
/// variance the passes removed per pixel.
fn nrc_row(src: &[[f32; 3]], y: usize, ups: &[NrcUp], off: f32, s: &mut NrcScratch) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut d: Vec<[f32; 3]> = src.iter().map(|c| nrc_chroma(*c, off)).collect();
    let mut removed = vec![[0.0; 3]; d.len()];
    for up in ups {
        up.apply_row(y, &mut d, &mut removed, &mut s.va, &mut s.vb, &mut s.vr);
    }
    (d, removed)
}

/// Colour noise reduction, second method: chroma as differences of square roots (noise of about
/// the same size at every brightness, and no blow-up near black as `c / Y` has), smoothed by a
/// self-guided filter whose threshold follows the noise measured in the image, the luminance kept.
///
/// No full-size intermediate is stored: each pass's subsampled input is gathered row by row from
/// the image (its chroma after the earlier passes, recomputed per row), and the last step applies
/// every pass and recombines a row at a time.
pub fn color_nr_v2(img: &mut Rgb32f, s: &DevelopSettings, src_long: usize, out_long: usize) {
    use lightcraft_raster::resample::{Filter, weights};
    let Some(p) = nrc_params(s, src_long, out_long) else { return };
    if nrc_joint_enabled() {
        crate::nrc_joint::denoise(img, &p, (s.detail.nr_color_detail / 100.0) as f32, (s.detail.nr_color / 100.0) as f32);
        return;
    }
    let _t = crate::profiling().then(std::time::Instant::now);
    let (w, h) = (img.width, img.height);
    let mut ups: Vec<NrcUp> = Vec::with_capacity(p.passes.len());
    for pass in &p.passes {
        let step = pass.step.max(1);
        let (lw, lh) = (w.div_ceil(step).max(1), h.div_ceil(step).max(1));
        // `resize(chroma after the earlier passes, lw, lh, Filter::Box)`
        let (bx, by) = (weights(w, lw, Filter::Box), weights(h, lh, Filter::Box));
        let mut lo = Rgb32f::new(lw, lh);
        let src = &*img;
        let done = &ups;
        for_rows(&mut lo.data, lw, |yo, row| {
            let mut scratch = NrcScratch::default();
            let Some((y0, wy)) = by.get(yo) else { return };
            for (j, &f) in wy.iter().enumerate() {
                if f == 0.0 || y0 + j >= h {
                    continue;
                }
                let (d, _) = nrc_row(src.row(y0 + j), y0 + j, done, p.off, &mut scratch);
                for (xo, o) in row.iter_mut().enumerate() {
                    let Some((x0, wx)) = bx.get(xo) else { continue };
                    let mut acc = [0.0; 3];
                    for (k, &g) in wx.iter().enumerate() {
                        if let Some(v) = d.get(x0 + k) {
                            acc = madd3(acc, *v, g);
                        }
                    }
                    *o = madd3(*o, acc, f);
                }
            }
        });
        // the pass's luminance at the same size (for the noise model)
        let mut ylo = Rgb32f::new(lw, lh);
        for_rows(&mut ylo.data, lw, |yo, row| {
            let Some((y0, wy)) = by.get(yo) else { return };
            for (j, &f) in wy.iter().enumerate() {
                if f == 0.0 || y0 + j >= h {
                    continue;
                }
                let srow = src.row(y0 + j);
                for (xo, o) in row.iter_mut().enumerate() {
                    let Some((x0, wx)) = bx.get(xo) else { continue };
                    let mut acc = 0.0;
                    for (k, &g) in wx.iter().enumerate() {
                        if let Some(c) = srow.get(x0 + k) {
                            acc += g * luminance_2020(*c);
                        }
                    }
                    o[0] += f * acc;
                }
            }
        });
        let (a, b, r) = nrc_coeffs(&lo, &ylo, pass);
        let coarse_only = !ups.is_empty() && nrc_linear() && knob("NRV2_COARSE_DETAIL", 1.0) > 0.5;
        let lo = coarse_only.then_some(lo);
        ups.push(NrcUp { a, b, r, lo, ux: weights(lw, w, Filter::Bilinear), uy: weights(lh, h, Filter::Bilinear) });
    }
    let off = p.off;
    for_rows(&mut img.data, w, |y, row| {
        let mut scratch = NrcScratch::default();
        let (q, removed) = nrc_row(row, y, &ups, off, &mut scratch);
        for ((px, qv), r) in row.iter_mut().zip(q).zip(removed) {
            *px = nrc_recombine(*px, qv, r, off);
        }
    });
    if let Some(t) = _t {
        eprintln!("    nr colour v2: {:.1} ms ({:?})", t.elapsed().as_secs_f64() * 1e3, p.passes.iter().map(|q| q.sigma).collect::<Vec<_>>());
    }
}

pub fn log_lum(c: [f32; 3]) -> f32 {
    (luminance_2020(c).max(1e-7) / crate::tone::GREY).log2()
}

/// Spatial planes of one (pre-exposure) image, each tagged with the radius it was computed at.
/// `key` identifies the image; see [`crate::StageCache`].
#[derive(Clone, Default)]
pub(crate) struct Planes {
    pub key: u64,
    /// The image's key without its noise reduction (`Plan::pre_nr_key`).
    pub pre_nr: u64,
    pub log_l: Option<Arc<Plane>>,
    /// Keyed by its radius, process and median radius ([`BaseKey`]).
    pub base: Option<(BaseKey, Arc<Plane>)>,
    pub clarity: Option<(u32, Arc<Plane>)>,
    pub texture: Option<(u32, Arc<Plane>)>,
    /// Dark channel and its airlight.
    pub dark: Option<(u32, Arc<Plane>, f32)>,
    /// Blurred chromaticity (local Moiré / Noise).
    pub chroma: Option<(u32, Arc<Rgb32f>)>,
}

impl Planes {
    /// The planes that still fit an image differing from this one only in noise reduction, close
    /// enough for a draft: the wide blurs (base, clarity, dehaze, chromaticity). Its `key` matches
    /// no image, so only another draft takes them.
    pub fn for_other_nr(self) -> Planes {
        Planes { key: 0, pre_nr: self.pre_nr, log_l: None, texture: None, ..self }
    }
}

/// What the highlights/shadows base plane was computed with: its radius (`f32` bits), the
/// rendering process and the median radius ([`tone_base_detail`]). The base differs between
/// processes, so a photo moved to another one (Update to Current Process, undo) recomputes it.
pub type BaseKey = (u32, Process, usize);

/// The [`BaseKey`] of a base at `sigma` for `s` (`detail`: [`PlaneSigmas::base_detail`]).
pub fn base_key(s: &DevelopSettings, sigma: f32, detail: usize) -> BaseKey {
    (sigma.to_bits(), s.process.process(), detail)
}

/// Reuse `slot` if it was computed with `key` (a radius's `f32` bits, or a [`BaseKey`]), else
/// compute and store it.
pub fn plane_at<K: Copy + PartialEq, P>(slot: &mut Option<(K, Arc<P>)>, key: K, f: impl FnOnce() -> P) -> Arc<P> {
    match slot {
        Some((k, p)) if *k == key => p.clone(),
        _ => {
            let p = Arc::new(f());
            *slot = Some((key, p.clone()));
            p
        }
    }
}

/// Guided-filter epsilons (EV²) of the highlights/shadows base and the clarity band.
pub const BASE_EPS: f32 = 0.35;
pub const CLARITY_EPS: f32 = 0.8;

/// Dark channel of a pixel (dehaze).
#[inline]
pub fn dark_of(c: [f32; 3]) -> f32 {
    c[0].min(c[1]).min(c[2])
}

/// Radius of the local Moiré / colour-noise chromaticity blur, as a fraction of the long edge.
pub const CHROMA_SIGMA: f32 = 0.004;

/// Radii (px) of the spatial planes the settings need (`None` = not needed).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlaneSigmas {
    /// Edge-aware base for highlights/shadows ([`tone_base`]: fast guided filter, [`BASE_EPS`]; from
    /// process V2 of the [`BASE_DETAIL`] median).
    pub base: Option<f32>,
    /// Median radius (px) of that base ([`tone_base_detail`]: 0 under V1).
    pub base_detail: usize,
    /// Clarity band (fast guided filter, [`CLARITY_EPS`]).
    pub clarity: Option<f32>,
    /// Texture / sharpening band (Gaussian).
    pub texture: Option<f32>,
    /// Dehaze dark channel (Gaussian).
    pub dark: Option<f32>,
    /// Chromaticity blur for local Moiré / colour noise (Gaussian).
    pub chroma: Option<f32>,
}

pub fn plane_sigmas(s: &DevelopSettings, px_per_long: f64, q: Quality) -> PlaneSigmas {
    let ppl = px_per_long as f32;
    let local_any = |f: fn(&lightcraft_develop::LocalAdjustments) -> f64| s.masks.iter().any(|m| f(&m.adjust) != 0.0);
    let tone_active =
        s.light.highlights != 0.0 || s.light.shadows != 0.0 || s.masks.iter().any(|m| m.adjust.highlights != 0.0 || m.adjust.shadows != 0.0);
    // Edge-aware base at ~1.5% of the long edge (EV² epsilon: edges of > ~0.6 EV are preserved)
    // From V2 it is taken of the median at 0.15%: lines and texture finer than that are detail,
    // not tone (#632).
    let base = tone_active.then(|| {
        let sigma = (0.015 * ppl).max(1.0);
        if q == Quality::Draft { sigma.min(24.0) } else { sigma }
    });
    let clarity = (s.effects.clarity != 0.0 || local_any(|a| a.clarity)).then(|| (0.012 * ppl).max(1.0));
    // local Noise and Defringe read the fine detail band too
    let texture = (s.effects.texture != 0.0
        || s.detail.sharpen_amount != 0.0
        || local_any(|a| a.texture)
        || local_any(|a| a.sharpness)
        || local_any(|a| a.noise)
        || s.masks.iter().any(|m| m.adjust.defringe > 0.0))
    .then(|| (0.0018 * ppl).max(0.6));
    let dark = (s.effects.dehaze != 0.0 || local_any(|a| a.dehaze)).then(|| (0.02 * ppl).max(1.0));
    let chroma = (local_any(|a| a.moire) || s.masks.iter().any(|m| m.adjust.noise > 0.0)).then(|| (CHROMA_SIGMA * ppl).max(1.0));
    let base_detail = if base.is_some() { tone_base_detail(s, px_per_long) } else { 0 };
    PlaneSigmas { base, base_detail, clarity, texture, dark, chroma }
}

/// The spatial planes the per-pixel stage needs for `img` (white-balanced, before exposure),
/// reusing whatever `planes` already holds for it. Missing planes are computed side by side (each
/// one alone scales poorly: the guided filters work on small subsampled grids).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare(
    img: Arc<Rgb32f>,
    s: &DevelopSettings,
    frame: &Frame,
    px_per_long: f64,
    q: Quality,
    planes: &mut Planes,
    mattes: Option<&masks::Mattes>,
) -> Prepared {
    let log_l = planes.log_l.get_or_insert_with(|| Arc::new(timed("log_l", || img.map(log_lum)))).clone();
    let PlaneSigmas { base: base_sigma, base_detail, clarity: clarity_sigma, texture: texture_sigma, dark: dark_sigma, chroma: chroma_sigma } =
        plane_sigmas(s, px_per_long, q);

    let Planes { base: sb, clarity: sc, texture: st, dark: sd, chroma: sch, .. } = planes;
    let chroma_blur = chroma_sigma.map(|sg| match sch {
        Some((k, c)) if *k == sg.to_bits() => c.clone(),
        _ => {
            let c = timed("chroma", || Arc::new(gaussian(&img.map(masks::chromaticity), sg)));
            *sch = Some((sg.to_bits(), c.clone()));
            c
        }
    });
    let l = &log_l;
    let (base, (clarity_blur, (texture_blur, dark))) = par_join(
        || base_sigma.map(|sg| plane_at(sb, base_key(s, sg, base_detail), || timed("base", || tone_base(l, sg, base_detail)))),
        || {
            par_join(
                || clarity_sigma.map(|sg| plane_at(sc, sg.to_bits(), || timed("clarity", || guided_fast(l, sg, CLARITY_EPS)))),
                || {
                    par_join(
                        || texture_sigma.map(|sg| plane_at(st, sg.to_bits(), || timed("texture", || gaussian(l, sg)))),
                        || {
                            dark_sigma.map(|sg| match sd {
                                Some((k, d, air)) if *k == sg.to_bits() => (d.clone(), *air),
                                _ => {
                                    let d = timed("dehaze", || Arc::new(gaussian(&img.map(dark_of), sg)));
                                    let air = airlight(&d);
                                    *sd = Some((sg.to_bits(), d.clone(), air));
                                    (d, air)
                                }
                            })
                        },
                    )
                },
            )
        },
    );
    let base = base.unwrap_or_else(|| log_l.clone());
    let (dark, air) = match dark {
        Some((d, air)) => (Some(d), air),
        None => (None, 1.0),
    };
    let ev = s.light.exposure as f32;
    let masks = timed("masks", || masks::evaluate(&s.masks, frame, img.width, img.height, &img, &log_l, ev, mattes));
    Prepared { img, log_l, base, clarity_blur, texture_blur, dark, chroma_blur, air, masks, px_per_long }
}

/// The airlight of the whole output frame, estimated on a small render of it (`proxy_w` px wide):
/// a windowed render must dehaze with the frame's airlight, not that of the pixels it holds.
pub(crate) fn frame_airlight(src: &Rgb32f, info: &SourceInfo, s: &DevelopSettings, frame: &Frame, proxy_w: usize, proxy_h: usize) -> f32 {
    let mut img = frame.sample(src, proxy_w, proxy_h);
    white_balance(&mut img, info, s);
    let sigma = (0.02 * frame.px_per_long(proxy_w)).max(1.0) as f32;
    airlight(&gaussian(&img.map(dark_of), sigma))
}

/// The airlight is estimated from every `AIRLIGHT_STEP`-th value of the dark channel.
pub const AIRLIGHT_STEP: usize = 7;

/// Airlight estimate: bright end of the dark channel.
pub fn airlight(dark: &Plane) -> f32 {
    airlight_of(dark.data.iter().step_by(AIRLIGHT_STEP).copied().collect())
}

/// Airlight from the sampled dark-channel values (see [`airlight`]): their 99.5th percentile.
pub fn airlight_of(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 1.0;
    }
    let k = ((v.len() as f32) * 0.995) as usize;
    let k = k.min(v.len() - 1);
    v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    v[k].max(0.05)
}

#[cfg(test)]
mod tests {
    /// Float mean of a flat, noisy dark patch over three noise realisations; linear modes assert drift bounds.
    #[test]
    #[ignore]
    fn probe_nrc_mean_drift() {
        use lightcraft_develop::DevelopSettings;
        let (w, h) = (768usize, 768usize);
        let mut worst = 0.0f64;
        for seed0 in [0x2545_f491_4f6c_dd1du64, 0x9e37_79b9_7f4a_7c15, 0x1234_5678_9abc_def1] {
            let mut seed = seed0;
            let mut gauss = || {
                (0..4)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        (seed >> 11) as f32 / (1u64 << 53) as f32 - 0.5
                    })
                    .sum::<f32>()
                    * 1.732
            };
            for (name, base, sigma) in [
                ("dark warm, blue near 0", [0.006f32, 0.004, 0.0008], [0.002f32, 0.0015, 0.003]),
                ("cat-like, red near 0", [0.0015, 0.004, 0.009], [0.004, 0.003, 0.005]),
            ] {
                let mut img = Rgb32f::new(w, h);
                for p in img.data.iter_mut() {
                    *p = [0, 1, 2].map(|i| base[i] + sigma[i] * gauss());
                }
                let mean = |im: &Rgb32f| {
                    let n = im.data.len() as f64;
                    [0, 1, 2].map(|i| im.data.iter().map(|p| p[i] as f64).sum::<f64>() / n)
                };
                let m0 = mean(&img);
                for nr in [25.0, 100.0] {
                    let mut s = DevelopSettings::default();
                    s.detail.nr_color = nr;
                    let mut a = img.clone();
                    color_nr_v2(&mut a, &s, w, w);
                    let m = mean(&a);
                    let clipped0 = mean(&img.map(|p| p.map(|v| v.max(0.0))));
                    let sd = |im: &Rgb32f, mm: [f64; 3]| {
                        [0, 1, 2].map(|i| (im.data.iter().map(|p| (p[i] as f64 - mm[i]).powi(2)).sum::<f64>() / im.data.len() as f64).sqrt())
                    };
                    if nrc_linear() {
                        worst = worst.max([0, 1, 2].iter().map(|&i| (m[i] / m0[i] - 1.0).abs()).fold(0.0, f64::max));
                    }
                    println!(
                        "{name} NR {nr}: true mean {m0:.5?}, clipped noisy mean {clipped0:.5?}, after v2 {m:.5?} → {:+.3?} % vs true; noise sd {:.5?} → {:.5?}",
                        [0, 1, 2].map(|i| 100.0 * (m[i] / m0[i] - 1.0)),
                        sd(&img, m0),
                        sd(&a, m)
                    );
                }
            }
        }
        if nrc_linear() {
            println!("worst mean deviation {:.4} %", 100.0 * worst);
            let limit = if nrc_joint_enabled() { 0.0005 } else { 0.01 };
            assert!(worst < limit, "linear mode moved a channel's mean by {:.3} %", 100.0 * worst);
        }
    }

    /// PROBE (prints, never fails): what colour NR v2 keeps of real colour. A clean synthetic scene (an
    /// equal-luminance red/green edge, a 2 px red line on grey, a saturated blue patch near black, a flat grey
    /// patch) plus noise; each measure compares the output with the clean scene.
    #[test]
    #[ignore]
    fn probe_nrc_real_colour() {
        use lightcraft_develop::DevelopSettings;
        let (w, h) = (768usize, 512usize);
        let y_of = |c: [f32; 3]| luminance_2020(c);
        // two colours of the same luminance 0.03
        let red = [0.08f32, 0.0155, 0.02];
        let green = {
            let g = [0.012f32, 0.04, 0.015];
            let k = y_of(red) / y_of(g);
            g.map(|v| v * k)
        };
        let grey = [0.03f32; 3];
        let blue_dark = [0.001f32, 0.0015, 0.02];
        // a bright saturated green next to black (a lit leaf against a dark background)
        let leaf = [0.02f32, 0.12, 0.01];
        let clean = |x: usize, y: usize| -> [f32; 3] {
            if y < 256 {
                if x < 384 { red } else { green }
            } else if x < 384 {
                if (190..192).contains(&x) { red } else { grey }
            } else if x < 576 {
                blue_dark
            } else if y < 384 {
                grey
            } else if x < 672 {
                leaf
            } else {
                [0.0005f32; 3]
            }
        };
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut gauss = || {
            (0..4)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 11) as f32 / (1u64 << 53) as f32 - 0.5
                })
                .sum::<f32>()
                * 1.732
        };
        let mut img = Rgb32f::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = clean(x, y);
                // shot + read noise per channel
                img.data[y * w + x] = c.map(|v| v + (0.00002 + 0.004 * v.max(0.0)).sqrt() * 0.6 * gauss());
            }
        }
        let chroma = |c: [f32; 3]| {
            let y = y_of(c);
            c.map(|v| v - y)
        };
        let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        for nr in [25.0, 100.0] {
            let mut s = DevelopSettings::default();
            s.detail.nr_color = nr;
            let mut o = img.clone();
            color_nr_v2(&mut o, &s, w, w);
            let px = |x: usize, y: usize| o.data[y * w + x];
            // edge: chroma error 1..8 px either side, relative to the step between the two colours
            let step = dist(chroma(red), chroma(green));
            let mut e = 0.0;
            let mut n = 0.0;
            for y in 40..216 {
                for d in 1..=8 {
                    for x in [384 - d, 383 + d] {
                        e += dist(chroma(px(x, y)), chroma(clean(x, y)));
                        n += 1.0;
                    }
                }
            }
            // line: share of the line's chroma kept (mean over its rows)
            let (mut kept, mut m) = (0.0, 0.0);
            for y in 300..480 {
                for x in 190..192 {
                    let c = chroma(px(x, y));
                    let r = chroma(red);
                    let rg = chroma(grey);
                    let full = [0, 1, 2].map(|i| r[i] - rg[i]);
                    let got = [0, 1, 2].map(|i| c[i] - rg[i]);
                    kept += (got[0] * full[0] + got[1] * full[1] + got[2] * full[2]) / (full[0] * full[0] + full[1] * full[1] + full[2] * full[2]);
                    m += 1.0;
                }
            }
            // saturated blue near black: chroma of the patch's mean kept
            let mut acc = [0.0f32; 3];
            let mut cnt = 0.0;
            for y in 300..480 {
                for x in 420..540 {
                    acc = [0, 1, 2].map(|i| acc[i] + px(x, y)[i]);
                    cnt += 1.0;
                }
            }
            let mean = acc.map(|v| v / cnt);
            let sat_kept = dist(chroma(mean), [0.0; 3]) / dist(chroma(blue_dark), [0.0; 3]);
            // grey: residual chroma noise
            let mut sd = 0.0;
            let mut k = 0.0;
            for y in 270..370 {
                for x in 600..760 {
                    sd += dist(chroma(px(x, y)), [0.0; 3]).powi(2);
                    k += 1.0;
                }
            }
            let mut sd0 = 0.0;
            for y in 270..370 {
                for x in 600..760 {
                    sd0 += dist(chroma(img.data[y * w + x]), [0.0; 3]).powi(2);
                }
            }
            // leaf next to black: chroma kept in the leaf's 12 px next to the black (mean over rows)
            let (mut lk, mut ln) = (0.0f32, 0.0f32);
            for y in 400..500 {
                for x in 660..672 {
                    let c = chroma(px(x, y));
                    let r = chroma(leaf);
                    lk += (c[0] * r[0] + c[1] * r[1] + c[2] * r[2]) / (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                    ln += 1.0;
                }
            }
            println!("NR {nr}: leaf next to black keeps {:.0} % of its chroma (12 px next to the black)", 100.0 * lk / ln);
            println!(
                "NR {nr}: edge error {:.1} % of the step | red line kept {:.0} % | dark blue chroma kept {:.0} % | grey chroma noise {:.5} → {:.5}",
                100.0 * e / n / step,
                100.0 * kept / m,
                100.0 * sat_kept,
                (sd0 / k).sqrt(),
                (sd / k).sqrt()
            );
        }
    }

    #[test]
    fn nrc_noise_fit_is_non_negative() {
        // a falling cloud would give a negative slope: the constant fits instead
        let (a, b) = fit_nonneg_line(&[(0.0, 4.0, 1.0), (1.0, 3.0, 1.0), (2.0, 2.0, 1.0)]);
        assert!(a == 0.0 && (b - 3.0).abs() < 1e-9);
        // a line through a negative intercept: refitted through the origin
        let (a, b) = fit_nonneg_line(&[(1.0, 1.0, 1.0), (2.0, 3.0, 1.0), (3.0, 5.0, 1.0)]);
        assert!(a > 0.0 && b == 0.0);
        // an exact shot + read noise line is found as it is
        let (a, b) = fit_nonneg_line(&[(0.0, 2.0, 1.0), (1.0, 5.0, 1.0), (2.0, 8.0, 1.0)]);
        assert!((a - 3.0).abs() < 1e-9 && (b - 2.0).abs() < 1e-9);
    }
    use super::*;

    #[test]
    fn guided_preserves_step_edges_and_flattens_texture() {
        let p = Plane::from_fn(80, 20, |x, y| if x < 40 { -2.0 } else { 2.0 } + if (x + y) % 2 == 0 { 0.05 } else { -0.05 });
        let q = guided(&p, 4.0, 0.3);
        // edge kept
        assert!(q.get(35, 10) < -1.5 && q.get(45, 10) > 1.5);
        // checkerboard texture removed away from the edge
        assert!((q.get(10, 10) - q.get(11, 10)).abs() < 0.02);
    }

    #[test]
    fn wb_as_shot_is_identity_and_warming_adds_red() {
        let s = DevelopSettings::default();
        let info = SourceInfo::default();
        let mut img = Rgb32f::filled(4, 4, [0.3, 0.3, 0.3]);
        scene_linear_pre(&mut img, &info, &s);
        assert!((img.get(0, 0)[0] - 0.3).abs() < 1e-6);
        let mut warm = s.clone();
        warm.wb.mode = lightcraft_develop::WbMode::Custom;
        warm.wb.temp = 9000.0;
        let mut img2 = Rgb32f::filled(4, 4, [0.3, 0.3, 0.3]);
        scene_linear_pre(&mut img2, &info, &warm);
        let c = img2.get(0, 0);
        assert!(c[0] > c[2], "{c:?}");
        // luminance preserved
        assert!((luminance_2020(c) - 0.3).abs() < 0.01);
    }

    #[test]
    fn exposure_doubles() {
        let mut s = DevelopSettings::default();
        s.light.exposure = 1.0;
        let mut img = Rgb32f::filled(2, 2, [0.1, 0.2, 0.3]);
        scene_linear_pre(&mut img, &SourceInfo::default(), &s);
        assert!((img.get(1, 1)[2] - 0.6).abs() < 1e-5);
    }
}

#[cfg(test)]
mod fast_tests {
    use super::*;

    #[test]
    fn fast_guided_close_to_reference() {
        let p = Plane::from_fn(160, 120, |x, y| ((x as f32 * 0.13).sin() + (y as f32 * 0.07).cos()) * 2.0 + if x > 80 { 1.5 } else { 0.0 });
        let a = guided(&p, 12.0, 0.3);
        let b = guided_fast(&p, 12.0, 0.3);
        let err: f32 = a.data.iter().zip(&b.data).map(|(u, v)| (u - v).abs()).sum::<f32>() / a.len() as f32;
        assert!(err < 0.08, "{err}");
    }
}

#[cfg(test)]
mod nr_tests {
    use super::*;

    #[test]
    fn luminance_nr_reduces_noise_keeps_mean() {
        let mut img = Rgb32f::from_fn(64, 64, |x, y| {
            let n = (((x * 7919 + y * 104729) % 97) as f32 / 97.0 - 0.5) * 0.06;
            [0.2 + n; 3]
        });
        let var = |im: &Rgb32f| {
            let m = im.data.iter().map(|p| p[1]).sum::<f32>() / im.len() as f32;
            (im.data.iter().map(|p| (p[1] - m).powi(2)).sum::<f32>() / im.len() as f32, m)
        };
        let (v0, m0) = var(&img);
        let mut s = DevelopSettings::default();
        s.detail.nr_luminance = 80.0;
        denoise(&mut img, &s, 64, 64);
        let (v1, m1) = var(&img);
        assert!(v1 < v0 * 0.5, "{v0} -> {v1}");
        assert!((m1 - m0).abs() < 0.01);
    }
}
