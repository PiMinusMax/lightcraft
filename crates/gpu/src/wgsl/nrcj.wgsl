// Joint colour noise reduction (`lightcraft_pipeline::nrc_joint`). Each kernel mirrors its CPU
// twin there, in the same order of operations; keep them in step.
//
// Buffers: `a`, `b` inputs, `dst` output. Images are interleaved: RGB / Y,R−Y,B−Y (3 per pixel),
// chroma pairs R−Y,B−Y (2 per pixel), or path/weight quads (path h, path v, weight h, weight v).
// Kernels with the noise model: P[0] = n, P[1] = w, P[2] = h, P[3..6] = alpha, P[6..9] = beta,
// P[9..12] = ymax, P[12] = tolerance.

fn ja(i: u32) -> vec3<f32> {
    return vec3<f32>(a[3u * i], a[3u * i + 1u], a[3u * i + 2u]);
}

fn jb3(i: u32) -> vec3<f32> {
    return vec3<f32>(b[3u * i], b[3u * i + 1u], b[3u * i + 2u]);
}

fn ja2(i: u32) -> vec2<f32> {
    return vec2<f32>(a[2u * i], a[2u * i + 1u]);
}

fn jvc(v: vec3<f32>, i: u32) -> f32 {
    return select(select(v.x, v.y, i == 1u), v.z, i == 2u);
}

// `JointNoise::var`.
fn nvar(i: u32, y: f32) -> f32 {
    let yy = min(y, pf(9u + i));
    return max(pf(3u + i) * max(yy, 0.0) + pf(6u + i), 1e-14);
}

// `nrc_joint::distance`.
fn jdist(p: vec3<f32>, q: vec3<f32>) -> f32 {
    let y = 0.5 * (p.x + q.x);
    let t = 0.025 * max(y, 0.0);
    let allowance = t * t;
    var sum = 0.0;
    for (var i = 0u; i < 3u; i++) {
        let v = nvar(i, y);
        let d = jvc(p, i) - jvc(q, i);
        let excess = max(d * d - v, 0.0);
        if (excess != 0.0) {
            sum += excess / (3.0 * pf(12u) * v + allowance);
        }
    }
    return sum;
}

// `nrc_joint::weight`.
fn jweight(d: f32, p: f32) -> f32 {
    let m = max(d, p);
    return select(exp(-m), 1.0, m == 0.0);
}

// `nrc_joint::yuv` of pixel `i` of image `a`.
fn jyuv(i: u32) -> vec3<f32> {
    let c = ja(i);
    let y = lum2020(c);
    return vec3<f32>(y, c.x - y, c.z - y);
}

// `nrc_joint::yuv` of `n` pixels of image `a` from pixel P[1] on (a band's rows): P[0] = n.
@compute @workgroup_size(256)
fn nrcj_yuv(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let c = jyuv(pu(1u) + i);
    dst[3u * i] = c.x;
    dst[3u * i + 1u] = c.y;
    dst[3u * i + 2u] = c.z;
}

// The chroma pair (R−Y, B−Y) of the transformed image `a` (P[0] = n).
@compute @workgroup_size(256)
fn nrcj_uv(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    dst[2u * i] = a[3u * i + 1u];
    dst[2u * i + 1u] = a[3u * i + 2u];
}

// Noise samples of image `a`, transformed on the fly (`nrc_joint::noise_model`): P[0] = sample count,
// P[1] = w, P[2] = h, P[3] = stride. Per sample 11 values: valid, brightness, 3 × 3 residuals.
@compute @workgroup_size(256)
fn nrcj_samples(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let k = lin_index(g, nw);
    if (k >= pu(0u)) {
        return;
    }
    let w = pu(1u);
    let h = pu(2u);
    let index = k * pu(3u);
    let x = index % w;
    let y = index / w;
    let o = 11u * k;
    if (x == 0u || y == 0u || x + 1u == w || y + 1u == h) {
        dst[o] = 0.0;
        return;
    }
    let c = jyuv(index);
    var brightness = 0.0;
    for (var s = 0u; s < 3u; s++) {
        let d = min(min(min(min(1u << s, x), y), w - x - 1u), h - y - 1u);
        var m = vec3<f32>(0.0);
        m = m + 0.25 * jyuv(index - d);
        m = m + 0.25 * jyuv(index + d);
        m = m + 0.25 * jyuv(index - d * w);
        m = m + 0.25 * jyuv(index + d * w);
        let r = c - m;
        dst[o + 2u + 3u * s] = r.x * r.x;
        dst[o + 3u + 3u * s] = r.y * r.y;
        dst[o + 4u + 3u * s] = r.z * r.z;
        brightness += m.x / 3.0;
    }
    dst[o] = 1.0;
    dst[o + 1u] = max(brightness, 0.0);
}

// `nrc_joint::pilot`: `a` = transformed image, `b` = its gaussian(0.8); writes the guide.
@compute @workgroup_size(256)
fn nrcj_pilot(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let w = pu(1u);
    let h = pu(2u);
    let x = i % w;
    let y = i / w;
    var out = jb3(i);
    if (!(x == 0u || y == 0u || x + 1u == w || y + 1u == h)) {
        let p = ja(i);
        var means: array<vec3<f32>, 2>;
        var variances = array<f32, 2>(0.0, 0.0);
        for (var axis = 0u; axis < 2u; axis++) {
            let off = select(1u, w, axis == 1u);
            let qa = ja(i - off);
            let qb = ja(i + off);
            let m = (qa + p + qb) / 3.0;
            means[axis] = m;
            var v = 0.0;
            for (var c = 0u; c < 3u; c++) {
                let mc = jvc(m, c);
                let da = jvc(qa, c) - mc;
                let dp = jvc(p, c) - mc;
                let db = jvc(qb, c) - mc;
                v += (da * da + dp * dp + db * db) / (9.0 * nvar(c, p.x));
            }
            variances[axis] = v;
        }
        if (variances[0] > max(4.0, 4.0 * variances[1])) {
            out = means[1];
        } else if (variances[1] > max(4.0, 4.0 * variances[0])) {
            out = means[0];
        }
    }
    dst[3u * i] = out.x;
    dst[3u * i + 1u] = out.y;
    dst[3u * i + 2u] = out.z;
}

// Paths and weights of the direct neighbours: `a` = guide; writes the quads.
@compute @workgroup_size(256)
fn nrcj_links1(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let w = pu(1u);
    let h = pu(2u);
    let x = i % w;
    let y = i / w;
    let p = ja(i);
    var q = vec4<f32>(0.0);
    if (x + 1u < w) {
        q.x = jdist(p, ja(i + 1u));
        q.z = jweight(q.x, q.x);
    }
    if (y + 1u < h) {
        q.y = jdist(p, ja(i + w));
        q.w = jweight(q.y, q.y);
    }
    dst[4u * i] = q.x;
    dst[4u * i + 1u] = q.y;
    dst[4u * i + 2u] = q.z;
    dst[4u * i + 3u] = q.w;
}

// Paths of length 2·delta from those of length delta (`b`) and the next scale's weights (guide `a`):
// P[13] = delta, P[14] = next.
@compute @workgroup_size(256)
fn nrcj_next(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let w = pu(1u);
    let h = pu(2u);
    let delta = pu(13u);
    let next = pu(14u);
    let x = i % w;
    let y = i / w;
    var ph = b[4u * i];
    var pv = b[4u * i + 1u];
    if (x + delta < w) {
        ph = max(ph, b[4u * (i + delta)]);
    }
    if (y + delta < h) {
        pv = max(pv, b[4u * (i + delta * w) + 1u]);
    }
    let p = ja(i);
    var lh = 0.0;
    var lv = 0.0;
    if (x + next < w) {
        lh = jweight(jdist(p, ja(i + next)), ph);
    }
    if (y + next < h) {
        lv = jweight(jdist(p, ja(i + next * w)), pv);
    }
    dst[4u * i] = ph;
    dst[4u * i + 1u] = pv;
    dst[4u * i + 2u] = lh;
    dst[4u * i + 3u] = lv;
}

// One horizontal diffusion step (`nrc_joint::h_row`): `a` = chroma pairs, `b` = quads.
// P[0] = n, P[1] = w, P[2] = h, P[3] = delta, P[4] = rate.
@compute @workgroup_size(256)
fn nrcj_dh(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let w = pu(1u);
    let delta = pu(3u);
    let rate = pf(4u);
    let x = i % w;
    let p = ja2(i);
    var o = p;
    if (x >= delta) {
        o += rate * b[4u * (i - delta) + 2u] * (ja2(i - delta) - p);
    }
    if (x + delta < w) {
        o += rate * b[4u * i + 2u] * (ja2(i + delta) - p);
    }
    dst[2u * i] = o.x;
    dst[2u * i + 1u] = o.y;
}

// One vertical diffusion step (`nrc_joint::v_row`), parameters as `nrcj_dh`.
@compute @workgroup_size(256)
fn nrcj_dv(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let w = pu(1u);
    let h = pu(2u);
    let delta = pu(3u);
    let rate = pf(4u);
    let y = i / w;
    let p = ja2(i);
    var o = p;
    if (y >= delta) {
        o += rate * b[4u * (i - delta * w) + 3u] * (ja2(i - delta * w) - p);
    }
    if (y + delta < h) {
        o += rate * b[4u * i + 3u] * (ja2(i + delta * w) - p);
    }
    dst[2u * i] = o.x;
    dst[2u * i + 1u] = o.y;
}

// Back to RGB with the image's own luminance, blended by the strength: `a` = image, `b` = a band's
// filtered chroma pairs, `dst` = the whole output. P[0] = pixels written, P[1] = strength,
// P[2] = first pixel written (in the image), P[3] = first pixel of the band (in the image).
@compute @workgroup_size(256)
fn nrcj_recon(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let k = lin_index(g, nw);
    if (k >= pu(0u)) {
        return;
    }
    let s = pf(1u);
    let i = pu(2u) + k;
    let bi = i - pu(3u);
    let p = ja(i);
    let y = lum2020(p);
    let r = y + b[2u * bi];
    let bl = y + b[2u * bi + 1u];
    let res = vec3<f32>(r, (y - 0.2627 * r - 0.0593 * bl) / 0.6780, bl);
    let o = p + s * (res - p);
    dst[3u * i] = o.x;
    dst[3u * i + 1u] = o.y;
    dst[3u * i + 2u] = o.z;
}
