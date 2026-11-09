//! WGSL compute shaders. Only core WGSL is used (no `shader-f16`, no subgroups) so the
//! same code runs on Adreno, Mali, Xclipse (RDNA), PowerVR, Apple GPUs and desktop GPUs.
//! f16 data (weights, KV cache) is packed in `u32` pairs and decoded with
//! `unpack2x16float`.

pub const STEP: &str = r#"
struct Step {
    n_tok: u32,
    pos0: u32,
    _p0: u32,
    _p1: u32,
};
@group(0) @binding(0) var<uniform> step: Step;
"#;

/// Matrix multiply `Y[t, row] = W[row] · X[t]`. `DEQUANT` is substituted per weight format.
/// One workgroup computes one row for up to 8 tokens; threads stride over 32-value sub-blocks.
pub const MATMUL: &str = r#"
struct P {
    rows: u32,          // rows in this chunk
    cols: u32,
    words_per_row: u32,
    row_offset: u32,    // first row of this chunk in the full matrix
    ld_out: u32,        // output stride between tokens
    x_stride: u32,      // input stride between tokens
    fixed_ntok: u32,    // 0: use step.n_tok
    accumulate: u32,    // 1: Y += result
};
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

const WG: u32 = 64u;
const TB: u32 = 8u;
var<workgroup> red: array<f32, 512>;
var<private> vals: array<f32, 32>;

fn byte_of(w: u32, b: u32) -> u32 { return (w >> (8u * b)) & 0xFFu; }

DEQUANT

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {
    let row = wg.x + wg.y * nwg.x;
    var ntok = step.n_tok;
    if (p.fixed_ntok != 0u) { ntok = p.fixed_ntok; }
    let t0 = wg.z * TB;
    if (row >= p.rows || t0 >= ntok) { return; }   // uniform per workgroup
    let nt = min(TB, ntok - t0);
    var acc: array<f32, 8>;
    for (var t = 0u; t < TB; t++) { acc[t] = 0.0; }
    let n_sub = p.cols / 32u;
    let rb = row * p.words_per_row;
    for (var s = tid; s < n_sub; s += WG) {
        dequant(rb, s);
        for (var t = 0u; t < nt; t++) {
            let xb = (t0 + t) * p.x_stride + s * 32u;
            var a = 0.0;
            for (var k = 0u; k < 32u; k += 4u) {
                a += vals[k] * X[xb + k] + vals[k + 1u] * X[xb + k + 1u]
                   + vals[k + 2u] * X[xb + k + 2u] + vals[k + 3u] * X[xb + k + 3u];
            }
            acc[t] += a;
        }
    }
    for (var t = 0u; t < TB; t++) { red[t * WG + tid] = acc[t]; }
    workgroupBarrier();
    for (var off = WG / 2u; off > 0u; off >>= 1u) {
        if (tid < off) {
            for (var t = 0u; t < TB; t++) { red[t * WG + tid] += red[t * WG + tid + off]; }
        }
        workgroupBarrier();
    }
    if (tid < nt) {
        let o = (t0 + tid) * p.ld_out + p.row_offset + row;
        if (p.accumulate != 0u) { Y[o] += red[tid * WG]; } else { Y[o] = red[tid * WG]; }
    }
}
"#;

pub const DEQ_F16: &str = r#"
fn dequant(rb: u32, s: u32) {
    let base = rb + s * 16u;
    for (var i = 0u; i < 16u; i++) {
        let v = unpack2x16float(W[base + i]);
        vals[2u * i] = v.x;
        vals[2u * i + 1u] = v.y;
    }
}
"#;

pub const DEQ_Q8_0: &str = r#"
fn dequant(rb: u32, s: u32) {
    let base = rb + s * 9u;
    let d = bitcast<f32>(W[base]);
    for (var i = 0u; i < 8u; i++) {
        let w = bitcast<i32>(W[base + 1u + i]);
        for (var b = 0u; b < 4u; b++) {
            vals[4u * i + b] = f32(extractBits(w, 8u * b, 8u)) * d;
        }
    }
}
"#;

pub const DEQ_Q4_0: &str = r#"
fn dequant(rb: u32, s: u32) {
    let base = rb + s * 5u;
    let d = bitcast<f32>(W[base]);
    for (var i = 0u; i < 4u; i++) {
        let w = W[base + 1u + i];
        for (var b = 0u; b < 4u; b++) {
            let q = byte_of(w, b);
            vals[4u * i + b] = (f32(q & 0xFu) - 8.0) * d;
            vals[16u + 4u * i + b] = (f32(q >> 4u) - 8.0) * d;
        }
    }
}
"#;

// Repacked Q4_K block (38 words / 256 values):
//   [0] d  [1] dmin  [2..4) 8 scales  [4..6) 8 mins  [6..38) 128 bytes of nibbles
pub const DEQ_Q4_K: &str = r#"
fn dequant(rb: u32, s: u32) {
    let base = rb + (s / 8u) * 38u;
    let j = s % 8u;
    let d = bitcast<f32>(W[base]);
    let dmin = bitcast<f32>(W[base + 1u]);
    let sc = f32(byte_of(W[base + 2u + j / 4u], j % 4u));
    let mn = f32(byte_of(W[base + 4u + j / 4u], j % 4u));
    let ds = d * sc;
    let ms = dmin * mn;
    let qb = base + 6u + (j / 2u) * 8u;
    let shift = (j % 2u) * 4u;
    for (var i = 0u; i < 8u; i++) {
        let w = W[qb + i];
        for (var b = 0u; b < 4u; b++) {
            vals[4u * i + b] = ds * f32((w >> (8u * b + shift)) & 0xFu) - ms;
        }
    }
}
"#;

// Repacked Q6_K block (53 words / 256 values):
//   [0] d  [1..5) 16 int8 scales  [5..37) ql (128 bytes)  [37..53) qh (64 bytes)
pub const DEQ_Q6_K: &str = r#"
fn dequant(rb: u32, s: u32) {
    let base = rb + (s / 8u) * 53u;
    let c = s % 8u;
    let n = c / 4u;
    let part = c % 4u;
    let d = bitcast<f32>(W[base]);
    let qlw = base + 5u + n * 16u + (part % 2u) * 8u;
    let qhw = base + 37u + n * 8u;
    let lshift = (part / 2u) * 4u;
    let hshift = 2u * part;
    let sidx0 = 8u * n + 2u * part;
    let s0 = d * f32(extractBits(bitcast<i32>(W[base + 1u + sidx0 / 4u]), 8u * (sidx0 % 4u), 8u));
    let sidx1 = sidx0 + 1u;
    let s1 = d * f32(extractBits(bitcast<i32>(W[base + 1u + sidx1 / 4u]), 8u * (sidx1 % 4u), 8u));
    for (var i = 0u; i < 8u; i++) {
        let lw = W[qlw + i];
        let hw = W[qhw + i];
        var sc = s0;
        if (i >= 4u) { sc = s1; }
        for (var b = 0u; b < 4u; b++) {
            let q = ((lw >> (8u * b + lshift)) & 0xFu) | (((hw >> (8u * b + hshift)) & 3u) << 4u);
            vals[4u * i + b] = sc * (f32(q) - 32.0);
        }
    }
}
"#;

/// RMSNorm over rows: `out[r] = in[row_in(r)] * rsqrt(mean(in²)+eps) * w`.
/// `last_only`: process only the last token row (output head), writing to row 0.
pub const RMSNORM: &str = r#"
struct P { n: u32, last_only: u32, eps: f32, _p: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    var r_in = wg.x;
    var r_out = wg.x;
    if (p.last_only != 0u) { r_in = step.n_tok - 1u; r_out = 0u; if (wg.x > 0u) { return; } }
    else if (wg.x >= step.n_tok) { return; }
    let bi = r_in * p.n;
    var ss = 0.0;
    for (var i = tid; i < p.n; i += 64u) { let v = inp[bi + i]; ss += v * v; }
    red[tid] = ss;
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (tid < off) { red[tid] += red[tid + off]; }
        workgroupBarrier();
    }
    let scale = inverseSqrt(red[0] / f32(p.n) + p.eps);
    let bo = r_out * p.n;
    for (var i = tid; i < p.n; i += 64u) { outp[bo + i] = inp[bi + i] * scale * w[i]; }
}
"#;

/// In-place RMSNorm (post-attention / post-FFN norms of Gemma 2/3).
pub const RMSNORM_INPLACE: &str = r#"
struct P { n: u32, _a: u32, eps: f32, _p: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    if (wg.x >= step.n_tok) { return; }
    let b = wg.x * p.n;
    var ss = 0.0;
    for (var i = tid; i < p.n; i += 64u) { let v = x[b + i]; ss += v * v; }
    red[tid] = ss;
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (tid < off) { red[tid] += red[tid + off]; }
        workgroupBarrier();
    }
    let scale = inverseSqrt(red[0] / f32(p.n) + p.eps);
    for (var i = tid; i < p.n; i += 64u) { x[b + i] = x[b + i] * scale * w[i]; }
}
"#;

/// Per-head: optional bias, optional RMSNorm, RoPE, scale. One thread per (token, head).
pub const HEAD_PREP: &str = r#"
struct P {
    n_heads: u32, hd: u32, rope_dim: u32, neox: u32,
    has_norm: u32, has_bias: u32, eps: f32, scale: f32,
};
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> inv_freq: array<f32>;
@group(0) @binding(4) var<storage, read> nw: array<f32>;   // [hd] norm weight, then [n_heads*hd] bias

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= step.n_tok * p.n_heads) { return; }
    let t = idx / p.n_heads;
    let h = idx % p.n_heads;
    let b = (t * p.n_heads + h) * p.hd;
    if (p.has_bias != 0u) {
        for (var i = 0u; i < p.hd; i++) { x[b + i] += nw[p.hd + h * p.hd + i]; }
    }
    if (p.has_norm != 0u) {
        var ss = 0.0;
        for (var i = 0u; i < p.hd; i++) { ss += x[b + i] * x[b + i]; }
        let sc = inverseSqrt(ss / f32(p.hd) + p.eps);
        for (var i = 0u; i < p.hd; i++) { x[b + i] = x[b + i] * sc * nw[i]; }
    }
    let pos = f32(step.pos0 + t);
    let half = p.rope_dim / 2u;
    for (var i = 0u; i < half; i++) {
        let theta = pos * inv_freq[i];
        let c = cos(theta);
        let s = sin(theta);
        var i0 = 2u * i;
        var i1 = 2u * i + 1u;
        if (p.neox != 0u) { i0 = i; i1 = i + half; }
        let a = x[b + i0];
        let bb = x[b + i1];
        x[b + i0] = a * c - bb * s;
        x[b + i1] = a * s + bb * c;
    }
    if (p.scale != 1.0) {
        for (var i = 0u; i < p.hd; i++) { x[b + i] *= p.scale; }
    }
}
"#;

/// Bias add for V (no RoPE): x[t, i] += bias[i].
pub const BIAS: &str = r#"
struct P { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= step.n_tok * p.n) { return; }
    x[gid.x] += bias[gid.x % p.n];
}
"#;

/// Store K and V rows into the packed-f16 cache ([k region | v region]).
pub const KV_STORE: &str = r#"
struct P { n_kv: u32, hd: u32, slots: u32, v_off: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> cache: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hp = p.hd / 2u;
    let per_tok = p.n_kv * hp;
    if (gid.x >= step.n_tok * per_tok) { return; }
    let t = gid.x / per_tok;
    let r = gid.x % per_tok;
    let g = r / hp;
    let j = r % hp;
    let slot = (step.pos0 + t) % p.slots;
    let src = t * p.n_kv * p.hd + g * p.hd + 2u * j;
    let dst = (g * p.slots + slot) * hp + j;
    cache[dst] = pack2x16float(vec2<f32>(k[src], k[src + 1u]));
    cache[p.v_off + dst] = pack2x16float(vec2<f32>(v[src], v[src + 1u]));
}
"#;

/// Causal / sliding-window GQA attention. One workgroup per (head, token).
pub const ATTENTION: &str = r#"
struct P {
    n_head: u32, n_kv: u32, hd: u32, slots: u32,
    window: u32, n_ctx: u32, v_off: u32, softcap: f32,
};
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> cache: array<u32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let h = wg.x;
    let t = wg.y;
    if (t >= step.n_tok) { return; }
    let g = h / (p.n_head / p.n_kv);
    let pos = step.pos0 + t;
    var start = 0u;
    if (p.window > 0u && pos + 1u > p.window) { start = pos + 1u - p.window; }
    let len = pos + 1u - start;
    let hp = p.hd / 2u;
    let qb = (t * p.n_head + h) * p.hd;
    let sb = (t * p.n_head + h) * p.n_ctx;
    // 1) scores + running max
    var mx = -3.0e38;
    for (var i = tid; i < len; i += 64u) {
        let slot = (start + i) % p.slots;
        let kb = (g * p.slots + slot) * hp;
        var s = 0.0;
        for (var j = 0u; j < hp; j++) {
            let kk = unpack2x16float(cache[kb + j]);
            s += q[qb + 2u * j] * kk.x + q[qb + 2u * j + 1u] * kk.y;
        }
        if (p.softcap > 0.0) { s = p.softcap * tanh(s / p.softcap); }
        scores[sb + i] = s;
        mx = max(mx, s);
    }
    red[tid] = mx;
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (tid < off) { red[tid] = max(red[tid], red[tid + off]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    // 2) exp + sum
    var sum = 0.0;
    for (var i = tid; i < len; i += 64u) {
        let e = exp(scores[sb + i] - m);
        scores[sb + i] = e;
        sum += e;
    }
    red[tid] = sum;
    workgroupBarrier();
    for (var off = 32u; off > 0u; off >>= 1u) {
        if (tid < off) { red[tid] += red[tid + off]; }
        workgroupBarrier();
    }
    let inv = 1.0 / red[0];
    storageBarrier();
    workgroupBarrier();
    // 3) weighted sum of V; each thread owns pairs of output dims
    for (var j = tid; j < hp; j += 64u) {
        var acc = vec2<f32>(0.0, 0.0);
        for (var i = 0u; i < len; i++) {
            let slot = (start + i) % p.slots;
            let vv = unpack2x16float(cache[p.v_off + (g * p.slots + slot) * hp + j]);
            acc += scores[sb + i] * vv;
        }
        outp[qb + 2u * j] = acc.x * inv;
        outp[qb + 2u * j + 1u] = acc.y * inv;
    }
}
"#;

/// gate = act(gate) * up
pub const GLU: &str = r#"
struct P { n: u32, gelu: u32, _a: u32, _b: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= step.n_tok * p.n) { return; }
    let g = gate[i];
    var a: f32;
    if (p.gelu != 0u) {
        a = 0.5 * g * (1.0 + tanh(0.7978845608 * g * (1.0 + 0.044715 * g * g)));
    } else {
        a = g / (1.0 + exp(-g));
    }
    gate[i] = a * up[i];
}
"#;

/// x += y * scale
pub const RESIDUAL: &str = r#"
struct P { n: u32, _a: u32, scale: f32, _b: u32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= step.n_tok * p.n) { return; }
    x[gid.x] += y[gid.x] * p.scale;
}
"#;

/// Final logit transforms: divide by logit_scale, optional tanh soft-capping.
pub const LOGITS_POST: &str = r#"
struct P { n: u32, _a: u32, inv_scale: f32, softcap: f32 };
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (i >= p.n || step.n_tok == 0u) { return; }
    var v = x[i] * p.inv_scale;
    if (p.softcap > 0.0) { v = p.softcap * tanh(v / p.softcap); }
    x[i] = v;
}
"#;
