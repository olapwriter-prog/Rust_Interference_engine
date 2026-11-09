//! # rie-gpu — GPU backend (Vulkan on Android, Metal on iOS/macOS) via wgpu
//!
//! The whole transformer step runs on the GPU: weights are uploaded once (repacked into
//! 4-byte-aligned layouts), activations and the f16 KV cache stay in GPU memory, and only
//! the final logits are read back. Per layer the command stream is
//! `rmsnorm → QKV matmul → bias/qk-norm/RoPE → KV store → attention → O matmul →
//! residual → rmsnorm → gate/up matmul → GLU → down matmul → residual`.
//!
//! Supported weight formats on the GPU: F16, Q8_0, Q4_0, Q4_K, Q6_K natively; other
//! types (Q4_1, Q5_x, Q2_K, Q3_K, Q5_K, BF16, F32) are converted at load time to Q8_0 or
//! F16.

mod shaders;

use rie_core::error::{Error, Result};
use rie_core::model::{Activation, Backend, Model};
use rie_core::quant::{dequantize_row, f16_at, GgmlType, QK_K};
use rie_core::tensor::Weight;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub n_ctx: usize,
    pub max_batch: usize,
    /// Prefer a low-power (integrated) adapter; phones only have one anyway.
    pub low_power: bool,
    /// 8-bit KV cache (Q8_0-style blocks, ~56% of the f16 cache). Needs head_dim % 32 == 0.
    pub kv_q8: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        GpuConfig { n_ctx: 4096, max_batch: 32, low_power: false, kv_q8: false }
    }
}

/// Describe the available GPU adapters.
pub fn list_adapters() -> Vec<String> {
    let instance = wgpu::Instance::default();
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .map(|a| {
            let i = a.get_info();
            let l = a.limits();
            format!(
                "{} ({:?}, {:?}, driver {} {}), max storage binding {} MB",
                i.name,
                i.backend,
                i.device_type,
                i.driver,
                i.driver_info,
                l.max_storage_buffer_binding_size >> 20
            )
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
enum GpuFmt {
    F16,
    Q8_0,
    Q4_0,
    Q4_K,
    Q6_K,
}

impl GpuFmt {
    /// u32 words per 32 values (or per 256 for k-quants, see `words_per_row`).
    fn words_per_row(self, cols: usize) -> usize {
        match self {
            GpuFmt::F16 => cols / 2,
            GpuFmt::Q8_0 => cols / 32 * 9,
            GpuFmt::Q4_0 => cols / 32 * 5,
            GpuFmt::Q4_K => cols / QK_K * 38,
            GpuFmt::Q6_K => cols / QK_K * 53,
        }
    }
    fn dequant_src(self) -> &'static str {
        match self {
            GpuFmt::F16 => shaders::DEQ_F16,
            GpuFmt::Q8_0 => shaders::DEQ_Q8_0,
            GpuFmt::Q4_0 => shaders::DEQ_Q4_0,
            GpuFmt::Q4_K => shaders::DEQ_Q4_K,
            GpuFmt::Q6_K => shaders::DEQ_Q6_K,
        }
    }
    const ALL: [GpuFmt; 5] = [GpuFmt::F16, GpuFmt::Q8_0, GpuFmt::Q4_0, GpuFmt::Q4_K, GpuFmt::Q6_K];
}

fn push_f32(out: &mut Vec<u32>, v: f32) {
    out.push(v.to_bits());
}

fn push_bytes(out: &mut Vec<u32>, b: &[u8]) {
    debug_assert!(b.len() % 4 == 0);
    for c in b.chunks_exact(4) {
        out.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
}

/// Repack one weight row into the GPU layout.
fn repack_row(w: &Weight, r: usize, fmt: GpuFmt, out: &mut Vec<u32>, tmp: &mut Vec<f32>) {
    let row = w.row(r);
    match (w.ty, fmt) {
        (GgmlType::F16, GpuFmt::F16) => push_bytes(out, row),
        (GgmlType::Q8_0, GpuFmt::Q8_0) => {
            for b in row.chunks_exact(34) {
                push_f32(out, f16_at(b, 0));
                push_bytes(out, &b[2..34]);
            }
        }
        (GgmlType::Q4_0, GpuFmt::Q4_0) => {
            for b in row.chunks_exact(18) {
                push_f32(out, f16_at(b, 0));
                push_bytes(out, &b[2..18]);
            }
        }
        (GgmlType::Q4_K, GpuFmt::Q4_K) => {
            for b in row.chunks_exact(144) {
                push_f32(out, f16_at(b, 0));
                push_f32(out, f16_at(b, 2));
                let (sc, mn) = rie_core::quant::reference::unpack_k4_scales(&b[4..16]);
                push_bytes(out, &sc);
                push_bytes(out, &mn);
                push_bytes(out, &b[16..144]);
            }
        }
        (GgmlType::Q6_K, GpuFmt::Q6_K) => {
            for b in row.chunks_exact(210) {
                push_f32(out, f16_at(b, 208));
                push_bytes(out, &b[192..208]);
                push_bytes(out, &b[0..128]);
                push_bytes(out, &b[128..192]);
            }
        }
        (_, GpuFmt::F16) => {
            tmp.resize(w.cols, 0.0);
            dequantize_row(w.ty, row, tmp);
            push_bytes(out, &rie_core::quant::quantize::f16(tmp));
        }
        (_, GpuFmt::Q8_0) => {
            tmp.resize(w.cols, 0.0);
            dequantize_row(w.ty, row, tmp);
            let q = rie_core::quant::quantize::q8_0(tmp);
            for b in q.chunks_exact(34) {
                push_f32(out, f16_at(b, 0));
                push_bytes(out, &b[2..34]);
            }
        }
        _ => unreachable!(),
    }
}

fn gpu_fmt_for(ty: GgmlType) -> GpuFmt {
    match ty {
        GgmlType::F16 | GgmlType::F32 | GgmlType::BF16 => GpuFmt::F16,
        GgmlType::Q8_0 => GpuFmt::Q8_0,
        GgmlType::Q4_0 => GpuFmt::Q4_0,
        GgmlType::Q4_K => GpuFmt::Q4_K,
        GgmlType::Q6_K => GpuFmt::Q6_K,
        _ => GpuFmt::Q8_0,
    }
}

struct Chunk {
    buf: wgpu::Buffer,
    row_start: usize,
    n_rows: usize,
}

#[allow(dead_code)]
struct GpuMatrix {
    fmt: GpuFmt,
    rows: usize,
    cols: usize,
    chunks: Vec<Chunk>,
}

/// One recorded dispatch.
struct Dispatch {
    pipeline: usize,
    bind: wgpu::BindGroup,
    /// Workgroup counts; `Tok` variants are resolved per forward from n_tok.
    size: Size,
}

#[derive(Clone, Copy)]
enum Size {
    Fixed(u32, u32, u32),
    /// (x, y) fixed, z = ceil(n_tok / 8)
    MatmulTok(u32, u32),
    /// x = n_tok
    PerTok,
    /// x = ceil(n_tok * n / 64) (possibly 2-D)
    Elems(u32),
    /// (n_head, n_tok)
    HeadsTok(u32),
}

pub struct GpuBackend {
    model: Arc<Model>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    pipelines: Vec<wgpu::ComputePipeline>,
    dispatches: Vec<Dispatch>,
    step_buf: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    logits_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    n_ctx: usize,
    max_batch: usize,
    embd_row: Vec<f32>,
    x_host: Vec<f32>,
    // keep resources alive
    _keep: Vec<wgpu::Buffer>,
    _mats: Vec<GpuMatrix>,
    vram_bytes: usize,
}

// pipeline indices
const P_RMS: usize = 5;
const P_RMS_IP: usize = 6;
const P_HEAD: usize = 7;
const P_BIAS: usize = 8;
const P_KV: usize = 9;
const P_ATT: usize = 10;
const P_KV_Q8: usize = 14;
const P_ATT_Q8: usize = 15;
const P_GLU: usize = 11;
const P_RES: usize = 12;
const P_LOGITS: usize = 13;

impl GpuBackend {
    pub fn new(model: Arc<Model>, cfg: GpuConfig) -> Result<Self> {
        let gerr = |e: &dyn std::fmt::Display| Error::Runtime(format!("GPU: {e}"));
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: if cfg.low_power { wgpu::PowerPreference::LowPower } else { wgpu::PowerPreference::HighPerformance },
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| gerr(&e))?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < 4 {
            return Err(Error::Unsupported("GPU supports fewer than 4 storage buffers per stage".into()));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rie"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .map_err(|e| gerr(&e))?;
        let max_binding = (limits.max_storage_buffer_binding_size).min(limits.max_buffer_size) as usize;

        if rie_core::model::is_moe(&model) {
            return Err(Error::Unsupported("mixture-of-experts models run on the CPU backend only".into()));
        }
        let c = model.cfg.clone();
        let n_ctx = cfg.n_ctx;
        let max_batch = cfg.max_batch.clamp(1, 64).min(n_ctx);
        let (e, qd, kvd, hd, ff, nv) = (c.n_embd, c.q_dim(), c.kv_dim(), c.head_dim, c.n_ff, c.n_vocab);
        if hd % 2 != 0 {
            return Err(Error::Unsupported("odd head dimension".into()));
        }
        let mut vram = 0usize;

        // ---- pipelines ----
        let mk = |src: String, label: &str| -> wgpu::ComputePipeline {
            let module = device
                .create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let mut pipelines = Vec::new();
        for f in GpuFmt::ALL {
            pipelines
                .push(mk(format!("{}{}", shaders::STEP, shaders::MATMUL.replace("DEQUANT", f.dequant_src())), &format!("matmul_{f:?}")));
        }
        for (src, name) in [
            (shaders::RMSNORM, "rmsnorm"),
            (shaders::RMSNORM_INPLACE, "rmsnorm_inplace"),
            (shaders::HEAD_PREP, "head_prep"),
            (shaders::BIAS, "bias"),
            (shaders::KV_STORE, "kv_store"),
            (&*shaders::ATTENTION.replace("KVREAD", shaders::KVREAD_F16), "attention"),
            (shaders::GLU, "glu"),
            (shaders::RESIDUAL, "residual"),
            (shaders::LOGITS_POST, "logits_post"),
            (shaders::KV_STORE_Q8, "kv_store_q8"),
            (&*shaders::ATTENTION.replace("KVREAD", shaders::KVREAD_Q8), "attention_q8"),
        ] {
            pipelines.push(mk(format!("{}{}", shaders::STEP, src), name));
        }
        let kv_q8 = cfg.kv_q8 && hd % 32 == 0;
        let (p_kv, p_att) = if kv_q8 { (P_KV_Q8, P_ATT_Q8) } else { (P_KV, P_ATT) };

        // ---- buffers ----
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let mut keep: Vec<wgpu::Buffer> = Vec::new();
        let new_buf = |size: usize, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (size.max(4) * 4) as u64,
                usage: storage,
                mapped_at_creation: false,
            })
        };
        let init_f32 = |data: &[f32], label: &str| -> wgpu::Buffer {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &bytes, usage: storage })
        };
        let uniform = |words: [u32; 8]| -> wgpu::Buffer {
            let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let step_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b = max_batch;
        let x_buf = new_buf(b * e, "x");
        let xn_buf = new_buf(b * e, "xn");
        let q_buf = new_buf(b * qd, "q");
        let k_buf = new_buf(b * kvd, "k");
        let v_buf = new_buf(b * kvd, "v");
        let att_buf = new_buf(b * qd, "att");
        let o_buf = new_buf(b * e, "o");
        let gate_buf = new_buf(b * ff, "gate");
        let up_buf = new_buf(b * ff, "up");
        let scores_buf = new_buf(b * c.n_head * n_ctx, "scores");
        let logits_buf = new_buf(nv, "logits");
        let dummy = new_buf(4, "dummy");
        vram += 4 * (2 * b * e * 2 + 2 * b * qd + 2 * b * kvd + 2 * b * ff + b * c.n_head * n_ctx + nv);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (nv * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- weights ----
        let mut mats: Vec<GpuMatrix> = Vec::new();
        let upload = |mats: &mut Vec<GpuMatrix>, vram: &mut usize, w: &Weight, label: &str| -> Result<usize> {
            let fmt = gpu_fmt_for(w.ty);
            let wpr = fmt.words_per_row(w.cols);
            let rows_per_chunk = (max_binding / (wpr * 4)).max(1);
            let mut chunks = Vec::new();
            let mut tmp = Vec::new();
            let mut r0 = 0;
            while r0 < w.rows {
                let n = rows_per_chunk.min(w.rows - r0);
                let mut words = Vec::with_capacity(n * wpr);
                for r in r0..r0 + n {
                    repack_row(w, r, fmt, &mut words, &mut tmp);
                }
                debug_assert_eq!(words.len(), n * wpr);
                let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
                *vram += bytes.len();
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });
                chunks.push(Chunk { buf, row_start: r0, n_rows: n });
                r0 += n;
            }
            mats.push(GpuMatrix { fmt, rows: w.rows, cols: w.cols, chunks });
            Ok(mats.len() - 1)
        };

        let mut dispatches: Vec<Dispatch> = Vec::new();
        let bind = |pl: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]| -> wgpu::BindGroup {
            let layout = pl.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry> =
                bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries })
        };
        // Records matmul dispatches for matrix `m` reading X (stride x_stride) into Y.
        let matmul = |mats: &Vec<GpuMatrix>,
                      dispatches: &mut Vec<Dispatch>,
                      keep: &mut Vec<wgpu::Buffer>,
                      m: usize,
                      x: &wgpu::Buffer,
                      x_stride: usize,
                      y: &wgpu::Buffer,
                      ld: usize,
                      fixed_ntok: u32,
                      rows_limit: usize| {
            let mat = &mats[m];
            let pi = GpuFmt::ALL.iter().position(|f| *f == mat.fmt).unwrap();
            for ch in &mat.chunks {
                if ch.row_start >= rows_limit {
                    break;
                }
                let n_rows = ch.n_rows.min(rows_limit - ch.row_start);
                let u = uniform([
                    n_rows as u32,
                    mat.cols as u32,
                    mat.fmt.words_per_row(mat.cols) as u32,
                    ch.row_start as u32,
                    ld as u32,
                    x_stride as u32,
                    fixed_ntok,
                    0,
                ]);
                let bg = bind(&pipelines[pi], &[&step_buf, &u, &ch.buf, x, y]);
                keep.push(u);
                let gx = (n_rows as u32).min(32768);
                let gy = (n_rows as u32).div_ceil(gx);
                let size = if fixed_ntok > 0 { Size::Fixed(gx, gy, 1) } else { Size::MatmulTok(gx, gy) };
                dispatches.push(Dispatch { pipeline: pi, bind: bg, size });
            }
        };
        let fbits = |v: f32| v.to_bits();

        // rope inverse frequencies
        let inv = |base: f32, scale: f32, factors: Option<&[f32]>| -> Vec<f32> {
            rie_core::ops::Rope::new(c.rope_dim, c.rope_neox, base, scale, factors).inv_freq
        };
        let inv_global = init_f32(&inv(c.rope_base, c.rope_freq_scale, model.rope_factors.as_deref()), "inv_freq");
        let inv_swa = c.rope_base_swa.map(|bs| init_f32(&inv(bs, 1.0, None), "inv_freq_swa"));

        for (il, lw) in model.layers.iter().enumerate() {
            let swa = c.is_swa(il);
            let slots = if swa { (c.swa_window + max_batch).min(n_ctx) } else { n_ctx };
            let row_words = if kv_q8 { hd / 32 * 9 } else { hd / 2 };
            let v_off = c.n_head_kv * slots * row_words;
            let cache = new_buf(2 * v_off, "kv");
            vram += 8 * v_off;
            // attention norm
            let attn_norm = init_f32(&lw.attn_norm, "attn_norm");
            let u = uniform([e as u32, 0, fbits(c.rms_eps), 0, 0, 0, 0, 0]);
            dispatches.push(Dispatch {
                pipeline: P_RMS,
                bind: bind(&pipelines[P_RMS], &[&step_buf, &u, &x_buf, &attn_norm, &xn_buf]),
                size: Size::PerTok,
            });
            keep.extend([u, attn_norm]);
            // QKV
            let mq = upload(&mut mats, &mut vram, &lw.wq, "wq")?;
            let mk_ = upload(&mut mats, &mut vram, &lw.wk, "wk")?;
            let mv = upload(&mut mats, &mut vram, &lw.wv, "wv")?;
            matmul(&mats, &mut dispatches, &mut keep, mq, &xn_buf, e, &q_buf, qd, 0, usize::MAX);
            matmul(&mats, &mut dispatches, &mut keep, mk_, &xn_buf, e, &k_buf, kvd, 0, usize::MAX);
            matmul(&mats, &mut dispatches, &mut keep, mv, &xn_buf, e, &v_buf, kvd, 0, usize::MAX);
            // head prep (bias + qk-norm + rope + q scale)
            let inv_buf = if swa { inv_swa.as_ref().unwrap_or(&inv_global) } else { &inv_global };
            for (is_q, buf, n_heads, norm, bias, scale) in
                [(true, &q_buf, c.n_head, &lw.q_norm, &lw.bq, c.attn_scale), (false, &k_buf, c.n_head_kv, &lw.k_norm, &lw.bk, 1.0)]
            {
                let mut nw = vec![0.0f32; hd];
                if let Some(n) = norm {
                    nw.copy_from_slice(n);
                }
                if let Some(bv) = bias {
                    nw.extend_from_slice(bv);
                }
                let nwb = init_f32(&nw, if is_q { "q_aux" } else { "k_aux" });
                let u = uniform([
                    n_heads as u32,
                    hd as u32,
                    c.rope_dim as u32,
                    c.rope_neox as u32,
                    norm.is_some() as u32,
                    bias.is_some() as u32,
                    fbits(c.rms_eps),
                    fbits(scale),
                ]);
                dispatches.push(Dispatch {
                    pipeline: P_HEAD,
                    bind: bind(&pipelines[P_HEAD], &[&step_buf, &u, buf, inv_buf, &nwb]),
                    size: Size::Elems(n_heads as u32),
                });
                keep.extend([u, nwb]);
            }
            if let Some(bv) = &lw.bv {
                let bb = init_f32(bv, "bv");
                let u = uniform([kvd as u32, 0, 0, 0, 0, 0, 0, 0]);
                dispatches.push(Dispatch {
                    pipeline: P_BIAS,
                    bind: bind(&pipelines[P_BIAS], &[&step_buf, &u, &v_buf, &bb]),
                    size: Size::Elems(kvd as u32),
                });
                keep.extend([u, bb]);
            }
            // KV store
            let u = uniform([c.n_head_kv as u32, hd as u32, slots as u32, v_off as u32, 0, 0, 0, 0]);
            dispatches.push(Dispatch {
                pipeline: p_kv,
                bind: bind(&pipelines[p_kv], &[&step_buf, &u, &k_buf, &v_buf, &cache]),
                size: Size::Elems((c.n_head_kv * if kv_q8 { hd / 32 } else { hd / 2 }) as u32),
            });
            keep.push(u);
            // attention
            let u = uniform([
                c.n_head as u32,
                c.n_head_kv as u32,
                hd as u32,
                slots as u32,
                if swa { c.swa_window as u32 } else { 0 },
                n_ctx as u32,
                v_off as u32,
                fbits(c.attn_softcap.unwrap_or(0.0)),
            ]);
            dispatches.push(Dispatch {
                pipeline: p_att,
                bind: bind(&pipelines[p_att], &[&step_buf, &u, &q_buf, &cache, &att_buf, &scores_buf]),
                size: Size::HeadsTok(c.n_head as u32),
            });
            keep.extend([u, cache]);
            // output projection + post norm + residual
            let mo = upload(&mut mats, &mut vram, &lw.wo, "wo")?;
            matmul(&mats, &mut dispatches, &mut keep, mo, &att_buf, qd, &o_buf, e, 0, usize::MAX);
            let post = |dispatches: &mut Vec<Dispatch>, keep: &mut Vec<wgpu::Buffer>, w: &Option<Vec<f32>>| {
                if let Some(w) = w {
                    let wb = init_f32(w, "post_norm");
                    let u = uniform([e as u32, 0, fbits(c.rms_eps), 0, 0, 0, 0, 0]);
                    dispatches.push(Dispatch {
                        pipeline: P_RMS_IP,
                        bind: bind(&pipelines[P_RMS_IP], &[&step_buf, &u, &wb, &o_buf]),
                        size: Size::PerTok,
                    });
                    keep.extend([u, wb]);
                }
                let u = uniform([e as u32, 0, fbits(c.residual_scale), 0, 0, 0, 0, 0]);
                dispatches.push(Dispatch {
                    pipeline: P_RES,
                    bind: bind(&pipelines[P_RES], &[&step_buf, &u, &x_buf, &o_buf]),
                    size: Size::Elems(e as u32),
                });
                keep.push(u);
            };
            post(&mut dispatches, &mut keep, &lw.post_attn_norm);
            // FFN
            let ffn_norm = init_f32(&lw.ffn_norm, "ffn_norm");
            let u = uniform([e as u32, 0, fbits(c.rms_eps), 0, 0, 0, 0, 0]);
            dispatches.push(Dispatch {
                pipeline: P_RMS,
                bind: bind(&pipelines[P_RMS], &[&step_buf, &u, &x_buf, &ffn_norm, &xn_buf]),
                size: Size::PerTok,
            });
            keep.extend([u, ffn_norm]);
            let rie_core::model::Ffn::Dense(dffn) = &lw.ffn else { unreachable!("checked above") };
            let mg = upload(&mut mats, &mut vram, &dffn.gate, "w_gate")?;
            let mu = upload(&mut mats, &mut vram, &dffn.up, "w_up")?;
            matmul(&mats, &mut dispatches, &mut keep, mg, &xn_buf, e, &gate_buf, ff, 0, usize::MAX);
            matmul(&mats, &mut dispatches, &mut keep, mu, &xn_buf, e, &up_buf, ff, 0, usize::MAX);
            let u = uniform([ff as u32, (c.act == Activation::GeluTanh) as u32, 0, 0, 0, 0, 0, 0]);
            dispatches.push(Dispatch {
                pipeline: P_GLU,
                bind: bind(&pipelines[P_GLU], &[&step_buf, &u, &gate_buf, &up_buf]),
                size: Size::Elems(ff as u32),
            });
            keep.push(u);
            let md = upload(&mut mats, &mut vram, &dffn.down, "w_down")?;
            matmul(&mats, &mut dispatches, &mut keep, md, &gate_buf, ff, &o_buf, e, 0, usize::MAX);
            post(&mut dispatches, &mut keep, &lw.post_ffn_norm);
        }
        // output head on the last token only
        let onorm = init_f32(&model.output_norm, "output_norm");
        let u = uniform([e as u32, 1, fbits(c.rms_eps), 0, 0, 0, 0, 0]);
        dispatches.push(Dispatch {
            pipeline: P_RMS,
            bind: bind(&pipelines[P_RMS], &[&step_buf, &u, &x_buf, &onorm, &xn_buf]),
            size: Size::Fixed(1, 1, 1),
        });
        keep.extend([u, onorm]);
        let mout = upload(&mut mats, &mut vram, &model.output, "output")?;
        matmul(&mats, &mut dispatches, &mut keep, mout, &xn_buf, e, &logits_buf, nv, 1, nv);
        if c.logit_scale != 1.0 || c.final_softcap.is_some() {
            let u = uniform([nv as u32, 0, fbits(1.0 / c.logit_scale), fbits(c.final_softcap.unwrap_or(0.0)), 0, 0, 0, 0]);
            let groups = (nv as u32).div_ceil(64);
            let gx = groups.min(32768);
            dispatches.push(Dispatch {
                pipeline: P_LOGITS,
                bind: bind(&pipelines[P_LOGITS], &[&step_buf, &u, &logits_buf]),
                size: Size::Fixed(gx, groups.div_ceil(gx), 1),
            });
            keep.push(u);
        }
        keep.extend([xn_buf, q_buf, k_buf, v_buf, att_buf, o_buf, gate_buf, up_buf, scores_buf, dummy, inv_global]);
        if let Some(b) = inv_swa {
            keep.push(b);
        }
        let adapter_name = format!("{} ({:?})", info.name, info.backend);
        Ok(GpuBackend {
            embd_row: vec![0.0; e],
            x_host: vec![0.0; max_batch * e],
            model,
            device,
            queue,
            adapter_name,
            pipelines,
            dispatches,
            step_buf,
            x_buf,
            logits_buf,
            staging,
            n_ctx,
            max_batch,
            _keep: keep,
            _mats: mats,
            vram_bytes: vram,
        })
    }

    /// Approximate GPU memory allocated (weights + KV cache + activations).
    pub fn vram_bytes(&self) -> usize {
        self.vram_bytes
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> String {
        format!("GPU {}", self.adapter_name)
    }
    fn max_batch(&self) -> usize {
        self.max_batch
    }
    fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    fn forward(&mut self, tokens: &[u32], pos: usize, all_logits: bool, logits: &mut [f32]) -> Result<()> {
        let c = &self.model.cfg;
        let n = tokens.len();
        if n == 0 || n > self.max_batch {
            return Err(Error::Runtime(format!("GPU batch of {n} tokens (max {})", self.max_batch)));
        }
        if all_logits && n > 1 {
            // Score tokens one by one (used only by perplexity).
            let nv = c.n_vocab;
            for (i, &t) in tokens.iter().enumerate() {
                self.forward(&[t], pos + i, false, &mut logits[i * nv..(i + 1) * nv])?;
            }
            return Ok(());
        }
        if pos + n > self.n_ctx {
            return Err(Error::Runtime(format!("context window full ({} tokens)", self.n_ctx)));
        }
        let e = c.n_embd;
        for (t, &tok) in tokens.iter().enumerate() {
            if tok as usize >= self.model.token_embd.rows {
                return Err(Error::Runtime(format!("token id {tok} out of range")));
            }
            self.model.token_embd.dequant_row(tok as usize, &mut self.embd_row);
            for (d, s) in self.x_host[t * e..(t + 1) * e].iter_mut().zip(&self.embd_row) {
                *d = s * c.embd_scale;
            }
        }
        let xb: Vec<u8> = self.x_host[..n * e].iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue.write_buffer(&self.x_buf, 0, &xb);
        let step: Vec<u8> = [n as u32, pos as u32, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue.write_buffer(&self.step_buf, 0, &step);

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("forward") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("forward"), timestamp_writes: None });
            let n = n as u32;
            for d in &self.dispatches {
                let (x, y, z) = match d.size {
                    Size::Fixed(x, y, z) => (x, y, z),
                    Size::MatmulTok(x, y) => (x, y, n.div_ceil(8)),
                    Size::PerTok => (n, 1, 1),
                    Size::Elems(k) => {
                        let groups = (n * k).div_ceil(64);
                        if groups <= 65535 {
                            (groups, 1, 1)
                        } else {
                            let gx = 32768;
                            (gx, groups.div_ceil(gx), 1)
                        }
                    }
                    Size::HeadsTok(h) => (h, n, 1),
                };
                pass.set_pipeline(&self.pipelines[d.pipeline]);
                pass.set_bind_group(0, &d.bind, &[]);
                pass.dispatch_workgroups(x, y, z);
            }
        }
        let nv = c.n_vocab;
        enc.copy_buffer_to_buffer(&self.logits_buf, 0, &self.staging, 0, (nv * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| Error::Runtime(format!("GPU poll: {e}")))?;
        rx.recv().map_err(|_| Error::Runtime("GPU map channel closed".into()))?.map_err(|e| Error::Runtime(format!("GPU map: {e}")))?;
        {
            let data = slice.get_mapped_range().map_err(|e| Error::Runtime(format!("GPU map: {e:?}")))?;
            for (o, c) in logits[..nv].iter_mut().zip(data.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        self.staging.unmap();
        Ok(())
    }
}
