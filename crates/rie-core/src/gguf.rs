//! Zero-copy GGUF (v2/v3) reader.
//!
//! The file is memory-mapped; tensor data is never copied for the CPU backend.
//! On mobile this matters: the OS can page weights in lazily and evict them under
//! memory pressure without the app being killed for its resident set size.

use crate::error::{bail, Error, Result};
use crate::quant::GgmlType;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little endian
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::I8(v) if v >= 0 => v as u64,
            Value::U16(v) => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::U32(v) => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::U64(v) => v,
            Value::I64(v) if v >= 0 => v as u64,
            Value::Bool(b) => b as u64,
            _ => return None,
        })
    }
    pub fn as_i64(&self) -> Option<i64> {
        Some(match *self {
            Value::I8(v) => v as i64,
            Value::I16(v) => v as i64,
            Value::I32(v) => v as i64,
            Value::I64(v) => v,
            _ => return self.as_u64().map(|v| v as i64),
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Value::F32(v) => v as f64,
            Value::F64(v) => v,
            _ => return self.as_i64().map(|v| v as f64),
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => self.as_u64().map(|v| v != 0),
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in ggml order: `dims[0]` is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte offset in the file.
    pub offset: u64,
    pub size_bytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    pub fn row_len(&self) -> usize {
        self.dims[0] as usize
    }
    pub fn n_rows(&self) -> usize {
        (self.n_elements() / self.dims[0]) as usize
    }
}

enum Storage {
    Mmap(memmap2::Mmap),
    Owned(Vec<u8>),
}

/// A parsed GGUF file backed by a memory map.
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    storage: Storage,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            bail!(Format, "unexpected end of file at byte {}", self.pos);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        if len > self.buf.len() {
            bail!(Format, "string length {len} out of range");
        }
        let bytes = self.take(len)?;
        // Some vocabularies contain invalid UTF-8 fragments; keep them lossily.
        Ok(match std::str::from_utf8(bytes) {
            Ok(s) => s.to_owned(),
            Err(_) => String::from_utf8_lossy(bytes).into_owned(),
        })
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => {
                if depth > 4 {
                    bail!(Format, "array nesting too deep");
                }
                let elem_ty = self.u32()?;
                let len = self.u64()? as usize;
                if len > self.buf.len() {
                    bail!(Format, "array length {len} out of range");
                }
                let mut v = Vec::with_capacity(len);
                for _ in 0..len {
                    v.push(self.value(elem_ty, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            t => bail!(Format, "unknown metadata value type {t}"),
        })
    }
}

impl GgufFile {
    /// Memory-map and parse a GGUF file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: the file must not be truncated while mapped; model files are read-only assets.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        #[cfg(unix)]
        {
            // Weights are streamed sequentially once per token: hint the kernel accordingly.
            let _ = mmap.advise(memmap2::Advice::WillNeed);
        }
        Self::parse(Storage::Mmap(mmap))
    }

    /// Parse a GGUF image held in memory (used by tests and by apps that load from assets).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(Storage::Owned(bytes))
    }

    fn parse(storage: Storage) -> Result<Self> {
        let buf: &[u8] = match &storage {
            Storage::Mmap(m) => &m[..],
            Storage::Owned(v) => &v[..],
        };
        let mut r = Reader { buf, pos: 0 };
        if r.u32()? != GGUF_MAGIC {
            bail!(Format, "not a GGUF file (bad magic)");
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            bail!(Unsupported, "GGUF version {version} (only v2 and v3 are supported)");
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;
        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let val = r.value(ty, 0)?;
            metadata.insert(key, val);
        }
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);

        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            if n_dims == 0 || n_dims > 4 {
                bail!(Format, "tensor {name} has {n_dims} dims");
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ty_id = r.u32()?;
            let ty = GgmlType::from_id(ty_id)
                .ok_or_else(|| Error::Unsupported(format!("tensor {name} uses ggml type id {ty_id}")))?;
            let rel = r.u64()?;
            tensors.push((name, dims, ty, rel));
        }
        let data_start = (r.pos as u64).div_ceil(alignment) * alignment;
        let mut out = Vec::with_capacity(tensors.len());
        let mut tensor_index = HashMap::with_capacity(tensors.len());
        for (name, dims, ty, rel) in tensors {
            let n: u64 = dims.iter().product();
            if dims[0] % ty.block_size() as u64 != 0 {
                bail!(Format, "tensor {name}: row length {} not a multiple of block size", dims[0]);
            }
            let size_bytes = n / ty.block_size() as u64 * ty.block_bytes() as u64;
            let offset = data_start + rel;
            if offset + size_bytes > buf.len() as u64 {
                bail!(Format, "tensor {name} extends past end of file");
            }
            tensor_index.insert(name.clone(), out.len());
            out.push(TensorInfo { name, dims, ty, offset, size_bytes });
        }
        Ok(GgufFile { version, metadata, tensors: out, tensor_index, storage })
    }

    pub fn data(&self) -> &[u8] {
        match &self.storage {
            Storage::Mmap(m) => &m[..],
            Storage::Owned(v) => &v[..],
        }
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        &self.data()[t.offset as usize..(t.offset + t.size_bytes) as usize]
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(Value::as_f64).map(|v| v as f32)
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(Value::as_bool)
    }
    pub fn get_arr(&self, key: &str) -> Option<&[Value]> {
        self.get(key).and_then(Value::as_array)
    }

    /// Best-effort hint to the OS that the whole weight region will be needed soon.
    pub fn prefetch(&self) {
        #[cfg(unix)]
        if let Storage::Mmap(m) = &self.storage {
            let _ = m.advise(memmap2::Advice::WillNeed);
        }
    }

    /// Optionally lock weights in RAM (mlock). Fails silently when not permitted.
    pub fn lock_in_memory(&self) -> bool {
        #[cfg(unix)]
        if let Storage::Mmap(m) = &self.storage {
            // SAFETY: valid mapping for the lifetime of self.
            return unsafe { libc::mlock(m.as_ptr() as *const libc::c_void, m.len()) == 0 };
        }
        false
    }
}

/// Minimal GGUF writer, used by tests, benchmarks and tooling to build synthetic models.
pub mod writer {
    use super::Value;
    use crate::quant::GgmlType;

    pub struct GgufWriter {
        kv: Vec<(String, Value)>,
        tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
    }

    impl Default for GgufWriter {
        fn default() -> Self {
            Self::new()
        }
    }

    impl GgufWriter {
        pub fn new() -> Self {
            GgufWriter { kv: Vec::new(), tensors: Vec::new() }
        }
        pub fn kv(&mut self, key: &str, v: Value) -> &mut Self {
            self.kv.push((key.to_string(), v));
            self
        }
        pub fn tensor(&mut self, name: &str, dims: &[u64], ty: GgmlType, data: Vec<u8>) -> &mut Self {
            let n: u64 = dims.iter().product();
            assert_eq!(
                data.len() as u64,
                n / ty.block_size() as u64 * ty.block_bytes() as u64,
                "tensor {name} data size mismatch"
            );
            self.tensors.push((name.to_string(), dims.to_vec(), ty, data));
            self
        }

        fn write_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        fn type_id(v: &Value) -> u32 {
            match v {
                Value::U8(_) => 0,
                Value::I8(_) => 1,
                Value::U16(_) => 2,
                Value::I16(_) => 3,
                Value::U32(_) => 4,
                Value::I32(_) => 5,
                Value::F32(_) => 6,
                Value::Bool(_) => 7,
                Value::String(_) => 8,
                Value::Array(_) => 9,
                Value::U64(_) => 10,
                Value::I64(_) => 11,
                Value::F64(_) => 12,
            }
        }
        fn write_val(out: &mut Vec<u8>, v: &Value) {
            match v {
                Value::U8(x) => out.push(*x),
                Value::I8(x) => out.push(*x as u8),
                Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::Bool(x) => out.push(*x as u8),
                Value::String(s) => Self::write_str(out, s),
                Value::Array(a) => {
                    let et = a.first().map(Self::type_id).unwrap_or(4);
                    out.extend_from_slice(&et.to_le_bytes());
                    out.extend_from_slice(&(a.len() as u64).to_le_bytes());
                    for e in a {
                        Self::write_val(out, e);
                    }
                }
                Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
            }
        }

        pub fn to_bytes(&self) -> Vec<u8> {
            const ALIGN: usize = 32;
            let mut out = Vec::new();
            out.extend_from_slice(&super::GGUF_MAGIC.to_le_bytes());
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
            for (k, v) in &self.kv {
                Self::write_str(&mut out, k);
                out.extend_from_slice(&Self::type_id(v).to_le_bytes());
                Self::write_val(&mut out, v);
            }
            let mut offset = 0usize;
            for (name, dims, ty, data) in &self.tensors {
                Self::write_str(&mut out, name);
                out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                out.extend_from_slice(&(ty.id()).to_le_bytes());
                out.extend_from_slice(&(offset as u64).to_le_bytes());
                offset += data.len().div_ceil(ALIGN) * ALIGN;
            }
            out.resize(out.len().div_ceil(ALIGN) * ALIGN, 0);
            for (_, _, _, data) in &self.tensors {
                out.extend_from_slice(data);
                out.resize(out.len().div_ceil(ALIGN) * ALIGN, 0);
            }
            out
        }
    }
}
