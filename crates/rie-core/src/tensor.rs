//! Weight tensor views.

use crate::quant::{dequantize_row, GgmlType};
use std::sync::Arc;

/// A 2-D weight matrix `[rows, cols]` in ggml block layout (row-major, `cols` contiguous).
/// Usually a zero-copy view into the memory-mapped model file.
#[derive(Clone)]
pub struct Weight {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    ptr: *const u8,
    len: usize,
    _owner: Option<Arc<Vec<u8>>>,
}

// SAFETY: the data is immutable and kept alive either by the model's mmap or by `_owner`.
unsafe impl Send for Weight {}
unsafe impl Sync for Weight {}

impl std::fmt::Debug for Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weight({:?}, {}x{})", self.ty, self.rows, self.cols)
    }
}

impl Weight {
    /// # Safety
    /// `data` must outlive the returned value (it is a view into the model mmap).
    pub unsafe fn from_raw(ty: GgmlType, rows: usize, cols: usize, data: &[u8]) -> Self {
        assert_eq!(data.len(), ty.row_bytes(cols) * rows, "weight size mismatch");
        Weight { ty, rows, cols, ptr: data.as_ptr(), len: data.len(), _owner: None }
    }

    pub fn owned(ty: GgmlType, rows: usize, cols: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), ty.row_bytes(cols) * rows, "weight size mismatch");
        let owner = Arc::new(data);
        Weight { ty, rows, cols, ptr: owner.as_ptr(), len: owner.len(), _owner: Some(owner) }
    }

    pub fn from_f32(rows: usize, cols: usize, data: &[f32]) -> Self {
        Self::owned(GgmlType::F32, rows, cols, crate::quant::quantize::f32(data))
    }

    #[inline]
    pub fn data(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub fn row_bytes(&self) -> usize {
        self.ty.row_bytes(self.cols)
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[u8] {
        let rb = self.row_bytes();
        &self.data()[r * rb..(r + 1) * rb]
    }

    /// A view of rows `[start, start+n)` (used to split fused QKV / gate-up tensors).
    pub fn slice_rows(&self, start: usize, n: usize) -> Weight {
        assert!(start + n <= self.rows);
        let rb = self.row_bytes();
        Weight {
            ty: self.ty,
            rows: n,
            cols: self.cols,
            ptr: unsafe { self.ptr.add(start * rb) },
            len: n * rb,
            _owner: self._owner.clone(),
        }
    }

    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        dequantize_row(self.ty, self.row(r), &mut out[..self.cols]);
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let mut v = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            self.dequant_row(r, &mut v[r * self.cols..(r + 1) * self.cols]);
        }
        v
    }

    pub fn size_bytes(&self) -> usize {
        self.len
    }
}
