// WMMA + PTX kernel glue for the WmmaGemm sample.
//
// This is *project-local* sugar — the raw PTX wrappers it builds on
// (hybridizer::ldmatrix::*, hybridizer::mma::*, __pipeline_memcpy_async)
// live in Hybridizer.Runtime.CUDAImports + Transcoder.Cuda/hybridizer.cuda.cuh.
// Each helper here folds the (array, offset) shape Hybridizer hands kernel
// authors into the (raw_pointer, ...) shape the upstream bindings take.
// When CUDAImports grows array-typed overloads on its pipeline/ldmatrix/mma
// modules, this whole file collapses to nothing.
#pragma once

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

// hybridizer::ldmatrix::* / hybridizer::mma::* — these symbols live in
// Transcoder.Cuda/hybridizer.cuda.cuh which the standalone Hybridizer build
// already includes unconditionally. No explicit #include needed here.

namespace matmul {

// ===== nvcuda::wmma helpers — global-pointer variants =====
// The nvcuda::wmma intrinsics only take (frag, ptr, ldm); for tile addressing
// inside a kernel we need (frag, base + off, ldm). These thin wrappers fold
// the offset into the pointer at the call site so Hybridizer can stay on the
// array-typed C# signatures.
//
// `base` is typed `const void*` because Hybridizer maps the C# IntPtr to
// `void*` on the device. We pass raw device pointers from .NET to avoid
// Hybridizer's array marshaller — it re-copies on every kernel call and
// dominates the WMMA timing.

__device__ inline void wmma_load_a_16x16x16_half_row(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& f,
    const void* base, int off, unsigned int ldm)
{
    nvcuda::wmma::load_matrix_sync(f, ((const half*)base) + off, ldm);
}

__device__ inline void wmma_load_b_16x16x16_half_row(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& f,
    const void* base, int off, unsigned int ldm)
{
    nvcuda::wmma::load_matrix_sync(f, ((const half*)base) + off, ldm);
}

__device__ inline void wmma_store_c_16x16x16_float(
    const void* base, int off,
    const nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& f,
    unsigned int ldm, nvcuda::wmma::layout_t layout)
{
    nvcuda::wmma::store_matrix_sync(((float*)base) + off, f, ldm, layout);
}

// ===== nvcuda::wmma helpers — shared-memory variants =====
// `base` is `half*` because the C# shmem allocation is declared with the
// element type and decays cleanly. Symmetric with the global-pointer variants
// above; the difference is only the storage class of the input pointer.

__device__ inline void wmma_load_a_shmem_16x16x16_half_row(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major>& f,
    const half* base, int off, unsigned int ldm)
{
    nvcuda::wmma::load_matrix_sync(f, base + off, ldm);
}

__device__ inline void wmma_load_b_shmem_16x16x16_half_row(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major>& f,
    const half* base, int off, unsigned int ldm)
{
    nvcuda::wmma::load_matrix_sync(f, base + off, ldm);
}

// ===== cp.async + scalar load glue =====

// Single-element global load used for the cooperative shmem stage in the
// non-async kernels. C# can't subscript an IntPtr, so this is the one
// intrinsic that lets the kernel fetch one half at a time from the
// device-buffer base address.
__device__ inline half global_load_half(const void* base, int off)
{
    return ((const half*)base)[off];
}

// Hybridizer hands us `half[] sDst` (decays to half*) and `IntPtr gSrc`
// (decays to void*) with separate `int` offsets. The upstream
// pipeline::op::memcpy_async binding takes raw (void*, void*, size_t) and
// has no offset args, so we fold the offsets into the pointers here and
// hard-code the 16-byte (8-half) granularity. Will become a half[] overload
// inside CUDAImports::pipeline once that lands.
__device__ inline void cp_async_16B_half(
    half* sDst, int sOff, const void* gSrc, int gOff)
{
    __pipeline_memcpy_async(sDst + sOff, ((const half*)gSrc) + gOff, 16);
}

// ===== PTX MMA kernel glue =====
//
// Paired with the Mma.cs / Ldmatrix.cs bindings. The MultiplyKernelWmmaBigPtx
// kernel holds A/B in uint32 register locals and C in float locals, then
// drives mma.sync.m16n8k16 directly — same approach as CUTLASS::Mma_sm80. We
// don't go through wmma::fragment here because its `x[]` storage layout per
// lane is documented as "unspecified" and writing ldmatrix output into it
// silently produces garbage on some configurations.
//
// Lane-to-tile-cell mapping for `ldmatrix.x4` over a row-major 16x16 half
// tile when the target is mma.m16n8k16. The 4 8x8 submatrices have to be
// arranged COLUMN-MAJOR within the 16x16 (top-left, bottom-left, top-right,
// bottom-right) so the per-lane register order matches what mma.m16n8k16
// expects:
//   reg[0] = top-left 8x8, reg[1] = bottom-left, reg[2] = top-right, reg[3] = bottom-right
// This is the CUTLASS convention. Mapping per lane L (within a warp):
//   L &  7              -> row within an 8x8
//   ((L >> 3) & 1) * 8  -> +8 row offset (bottom submatrix vs top)
//   ((L >> 4) & 1) * 8  -> +8 col offset (right submatrix vs left)
__device__ inline int ldmatrix_x4_row_offset(int lane)
{
    return (lane & 7) + (((lane >> 3) & 1) << 3);
}
__device__ inline int ldmatrix_x4_col_offset(int lane)
{
    return ((lane >> 4) & 1) << 3;
}

// Load one 16x16 half row-major tile of A. Each lane provides one row-of-one-
// 8x8-submatrix address; ldmatrix.x4 distributes the 4 submatrices across the
// warp's regs, which are the m16k16 row-major A operand of mma.m16n8k16.
__device__ inline void ptx_load_a_m16k16(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const half* sBase, int row_base, int col_base, int ldm)
{
    int lane = threadIdx.x & 31;
    int r = ldmatrix_x4_row_offset(lane);
    int c = ldmatrix_x4_col_offset(lane);
    uint32_t addr = hybridizer::ldmatrix::generic_to_shared(sBase + (row_base + r) * ldm + (col_base + c));
    hybridizer::ldmatrix::x4_b16(r0, r1, r2, r3, addr);
}

// Load one 16x16 half row-major tile of B as the k16n16 col-major operand
// expected by mma.m16n8k16. `.trans` flips each 8x8 in-register. The 4 regs
// hold (in lane-broadcast layout): reg0=(k=0..7, n=0..7), reg1=(k=8..15, n=0..7),
// reg2=(k=0..7, n=8..15), reg3=(k=8..15, n=8..15). So mma cols 0..7 uses
// (r0, r1) and cols 8..15 uses (r2, r3) — see the call sites in the kernel.
__device__ inline void ptx_load_b_k16n16(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const half* sBase, int row_base, int col_base, int ldm)
{
    int lane = threadIdx.x & 31;
    int r = ldmatrix_x4_row_offset(lane);
    int c = ldmatrix_x4_col_offset(lane);
    uint32_t addr = hybridizer::ldmatrix::generic_to_shared(sBase + (row_base + r) * ldm + (col_base + c));
    hybridizer::ldmatrix::x4_trans_b16(r0, r1, r2, r3, addr);
}

// Write one m16n8 fp32 accumulator back to row-major C. Per PTX spec, lane T
// holds 4 elements at: (row T/4, col 2*(T%4)),       (row T/4, col 2*(T%4)+1),
//                      (row T/4 + 8, col 2*(T%4)),  (row T/4 + 8, col 2*(T%4)+1).
//
// The (c0, c1) and (c2, c3) pairs sit at consecutive even-aligned column
// indices, so we emit them as float2 (st.global.v2.f32) — one 8-byte store
// instead of two 4-byte stores. The scalar form hits only 16/32 bytes per
// sector (lanes 0..3 cover cols 0,2,4,6 of a row); float2 has lanes 0..3
// cover cols 0..7 contiguously and the cacheline is fully utilized.
__device__ inline void ptx_store_c_m16n8(
    void* cBase, int row_base, int col_base, int n,
    float c0, float c1, float c2, float c3)
{
    float* C = (float*)cBase;
    int lane = threadIdx.x & 31;
    int row = row_base + (lane >> 2);
    int col = col_base + ((lane & 3) << 1);
    *reinterpret_cast<float2*>(&C[ row      * n + col]) = make_float2(c0, c1);
    *reinterpret_cast<float2*>(&C[(row + 8) * n + col]) = make_float2(c2, c3);
}

} // namespace matmul
