/* (c) ALTIMESH 2026 -- all rights reserved */
//
// `mma.sync.aligned.m16n8k16.*` bindings — Ampere/Ada/Blackwell tensor-core
// MMA at the PTX level. This is the building block CUTLASS uses to compose
// fp16/bf16/tf32/fp8 GEMMs at all sizes; for hand-written kernels that
// want full control over the shmem→register→tensor-core path (paired
// with ldmatrix), it's the right primitive.
//
// Available since sm_80 for the fp16 / tf32 / bf16 forms; sm_89+ for fp8.
//
// Each call is a warp-collective operation. For
// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`:
//   * A operand: 4 uint32 per lane (8 halves, m16 × k16, row-major)
//   * B operand: 2 uint32 per lane (4 halves, k16 × n8, col-major)
//   * C operand: 4 fp32 per lane (m16 × n8 accumulator)
//   * D output : 4 fp32 per lane (in-place accumulate when D == C)
// Per-lane storage is the registered layout `ldmatrix.x4(.trans)` produces.
//
// Phase 1 scope (this file):
//   * m16n8k16 f32 = f16 * f16 + f32
//
// Deferred to follow-up PRs:
//   * m16n8k8  f32 = f16 * f16 + f32  (smaller K)
//   * m16n8k16 f16 = f16 * f16 + f16  (half accumulator)
//   * m16n8k16 f32 = bf16 * bf16 + f32
//   * m16n8k8  f32 = tf32 * tf32 + f32
//   * m16n8k32 / m16n8k64 fp8/int8 (Ada+)
//
using System;

#pragma warning disable 1591
namespace Hybridizer.Runtime.CUDAImports.mma
{
    /// <summary>
    /// Tensor-core matrix multiply-accumulate. Warp-collective: every lane
    /// must call uniformly with its own 4+2+4 register operand split. The
    /// D and C operands are folded into the same <c>ref float</c> args for
    /// in-place accumulate (matches CUTLASS's <c>Mma_sm80</c> pattern).
    /// </summary>
    public static class op
    {
        /// <summary>
        /// <c>mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32</c>:
        /// 16×8 fp32 = 16×16 fp16 × 16×8 fp16 + 16×8 fp32. Operates in-place
        /// on the accumulator (D == C); the 4 <c>ref float</c> args are both
        /// the input C and the output D.
        /// </summary>
        [IntrinsicFunction("hybridizer::mma::m16n8k16_f32_f16_f16_f32")]
        public static void m16n8k16_f32_f16_f16_f32(
            ref float dc0, ref float dc1, ref float dc2, ref float dc3,
            uint a0, uint a1, uint a2, uint a3,
            uint b0, uint b1) { }
    }
}
#pragma warning restore 1591
