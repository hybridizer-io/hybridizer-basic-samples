/* (c) ALTIMESH 2026 -- all rights reserved */
//
// `ldmatrix.sync.aligned.m8n8.*` bindings — warp-collective shared-memory
// matrix loads that distribute 8x8 b16 submatrices across the 32 lanes of
// a warp. Targets the layout that `mma.sync.aligned.m16n8k16` expects,
// avoiding the unspecified-layout pitfalls of populating
// nvcuda::wmma::fragment via raw inline asm.
//
// Available since sm_75. No toolkit header exposes these as named
// functions in CUDA 13.0 (cccl/cuda/__ptx covers cp.async, mbarrier,
// tcgen05, etc. but not ldmatrix), so the underlying helpers live in
// hybridizer.cuda.cuh under namespace `hybridizer::ldmatrix` — kept in
// the runtime header alongside the other PTX wrappers Hybridizer ships.
//
// `.trans` variants transpose each 8x8 in-flight during the load — used
// when the data is in shared memory in the layout opposite to what the
// downstream mma.sync operand expects (e.g. a row-major B against a
// col-major MMA operand).
//
// Per-lane shared-memory address must be a `.shared` 32-bit value, not a
// generic pointer — use <see cref="op.generic_to_shared"/> to convert.
//
using System;

#pragma warning disable 1591
namespace Hybridizer.Runtime.CUDAImports.ldmatrix
{
    /// <summary>
    /// Warp-collective <c>ldmatrix.sync.aligned</c> helpers. Every lane must
    /// call (uniformly) and provide a shared-memory address. The load is a
    /// single PTX instruction that distributes the loaded 8x8 b16 matrices
    /// across the warp's registers; each output uint32 holds two halves.
    /// </summary>
    public static class op
    {
        /// <summary>Convert a generic pointer to a shared-memory address (PTX <c>cvta.to.shared</c>).</summary>
        [IntrinsicFunction("hybridizer::ldmatrix::generic_to_shared")]
        public static unsafe uint generic_to_shared(void* p) => 0;

        /// <summary>Load 1 of 8x8 b16. Lane <c>i</c> provides address for row <c>i</c> of the matrix.</summary>
        [IntrinsicFunction("hybridizer::ldmatrix::x1_b16")]
        public static void x1_b16(out uint d0, uint shmem_addr) { d0 = 0; }

        /// <summary>Load 2 of 8x8 b16. Lanes 0..7 → matrix 0, lanes 8..15 → matrix 1.</summary>
        [IntrinsicFunction("hybridizer::ldmatrix::x2_b16")]
        public static void x2_b16(out uint d0, out uint d1, uint shmem_addr) { d0 = d1 = 0; }

        /// <summary>
        /// Load 4 of 8x8 b16. Lanes 0..7 / 8..15 / 16..23 / 24..31 provide
        /// row addresses for matrices 0/1/2/3 respectively. Each lane ends
        /// up holding 2 halves from each of the 4 matrices (4 uint32 = 8 b16).
        /// </summary>
        [IntrinsicFunction("hybridizer::ldmatrix::x4_b16")]
        public static void x4_b16(out uint d0, out uint d1, out uint d2, out uint d3, uint shmem_addr)
        {
            d0 = d1 = d2 = d3 = 0;
        }

        /// <summary>Load 4 of 8x8 b16, transposing each 8x8 during the load.</summary>
        [IntrinsicFunction("hybridizer::ldmatrix::x4_trans_b16")]
        public static void x4_trans_b16(out uint d0, out uint d1, out uint d2, out uint d3, uint shmem_addr)
        {
            d0 = d1 = d2 = d3 = 0;
        }
    }
}
#pragma warning restore 1591
