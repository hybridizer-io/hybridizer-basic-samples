/* (c) ALTIMESH 2026 -- all rights reserved */
//
// nvcuda::wmma sibling — async memcpy helpers built on `cp.async` PTX
// (sm_80+). Binds the global-namespace free functions in <cuda_pipeline.h>
// directly, matching the shape of wmma.cs: untemplated, primitive args,
// single [IntrinsicIncludeCUDA] on the static class.
//
// On sm < 80 the toolkit header's fallback degrades to a synchronous
// memcpy, so the binding is safe to surface unconditionally — same
// architecture-gating model as the WMMA bindings.
//
// All three calls are warp/lane-uniform: every thread that issues an
// op.memcpy_async contributes exactly one in-flight copy to the current
// commit group; op.commit closes the group and op.wait_prior(N) blocks
// until at most N groups remain outstanding (so wait_prior(0) drains all
// pending copies).
//
using System;

#pragma warning disable 1591
namespace Hybridizer.Runtime.CUDAImports.pipeline
{
    /// <summary>
    /// Warp/lane-uniform async memcpy helpers driven by <c>cp.async</c>.
    /// Producer/consumer pipelines layer these with shared-memory tile
    /// staging: issue a wave of <see cref="op.memcpy_async"/> calls, close
    /// the wave with <see cref="op.commit"/>, then <see cref="op.wait_prior"/>
    /// when the consumer needs the data.
    /// </summary>
    [IntrinsicIncludeCUDA("<cuda_pipeline.h>")]
    public static class op
    {
        /// <summary>
        /// Asynchronously copy <paramref name="size_and_align"/> bytes from
        /// generic device memory at <paramref name="src_global"/> into shared
        /// memory at <paramref name="dst_shared"/>.
        ///
        /// <para><paramref name="size_and_align"/> must be one of 4, 8, or 16.
        /// The 16-byte form maps to a single <c>cp.async.ca.shared.global</c>
        /// PTX instruction and is the natural granularity for tile staging.
        /// </para>
        /// </summary>
        [IntrinsicFunction("__pipeline_memcpy_async")]
        public static unsafe void memcpy_async(
            void* dst_shared, void* src_global, ulong size_and_align) { }

        /// <summary>
        /// Close the current commit group. Every <see cref="memcpy_async"/>
        /// issued since the previous <c>commit</c> becomes one in-flight group.
        /// </summary>
        [IntrinsicFunction("__pipeline_commit")]
        public static void commit() { }

        /// <summary>
        /// Block the calling warp/lane until at most <paramref name="N"/>
        /// previously committed groups are still in flight. Use
        /// <c>wait_prior(0)</c> to fully drain pending copies.
        /// </summary>
        [IntrinsicFunction("__pipeline_wait_prior")]
        public static void wait_prior(ulong N) { }
    }
}
#pragma warning restore 1591
