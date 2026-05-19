using System.Runtime.InteropServices;
using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Math;

/// <summary>
/// Q8_0 matrix-vector kernels — Hybridizer entry points for OMP + CUDA flavors.
///
/// Layout assumption: the FP16 scale of each block has been pre-decoded to FP32
/// on the host side and lives in a separate `scales` array. The `values` array
/// holds only the int8 quantized values (32 per block, stored as `byte` because
/// .NET does not have an `sbyte[]` Hybridizer-compatible primitive — we cast
/// per-element inside the kernel). This split layout avoids needing a builtins
/// mapping for `BitConverter.UInt16BitsToHalf` or `__half2float` in the hot path.
///
/// Shape: per-row reduction, embarrassingly parallel across rows.
///   `Parallel.For(0, rows, row => result[row] = dot(row, vector))`
/// transcodes to:
///   • OMP: `#pragma omp parallel for` — one row per OMP thread
///   • CUDA: grid distribution — one thread (or thread block, depending on
///     SetDistrib) per row, sequential inner loop. No shared memory tiling
///     in this step-5 baseline; tile-and-reduce is a step-8 optimization.
/// </summary>
public class Q8Kernels
{
    public const int BlockElements = 32;

    /// <summary>
    /// result[row] = scale-weighted dot of quantized row `row` against `vector`.
    /// </summary>
    /// <param name="result">[rows] — output, overwritten.</param>
    /// <param name="values">[rows * blocksPerRow * 32] — int8 values stored as bytes.</param>
    /// <param name="scales">[rows * blocksPerRow] — per-block FP32 scales (pre-decoded from FP16 at load).</param>
    /// <param name="vector">[blocksPerRow * 32] — input vector.</param>
    /// <param name="rows">Number of output rows = number of rows of the quantized matrix.</param>
    /// <param name="blocksPerRow">Number of 32-element blocks per row. cols must equal blocksPerRow * 32.</param>
    [EntryPoint]
    public static void MatVecMul(
        [Out] float[] result,
        [In] byte[] values,
        [In] float[] scales,
        [In] float[] vector,
        int rows,
        int blocksPerRow)
    {
        Parallel.For(0, rows, row =>
        {
            int valuesRowOffset = row * blocksPerRow * BlockElements;
            int scalesRowOffset = row * blocksPerRow;
            float sum = 0f;

            for (int b = 0; b < blocksPerRow; b++)
            {
                float scale = scales[scalesRowOffset + b];
                int blockOffset = valuesRowOffset + b * BlockElements;
                int vecOffset = b * BlockElements;

                float blockSum = 0f;
                for (int i = 0; i < BlockElements; i++)
                {
                    blockSum += (sbyte)values[blockOffset + i] * vector[vecOffset + i];
                }

                sum += scale * blockSum;
            }

            result[row] = sum;
        });
    }

    /// <summary>
    /// Same kernel body as <see cref="MatVecMul"/> but takes device-resident weight
    /// arrays — values + scales never leave the GPU after the initial host→device
    /// upload. Only `vector` (current activation) and `result` (output logits/row)
    /// still cross the bus per call. This is the step-6 memcpy-reduction shape.
    ///
    /// Resident arrays carry no [In]/[Out] hint — they manage their own H/D sync
    /// status via RefreshHost()/RefreshDevice(); HybRunner.Wrap recognises them
    /// and passes the device pointer directly into the launched kernel.
    /// </summary>
    [EntryPoint]
    public static void MatVecMulResident(
        [Out] float[] result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        [In] float[] vector,
        int rows,
        int blocksPerRow)
    {
        Parallel.For(0, rows, row =>
        {
            int valuesRowOffset = row * blocksPerRow * BlockElements;
            int scalesRowOffset = row * blocksPerRow;
            float sum = 0f;

            for (int b = 0; b < blocksPerRow; b++)
            {
                float scale = scales[scalesRowOffset + b];
                int blockOffset = valuesRowOffset + b * BlockElements;
                int vecOffset = b * BlockElements;

                float blockSum = 0f;
                for (int i = 0; i < BlockElements; i++)
                {
                    blockSum += (sbyte)values[blockOffset + i] * vector[vecOffset + i];
                }

                sum += scale * blockSum;
            }

            result[row] = sum;
        });
    }

    /// <summary>
    /// Q8 matvec with every array resident — vector input and result output both
    /// live on the device. Only the bus traffic per call is scalar args. Used by
    /// the CUDA path once <see cref="LlamaCsharp.Model.LlamaTransformer"/> has
    /// promoted all its activations.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void MatVecMulFullyResident(
        FloatResidentArray result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        FloatResidentArray vector,
        int rows,
        int blocksPerRow)
    {
        Parallel.For(0, rows, row =>
        {
            int valuesRowOffset = row * blocksPerRow * BlockElements;
            int scalesRowOffset = row * blocksPerRow;
            float sum = 0f;

            for (int b = 0; b < blocksPerRow; b++)
            {
                float scale = scales[scalesRowOffset + b];
                int blockOffset = valuesRowOffset + b * BlockElements;
                int vecOffset = b * BlockElements;

                float blockSum = 0f;
                for (int i = 0; i < BlockElements; i++)
                {
                    blockSum += (sbyte)values[blockOffset + i] * vector[vecOffset + i];
                }

                sum += scale * blockSum;
            }

            result[row] = sum;
        });
    }

    /// <summary>
    /// Bridge variant: vector still host <c>float[]</c> (e.g. <c>_xNorm</c>
    /// not yet promoted), result resident (e.g. <c>_gate</c> / <c>_up</c>
    /// after the FFN step-3 promotion). Used until <c>_xNorm</c> is promoted
    /// in step 5, at which point callers switch to <see cref="MatVecMulFullyResident"/>.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void MatVecMulResidentOut(
        FloatResidentArray result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        [In] float[] vector,
        int rows,
        int blocksPerRow)
    {
        Parallel.For(0, rows, row =>
        {
            int valuesRowOffset = row * blocksPerRow * BlockElements;
            int scalesRowOffset = row * blocksPerRow;
            float sum = 0f;

            for (int b = 0; b < blocksPerRow; b++)
            {
                float scale = scales[scalesRowOffset + b];
                int blockOffset = valuesRowOffset + b * BlockElements;
                int vecOffset = b * BlockElements;

                float blockSum = 0f;
                for (int i = 0; i < BlockElements; i++)
                {
                    blockSum += (sbyte)values[blockOffset + i] * vector[vecOffset + i];
                }

                sum += scale * blockSum;
            }

            result[row] = sum;
        });
    }

    /// <summary>
    /// Dequantize one row of a Q8_0 matrix into a device-resident destination.
    /// Replaces the host-side <see cref="Q8Decode.DequantizeRow"/> for the token
    /// embedding lookup once the destination buffer (<c>_x</c>) is resident.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void DequantizeRowFullyResident(
        FloatResidentArray dst,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        int rowIndex,
        int cols,
        int blocksPerRow)
    {
        Parallel.For(0, cols, i =>
        {
            int blockIdx = i / BlockElements;
            float scale = scales[rowIndex * blocksPerRow + blockIdx];
            dst[i] = scale * (sbyte)values[rowIndex * cols + i];
        });
    }

    /// <summary>
    /// Same body as <see cref="DequantizeRowFullyResident"/> but reads the row
    /// index from <paramref name="rowIdx"/>[0] on the device. Lets the GPU
    /// greedy sampler feed the next-token id straight into the embedding
    /// lookup with no D→H roundtrip — used by
    /// <c>LlamaTransformer.ForwardArgmaxGpuFromDevice</c> from Iter 3a onward.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void DequantizeRowByDeviceIdxFullyResident(
        FloatResidentArray dst,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        ResidentArrayGeneric<int> rowIdx,
        int cols,
        int blocksPerRow)
    {
        Parallel.For(0, cols, i =>
        {
            int rowIndex = rowIdx[0];
            int blockIdx = i / BlockElements;
            float scale = scales[rowIndex * blocksPerRow + blockIdx];
            dst[i] = scale * (sbyte)values[rowIndex * cols + i];
        });
    }

    /// <summary>
    /// Cooperative-row Q8 matvec: <strong>one CUDA block per output row</strong>
    /// with threads inside the block striding the column dot product and
    /// reducing via shared memory. Replaces the one-thread-per-row layout of
    /// <see cref="MatVecMulFullyResident"/> which under-utilizes the GPU when
    /// the matrix has few rows (e.g. TinyLlama Wk/Wv with 256 rows — 0.8%
    /// thread occupancy in the old layout).
    ///
    /// Launch shape (set by <see cref="LlamaCsharp.Utils.CudaInvoke.MatVecMulCoopRow"/>):
    ///   <c>gridDim.x = rows</c>, <c>blockDim.x = 128</c>,
    ///   shared = <c>blockDim.x * sizeof(float)</c>.
    /// Each thread accumulates a strided partial sum across the row; a
    /// log₂(blockDim.x) shared-memory tree reduction collapses to thread 0
    /// which writes <paramref name="result"/>[row].
    /// Pattern lifted from Hybridizer's <c>1.Simple/Reduction</c> sample.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void MatVecMulCoopRow(
        FloatResidentArray result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        FloatResidentArray vector,
        int rows,
        int blocksPerRow)
    {
        int row = blockIdx.x;
        if (row >= rows) return;

        int tid = threadIdx.x;
        int nThreads = blockDim.x;
        int cols = blocksPerRow * BlockElements;
        int valuesRowOffset = row * cols;
        int scalesRowOffset = row * blocksPerRow;

        // Per-thread partial sum over a strided slice of the row.
        float partial = 0f;
        for (int i = tid; i < cols; i += nThreads)
        {
            int b = i / BlockElements;
            float s = scales[scalesRowOffset + b];
            partial += s * (sbyte)values[valuesRowOffset + i] * vector[i];
        }

        // Shared-memory tree reduction.
        var cache = new SharedMemoryAllocator<float>().allocate(blockDim.x);
        cache[tid] = partial;
        CUDAIntrinsics.__syncthreads();

        int stride = nThreads >> 1;
        while (stride > 0)
        {
            if (tid < stride)
            {
                cache[tid] = cache[tid] + cache[tid + stride];
            }
            CUDAIntrinsics.__syncthreads();
            stride = stride >> 1;
        }

        if (tid == 0)
        {
            result[row] = cache[0];
        }
    }

    /// <summary>
    /// Warp-shuffle variant of <see cref="MatVecMulCoopRow"/>. Replaces the
    /// shared-memory tree reduction (log₂(blockDim) shared-mem writes + the
    /// same number of <c>__syncthreads</c>) with a 2-stage shuffle:
    /// <list type="number">
    /// <item>Within-warp 5-step <c>shfl_down</c> reduction (no
    /// <c>__syncthreads</c>) → lane 0 of each warp holds the warp sum.</item>
    /// <item>Cross-warp via tiny shared-mem array of <c>blockDim/32</c>
    /// floats; warp 0 reads back and shuffle-reduces to lane 0 which writes
    /// <paramref name="result"/>[row].</item>
    /// </list>
    /// For blockDim = 128 (4 warps) this saves 5 of the 7
    /// <c>__syncthreads</c> in the original kernel — each sync is a few
    /// hundred ns on WDDM, so the saving is roughly ~1 µs per matvec call,
    /// ~150 µs/token at 155 matvec calls/token.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void MatVecMulCoopRowShfl(
        FloatResidentArray result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        FloatResidentArray vector,
        int rows,
        int blocksPerRow)
    {
        int row = blockIdx.x;
        if (row >= rows) return;

        int tid = threadIdx.x;
        int nThreads = blockDim.x;
        int cols = blocksPerRow * BlockElements;
        int valuesRowOffset = row * cols;
        int scalesRowOffset = row * blocksPerRow;

        int laneId = tid & 31;
        int warpId = tid >> 5;

        // Per-thread partial sum (same as 7.A.1 kernel).
        float partial = 0f;
        for (int i = tid; i < cols; i += nThreads)
        {
            int b = i >> 5;
            float s = scales[scalesRowOffset + b];
            partial = partial + s * (sbyte)values[valuesRowOffset + i] * vector[i];
        }

        // Within-warp 5-step shfl_down reduction → lane 0 of each warp
        // holds the warp's sum.
        var tb = cooperative_groups.this_thread_block();
        var warp = cooperative_groups.tile_partition_32(tb);
        partial = partial + warp.shfl_down(partial, 16);
        partial = partial + warp.shfl_down(partial, 8);
        partial = partial + warp.shfl_down(partial, 4);
        partial = partial + warp.shfl_down(partial, 2);
        partial = partial + warp.shfl_down(partial, 1);

        // Lane 0 of each warp writes its sum to shared mem.
        // Cache size = nWarps = blockDim/32. For blockDim=128 → 4 floats.
        var cache = new SharedMemoryAllocator<float>().allocate(32);
        if (laneId == 0)
        {
            cache[warpId] = partial;
        }
        CUDAIntrinsics.__syncthreads();

        // Warp 0 reduces the per-warp sums via another shuffle.
        if (warpId == 0)
        {
            int nWarps = nThreads >> 5;
            float warpSum = laneId < nWarps ? cache[laneId] : 0f;
            warpSum = warpSum + warp.shfl_down(warpSum, 16);
            warpSum = warpSum + warp.shfl_down(warpSum, 8);
            warpSum = warpSum + warp.shfl_down(warpSum, 4);
            warpSum = warpSum + warp.shfl_down(warpSum, 2);
            warpSum = warpSum + warp.shfl_down(warpSum, 1);
            if (laneId == 0)
            {
                result[row] = warpSum;
            }
        }
    }

    /// <summary>
    /// CUB-reduction variant of <see cref="MatVecMulCoopRowShfl"/>. Same launch
    /// shape (one block per output row, blockDim = 128) but the manual 2-stage
    /// shuffle is replaced by a single <c>cub::BlockReduce&lt;float,128&gt;::Sum</c>
    /// call via <see cref="CubReduce.SumBlock128"/>. CUB internally uses warp
    /// shuffles + butterfly cross-warp; functionally equivalent, just less
    /// hand-rolled code. No dynamic shared memory needed — CUB's TempStorage is
    /// declared static <c>__shared__</c> inside the intrinsic helper.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void MatVecMulCoopRowCub(
        FloatResidentArray result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        FloatResidentArray vector,
        int rows,
        int blocksPerRow)
    {
        int row = blockIdx.x;
        if (row >= rows) return;

        int tid = threadIdx.x;
        int nThreads = blockDim.x;
        int cols = blocksPerRow * BlockElements;
        int valuesRowOffset = row * cols;
        int scalesRowOffset = row * blocksPerRow;

        float partial = 0f;
        for (int i = tid; i < cols; i += nThreads)
        {
            int b = i >> 5;
            float s = scales[scalesRowOffset + b];
            partial = partial + s * (sbyte)values[valuesRowOffset + i] * vector[i];
        }

        float sum = CubReduce.SumBlock128(partial);
        if (tid == 0)
        {
            result[row] = sum;
        }
    }

    /// <summary>
    /// Step 7.A.8 + 7.A.9: <strong>fully vectorized-load</strong> variant of
    /// <see cref="MatVecMulCoopRowCub"/>. Same launch shape (one block per
    /// output row, blockDim = 128) and same CUB reduction. Two vectorized
    /// loads per inner-loop iteration:
    /// <list type="bullet">
    ///   <item><see cref="Q8VecLoad.LoadPacked4"/> — 4 contiguous Q8 bytes
    ///   via <c>ld.global.v4.u8</c> + PRMT byte extracts (iter 7.A.8).</item>
    ///   <item><see cref="F32VecLoad.LoadVec4"/> — 4 contiguous activation
    ///   floats via <c>ld.global.v4.f32</c> (iter 7.A.9). Replaces what was
    ///   previously 4 separate <c>LD.E</c> 32-bit loads emitted by
    ///   <c>Parallel.For</c>'s natural lowering.</item>
    /// </list>
    /// The Nsight Compute profile (3.ncu-rep, 2026-05-18 PM) showed the
    /// post-7.A.8 matvec at 75% DRAM throughput with the activation-side
    /// loads still emitted byte-by-byte; the float-side vectorization
    /// targets that remaining LSU bandwidth.
    ///
    /// Within-warp threads are 4 bytes apart in the byte scheme and 16
    /// bytes apart in the float scheme, so per warp-iteration: 128 bytes
    /// (4 Q8 blocks of 32 bytes) + 512 bytes (32 floats × 16 bytes). Each
    /// thread's 4 bytes stay inside one Q8 block → one <c>scales[]</c>
    /// fetch per thread per iteration is correct.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void MatVecMulCoopRowCubVec4(
        FloatResidentArray result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        FloatResidentArray vector,
        int rows,
        int blocksPerRow)
    {
        int row = blockIdx.x;
        if (row >= rows) return;

        int tid = threadIdx.x;
        int nThreads = blockDim.x;
        int cols = blocksPerRow * BlockElements;
        int valuesRowOffset = row * cols;
        int scalesRowOffset = row * blocksPerRow;

        float partial = 0f;
        for (int i = tid * 4; i < cols; i += nThreads * 4)
        {
            int b = i >> 5;
            float s = scales[scalesRowOffset + b];
            F32VecLoad.LoadVec4(vector, i, out float v0, out float v1, out float v2, out float v3);
            uint packed = Q8VecLoad.LoadPacked4(values, valuesRowOffset + i);
            int q0 = (sbyte)(packed         & 0xFF);
            int q1 = (sbyte)((packed >>  8) & 0xFF);
            int q2 = (sbyte)((packed >> 16) & 0xFF);
            int q3 = (sbyte)((packed >> 24) & 0xFF);
            partial = partial + s * (q0 * v0 + q1 * v1 + q2 * v2 + q3 * v3);
        }

        float sum = CubReduce.SumBlock128(partial);
        if (tid == 0)
        {
            result[row] = sum;
        }
    }

    /// <summary>
    /// Step 7.A.3: <strong>4-rows-per-block</strong> variant of
    /// <see cref="MatVecMulCoopRow"/>. Each thread loads <c>vector[i]</c>
    /// once and uses it to accumulate into FOUR partial sums (one per
    /// output row in the block's row-block). Cuts per-thread vector-load
    /// instructions and L1 pressure by 4× compared to the 1-row variant.
    /// L2 already caches the vector across blocks, so DRAM traffic for
    /// xNorm is unchanged; the win is on instruction count + register
    /// allocation + cache pressure.
    ///
    /// Launch shape (see <see cref="LlamaCsharp.Utils.CudaInvoke.MatVecMulCoopRow4"/>):
    ///   <c>gridDim.x = ceil(rows / 4)</c>, <c>blockDim.x = 128</c>,
    ///   shared = <c>4 * blockDim.x * sizeof(float)</c> (4 reduction
    ///   slots per thread, packed as cache[k * blockDim + tid]).
    /// Bounds-checked: the last block may handle fewer than 4 rows if
    /// <c>rows</c> isn't a multiple of 4 — only valid output indices are
    /// written.
    /// </summary>
    [EntryPoint]
    [HybridizerIgnore("OMP")]
    public static void MatVecMulCoopRow4(
        FloatResidentArray result,
        ResidentArrayGeneric<byte> values,
        FloatResidentArray scales,
        FloatResidentArray vector,
        int rows,
        int blocksPerRow)
    {
        int rowBase = blockIdx.x * 4;
        if (rowBase >= rows) return;

        int tid = threadIdx.x;
        int nThreads = blockDim.x;
        int cols = blocksPerRow * BlockElements;

        int valuesRowOffset0 = (rowBase + 0) * cols;
        int valuesRowOffset1 = (rowBase + 1) * cols;
        int valuesRowOffset2 = (rowBase + 2) * cols;
        int valuesRowOffset3 = (rowBase + 3) * cols;
        int scalesRowOffset0 = (rowBase + 0) * blocksPerRow;
        int scalesRowOffset1 = (rowBase + 1) * blocksPerRow;
        int scalesRowOffset2 = (rowBase + 2) * blocksPerRow;
        int scalesRowOffset3 = (rowBase + 3) * blocksPerRow;

        // Mask out rows that fall off the end (last block on a non-multiple-of-4 size).
        bool row0Valid = rowBase + 0 < rows;
        bool row1Valid = rowBase + 1 < rows;
        bool row2Valid = rowBase + 2 < rows;
        bool row3Valid = rowBase + 3 < rows;

        float partial0 = 0f;
        float partial1 = 0f;
        float partial2 = 0f;
        float partial3 = 0f;

        for (int i = tid; i < cols; i += nThreads)
        {
            int b = i >> 5;           // i / BlockElements, BlockElements = 32 = 2^5
            float v = vector[i];

            if (row0Valid)
            {
                float s = scales[scalesRowOffset0 + b];
                partial0 = partial0 + s * (sbyte)values[valuesRowOffset0 + i] * v;
            }
            if (row1Valid)
            {
                float s = scales[scalesRowOffset1 + b];
                partial1 = partial1 + s * (sbyte)values[valuesRowOffset1 + i] * v;
            }
            if (row2Valid)
            {
                float s = scales[scalesRowOffset2 + b];
                partial2 = partial2 + s * (sbyte)values[valuesRowOffset2 + i] * v;
            }
            if (row3Valid)
            {
                float s = scales[scalesRowOffset3 + b];
                partial3 = partial3 + s * (sbyte)values[valuesRowOffset3 + i] * v;
            }
        }

        // Shared memory layout: 4 packed reduction slots, slot k at
        // offset k * blockDim.x. Allocate as one contiguous buffer.
        var cache = new SharedMemoryAllocator<float>().allocate(4 * blockDim.x);
        cache[0 * nThreads + tid] = partial0;
        cache[1 * nThreads + tid] = partial1;
        cache[2 * nThreads + tid] = partial2;
        cache[3 * nThreads + tid] = partial3;
        CUDAIntrinsics.__syncthreads();

        int stride = nThreads >> 1;
        while (stride > 0)
        {
            if (tid < stride)
            {
                cache[0 * nThreads + tid] = cache[0 * nThreads + tid] + cache[0 * nThreads + tid + stride];
                cache[1 * nThreads + tid] = cache[1 * nThreads + tid] + cache[1 * nThreads + tid + stride];
                cache[2 * nThreads + tid] = cache[2 * nThreads + tid] + cache[2 * nThreads + tid + stride];
                cache[3 * nThreads + tid] = cache[3 * nThreads + tid] + cache[3 * nThreads + tid + stride];
            }
            CUDAIntrinsics.__syncthreads();
            stride = stride >> 1;
        }

        if (tid == 0)
        {
            if (row0Valid) result[rowBase + 0] = cache[0 * nThreads];
            if (row1Valid) result[rowBase + 1] = cache[1 * nThreads];
            if (row2Valid) result[rowBase + 2] = cache[2 * nThreads];
            if (row3Valid) result[rowBase + 3] = cache[3 * nThreads];
        }
    }
}
