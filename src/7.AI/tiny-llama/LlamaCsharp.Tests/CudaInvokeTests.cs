using System;
using Hybridizer.Runtime.CUDAImports;
using LlamaCsharp.Math;
using LlamaCsharp.Utils;
using Xunit;
using Xunit.Abstractions;

namespace LlamaCsharp.Tests;

/// <summary>
/// Isolated bring-up tests for the DllImport-based dispatch (Utils/CudaInvoke,
/// Utils/ResidentStructCache). Bypasses HybRunner's per-call marshaller by
/// pre-allocating the 32-byte native struct mirror of each
/// <see cref="FloatResidentArray"/> / <see cref="ResidentArrayGeneric{T}"/>
/// and calling the generated <c>_ExternCWrapperStream_CUDA</c> entry directly.
///
/// Verifies the bypass mechanism in isolation with known data so we can pin
/// any layout / calling-convention / runtime-init bug without dragging in
/// the full forward pass.
/// </summary>
[Trait("Category", "Hybridizer")]
public class CudaInvokeTests
{
    private readonly ITestOutputHelper _output;
    public CudaInvokeTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// Roundtrip: fill two resident arrays via HostPointer bulk-copy, call
    /// AccumulateFullyResident via CudaInvoke, RefreshHost the result, compare
    /// to the managed reference. Smallest test surface that exercises:
    ///   - ResidentStructCache.Materialise (host-side struct image).
    ///   - cudaMemcpy of the struct image to device.
    ///   - NativeLibrary.GetExport symbol resolution.
    ///   - Delegate invocation with the right calling convention.
    ///   - The kernel actually reading our cached struct's `tab` field.
    /// </summary>
    [SkippableFact]
    public unsafe void AccumulateFullyResident_RoundTrips()
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(), "CUDA satellite not built");

        GpuBackend.Activate(ComputeBackend.Cuda);
        CudaInvoke.Initialize();
        try
        {
            const int Size = 16;
            var a = new FloatResidentArray(Size);
            var b = new FloatResidentArray(Size);

            // Populate host buffers via HostPointer (auto-allocates host),
            // then mark DeviceNeedsRefresh so the first kernel call uploads.
            float[] hostA = new float[Size];
            float[] hostB = new float[Size];
            for (int i = 0; i < Size; i++) { hostA[i] = i; hostB[i] = i * 10; }

            long bytes = (long)Size * sizeof(float);
            fixed (float* src = hostA)
                Buffer.MemoryCopy(src, (void*)a.HostPointer, bytes, bytes);
            fixed (float* src = hostB)
                Buffer.MemoryCopy(src, (void*)b.HostPointer, bytes, bytes);
            a.Status = ResidentArrayStatus.DeviceNeedsRefresh;
            b.Status = ResidentArrayStatus.DeviceNeedsRefresh;
            a.RefreshDevice();
            b.RefreshDevice();
            // After the explicit RefreshDevice the device buffer holds our
            // data and the marshaller would mark the array HostNeedsRefresh
            // on its first call. We do it manually for the bypass path.
            a.Status = ResidentArrayStatus.HostNeedsRefresh;
            b.Status = ResidentArrayStatus.HostNeedsRefresh;

            // Dump the cached struct's host image before the kernel call.
            nint aStructPtr = ResidentStructCache.Get(a);
            nint bStructPtr = ResidentStructCache.Get(b);
            byte[] aStructBytes = new byte[32];
            fixed (byte* dst = aStructBytes)
                cuda.ERROR_CHECK(cuda.Memcpy((nint)dst, aStructPtr, (size_t)32, cudaMemcpyKind.cudaMemcpyDeviceToHost));
            _output.WriteLine($"a struct device bytes: {BitConverter.ToString(aStructBytes)}");
            _output.WriteLine($"a.DevicePointer (expected at offset 8): 0x{a.DevicePointer.ToInt64():X16}");

            // Call the kernel via CudaInvoke (DllImport bypass path).
            CudaInvoke.AccumulateFullyResident(a, b, Size);
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());

            // Read result back.
            a.RefreshHost();
            fixed (float* dst = hostA)
                Buffer.MemoryCopy((void*)a.HostPointer, dst, bytes, bytes);

            _output.WriteLine($"a after Accumulate: [{string.Join(", ", hostA)}]");
            for (int i = 0; i < Size; i++)
            {
                float expected = i + i * 10f;
                Assert.True(System.Math.Abs(hostA[i] - expected) < 1e-5f,
                    $"index {i}: expected {expected}, got {hostA[i]}");
            }
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }
    }

    /// <summary>
    /// Two-phase resident argmax bring-up — verifies the GPU greedy sampler
    /// kernels produce the same index as the host
    /// <see cref="ParallelMath.Argmax"/> reference across three cases:
    /// unique max, two tied at the top with the smaller index winning, and all
    /// equal (returns 0). Tie-breaking is load-bearing — if a future kernel
    /// rewrite drops the smallest-index guarantee, this test fails and Lily
    /// golden drifts in subsequent iterations.
    /// </summary>
    [SkippableTheory]
    [InlineData(new float[] { 1f, 3f, 2f, 5f, 4f }, 3)]
    [InlineData(new float[] { 1f, 5f, 2f, 5f, 4f }, 1)] // tie at top: smaller index wins
    [InlineData(new float[] { 7f, 7f, 7f, 7f, 7f }, 0)] // all-equal → 0
    [InlineData(new float[] { -3f, -1f, -2f, -5f }, 1)] // negatives
    public unsafe void ArgmaxFullyResident_MatchesParallelMathArgmax(float[] values, int expected)
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(), "CUDA satellite not built");

        GpuBackend.Activate(ComputeBackend.Cuda);
        CudaInvoke.Initialize();
        try
        {
            int n = values.Length;
            var logits = new FloatResidentArray(n);
            long bytes = (long)n * sizeof(float);
            fixed (float* src = values)
                Buffer.MemoryCopy(src, (void*)logits.HostPointer, bytes, bytes);
            logits.Status = ResidentArrayStatus.DeviceNeedsRefresh;
            logits.RefreshDevice();
            logits.Status = ResidentArrayStatus.HostNeedsRefresh;

            // Persistent boxes — same lifetime as logits in production. The
            // dispatcher's init kernel seeds them on device on every call,
            // so we deliberately leave them uninitialized here to exercise
            // that seed step alongside the rest.
            cuda.ERROR_CHECK(cuda.Malloc(out nint maxBoxDev, (size_t)sizeof(float)));
            cuda.ERROR_CHECK(cuda.Malloc(out nint idxOutDev, (size_t)sizeof(int)));

            CudaInvoke.ArgmaxFullyResident(logits, n, maxBoxDev, idxOutDev);
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());

            int gpuIdx = 0;
            cuda.ERROR_CHECK(cuda.Memcpy((nint)(&gpuIdx), idxOutDev, (size_t)sizeof(int), cudaMemcpyKind.cudaMemcpyDeviceToHost));
            int hostIdx = ParallelMath.Argmax(values, n);

            _output.WriteLine($"values=[{string.Join(", ", values)}] expected={expected} hostIdx={hostIdx} gpuIdx={gpuIdx}");
            Assert.Equal(expected, hostIdx);
            Assert.Equal(expected, gpuIdx);

            cuda.ERROR_CHECK(cuda.Free(maxBoxDev));
            cuda.ERROR_CHECK(cuda.Free(idxOutDev));
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }
    }

    /// <summary>
    /// Cooperative-row Q8 matvec correctness check: build a small Q8_0 matrix
    /// (8 rows × 64 cols = 2 blocks/row, BlockElements=32), upload values +
    /// pre-decoded scales + an input vector to resident arrays, run
    /// <see cref="CudaInvoke.MatVecMulCoopRow"/>, and assert each output row
    /// is numerically equivalent to <see cref="Q8Kernels.MatVecMul"/>'s host
    /// implementation. Tolerance allows for the small drift introduced by the
    /// tree-reduction reassociation versus the host's sequential block sums.
    /// </summary>
    [SkippableFact]
    public unsafe void MatVecMulCoopRow_MatchesParallelMathMatVecMul()
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(), "CUDA satellite not built");

        GpuBackend.Activate(ComputeBackend.Cuda);
        CudaInvoke.Initialize();
        try
        {
            const int Rows = 8;
            const int BlocksPerRow = 2;
            const int BlockElements = Q8Kernels.BlockElements; // 32
            const int Cols = BlocksPerRow * BlockElements;     // 64

            // Build deterministic Q8 weight: values in [-100, 100] cast to sbyte,
            // scales small positive floats.
            byte[] hostValues = new byte[Rows * Cols];
            float[] hostScales = new float[Rows * BlocksPerRow];
            float[] hostVector = new float[Cols];
            float[] hostResult = new float[Rows];

            var rng = new Random(42);
            for (int i = 0; i < hostValues.Length; i++)
                hostValues[i] = unchecked((byte)(sbyte)(rng.Next(-100, 100)));
            for (int i = 0; i < hostScales.Length; i++)
                hostScales[i] = 0.01f + (float)rng.NextDouble() * 0.05f;
            for (int i = 0; i < hostVector.Length; i++)
                hostVector[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            Q8Kernels.MatVecMul(hostResult, hostValues, hostScales, hostVector, Rows, BlocksPerRow);

            // Upload to resident arrays + invoke the cooperative kernel.
            var residentValues = new ResidentArrayGeneric<byte>(hostValues.Length);
            var residentScales = new FloatResidentArray(hostScales.Length);
            var residentVector = new FloatResidentArray(hostVector.Length);
            var residentResult = new FloatResidentArray(Rows);

            fixed (byte* src = hostValues)
                Buffer.MemoryCopy(src, (void*)residentValues.HostPointer, hostValues.Length, hostValues.Length);
            fixed (float* src = hostScales)
                Buffer.MemoryCopy(src, (void*)residentScales.HostPointer, hostScales.Length * sizeof(float), hostScales.Length * sizeof(float));
            fixed (float* src = hostVector)
                Buffer.MemoryCopy(src, (void*)residentVector.HostPointer, hostVector.Length * sizeof(float), hostVector.Length * sizeof(float));
            residentValues.Status = ResidentArrayStatus.DeviceNeedsRefresh;
            residentScales.Status = ResidentArrayStatus.DeviceNeedsRefresh;
            residentVector.Status = ResidentArrayStatus.DeviceNeedsRefresh;
            residentValues.RefreshDevice();
            residentScales.RefreshDevice();
            residentVector.RefreshDevice();
            residentValues.Status = ResidentArrayStatus.HostNeedsRefresh;
            residentScales.Status = ResidentArrayStatus.HostNeedsRefresh;
            residentVector.Status = ResidentArrayStatus.HostNeedsRefresh;
            _ = residentResult.DevicePointer;
            residentResult.Status = ResidentArrayStatus.HostNeedsRefresh;

            CudaInvoke.MatVecMulCoopRow(residentResult, residentValues, residentScales, residentVector, Rows, BlocksPerRow);
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());

            float[] gpuResult = new float[Rows];
            residentResult.RefreshHost();
            fixed (float* dst = gpuResult)
                Buffer.MemoryCopy((void*)residentResult.HostPointer, dst, Rows * sizeof(float), Rows * sizeof(float));

            for (int r = 0; r < Rows; r++)
            {
                float host = hostResult[r];
                float gpu = gpuResult[r];
                float tol = System.Math.Max(1e-4f, System.Math.Abs(host) * 1e-4f);
                _output.WriteLine($"row {r}: host={host} gpu={gpu} delta={System.Math.Abs(host - gpu)}");
                Assert.True(System.Math.Abs(host - gpu) <= tol,
                    $"row {r}: host={host} gpu={gpu} delta={System.Math.Abs(host - gpu)} tol={tol}");
            }
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }
    }

    /// <summary>
    /// Int-resident plumbing round-trip: allocate a length-1
    /// <see cref="ResidentArrayGeneric{Int32}"/>, write a known value with the
    /// <c>WriteOneInt</c> bypassed kernel, read it back via a 4-byte
    /// <c>cudaMemcpy(D→H)</c>, assert equal. Exercises the new
    /// <c>ResidentStructCache.Get(ResidentArrayGeneric&lt;int&gt;)</c> overload
    /// and the int element-type path through the Hybridizer codegen before any
    /// dependent kernel relies on it.
    /// </summary>
    [SkippableFact]
    public unsafe void WriteOneInt_RoundTrips_DeviceInt()
    {
        Skip.IfNot(SatelliteLoader.CudaAvailable(), "CUDA satellite not built");

        GpuBackend.Activate(ComputeBackend.Cuda);
        CudaInvoke.Initialize();
        try
        {
            var dst = new ResidentArrayGeneric<int>(1);
            // Eager device alloc + HostNeedsRefresh: identical to the
            // pattern used for kernel-write FloatResidentArrays elsewhere.
            _ = dst.DevicePointer;
            dst.Status = ResidentArrayStatus.HostNeedsRefresh;

            CudaInvoke.WriteOneInt(dst, unchecked((int)0xCAFEBABE));
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());

            int hostValue = 0;
            cuda.ERROR_CHECK(cuda.Memcpy((nint)(&hostValue), dst.DevicePointer, (size_t)sizeof(int), cudaMemcpyKind.cudaMemcpyDeviceToHost));

            _output.WriteLine($"hostValue after WriteOneInt: 0x{hostValue:X8}");
            Assert.Equal(unchecked((int)0xCAFEBABE), hostValue);
        }
        finally
        {
            GpuBackend.Activate(ComputeBackend.Managed);
        }
    }
}
