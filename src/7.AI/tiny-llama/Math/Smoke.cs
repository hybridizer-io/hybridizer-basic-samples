using System.Runtime.InteropServices;
using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Math;

/// <summary>
/// Step-4 smoke kernel: SAXPY (Single-precision A·X Plus Y).
/// Embarrassingly-parallel <see cref="Parallel.For"/> body transcodes to both
/// OMP (#pragma omp parallel for) and CUDA (grid-distributed) without changes.
/// The same managed code runs end-to-end as the dual-path reference for tests.
/// </summary>
public class Smoke
{
    [EntryPoint]
    public static void Saxpy(int n, float a, [In] float[] x, float[] y)
    {
        Parallel.For(0, n, i => { y[i] = a * x[i] + y[i]; });
    }
}
