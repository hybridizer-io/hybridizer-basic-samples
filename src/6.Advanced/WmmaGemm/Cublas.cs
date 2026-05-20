// Thin P/Invoke wrapper around cublasGemmEx for the half→float GEMM
// comparison. cuBLAS uses column-major; our other kernels use row-major
// (C[i,j] = sum_k A[i,k]*B[k,j]). The standard remap is to compute
// (column-major) C^T = B^T * A^T, which means we pass our row-major A and B
// to cublasGemmEx with B and A swapped. The leading dimension on each side
// is the row stride in our row-major layout, which equals n for square
// matrices.
//
// Lives in the project for now; lifts cleanly into CUDAImports once we want
// a first-class binding. The cudaimports tree already has the cublas types
// (cublasHandle_t, cublasStatus_t, etc.) but no GemmEx surface yet.

using System;
using System.Runtime.InteropServices;
using Hybridizer.Runtime.CUDAImports;

namespace WmmaGemm;

internal static class Cublas
{
    private const string LIBCUBLAS_WIN = "cublas64_13";
    private const string LIBCUBLAS_LIN = "libcublas.so.13";

    // cuBLAS constants we need. Values from cublas_v2.h / cublas_api.h:
    //   CUBLAS_OP_N            = 0
    //   CUDA_R_16F             = 2   (cudaDataType)
    //   CUDA_R_32F             = 0
    //   CUBLAS_COMPUTE_32F     = 68
    //   CUBLAS_GEMM_DEFAULT    = -1
    private const int CUBLAS_OP_N = 0;
    private const int CUDA_R_16F  = 2;
    private const int CUDA_R_32F  = 0;
    private const int CUBLAS_COMPUTE_32F = 68;
    private const int CUBLAS_GEMM_DEFAULT = -1;

    // Linux entry points (libcublas.so.13)
    [DllImport(LIBCUBLAS_LIN, EntryPoint = "cublasCreate_v2")]
    private static extern int Lin_Create(out IntPtr handle);
    [DllImport(LIBCUBLAS_LIN, EntryPoint = "cublasDestroy_v2")]
    private static extern int Lin_Destroy(IntPtr handle);
    [DllImport(LIBCUBLAS_LIN, EntryPoint = "cublasGemmEx")]
    private static extern int Lin_GemmEx(
        IntPtr handle, int transa, int transb, int m, int n, int k,
        ref float alpha, IntPtr A, int Atype, int lda,
                         IntPtr B, int Btype, int ldb,
        ref float beta,  IntPtr C, int Ctype, int ldc,
        int computeType, int algo);

    // Windows entry points (cublas64_13.dll)
    [DllImport(LIBCUBLAS_WIN, EntryPoint = "cublasCreate_v2")]
    private static extern int Win_Create(out IntPtr handle);
    [DllImport(LIBCUBLAS_WIN, EntryPoint = "cublasDestroy_v2")]
    private static extern int Win_Destroy(IntPtr handle);
    [DllImport(LIBCUBLAS_WIN, EntryPoint = "cublasGemmEx")]
    private static extern int Win_GemmEx(
        IntPtr handle, int transa, int transb, int m, int n, int k,
        ref float alpha, IntPtr A, int Atype, int lda,
                         IntPtr B, int Btype, int ldb,
        ref float beta,  IntPtr C, int Ctype, int ldc,
        int computeType, int algo);

    public static IntPtr Create()
    {
        int rc = OperatingSystem.IsWindows()
            ? Win_Create(out var h)
            : Lin_Create(out h);
        if (rc != 0)
            throw new ApplicationException($"cublasCreate_v2 returned {rc}");
        return h;
    }

    public static void Destroy(IntPtr handle)
    {
        int rc = OperatingSystem.IsWindows() ? Win_Destroy(handle) : Lin_Destroy(handle);
        if (rc != 0)
            throw new ApplicationException($"cublasDestroy_v2 returned {rc}");
    }

    // Row-major C = A * B for square nxn half-input float-output matrices.
    // Implemented as cuBLAS (column-major) C^T = B^T * A^T by swapping A/B.
    public static void Hgemm_RowMajor(IntPtr handle, int n, IntPtr aHalfDev, IntPtr bHalfDev, IntPtr cFloatDev)
    {
        float alpha = 1.0f;
        float beta  = 0.0f;
        int rc = OperatingSystem.IsWindows()
            ? Win_GemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                         ref alpha, bHalfDev, CUDA_R_16F, n,
                                    aHalfDev, CUDA_R_16F, n,
                         ref beta,  cFloatDev, CUDA_R_32F, n,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
            : Lin_GemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                         ref alpha, bHalfDev, CUDA_R_16F, n,
                                    aHalfDev, CUDA_R_16F, n,
                         ref beta,  cFloatDev, CUDA_R_32F, n,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (rc != 0)
            throw new ApplicationException($"cublasGemmEx returned {rc}");
    }
}
