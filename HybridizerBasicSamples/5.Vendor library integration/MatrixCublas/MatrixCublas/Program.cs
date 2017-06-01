using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Hybridizer.Basic.Integration
{
    unsafe class Program
    {
        static void Main(string[] args)
        {
            const int N = 2048;

            Console.WriteLine("Execution cublas matrix mul with sizes ({0}, {1}) x ({2}, {3})", N, N, N, N);

            NaiveMatrix matrixA = new NaiveMatrix(N, N);
            NaiveMatrix matrixB = new NaiveMatrix(N, N);

            NaiveMatrix res = new NaiveMatrix(N, N);
            NaiveMatrix res_net = new NaiveMatrix(N, N);

            matrixA.FillMatrix();
            matrixB.FillMatrix();

            float alpha = 1.0f;
            float beta = 0.0f;

            cublas cublas = new cublas();
            cublasHandle_t handle;
            cublas.Create(out handle);

            cublasOperation_t transA = cublasOperation_t.CUBLAS_OP_N;
            cublasOperation_t transB = cublasOperation_t.CUBLAS_OP_N;

            cublasSgemm(handle, transA, transB, N, N, N, &alpha, matrixA.Values, N, matrixB.Values, N, &beta, res.Values, N);

            cublas.Destroy(handle);

            reference(matrixA, matrixB, res_net, N);

            for (int i = 0; i < N * N; ++i)
            {
                if (Math.Abs(res[i] - res_net[i]) >= 1.0E-3)
                {
                    Console.WriteLine("Error at {0}, expected {1}, got {2}", i, res_net[i], res[i]);
                    Environment.Exit(1);
                }
            }

            Console.Out.WriteLine("DONE");
        }

        [DllImport("cublas64_80.dll", EntryPoint = "cublasSgemm_v2", CallingConvention = CallingConvention.Cdecl), IntrinsicFunction("cublasSgemm")]
        public static extern cublasStatus_t cublasSgemm(cublasHandle_t handle,
                                              cublasOperation_t transA,
                                              cublasOperation_t transB,
                                              int m,
                                              int n,
                                              int k,
                                              float* alpha,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] float[] matrixA,
                                              int lda,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] float[] matrixB,
                                              int ldb,
                                              float* beta,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] float[] matrixC,
                                              int ldc);

        public static void displayMatrix(NaiveMatrix M)
        {

            for (int i = 0; i < M.Width; ++i)
            {
                for (int j = 0; j < M.Width; ++j)
                {
                    Console.Write(M[i * M.Width + j] + ", ");
                }
                Console.WriteLine();
            }

        }

        public static void reference(NaiveMatrix A, NaiveMatrix B, NaiveMatrix res, int N)
        {
            for (int i = 0; i < N; ++i)
            {
                for(int j = 0; j < N; ++j)
                {
                    float tmp = 0.0F;
                    for(int k = 0; k < N; ++k)
                    {
                        tmp += A.Values[i * N + k] * B.Values[k * N + j];
                    }
                    res.Values[i * N + j] = tmp;
                }
            }
        }
    }
}
