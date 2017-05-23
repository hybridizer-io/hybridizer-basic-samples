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
    //[IntrinsicInclude("cublas.h")]
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                args = new string[] { "512", "512", "512", "512" };
            }
            const int redo = 30;

            int heightA = Convert.ToInt32(args[0]);
            int widthA = Convert.ToInt32(args[1]);
            int heightB = Convert.ToInt32(args[2]);
            int widthB = Convert.ToInt32(args[3]);
            if (widthA != heightB)
            {
                throw new ArgumentException("invalid data -- incompatible matrices");
            }

            Console.WriteLine("Execution cublas matrix mul with sizes ({0}, {1}) x ({2}, {3})", heightA, widthA, heightB, widthB);

            NaiveMatrix matrixA = new NaiveMatrix(widthA, heightA);
            NaiveMatrix matrixB = new NaiveMatrix(widthB, heightB);

            NaiveMatrix res_net = new NaiveMatrix(widthB, heightA);
            NaiveMatrix res_cuda = new NaiveMatrix(widthB, heightA);

            double numberCompute = ((double)matrixA.Height * (double)matrixA.Width * (double)matrixB.Width) * 3.0E-9;

            Stopwatch watch = new Stopwatch();

            matrixA.FillMatrix();
            matrixB.FillMatrix();

            #region CUDA 
            float alpha = 1.0f ;
            float beta = 0.0f ;

            cublas cublas = new cublas();
            cublasHandle_t handle;
            cublas.Create(out handle);
            
            cublasOperation_t transA = cublasOperation_t.CUBLAS_OP_N;
            cublasOperation_t transB = cublasOperation_t.CUBLAS_OP_N;

            HybRunner runner = HybRunner.Cuda("MatrixCublas_CUDA.dll").SetDistrib(4, 5, 32, 32, 1, 0);
            dynamic wrapper = runner.Wrap(new Program());

            watch.Start();

            for (int i = 0; i < redo; ++i)
            {
                MatrixMul(cublas, handle, transA, transB, matrixA.Values, matrixB.Values, res_cuda.Values, alpha, beta, matrixB.Width, matrixA.Width, matrixA.Height);
            }

            watch.Stop();
            cublas.Destroy(handle);

            float msecPerMatrixMul = (float)watch.ElapsedMilliseconds / (float)redo;
            double flopsPerMatrixMul = 2.0 * (double)(res_cuda.Height * res_cuda.Width * matrixB.Height);
            double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
            Console.WriteLine(
                "CUDA : Performance= {2} GFlop/s, Time= {1} msec, Size= {0} Ops\n",
                flopsPerMatrixMul,
                msecPerMatrixMul,
                gigaFlops);

            gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / ((double)runner.LastKernelDuration.ElapsedMilliseconds / 1000.0f);
            Console.WriteLine(
                "CUDA : Performance= {0} GFlop/s, Time= {1} msec, Size= {2} Ops\n",
                gigaFlops,
                msecPerMatrixMul,
                flopsPerMatrixMul);

            #endregion

            #region C#
            watch.Restart();

            Reference(res_net, matrixA, matrixB);

            watch.Stop();
            Console.WriteLine("C#   GFlop/s :   {0}", numberCompute * (double)1 / (watch.ElapsedMilliseconds * 1.0E-03));
            #endregion

            // verify the results

            if (!res_net.Equals(res_cuda))
                Console.Out.WriteLine("ERROR !");

            //Write the matrix on the console
            //res_cuda.WriteMatrix();

            Console.Out.WriteLine("DONE");

        }
        
        public static void Reference(NaiveMatrix result, NaiveMatrix A, NaiveMatrix B)
        {
            Parallel.For(0, A.Height, (i) =>
            {
                for (int j = 0; j < B.Width; ++j)
                {
                    float accum = 0.0F;

                    for (int k = 0; k < A.Width; ++k)
                    {
                        accum += A[A.Width * i + k] * B[B.Width * k + j];
                    }

                    result[B.Width * i + j] = accum;
                }
            });
        }

        [IntrinsicFunction("fprintf")]
        public static void fprintf(string s)
        {
            Console.WriteLine(s);
        }

        [EntryPoint]
        public static void MatrixMul(cublas cublas, 
                                     cublasHandle_t handle, 
                                     cublasOperation_t transA, 
                                     cublasOperation_t transB,
                                     float[] matrixA, 
                                     float[] matrixB, 
                                     float[] matrixC, 
                                     float alpha, 
                                     float beta, 
                                     int wB,
                                     int wA, int hA)
        {
            cublasSgemm(handle, transA, transB, wB, hA, wA, alpha, matrixB, wB, matrixA, wA, beta, matrixC, wB);
        }

        [DllImport("cublas64_80.dll", EntryPoint = "cublasSgemm", CallingConvention = CallingConvention.Cdecl), IntrinsicFunction("cublasSgemm")]
        public static extern cublasStatus_t cublasSgemm(cublasHandle_t handle,
                                              cublasOperation_t transA ,
                                              cublasOperation_t transB,
                                              int m,
                                              int n,
                                              int k,
                                              [MarshalAs(UnmanagedType.R4, MarshalTypeRef = typeof(CudaMarshaler))] float alpha,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] float[] matrixB,
                                              int lda,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] float[] matrixA,
                                              int ldb,
                                              [MarshalAs(UnmanagedType.R4, MarshalTypeRef = typeof(CudaMarshaler))] float beta,
                                              [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(CudaMarshaler))] float[] matrixC,
                                              int ldc);
    }
}
