using System.Runtime.InteropServices;
using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;

namespace Hybridizer.Basic.Maths
{
    unsafe class Program
    {
        static HybRunner runner;
        static dynamic wrapper;

        static void Main(string[] args)
        {
            // configure CUDA
            gpuDeviceProp prop;
            gpu.GetDeviceProperties(out prop, 0);
            runner = SatelliteLoader.Load().SetDistrib(prop.multiProcessorCount * 16, 1, 128, 1, 1, 128 * sizeof(float));
            wrapper = runner.Wrap(new Program());

            int size = 10000; // very slow convergence (or even not at all) with no preconditioner
            SparseMatrix A = SparseMatrix.Laplacian_1D(size);
            FloatResidentArray B = new(size);
            FloatResidentArray X = new(size);

            int maxiter = 1000000;
            float eps = 1.0e-08f;

            for (int i = 0; i < size; ++i)
            {
                B[i] = 1.0f; // right side
                X[i] = 0.0f; // starting point
            }
            
            ConjugateGradient(X, A, B, maxiter, eps);

            // the exact solution of this system is x[i] = (i + 1) (size - i) / 2
            double error = 0.0, norm = 0.0;
            for (int i = 0; i < size; ++i)
            {
                double exact = (i + 1.0) * (size - i) / 2.0;
                error += (X[i] - exact) * (X[i] - exact);
                norm += exact * exact;
            }
            double relativeError = Math.Sqrt(error / norm);
            Console.WriteLine($"relative error to the exact solution : {relativeError}");
            if (!(relativeError < 1.0E-3))
            {
                Console.Error.WriteLine("ERROR : the solution is not accurate enough");
                Environment.Exit(6); // abort
            }
        }

        public static void ConjugateGradient(FloatResidentArray X, SparseMatrix A, FloatResidentArray B, int maxiter, float eps)
        {
            float[] scalBuf = [0];
            int N = (int) B.Count;
            FloatResidentArray R = new(N);
            FloatResidentArray P = new(N);
            FloatResidentArray AP = new(N);
            A.RefreshDevice();
            X.RefreshDevice();
            B.RefreshDevice();

            wrapper.Fmsub(R, B, A, X, N);                       // R = B - A*X
            wrapper.Copy(P, R, N); 
            int k = 0;
            while(k < maxiter)
            {
                wrapper.Multiply(AP, A, P, N);                  // AP = A*P
                scalBuf[0] = 0; wrapper.ScalarProd(scalBuf, R, R, N); float r = scalBuf[0];               // save <R|R>
                scalBuf[0] = 0; wrapper.ScalarProd(scalBuf, P, AP, N); float alpha = r / scalBuf[0];       // alpha = <R|R> / <P|AP>
                wrapper.Saxpy(X, X, alpha, P, N);               // X = X - alpha*P
                wrapper.Saxpy(R, R, -alpha, AP, N);             // RR = R-alpha*AP
                scalBuf[0] = 0; wrapper.ScalarProd(scalBuf, R, R, N); float rr = scalBuf[0];
                if (k % 10 == 0)
                    Console.WriteLine(Math.Sqrt(rr));
                if(rr < eps*eps)
                {
                    break;
                }

                float beta = rr / r;
                wrapper.Saxpy(P, R, beta, P, N);                // P = R + beta*P
                ++k;
            }

            X.RefreshHost();
        }

        [EntryPoint]
        private static void ScalarProd([Out] float[] result, [In] FloatResidentArray r1, [In] FloatResidentArray r2, int N)
        {
            var cache = new SharedMemoryAllocator<float>().allocate(blockDim.x);
            int tid = threadIdx.x + blockDim.x * blockIdx.x;
            int cacheIndex = threadIdx.x;

            float tmp = 0.0F;
            while (tid < N)
            {
                tmp += r1[tid] * r2[tid];
                tid += blockDim.x * gridDim.x;
            }

            cache[cacheIndex] = tmp;

            CUDAIntrinsics.__syncthreads();

            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (cacheIndex < i)
                {
                    cache[cacheIndex] += cache[cacheIndex + i];
                }

                CUDAIntrinsics.__syncthreads();
                i >>= 1;
            }

            if (cacheIndex == 0)
            {
                AtomicExpr.apply(ref result[0], cache[0], (x, y) => x + y);
            }
        }

        [EntryPoint]
        public static void Copy([In] FloatResidentArray res, [In] FloatResidentArray src, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                res[i] = src[i];
            });
        }

        [EntryPoint]
        public static void Saxpy([In] FloatResidentArray res, [In] FloatResidentArray x, float alpha, [In] FloatResidentArray y, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                res[i] = x[i] + alpha * y[i];
            });
        }

        // res = A - m*v
        [EntryPoint]
        public static void Fmsub([In] FloatResidentArray res, [In] FloatResidentArray A, [In] SparseMatrix m, [In] FloatResidentArray v, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                int rowless = m.rows[i];
                int rowup = m.rows[i + 1];
                float tmp = A[i];
                for (int j = rowless; j < rowup; ++j)
                {
                    tmp -= v[m.indices[j]] * m.data[j];
                }

                res[i] = tmp;
            });
        }

        [EntryPoint]
        public static void Multiply([In] FloatResidentArray res, [In] SparseMatrix m, [In] FloatResidentArray v, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                int rowless = m.rows[i];
                int rowup = m.rows[i + 1];
                float tmp = 0.0F;
                for (int j = rowless; j < rowup; ++j)
                {
                    tmp += v[m.indices[j]] * m.data[j];
                }

                res[i] = tmp;
            });
        }
    }
}