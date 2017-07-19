using System;
using System.Threading.Tasks;
using System.Linq;
using Hybridizer.Runtime.CUDAImports;
using System.Runtime.InteropServices;

namespace Hybridizer.Basic.Maths
{
    unsafe class Program
    {
        static HybRunner runner;
        static dynamic wrapper;

        static void Main(string[] args)
        {
            // configure CUDA
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            runner = HybRunner.Cuda("ConjugateGradient_CUDA.dll").SetDistrib(prop.multiProcessorCount * 16, 128);
            wrapper = runner.Wrap(new Program());

            int size = 1000000; // very slow convergence with no preconditioner
            SparseMatrix A = SparseMatrix.Laplacian_1D(size);
            FloatResidentArray B = new FloatResidentArray(size);
            FloatResidentArray X = new FloatResidentArray(size);

            int maxiter = size;
            float eps = 1.0e-09f;

            for (int i = 0; i < size; ++i)
            {
                B[i] = 1.0f;
                X[i] = 0.0f;
            }

            ConjugateGradient(X, A, B, maxiter, eps);

            for(int i = 0; i < Math.Min(size, 6); ++i)
            {
                Console.WriteLine(X[i]);
            }
        }

        static public void ConjugateGradient(FloatResidentArray X, SparseMatrix A, FloatResidentArray B, int maxiter, float eps)
        {
            int N = (int) B.Count;
            FloatResidentArray R = new FloatResidentArray(N);
            FloatResidentArray P = new FloatResidentArray(N);
            FloatResidentArray AP = new FloatResidentArray(N);
            R.RefreshDevice();
            P.RefreshDevice();
            AP.RefreshDevice();
            A.RefreshDevice();
            X.RefreshDevice();
            B.RefreshDevice();

            wrapper.Fmsub(R, B, A, X, N);                       // R = B - A*X
            wrapper.Copy(P, R, N); 
            int k = 0;
            while(k < maxiter)
            {
                wrapper.Multiply(AP, A, P, N);                  // AP = A*P
                float r = ScalarProd(R, R, N);          // save <R|R>
                float alpha = r / ScalarProd(P, AP, N); // alpha = <R|R> / <P|AP>
                wrapper.Saxpy(X, X, alpha, P, N);               // X = X - alpha*P
                wrapper.Saxpy(R, R, -alpha, AP, N);             // RR = R-alpha*AP
                float rr = ScalarProd(R, R, N);
                if(rr < eps*eps)
                {
                    break;
                }

                Console.Write("\riter : {0}, err : {1}", k, Math.Sqrt(rr));

                float beta = rr / r;
                wrapper.Saxpy(P, R, beta, P, N);                // P = R + beta*P
                ++k;
            }
        }

        [EntryPoint]
        public static void Copy(FloatResidentArray res, FloatResidentArray src, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                res[i] = src[i];
            });
        }

        [EntryPoint]
        public static void Saxpy(FloatResidentArray res, FloatResidentArray x, float alpha, FloatResidentArray y, int N)
        {
            Parallel.For(0, N, (i) =>
            {
                res[i] = x[i] + alpha * y[i];
            });
        }

        
        public static float ScalarProd(FloatResidentArray X, FloatResidentArray Y, int N)
        {
            return inner_scalar_prod((float*) X.DevicePointer, (float*) Y.DevicePointer, N);
            //return ParallelEnumerable.Range(0, N).Sum(i => X[i] * Y[i]);
        }

        [DllImport("hybridizer_cuda_reduction.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "ScalarProd_float")]
        public extern static float inner_scalar_prod(float* X, float* Y, int N);

        // res = A - m*v
        [EntryPoint]
        public static void Fmsub(FloatResidentArray res, FloatResidentArray A, SparseMatrix m, FloatResidentArray v, int N)
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
        public static void Multiply(FloatResidentArray res, SparseMatrix m, FloatResidentArray v, int N)
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
