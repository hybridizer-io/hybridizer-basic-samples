using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace Hybridizer.Basic.Maths
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                args = new string[] { "512", "512", "512", "512" };
            }
            const int redo = 10;

            int heightA = Convert.ToInt32(args[0]);
            int widthA = Convert.ToInt32(args[1]);
            int heightB = Convert.ToInt32(args[2]);
            int widthB = Convert.ToInt32(args[3]);
            if (widthA != heightB)
            {
                throw new ArgumentException("invalid data -- incompatible matrices");
            }

            Console.WriteLine("Execution Naive matrix mul with sizes ({0}, {1}) x ({2}, {3})", heightA, widthA, heightB, widthB);

            NaiveMatrix matrixA = new NaiveMatrix(widthA, heightA);
            NaiveMatrix matrixB = new NaiveMatrix(widthB, heightB);

            NaiveMatrix res_net = new NaiveMatrix(widthB, heightA);
            NaiveMatrix res_cuda = new NaiveMatrix(widthB, heightA);

            double numberCompute = ((double)matrixA.Height * (double)matrixA.Width * (double)matrixB.Width) * 3.0E-9;

            matrixA.FillMatrix();
            matrixB.FillMatrix();

            #region CUDA 

            HybRunner runner = HybRunner.Cuda("SharedMatrix_CUDA.dll").SetDistrib(4, 5, 32, 32, 1, 1024*2*8);
            dynamic wrapper = runner.Wrap(new Program());
            
            for (int i = 0; i < redo; ++i)
            {
                wrapper.Multiply(res_cuda, matrixA, matrixB, matrixA.Width);
            }
            #endregion

            #region C#
            Reference(res_net, matrixA, matrixB);
            #endregion
            
            Console.Out.WriteLine("DONE");
        }

        [IntrinsicFunction("__syncthreads")]
        static void SyncThreads() { }


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
        
        [EntryPoint]
        public static void Multiply(NaiveMatrix result, NaiveMatrix A, NaiveMatrix B, int size)
        {
            SharedMemoryAllocator<float> allocator = new SharedMemoryAllocator<float>();
            
            float[] cacheA = allocator.allocate(blockDim.y * blockDim.x);
            float[] cacheB = allocator.allocate(blockDim.y * blockDim.x);
            
            for (int by = blockIdx.y; by < size / blockDim.y; by += gridDim.y)
            {
                for (int bx = blockIdx.x; bx < size / blockDim.x; bx += gridDim.x)
                {
                    int tx = threadIdx.x, ty = threadIdx.y;

                    int i = by * blockDim.y + ty;
                    int j = bx * blockDim.x + tx;

                    if (i >= size || j >= size)
                    {
                        return;
                    }

                    float Pvalue = 0;
                    for (int blockIdread = 0; blockIdread < size / blockDim.x; ++blockIdread)
                    {
                        cacheA[ty * blockDim.y + tx] = A[i * size + (blockIdread * blockDim.x + tx)];
                        cacheB[ty * blockDim.y + tx] = B[(blockIdread * blockDim.x + ty) * size + j];

                        SyncThreads();

                        for (int k = 0; k < blockDim.x; ++k)
                        {
                            Pvalue += cacheA[ty * blockDim.x + k] * cacheB[k * blockDim.x + tx];
                        }

                        SyncThreads();
                    }
                    
                    result[i * size + j] = Pvalue;
                }
            }
        }
    }
}

