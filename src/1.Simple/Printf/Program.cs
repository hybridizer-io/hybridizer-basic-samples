using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic;
using System;
using Hybridizer.Basic.Utilities;

namespace SimplePrintf
{
    class Program
    {
        [EntryPoint]
        public static void Run(int N, int[] a)
        {
            for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < N; i += blockDim.x * gridDim.x)
            {
                Console.Out.Write("hello from thread = {0} in block = {1} a[i] = {2}\n", threadIdx.x, blockIdx.x, a[i]);
            };
        }

        static void Main(string[] args)
        {
            int[] a = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };

            // create an instance of HybRunner object to wrap calls on GPU
            HybRunner runner = SatelliteLoader.Load().SetDistrib(4,4);

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());

            // run the method on GPU
            wrapped.Run(a.Length, a);

            // synchronize the GPU to flush stdout on the device
            // add error checking
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
        }
    }
}