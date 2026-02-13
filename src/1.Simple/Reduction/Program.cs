using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic.Utilities;
using System.Runtime.InteropServices;

namespace Reduction
{
    class Program
    {
        [EntryPoint]
        public static void ReduceAdd(int N, [In] int[] a, [Out] int[] result)
        {
            var cache = new SharedMemoryAllocator<int>().allocate(blockDim.x);
            int tid = threadIdx.x + blockDim.x * blockIdx.x;
            int cacheIndex = threadIdx.x;

            int tmp = 0;
            while (tid < N)
            {
                tmp += a[tid];
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
                Interlocked.Add(ref result[0], cache[0]);
            }
        }

        static void Main(string[] args)
        {
            Random random = new Random();
            const int N = 1024 * 1024 * 32;
            int[] a = new int[N];
            for (int i = 0; i < N; ++i)
            {
                a[i] = (random.NextDouble() < 0.2) ? 1 : 0;
            }

            int[] result = new int[1];

            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);

            const int BLOCK_DIM = 256;
            HybRunner runner = SatelliteLoader.Load().SetDistrib(16 * prop.multiProcessorCount, 1, BLOCK_DIM, 1, 1, BLOCK_DIM * sizeof(int));

            dynamic wrapped = runner.Wrap(new Program());

            wrapped.ReduceAdd(N, a, result);

            cuda.DeviceSynchronize();
            Console.Out.WriteLine("sum =      {0}", result[0]);
            Console.Out.WriteLine("expected = {0}", a.Aggregate((i, j) => i + j));
        }
    }
}