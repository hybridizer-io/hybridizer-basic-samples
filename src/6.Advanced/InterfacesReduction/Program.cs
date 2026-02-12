using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace InterfacesReduction
{
    interface ILocalReductor
    {
        [Kernel] // mandatory on interface
        float neutral { get; }
        [Kernel] // mandatory on interface
        float func(float x, float y);
    }

    class AddLocalReductor : ILocalReductor
    {
        [Kernel] // mandatory on implementation
        public float neutral { get => 0.0F;  }

        [Kernel] // mandatory on implementation
        public float func(float x, float y)
        {
            return x + y;
        }
    }

    class MaxLocalReductor : ILocalReductor
    {
        [Kernel] // mandatory on implementation
        public float neutral { get => float.NegativeInfinity; }

        [Kernel] // mandatory on implementation
        public float func(float x, float y)
        {
            return Math.Max(x, y);
        }
    }


    class Program
    {
        [EntryPoint]
        public static void Reduce([Out] float[] result, [In] float[] input, int N, ILocalReductor localReductor)
        {
            var cache = new SharedMemoryAllocator<float>().allocate(blockDim.x);
            int tid = threadIdx.x + blockDim.x * blockIdx.x;
            int cacheIndex = threadIdx.x;

            float tmp = localReductor.neutral;
            while (tid < N)
            {
                tmp = localReductor.func(tmp, input[tid]);
                tid += blockDim.x * gridDim.x;
            }

            cache[cacheIndex] = tmp;

            CUDAIntrinsics.__syncthreads();

            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (cacheIndex < i)
                {
                    cache[cacheIndex] = localReductor.func(cache[cacheIndex], cache[cacheIndex + i]);
                }

                CUDAIntrinsics.__syncthreads();
                i >>= 1;
            }

            if (cacheIndex == 0)
            {
                AtomicExpr.apply(ref result[0], cache[0], localReductor.func);
            }
        }

        static void Main(string[] args)
        {
            const int N = 1024 * 1024 * 32;
            float[] a = new float[N];

            // initialization
            Random random = new Random(42);
            Parallel.For(0, N, i => a[i] = (float)random.NextDouble());

            // hybridizer configuration
            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            int gridDimX = 8 * prop.multiProcessorCount;
            int blockDimX = 128;
            cuda.DeviceSetCacheConfig(cudaFuncCache.cudaFuncCachePreferShared);
            HybRunner runner = SatelliteLoader.Load().SetDistrib(gridDimX, 1, blockDimX, 1, 1, blockDimX * sizeof(float));
            float[] buffMax = new float[1];
            float[] buffAdd = new float[1];
            dynamic wrapped = runner.Wrap(new Program());

            // device reduction
            cuda.ERROR_CHECK((cudaError_t) wrapped.Reduce(buffMax, a, N, new MaxLocalReductor()));
            cuda.ERROR_CHECK((cudaError_t) wrapped.Reduce(buffAdd, a, N, new AddLocalReductor()));

            // check results
            float expectedMax = a.AsParallel().Aggregate((x, y) => Math.Max(x, y));
            float expectedAdd = a.AsParallel().Aggregate((x, y) => x + y);
            bool hasError = false;
            if (buffMax[0] != expectedMax)
            {
                Console.Error.WriteLine($"MAX Error : {buffMax[0]} != {expectedMax}");
                hasError = true;
            }

            // addition is not associative, so results cannot be exactly the same
            // https://en.wikipedia.org/wiki/Associative_property#Nonassociativity_of_floating_point_calculation
            if (Math.Abs(buffAdd[0] - expectedAdd) / expectedAdd > 1.0E-5F)
            {
                Console.Error.WriteLine($"ADD Error : {buffAdd[0]} != {expectedAdd}");
                hasError = true;
            }

            if (hasError)
                Environment.Exit(1);

            Console.Out.WriteLine("OK");
        }
    }
}