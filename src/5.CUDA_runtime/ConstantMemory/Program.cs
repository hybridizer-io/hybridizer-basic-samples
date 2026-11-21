using System.Runtime.InteropServices;
using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;

namespace ConstantMemory
{
    class Program
    {
        [HybridConstant(Location = ConstantLocation.ConstantMemory)]
        public static float[] data = [-2.0F, -1.0F, 0.0F, 1.0F, 2.0F];

        [EntryPoint]
        public static void Run([Out] float[] output, [In] float[] input, int N) 
        {
            for(int k = 2 + threadIdx.x + blockDim.x * blockIdx.x; k < N - 2; k += blockDim.x * gridDim.x)
            {
                float tmp = 0;
                for(int p = -2; p <= 2; ++p)
                {
                    tmp += data[p + 2] * input[k];
                }

                output[k] = tmp;
            }
        }

        static void Main(string[] args)
        {
            const int N = 1024 * 1024 * 32;
            float[] input = new float[N];
            float[] output = new float[N];
            Random rand = new();
            for(int k = 0; k < N; ++k)
            {
                output[k] = (float)rand.NextDouble();
            }

            HybRunner runner = SatelliteLoader.Load();

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());

            wrapped.Run(input, output, N);

            // the same computation in C#: Run writes its first argument from its second
            float[] expected = new float[N];
            Run(expected, output, N);
            for (int k = 2; k < N - 2; ++k)
            {
                if (input[k] != expected[k])
                {
                    Console.Error.WriteLine($"ERROR at {k} : {input[k]} != {expected[k]}");
                    Environment.Exit(6); // abort
                }
            }

            Console.Out.WriteLine("DONE");
        }
    }
}