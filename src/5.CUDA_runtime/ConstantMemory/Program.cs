using System.Runtime.InteropServices;
using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;
using System.Diagnostics;

namespace ConstantMemory
{
    class Program
    {
        [HybridConstant(Location = ConstantLocation.ConstantMemory)]
        public static float[] data = [-2.0F, -1.0F, 0.0F, 1.0F, 2.0F];

        [EntryPoint]
        public static void Run([Out] float[] output, [In] float[] input, int N)
        {
            for (int k = 2 + threadIdx.x + blockDim.x * blockIdx.x; k < N - 2; k += blockDim.x * gridDim.x)
            {
                float tmp = 0;
                for (int p = -2; p <= 2; ++p)
                {
                    tmp += data[p + 2] * input[k + p];
                }

                output[k] = tmp;
            }
        }

        public static void RunCPU(float[] output, float[] input, int N)
        {
            for (int k = 2; k < N - 2; ++k)
            {
                float tmp = 0;
                for (int p = -2; p <= 2; ++p)
                {
                    tmp += data[p + 2] * input[k + p];
                }
                output[k] = tmp;
            }
        }

        static void Main(string[] args)
        {
            const int N = 1024 * 1024 * 32;
            float[] input = new float[N];
            float[] output = new float[N];
            float[] outputCPU = new float[N];
            Random rand = new();
            for (int k = 0; k < N; ++k)
            {
                input[k] = (float)rand.NextDouble();
            }

            HybRunner runner = SatelliteLoader.Load();

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());
            Stopwatch swGpu = new Stopwatch();
            swGpu.Start();
            wrapped.Run(output, input, N);
            swGpu.Stop();
            Console.WriteLine("GPU time : {0} ms", swGpu.ElapsedMilliseconds);

            Stopwatch swCpu = new Stopwatch();
            swCpu.Start();
            RunCPU(outputCPU, input, N);
            swCpu.Stop();
            Console.WriteLine("CPU time : {0} ms", swCpu.ElapsedMilliseconds);

            Console.WriteLine("\nAperçu des premières valeurs :");
            for (int k = 0; k < 10; ++k)
            {
                Console.WriteLine("input[{0}] = {1,10:F4}   output[{0}] = {2,10:F4}   outputCPU[{0}] = {3,10:F4}",
                    k, input[k], output[k], outputCPU[k]);
            }

            Console.Out.WriteLine("DONE");
        }
    }
}