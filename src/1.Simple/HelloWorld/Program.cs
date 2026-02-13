using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic.Utilities;
using System;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace HelloWorld
{
    class Program
    {
        [EntryPoint]
        public static void Run(int N, double[] a, [In] double[] b)
        {
            Parallel.For(0, N, i => { a[i] += b[i]; });
        }

        static void Main(string[] args)
        {
            // 268 MB allocated on device -- should fit in every CUDA compatible GPU
            int N = 1024 * 1024 * 16;
            double[] acuda = new double[N];
            double[] adotnet = new double[N];

            double[] b = new double[N];

            Random rand = new();

            //Initialize acuda et adotnet and b by some doubles randoms, acuda and adotnet have same numbers. 
            for(int i = 0; i < N; ++i)
            {
                acuda[i] = rand.NextDouble();
                adotnet[i] = acuda[i];
                b[i] = rand.NextDouble();
            }

            cuda.GetDeviceProperties(out cudaDeviceProp prop, 0);
            HybRunner runner = SatelliteLoader.Load().SetDistrib(prop.multiProcessorCount * 16, 128);

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());
            
            // run the method on GPU
            wrapped.Run(N, acuda, b);
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
            
            // run .Net method
            Run(N, adotnet, b);

            // verify the results
            for (int k = 0; k < N; ++k)
            {
                if (acuda[k] != adotnet[k]) {
                    Console.Out.WriteLine("ERROR !");
                    return;
                }
            }
            Console.Out.WriteLine("DONE");
        }
    }
}