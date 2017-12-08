using Hybridizer.Runtime.CUDAImports;
using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace HelloWorld
{
    class Program
    {
        [EntryPoint("run")]
        public static void Run(int N, double[] a, double[] b)
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

            Random rand = new Random();

            //Initialize acuda et adotnet and b by some doubles randoms, acuda and adotnet have same numbers. 
            for(int i = 0; i < N; ++i)
            {
                acuda[i] = rand.NextDouble();
                adotnet[i] = acuda[i];
                b[i] = rand.NextDouble();
            }

            cudaDeviceProp prop;
            cuda.GetDeviceProperties(out prop, 0);
            HybRunner runner = HybRunner.Cuda("HelloWorld_CUDA.dll").SetDistrib(prop.multiProcessorCount * 16, 128);

            // create a wrapper object to call GPU methods instead of C#
            dynamic wrapped = runner.Wrap(new Program());
            
            // run the method on GPU
            wrapped.Run(N, acuda, b);
            
            // run .Net method
            Run(N, adotnet, b);

            // verify the results
            for (int k = 0; k < N; ++k)
            {
                if (acuda[k] != adotnet[k])
                    Console.Out.WriteLine("ERROR !");
            }
            Console.Out.WriteLine("DONE");
        }
    }
}
