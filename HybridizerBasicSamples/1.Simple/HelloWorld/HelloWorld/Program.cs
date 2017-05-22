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

            double dataGo = (double) N * 3.0 * 8.0 * 1e-09;

            Random rand = new Random();

            //Initialize acuda et adotnet and b by some doubles randoms, acuda and adotnet have same numbers. 
            for(int i = 0; i < N; ++i)
            {
                acuda[i] = rand.NextDouble();
                adotnet[i] = acuda[i];
                b[i] = rand.NextDouble();
            }

            // create an instance of HybRunner object to wrap calls on GPU
            HybRunner runner = HybRunner.Cuda("HelloWorld_CUDA.dll").SetDistrib(20,256);

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
