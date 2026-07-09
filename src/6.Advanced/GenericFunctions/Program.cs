using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;
using System.Diagnostics;

/// <summary>
/// experimental feature -- no extensive testing
/// </summary>
namespace GenericFunctions
{
    class Vector<T> where T : struct
    {
        readonly T[] _data;

        public Vector(T[] data)
        {
            _data = data;
            Length = data.Length;
        }

        [Kernel]
        public int Length { get; }
        
        [Kernel]
        public T this[int i] { get => _data[i]; set { _data[i] = value; } }
    }

    class Operations
    {
        [Kernel]
        public static void Invoke<T>(Vector<T> v, int i, T val, Func<T, T, T> func) where T: struct
        {
            v[i] = func(v[i], val);
        }

        [EntryPoint]
        public static void Add(Vector<float> v, float a)
        {
            Parallel.For(0, v.Length, i => Invoke(v, i, a, (x, y) => x + y));
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            const int N = 1024*1024*32;
            float[] a = new float[N];
            float[] aCPU = new float[N];
            var vect = new Vector<float>(a);

            Stopwatch swGpu = new Stopwatch();
            swGpu.Start();
            SatelliteLoader.Load().Wrap(new Operations()).Add(vect, 1.0F);
            cuda.DeviceSynchronize();
            swGpu.Stop();
            Console.WriteLine("GPU time : {0} ms", swGpu.ElapsedMilliseconds);

            Stopwatch swCpu = new Stopwatch();
            swCpu.Start();
            Parallel.For(0, N, i => aCPU[i] = 0.0F + 1.0F);
            swCpu.Stop();
            Console.WriteLine("CPU time : {0} ms", swCpu.ElapsedMilliseconds);

            int errorCount = 0;
            for (int i = 0; i < N; ++i)
            {
                if (a[i] != 1.0F)
                {
                    errorCount++;   
                }
            }
            if (errorCount > 0)
            {
                Console.Error.WriteLine("{0} error(s) out of {1:N0} values ({2:P4})", errorCount, N, errorCount / (double)N);
                Environment.Exit(6); // abort
            }

            Console.Out.WriteLine("OK");
        }
    }
}