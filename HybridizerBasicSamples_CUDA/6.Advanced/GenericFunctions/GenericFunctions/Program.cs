using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;

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
            Parallel.For(0, v.Length, i => Invoke<float>(v, i, a, (x, y) => x + y));
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            const int N = 1024*1024*32;
            float[] a = new float[N];
            var vect = new Vector<float>(a);
            SatelliteLoader.Load().Wrap(new Operations()).Add(vect, 1.0F);
            cuda.DeviceSynchronize();
            
            for (int i = 0; i < N; ++i)
            {
                if (a[i] != 1.0F)
                {
                    Console.Error.WriteLine($"ERROR at {i} : got {a[i]} instead of 1.0F");
                    Environment.Exit(6); // abort
                }
            }

            Console.Out.WriteLine("OK");
        }
    }
}