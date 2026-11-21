using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic;
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Hybridizer.Basic.Utilities;

namespace Intrinsics
{
    class Program
    {
        [HybridArithmeticFunction]
        public static half2 getHalf2(float x)
        {
            return new half2(x, x);
        }

        [HybridArithmeticFunction]
        public static half2 exp(half2 x)
        {
            return ((((((((((((((getHalf2(15.0F) + x)
                * x + getHalf2(210.0F))
                * x + getHalf2(2730.0F))
                * x + getHalf2(32760.0F))
                * x + getHalf2(360360.0F))
                * x + getHalf2(3603600.0F))
                * x + getHalf2(32432400.0F))
                * x + getHalf2(259459200.0F))
                * x + getHalf2(1816214400.0F))
                * x + getHalf2(10897286400.0F))
                * x + getHalf2(54486432000.0F))
                * x + getHalf2(217945728000.0F))
                * x + getHalf2(653837184000.0F))
                * x + getHalf2(1307674368000.0F))
                * x * getHalf2(7.6471637318198164759011319857881e-13F);
        }

        public static half2 exp12(half2 x)
        {
            return exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(exp(x))))))))))));
        }

        [EntryPoint]
        public static void Compute(half2[] input, int N)
        {
            Parallel.For(0, N, i =>
            {
                input[i] = exp12(input[i]);
            });
        }

        static void Main(string[] args)
        {
            const int N = 1024 * 1024 * 32;
            half2[] input = new half2[N];

            HybRunner runner = SatelliteLoader.Load();
            dynamic wrapped = runner.Wrap(new Program());
            wrapped.Compute(input, N);
            gpu.Check(gpu.DeviceSynchronize());

            // half2 has no arithmetic in .NET: the same operations, on System.Half (both halves are the same)
            Half expected = Exp12Reference((Half)0.0F);
            Span<uint> bits = MemoryMarshal.Cast<half2, uint>(input.AsSpan());
            for (int i = 0; i < N; ++i)
            {
                foreach (Half r in new[] { BitConverter.UInt16BitsToHalf((ushort)bits[i]), BitConverter.UInt16BitsToHalf((ushort)(bits[i] >> 16)) })
                {
                    // Equals, unlike ==, holds for two NaN
                    if (!r.Equals(expected))
                    {
                        Console.Error.WriteLine($"ERROR at {i} : got {r} instead of {expected}");
                        Environment.Exit(6); // abort
                    }
                }
            }
            Console.Out.WriteLine($"OK ({expected})");
        }

        static Half ExpReference(Half x)
        {
            Half h(float f) => (Half)f;
            return ((((((((((((((h(15.0F) + x)
                * x + h(210.0F))
                * x + h(2730.0F))
                * x + h(32760.0F))
                * x + h(360360.0F))
                * x + h(3603600.0F))
                * x + h(32432400.0F))
                * x + h(259459200.0F))
                * x + h(1816214400.0F))
                * x + h(10897286400.0F))
                * x + h(54486432000.0F))
                * x + h(217945728000.0F))
                * x + h(653837184000.0F))
                * x + h(1307674368000.0F))
                * x * h(7.6471637318198164759011319857881e-13F);
        }

        static Half Exp12Reference(Half x)
        {
            for (int k = 0; k < 12; ++k)
                x = ExpReference(x);
            return x;
        }
    }
}