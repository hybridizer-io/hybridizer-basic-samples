using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic;
using System;
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
            cuda.ERROR_CHECK(cuda.DeviceSynchronize());
        }
    }
}