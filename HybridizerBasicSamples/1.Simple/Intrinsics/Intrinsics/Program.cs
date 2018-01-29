using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Intrinsics
{
    class Program
    {
        public static half2 exp(half2 x)
        {
            return ((((((((((((((new half2(15.0F) + x)
                * x + new half2(210.0F))
                * x + new half2(2730.0F))
                * x + new half2(32760.0F))
                * x + new half2(360360.0F))
                * x + new half2(3603600.0F))
                * x + new half2(32432400.0F))
                * x + new half2(259459200.0F))
                * x + new half2(1816214400.0F))
                * x + new half2(10897286400.0F))
                * x + new half2(54486432000.0F))
                * x + new half2(217945728000.0F))
                * x + new half2(653837184000.0F))
                * x + new half2(1307674368000.0F))
                * x * new half2(7.6471637318198164759011319857881e-13F);
        }

        [EntryPoint]
        public static void Compute(half2[] input, int N)
        {
            Parallel.For(0, N, i =>
            {
                input[i] = exp(input[i]);
            });
        }

        static void Main(string[] args)
        {
        }
    }
}
