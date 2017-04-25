using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Hybridizer.Basic.Utilities
{
    public static class RandomExtension
    {
        public static float NextFloat(this Random a, float min = 0.0f, float max = 1.0f)
        {
            return ((float)a.NextDouble()) * (max - min) + min;
        }

        public static float4 NextFloat4(this Random a, float min = 0.0f, float max = 1.0f)
        {
            return new float4(a.NextFloat(min, max), a.NextFloat(min, max), a.NextFloat(min, max), a.NextFloat(min, max));
        }

        public static float2 NextFloat2(this Random a, float min = 0.0f, float max = 1.0f)
        {
            return new float2(a.NextFloat(min, max), a.NextFloat(min, max));
        }
    }

}
