using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonteCarloHeatEquation
{

    public struct SimpleBoundaryCondition : IBoundaryCondition
    {
        [Kernel]
        public float Temperature(float x, float y)
        {
            if ((x == 1.0F && y >= 0.5F) || (x == 0.0F && y <= 0.5F))
                return 1.0F;
            return 0.0F;
        }
    }
}
