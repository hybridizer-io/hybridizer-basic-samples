using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonteCarloHeatEquation
{
    public struct TetrisBoundaryCondition: IBoundaryCondition
    {
        [Kernel]
        public float Temperature(float x, float y)
        {
            if (y > 0.9F)
                return 1.0F;
            return 0.0F;
        }
    }
}
