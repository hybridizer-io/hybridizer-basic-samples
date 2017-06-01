using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonteCarloHeatEquation
{
    [HybridTemplateConcept]
    public interface IBoundaryCondition
    {
        [Kernel]
        float Temperature(float x, float y);
    }
}
