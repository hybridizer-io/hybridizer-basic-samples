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
        /// <summary>
        /// must return a number between 0 and 1
        /// </summary>
        /// <param name="x">between 0 and 1</param>
        /// <param name="y">between 0 and 1</param>
        /// <returns></returns>
        [Kernel]
        float Temperature(float x, float y);
    }
}
