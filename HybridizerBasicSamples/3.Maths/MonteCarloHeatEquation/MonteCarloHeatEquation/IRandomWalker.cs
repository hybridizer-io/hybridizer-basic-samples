using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonteCarloHeatEquation
{
    [HybridTemplateConcept]
    public interface IRandomWalker
    {
        [Kernel]
        void Init();
        [Kernel]
        void Walk(float fx, float fy, out float tx, out float ty);
    }
}
