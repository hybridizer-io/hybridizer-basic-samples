using Hybridizer.Runtime.CUDAImports;

namespace MonteCarloHeatEquation
{
    public struct SimpleBoundaryCondition : IBoundaryCondition
    {
        [Kernel]
        public readonly float Temperature(float x, float y)
        {
            if ((x == 1.0F && y >= 0.5F) || (x == 0.0F && y <= 0.5F))
                return 1.0F;
            return 0.0F;
        }
    }
}
