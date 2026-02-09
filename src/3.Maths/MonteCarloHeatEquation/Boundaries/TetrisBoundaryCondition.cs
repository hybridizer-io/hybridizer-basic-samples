using Hybridizer.Runtime.CUDAImports;

namespace MonteCarloHeatEquation
{
    public struct TetrisBoundaryCondition: IBoundaryCondition
    {
        [Kernel]
        public readonly float Temperature(float x, float y)
        {
            if (y > 0.9F)
                return 1.0F;
            return 0.0F;
        }
    }
}
