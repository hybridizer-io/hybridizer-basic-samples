using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonteCarloHeatEquation
{
    public class MonteCarloHeatSolver
    {
        I2DProblem _problem;

        public MonteCarloHeatSolver(I2DProblem problem)
        {
            _problem = problem;
        }

        [EntryPoint]
        public void Solve()
        {
            int start = threadIdx.x + blockIdx.x * blockDim.x;
            int stop = _problem.MaxIndex();
            int step = blockDim.x * gridDim.x;
            for (int i = start; i < stop; i += step)
            {
                int ii, jj;
                _problem.Coordinates(i, out ii, out jj);
                _problem.Solve(ii, jj);
            }
        }
    }
}
