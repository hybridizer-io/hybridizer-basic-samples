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
            Parallel.For(0, _problem.MaxIndex(), i =>
            {
                _problem.Coordinates(i, out int ii, out int jj);
                _problem.Solve(ii, jj);
            });
        }
    }
}
