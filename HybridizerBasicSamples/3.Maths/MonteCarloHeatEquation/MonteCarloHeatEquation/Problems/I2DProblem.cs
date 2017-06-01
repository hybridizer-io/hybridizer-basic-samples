using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MonteCarloHeatEquation
{
    public interface I2DProblem
    {
        [Kernel]
        int MaxIndex();
        [Kernel]
        void Coordinates(int i, out int ii, out int jj);
        [Kernel]
        void Solve(float x, float y);
        [HybridizerIgnore]
        void SaveImage(string fileName, Func<float, Color> GetColor);
    }
}
