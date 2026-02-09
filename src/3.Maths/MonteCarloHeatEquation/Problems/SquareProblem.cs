using Hybridizer.Runtime.CUDAImports;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;

namespace MonteCarloHeatEquation
{
    [HybridRegisterTemplate(Specialize = typeof(SquareProblem<SimpleWalker, SimpleBoundaryCondition>))]
    public class SquareProblem<TRandomWalker, TBoundaryCondition>: I2DProblem 
        where TRandomWalker : struct, IRandomWalker 
        where TBoundaryCondition: struct, IBoundaryCondition
    {

        private FloatResidentArray _inner;
        private int _N;   // resolution
        private int _iter;
        private float _h;
        private float _invIter;

        [HybridizerIgnore]
        public SquareProblem(int N, int iter)
        {
            _N = N;
            _h = 1.0F / (float)_N;
            _invIter = 1.0F / iter;
            _inner = new FloatResidentArray((N-1) * (N-1));
            _iter = iter;
        }

        public void RefreshHost()
        {
            _inner.RefreshHost();
        }

        [Kernel]
        public int MaxIndex()
        {
            return (_N - 1) * (_N - 1);
        }

        [Kernel] 
        public void Coordinates(int i, out int ii, out int jj)
        {
            ii = (i % (_N - 1)) + 1;
            jj = (i / (_N - 1)) + 1;
        }
        

        [Kernel]
        public void Solve(float x, float y)
        {
            TRandomWalker walker = default;
            TBoundaryCondition boundaryCondition = default;
            walker.Init();
            float temperature = 0.0F;
            float size = _N;
            for (int iter = 0; iter < _iter; ++iter)
            {
                float fx = x;
                float fy = y;
                
                while (true)
                {
                    walker.Walk(fx, fy, out float tx, out float ty);

                    // when on border, break
                    if (tx == 0.0F || ty == size || tx == size || ty == 0.0F)
                    {
                        temperature += boundaryCondition.Temperature((float)tx * _h, (float)ty * _h);
                        break;
                    }

                    // otherwise continue walk
                    fx = tx;
                    fy = ty;
                }
            }

            _inner[((int)(y - 1)) * (_N - 1) + (int)(x - 1)] = temperature * _invIter;
        }

        [HybridizerIgnore]
        public void SaveImage(string fileName, Func<float, Color> GetColor)
        {
            var image = new Image<Argb32>(_N - 1, _N - 1);
            for (int j = 0; j <= _N - 2; ++j)
            {
                for (int i = 0; i <= _N - 2; ++i)
                {
                    float temp = _inner[j * (_N - 1) + i];
                    image[i, j] = GetColor(temp);
                }
            }

            image.Save(fileName, new PngEncoder());
        }
    }
}
