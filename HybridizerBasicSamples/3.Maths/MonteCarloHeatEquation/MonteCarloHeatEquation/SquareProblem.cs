using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace MonteCarloHeatEquation
{
    [HybridRegisterTemplate(Specialize = typeof(SquareProblem<SimpleWalker>))]
    public class SquareProblem<TRandomWalker>: I2DProblem where TRandomWalker : struct, IRandomWalker
    {
        public FloatResidentArray _bottom;
        public FloatResidentArray _right;
        public FloatResidentArray _top;
        public FloatResidentArray _left;
        public FloatResidentArray _inner;
        public int _N;   // resolution
        public int _iter;

        [HybridizerIgnore]
        public SquareProblem(int N, Func<float, float, float> outsideTemperature, int iter)
        {
            _N = N;
            _bottom = new FloatResidentArray(N + 1);
            _right = new FloatResidentArray(N + 1);
            _top = new FloatResidentArray(N + 1);
            _left = new FloatResidentArray(N + 1);
            _inner = new FloatResidentArray((N-1) * (N-1));
            for(int i = 0; i <= N; ++i)
            {
                _bottom[i] = outsideTemperature((float)i / N, 0.0F);
                _top[i] = outsideTemperature((float)i / N, 1.0F);
                _left[i] = outsideTemperature(0.0F, (float) i / N);
                _right[i] = outsideTemperature(1.0F, (float) i / N);
            }
            
            _iter = iter;
        }

        public void RefreshDevice()
        {
            _bottom.RefreshDevice();
            _top.RefreshDevice();
            _left.RefreshDevice();
            _right.RefreshDevice();
        }

        public void RefreshHost()
        {
            _inner.RefreshHost();
        }

        [Kernel]
        public float GetBoundaryTemperature(int i, int j)
        {
            if(i == 0)
            {
                return _left[j];
            } 
            if (i == _N)
            {
                return _right[j];
            }
            if (j == 0)
            {
                return _bottom[i];
            }
            if (j == _N)
            {
                return _top[i];
            }

            return 0.0F;
        }

        [Kernel]
        public void WriteTemperature(int i, int j, float temperature)
        {
            _inner[j * (_N - 1) + i] = temperature;
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
            TRandomWalker walker = default(TRandomWalker);
            walker.Init();
            float temperature = 0.0F;
            float size = (float)_N;
            for (int iter = 0; iter < _iter; ++iter)
            {
                float fx = x;
                float fy = y;
                
                while (true)
                {
                    float tx, ty;
                    walker.Walk(fx, fy, out tx, out ty);
                    if(tx == 0.0F || ty == size || tx == size || ty == 0.0F)
                    {
                        temperature += GetBoundaryTemperature((int) tx, (int) ty);
                        break;
                    }

                    fx = tx;
                    fy = ty;
                }
            }

            WriteTemperature((int) (x - 1), (int) (y - 1), temperature / (float) _iter);
        }

        [HybridizerIgnore]
        public void SaveImage(string fileName, Func<float, Color> GetColor)
        {
            Bitmap image = new Bitmap(_N - 1, _N - 1);
            for (int j = 0; j <= _N - 2; ++j)
            {
                for (int i = 0; i <= _N - 2; ++i)
                {
                    float temp = _inner[j * (_N - 1) + i];
                    image.SetPixel(i, j, GetColor(temp));
                }
            }

            image.Save(fileName);
        }
    }
}
