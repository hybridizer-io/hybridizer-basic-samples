using Hybridizer.Runtime.CUDAImports;
using System;

namespace MonteCarloHeatEquation
{
    
    public struct SimpleWalker: IRandomWalker
    {
        public int _seed;

        [IntrinsicFunction("clock")]
        public static int clock()
        {
            return Guid.NewGuid().GetHashCode();
        }

        [Kernel]
        public void Init()
        {
            _seed = threadIdx.x + (int)clock() + blockIdx.x * blockDim.x;
        }

        [Kernel]
        public int Next()
        {
            _seed ^= _seed << 13;
            _seed ^= _seed >> 17;
            _seed ^= _seed << 5;
            return _seed;
        }

        [Kernel]
        public void Walk(float fx, float fy, out float tx, out float ty)
        {
            int tmp = Next() & 3;

            tx = fx;
            ty = fy;
            switch(tmp)
            {
                case 0:
                    tx = fx + 1.0F;
                    break;
                case 1:
                    ty = fy + 1.0F;
                    break;
                case 2:
                    tx = fx - 1.0F;
                    break;
                default:
                    ty = fy - 1.0F;
                    break;
            }
        }
    }
}
