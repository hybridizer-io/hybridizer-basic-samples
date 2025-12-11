using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbulb
{
    public class TextureHelpers
    {
        [IntrinsicFunction("surf2Dwrite")]
        public static void surf2Dwrite(uchar4 data, cudaSurfaceObject_t surfObj, int x, int y)
        {
        }

        [IntrinsicFunction("cudaCreateChannelDesc")]
        public static cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind kind)
        {
            cudaChannelFormatDesc res;
            res.x = x;
            res.y = y;
            res.z = z;
            res.w = w;
            res.f = kind;
            return res;
        }

        public static unsafe cudaResourceDesc CreateCudaResourceDesc(cudaArray_t cuArrayTex)
        {
            cudaResourceDesc resDescTex;
            memset((byte*)&resDescTex, Marshal.SizeOf(resDescTex), 0);
            resDescTex.arrayStruct = cuArrayTex;
            resDescTex.resType = cudaResourceType.cudaResourceTypeArray;
            return resDescTex;
        }

        public static unsafe void memset(byte* data, int bytes, byte val)
        {
            for (int i = 0; i < bytes; ++i)
            {
                data[i] = val;
            }
        }
    }
}
