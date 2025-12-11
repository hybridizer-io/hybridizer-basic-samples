using Hybridizer.Basic.Utilities;
using Hybridizer.Runtime.CUDAImports;
using System;
using System.Runtime.InteropServices;

namespace Streams
{
    public class Program
    {
        [EntryPoint]
        public static void Add(float[] a, float[] b, int start, int stop, int iter)
        {
            for(int k = start + threadIdx.x + blockDim.x * blockIdx.x; k < stop; k += blockDim.x * gridDim.x)
            {
                for (int p = 0; p < iter; ++p)
                {
                    a[k] += b[k];
                }
            }
        }

        static void Main(string[] args)
        {
            int nStreams = 8;
            cudaStream_t[] streams = new cudaStream_t[nStreams];
            dynamic wrapped = SatelliteLoader.Load().Wrap(new Program());
            for (int k = 0; k < nStreams; ++k)
            {
                cuda.StreamCreate(out streams[k]);
            }

            int N = 1024 * 1024 * 32;
            IntPtr d_a, d_b; // device pointers
            float[] a = new float[N];
            float[] b = new float[N];

            cuda.Malloc(out d_a, N * sizeof(float));
            cuda.Malloc(out d_b, N * sizeof(float));

            for(int k = 0; k < N; ++k)
            {
                a[k] = (float)k;
                b[k] = 1.0F;
            }

            GCHandle handle_a = GCHandle.Alloc(a, GCHandleType.Pinned);
            GCHandle handle_b = GCHandle.Alloc(b, GCHandleType.Pinned);
            IntPtr h_a = handle_a.AddrOfPinnedObject();
            IntPtr h_b = handle_b.AddrOfPinnedObject();

            int slice = N / nStreams;

            cuda.DeviceSynchronize();

            cuda.Memcpy(d_a, h_a, N * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
            cuda.Memcpy(d_b, h_b, N * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
            
            for (int k = 0; k < nStreams; ++k)
            {
                int start = k * slice;
                int stop = start + slice;
                wrapped.SetStream(streams[k]).Add(d_a, d_b, start, stop, 100);
            }

            for (int k = 0; k < nStreams; ++k)
            {
                int start = k * slice;
                cuda.MemcpyAsync(h_a + start * sizeof(float), d_a + start * sizeof(float), slice * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost, streams[k]);
            }

            for (int k = 0; k < nStreams; ++k)
            {
                cuda.StreamSynchronize(streams[k]);
                cuda.StreamDestroy(streams[k]);
            }

            for(int k = 0; k < 10; ++k)
            {
                Console.WriteLine(a[k]);
            }

            handle_a.Free();
            handle_b.Free();
        }
    }
}