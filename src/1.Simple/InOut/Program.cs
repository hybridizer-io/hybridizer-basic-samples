using Hybridizer.Runtime.CUDAImports;
using Hybridizer.Basic.Utilities;
using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading.Tasks;


namespace InOut
{
    public class Program
    {
        [EntryPoint]
        public static void NoAttributes (int[] dst, int[] src, int N)
        {
            for(int i = threadIdx.x + blockDim.x * blockIdx.x; i < N; i += blockDim.x * gridDim.x) 
            {
                dst[i] = src[i] + i;
            };
        }
        
        [EntryPoint]
        public static void Attributes ([Out] int[] dst, [In] int[] src, int N)
        {
            for(int i = threadIdx.x + blockDim.x * blockIdx.x; i < N; i += blockDim.x * gridDim.x) 
            {
                dst[i] = src[i] + i;
            };
        }

        public static void Main(string[] args)
        {
            // setup the GPU and the generated satellite
            gpuDeviceProp prop = DetectAndSelectDevice();
            dynamic wrapped = WrapSatellite(prop);

            // prepare data
            const int N = 1 << 24; // 4 millions random integers
            int[] src = new int[N];
            int[] cuda_dst_1 = new int[N];
            int[] cuda_dst_2 = new int[N];
            var rand = new Random();
            for(int i = 0; i < N; ++i)
            {
                src[i] = rand.Next();
            }

            // run
            Console.WriteLine("running generated GPU code (no attributes)");
            Stopwatch w = new ();
            w.Start();
            wrapped.NoAttributes(cuda_dst_1, src, N);
            gpu.DeviceSynchronize();
            w.Stop();
            Console.WriteLine($"no in/out attribute time : {w.ElapsedMilliseconds} ms");
            Console.WriteLine("running generated GPU code (attributes)");
            w.Reset();
            w.Start();
            wrapped.Attributes(cuda_dst_2, src, N);
            gpu.DeviceSynchronize();
            w.Stop();
            Console.WriteLine($"in/out attributes time : {w.ElapsedMilliseconds} ms");

            if (!GpuErrorCheck())
            {
                return;
            }

            Console.WriteLine("OK");
        }

        /// <summary>
        /// Selects the GPU with highest compute capability (Apple GPU family on Metal)
        /// </summary>
        /// <returns>its device properties</returns>
        private static gpuDeviceProp DetectAndSelectDevice()
        {
            gpu.GetDeviceCount(out int deviceCount);
            if(deviceCount <= 0)
            {
                Console.Error.WriteLine("No GPU detected -- aborting");
                Environment.Exit(6);
            }

            int maxCC = -1;
            int deviceId = -1;
            gpu.GetDeviceProperties(out gpuDeviceProp result, 0);
            for(int i = 0; i < deviceCount; ++i)
            {
                gpu.GetDeviceProperties(out gpuDeviceProp prop, i);
                int cc = 10 * prop.major + prop.minor;
                if(cc > maxCC)
                {
                    maxCC = cc;
                    deviceId = i;
                    result = prop;
                }
            }

            Console.WriteLine($"Selecting device {result.name} with compute capability {maxCC}");
            gpu.SetDevice(deviceId);
            return result;
        }

        private static dynamic WrapSatellite(gpuDeviceProp deviceProp)
        {
            // register the satellite and configure default execution grid
            HybRunner runner = SatelliteLoader.Load().SetDistrib(deviceProp.multiProcessorCount * 4, 1, 1, 256, 1, 1, 0);
            dynamic wrapped = runner.Wrap(new Program());
            return wrapped;
        }

        private static bool GpuErrorCheck()
        {
            gpuError_t err = gpu.PeekAtLastError();
            if (err != gpuError_t.gpuSuccess)
            {
                Console.Error.WriteLine($"GPUAssert (peek at last error): {err} -- {gpu.GetErrorString(err)}");
                return false;
            }
            err = gpu.DeviceSynchronize();
            if (err != gpuError_t.gpuSuccess)
            {
                Console.Error.WriteLine($"GPUAssert (device synchronize): {err} -- {gpu.GetErrorString(err)}");
                return false;
            }

            return true;
        }
    }
}