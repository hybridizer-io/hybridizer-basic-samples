using Hybridizer.Runtime.CUDAImports;
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
            // setup cuda and generated dll
            cudaDeviceProp prop = DetectAndSelectCudaDevice();
            dynamic wrapped = WrapCudaDll(prop);

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
            Console.WriteLine("running generated CUDA (no attributes)");
            Stopwatch w = new ();
            w.Start();
            wrapped.NoAttributes(cuda_dst_1, src, N);
            cuda.DeviceSynchronize();
            w.Stop();
            Console.WriteLine($"no in/out attribute time : {w.ElapsedMilliseconds} ms");
            Console.WriteLine("running generated CUDA (attributes)");
            w.Reset();
            w.Start();
            wrapped.Attributes(cuda_dst_2, src, N);
            cuda.DeviceSynchronize();
            w.Stop();
            Console.WriteLine($"in/out attributes time : {w.ElapsedMilliseconds} ms");

            if (!CudaErrorCheck())
            {
                return;
            }

            Console.WriteLine("OK");
        }

        /// <summary>
        /// Selects CUDA enabled device with highest compute capability
        /// </summary>
        /// <returns>its cuda device properties</returns>
        private static cudaDeviceProp DetectAndSelectCudaDevice()
        {
            cuda.GetDeviceCount(out int deviceCount);
            if(deviceCount <= 0)
            {
                Console.Error.WriteLine("No CUDA-capable device detected -- aborting");
                Environment.Exit(6);
            }

            int maxCC = -1;
            int deviceId = -1;
            cuda.GetDeviceProperties(out cudaDeviceProp result, 0);
            for(int i = 0; i < deviceCount; ++i)
            {
                cuda.GetDeviceProperties(out cudaDeviceProp prop, i);
                int cc = 10 * prop.major + prop.minor;
                if(cc > maxCC)
                {
                    maxCC = cc;
                    deviceId = i;
                    result = prop;
                }
            }

            Console.WriteLine($"Selecting device {new string(result.name)} with compute capability {maxCC}");
            cuda.SetDevice(deviceId);
            return result;
        }

        private static dynamic WrapCudaDll(cudaDeviceProp deviceProp)
        {
            var executing_assembly = new FileInfo(Assembly.GetExecutingAssembly().Location).Directory;
            if(executing_assembly == null)
            {
                Console.Error.WriteLine("Cannot find executing assembly");
                Environment.Exit(6); // abort
            }

            string cuda_dll = Path.Combine(executing_assembly.FullName, "InOut_CUDA.dll");
            if(!File.Exists(cuda_dll))
            {
                Console.Error.WriteLine($"CUDA dll({cuda_dll}) not found");
                Environment.Exit(6); // abort
            }

            // register dll and configure default execution grid
            HybRunner runner = HybRunner.Cuda(cuda_dll).SetDistrib(deviceProp.multiProcessorCount * 4, 1, 1, 256, 1, 1, 0);
            dynamic wrapped = runner.Wrap(new Program());
            return wrapped;
        }

        private static bool CudaErrorCheck()
        {
            cudaError_t err = cuda.GetPeekAtLastError();
            if (err != cudaError_t.cudaSuccess)
            {
                Console.Error.WriteLine($"GPUAssert (peek at last error): {err} -- {cuda.GetErrorString(err)}");
                return false;
            }
            err = cuda.DeviceSynchronize();
            if (err != cudaError_t.cudaSuccess)
            {
                Console.Error.WriteLine($"GPUAssert (device synchronize): {err} -- {cuda.GetErrorString(err)}");
                return false;
            }

            return true;
        }
    }
}