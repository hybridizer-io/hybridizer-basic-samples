using Hybridizer.Runtime.CUDAImports;
using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading.Tasks;


namespace GenericMemoryAccess
{
    public class Program
    {
        
        [EntryPoint] // marks the method to be hybridized
        public static void MyAutoEntryPoint (int[] dst, int[] src, int N)
        {
            Parallel.For(0, N, i =>
            {
                dst[i] += src[i];
            });
        }
        
        [EntryPoint] // marks the method to be hybridized
        public static void MyFineGrainedEntryPoint (int[] dst, int[] src, int N)
        {
            // explicit control of threads
            for(int i = threadIdx.x + blockDim.x * blockIdx.x; i < N; i += blockDim.x * gridDim.x)
            {
                dst[i] += src[i];
            }
        }

        public static void Main(string[] args)
        {
            // setup cuda and generated dll
            cudaDeviceProp prop = DetectAndSelectCudaDevice();
            dynamic wrapped = WrapCudaDll(prop);

            // prepare data
            const int N = 1 << 24; // 4 millions random integers
            int[] src = new int[N];
            int[] dotnet_dst_1 = new int[N];
            int[] dotnet_dst_2 = new int[N];
            int[] cuda_dst_1 = new int[N];
            int[] cuda_dst_2 = new int[N];
            var rand = new Random();
            for(int i = 0; i < N; ++i)
            {
                src[i] = rand.Next();
                dotnet_dst_1[i] = rand.Next();
                dotnet_dst_2[i] = dotnet_dst_1[i];
                cuda_dst_1[i] = dotnet_dst_1[i];
                cuda_dst_2[i] = dotnet_dst_1[i];
            }

            // run
            Console.Out.WriteLine("running dotnet");
            MyAutoEntryPoint(dotnet_dst_1, src, N);
            MyFineGrainedEntryPoint(dotnet_dst_2, src, N);
            Console.Out.WriteLine("running generated CUDA");
            wrapped.MyAutoEntryPoint(cuda_dst_1, src, N);
            wrapped.MyFineGrainedEntryPoint(cuda_dst_2, src, N);

            if (!CudaErrorCheck())
            {
                return;
            }

            if (!CheckResults(N, dotnet_dst_1, dotnet_dst_2, cuda_dst_1, cuda_dst_2))
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

            string cuda_dll = Path.Combine(executing_assembly.FullName, "GenericMemoryAccess_CUDA.dll");
            if(!File.Exists(cuda_dll))
            {
                Console.Error.WriteLine($"CUDA dll({cuda_dll}) not found");
                Environment.Exit(6); // abort
            }

            // register dll and configure default execution grid
            HybRunner runner = HybRunner.Cuda(cuda_dll).SetDistrib(deviceProp.multiProcessorCount * 4, 1, 256, 1, 1, 0);
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

        private static bool CheckResults(int N, int[] dotnet_dst_1, int[] dotnet_dst_2, int[] cuda_dst_1, int[] cuda_dst_2)
        {
            for (int i = 0; i < N; ++i)
            {
                if (dotnet_dst_2[i] != dotnet_dst_1[i])
                {
                    Console.Error.WriteLine($"Dotnet Error at index {i}");
                }
                if (cuda_dst_1[i] != dotnet_dst_1[i])
                {
                    Console.Error.WriteLine($"CUDA Error at index {i} for method MyAutoEntryPoint");
                    return false;
                }
                if (cuda_dst_2[i] != dotnet_dst_1[i])
                {
                    Console.Error.WriteLine($"CUDA Error at index {i} for method MyFineGrainedEntryPoint");
                    return false;
                }
            }

            return true;
        }
    }
}