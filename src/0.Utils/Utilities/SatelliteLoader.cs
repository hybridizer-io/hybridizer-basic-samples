using Hybridizer.Runtime.CUDAImports;
using System.Reflection;

namespace Hybridizer.Basic.Utilities
{
    public class SatelliteLoader
    {
        public static HybRunner Load()
        {
            var executing_directory = new FileInfo(Assembly.GetExecutingAssembly().Location).Directory.FullName;
            var satellite = Directory.GetFiles(executing_directory, "*_CUDA.dll").First();
            return HybRunner.Cuda(satellite);
        }
    }
}