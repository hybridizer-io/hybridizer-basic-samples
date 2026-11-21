using Hybridizer.Runtime.CUDAImports;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Hybridizer.Basic.Utilities
{
    public class SatelliteLoader
    {
        /// <summary>
        /// Runner of the satellite the build put next to the program: <c>&lt;Name&gt;_CUDA.dll</c>, or on macOS, which
        /// has no CUDA, <c>&lt;Name&gt;_METAL.dylib</c>, for the Apple GPU
        /// </summary>
        public static HybRunner Load()
        {
            var executing_directory = new FileInfo(Assembly.GetExecutingAssembly().Location).Directory.FullName;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                var metal = Directory.GetFiles(executing_directory, "*_METAL.dylib").FirstOrDefault();
                if (metal != null)
                    return HybRunner.Create(HybridizerFlavor.METAL, metal);
            }
            var satellite = Directory.GetFiles(executing_directory, "*_CUDA.dll").First();
            return HybRunner.Cuda(satellite);
        }
    }
}
