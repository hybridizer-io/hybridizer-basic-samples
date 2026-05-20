using Hybridizer.Runtime.CUDAImports;
using System.IO;
using System.Linq;
using System.Reflection;

namespace WmmaGemm;

internal static class SatelliteLoader
{
    public static HybRunner Load()
    {
        var executingDirectory = new FileInfo(Assembly.GetExecutingAssembly().Location).Directory!.FullName;
        var satellite = Directory.GetFiles(executingDirectory, "*_CUDA.dll").First();
        return HybRunner.Cuda(satellite);
    }
}
