using System.Reflection;
using Hybridizer.Runtime.CUDAImports;

namespace LlamaCsharp.Utils;

/// <summary>
/// Loads the Hybridizer satellite library compiled alongside this assembly.
///
/// Built by `Directory.Build.targets`:
///   • CUDA: $(TargetName)_CUDA.dll  — PE on Windows, ELF on Linux (extension
///     kept identical because <see cref="HybRunner.Cuda(string)"/> uses dlopen
///     on Linux which ignores the extension; matches the Windows convention).
///   • OMP : lib$(TargetName)_OMP.so on Linux, $(TargetName)_OMP.dll on Windows.
///
/// The OMP pattern is selected per-OS so we never try to LoadLibrary a Linux
/// ELF on Windows (yields error 193 ERROR_BAD_EXE_FORMAT) or vice versa.
/// Patterned after /mnt/d/hybridizer-basic-samples/src/0.Utils/Utilities/SatelliteLoader.cs.
/// </summary>
public static class SatelliteLoader
{
    private static string OmpPattern => OperatingSystem.IsWindows() ? "*_OMP.dll" : "lib*_OMP.so";

    public static HybRunner LoadCuda()
    {
        string path = FindSatellite("*_CUDA.dll")
            ?? throw new FileNotFoundException(
                $"No *_CUDA.dll found next to {ExecutingDirectory}. Build with <CompileCUDA>enable</CompileCUDA>.");
        return HybRunner.Cuda(path);
    }

    public static HybRunner LoadOmp()
    {
        string path = FindSatellite(OmpPattern)
            ?? throw new FileNotFoundException(
                $"No {OmpPattern} found next to {ExecutingDirectory}. Build with <CompileOMP>enable</CompileOMP>.");
        return HybRunner.OMP(path);
    }

    public static bool CudaAvailable() => FindSatellite("*_CUDA.dll") != null;
    public static bool OmpAvailable() => FindSatellite(OmpPattern) != null;

    /// <summary>Absolute path to the CUDA satellite, or null if not built.</summary>
    public static string? CudaSatellitePath() => FindSatellite("*_CUDA.dll");

    private static string? FindSatellite(string pattern)
        => Directory.GetFiles(ExecutingDirectory, pattern).FirstOrDefault();

    private static string ExecutingDirectory
        => new FileInfo(Assembly.GetExecutingAssembly().Location).Directory!.FullName;
}
