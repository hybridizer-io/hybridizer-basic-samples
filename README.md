Hybridizer Essentials is a compiler targeting CUDA-enabled GPUs from .NET. Using parallelization patterns such as `Parallel.For`, or distributing parallel work by hand, you can benefit from the compute power of GPUs without entering the learning curve of CUDA — all within the dotnet environment.

### hybridizer-basic-samples
This repo illustrates a few samples for Hybridizer.

These samples may be used with Hybridizer Essentials. C# code can run with any version of Hybridizer. They illustrate features of the toolchain and are a good starting point for experimenting and developing software based on Hybridizer.

## Requirements
- A CUDA-enabled NVIDIA GPU and an up-to-date NVIDIA driver.
- A CUDA toolkit installed (any 13.x; later majors as they ship). `nvcc` must be on `PATH` (Linux) or reachable via `%CUDA_PATH%` (Windows — the CUDA installer sets this).
- On Windows, Visual Studio 2022 with the "Desktop development with C++" workload (for `cl.exe` 16.5+ used by `nvcc`).
- .NET 8 SDK.
- The Hybridizer dotnet tool, installed globally from nuget.org:

  ```
  dotnet tool install -g Hybridizer
  ```

The samples auto-detect your installed CUDA version from `nvcc -V`; you do not need to set any env var. If you keep several toolkits side by side and want to pin one, set the MSBuild property `CUDAVersion` (e.g. `dotnet build -p:CUDAVersion=13.1`) — the launcher will resolve nvrtc against that specific install.

## Run
```
git clone https://github.com/hybridizer-io/hybridizer-basic-samples.git
cd hybridizer-basic-samples/src/<chapter>/<sample>
dotnet run
```

## A build of Hybridizer in place of the packages (`HybridizerDir`)
The samples call the GPU API common to every backend (`gpu.*`, `HybRunner.Create`), so that the same sample runs on CUDA and on Apple GPUs. No published `Hybridizer.Runtime.CUDAImports` package has it yet (3.7.1 is the latest): build the runtime and the transcoder of [hybridizer-software-suite](https://github.com/hybridizer-io/hybridizer-software-suite) into one directory, and point the samples at it with `HybridizerDir` (an MSBuild property or an environment variable). It then replaces the package and the `hybridizer` dotnet tool, for CUDA as for Metal; without it, the build stops and says so.

```
SUITE=<hybridizer-software-suite checkout>
DIR=<a directory for the build>
dotnet build $SUITE/Hybridizer/Runtime/CUDAImports/src/Hybridizer.Runtime.CUDAImports.csproj -c Release -property:Platform=x64 -p:PlatformTarget=AnyCPU -p:VERSION=3.7.0.0 -o $DIR
dotnet build $SUITE/Hybridizer.Application/Hybridizer.Application.csproj -c Release -property:Platform=x64 -p:PlatformTarget=AnyCPU -p:MyConstants=HYBRIDIZER_MAIN -o $DIR
export HybridizerDir=$DIR
cd src/<chapter>/<sample> && dotnet run
```

## Apple GPUs (macOS, Metal)
On macOS the samples are built for the METAL flavor (`HybridizerFlavor`, see `src/Directory.Build.props`): after the build, `src/Hybridizer.Metal.targets` transcodes the sample for Metal and builds its satellite, `<Name>_METAL.dylib`, next to the program; `SatelliteLoader` loads it. The Metal Shading Language it embeds is compiled by the Metal runtime at the first launch of a kernel: no Metal toolchain is needed, only the Xcode command line tools (`clang++`) and, to build the runtime of the suite, the [metal-cpp](https://developer.apple.com/metal/cpp/) headers (`/opt/homebrew/include/metal_cpp`, or `METAL_CPP`). Apple GPUs have no double precision: `double` is computed in software, IEEE-754 to the bit, much slower than `float`.

CUDA only: the window of NBody (`dotnet run -- --check` runs its simulation without one), Mandelbulb (OpenGL interop and CUDA surfaces), WmmaGemm (tensor cores) and tiny-llama. Not yet on Metal: Malloc (`new` in a kernel), SimpleMetadataDecorator (arrays of objects, strings). What kernels write to the console goes through `os_log`, macOS 15 and later.

## Checking every sample
`scripts/run-samples.sh` builds every sample, runs it on the GPU and tells which ones compute what their C# computes (each sample compares its results with C# and exits with an error when they differ): `HybridizerDir=$DIR scripts/run-samples.sh [sample ...]`, results in `/tmp/hyb-samples-run/matrix.tsv` (`OUT`).

## Troubleshooting
- `cuLinkAddData (222)` "unsupported toolchain" — your driver is older than the nvrtc that produced the PTX. Update the NVIDIA driver, or build with `-p:CUDAVersion=<a version your driver supports>` (and have that CUDA toolkit installed).
- `MSVC/cl.exe with traditional preprocessor` (CCCL header error on Windows) — already handled by the samples (they pass `/Zc:preprocessor` to `cl.exe`). If you adapt the build outside the samples, add `-Xcompiler /Zc:preprocessor` to your `nvcc` invocation.

## Documentation
Samples are explained in the [wiki](https://github.com/altimesh/hybridizer-basic-samples/wiki).

API documentation: [docs.hybridizer.io](https://docs.hybridizer.io/).
