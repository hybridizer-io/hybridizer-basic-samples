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

## Troubleshooting
- `cuLinkAddData (222)` "unsupported toolchain" — your driver is older than the nvrtc that produced the PTX. Update the NVIDIA driver, or build with `-p:CUDAVersion=<a version your driver supports>` (and have that CUDA toolkit installed).
- `MSVC/cl.exe with traditional preprocessor` (CCCL header error on Windows) — already handled by the samples (they pass `/Zc:preprocessor` to `cl.exe`). If you adapt the build outside the samples, add `-Xcompiler /Zc:preprocessor` to your `nvcc` invocation.

## Documentation
Samples are explained in the [wiki](https://github.com/altimesh/hybridizer-basic-samples/wiki).

API documentation: [docs.hybridizer.io](https://docs.hybridizer.io/).
