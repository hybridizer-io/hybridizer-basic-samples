# Llama.CSharp

This project is a pure-C# LLM inference engine. It requires a GGUF model file to run.

The hot-path kernels are written in C# and transcoded to CUDA by [Hybridizer](https://www.altimesh.com/) at build time, so the same source runs on the .NET thread pool or on the GPU depending on the selected backend.

## Prerequisites

- **.NET 8 SDK** ([download](https://dotnet.microsoft.com/download/dotnet/8.0))
- **Hybridizer dotnet tool** (installed globally from nuget.org — see the [repo root README](../../../README.md)):
  ```bash
  dotnet tool install -g Hybridizer
  ```
- **CUDA Toolkit** with `nvcc` on the path. Any 13.x is fine; the build auto-detects the version from `nvcc -V` and the GPU compute capability from `nvidia-smi`.
- **C++ compiler**:
  - Linux / WSL: `g++` (invoked by `nvcc`).
  - Windows: Visual Studio 2022 with the "Desktop development with C++" workload (`cl.exe` is auto-located via `vswhere` → `vcvarsall.bat`).

## Building

```bash
dotnet build -c Release
```

The build:
1. Transcodes `[EntryPoint]`-annotated methods into `.cu` files under `generated-sources/`.
2. Compiles the CUDA satellite (`LlamaCsharp_CUDA.dll`) via `nvcc`.
3. Produces the `llama-csharp` executable along with the satellite DLL in `bin/Release/net8.0/`.

### CUDA satellite — extra translation units

The CUDA backend needs a handful of `extern "C"` host wrappers around `cudaStream*` / `cudaGraph*` APIs that `Hybridizer.Runtime.CUDAImports` does not expose (used by `Utils/GraphInvoke.cs` to capture the per-token forward pass as a CUDA Graph). Those wrappers live in `intrinsics.cuh`, guarded by `#ifndef __CUDACC_RTC__` so NVRTC's JIT path ignores them.

The auto-generated `hybridizer.wrappers.cu` does not include `intrinsics.cuh`, so a tiny dedicated TU pulls the header into the static nvcc build:

- `host_wrappers.cu` — single-line `#include "intrinsics.cuh"`.
- Declared in `LlamaCsharp.csproj` via `<CudaExtraSources Include="host_wrappers.cu" />`.
- The repo-shared `CompileCUDA` target in `src/Directory.Build.targets` splices `@(CudaExtraSources)` into the nvcc command line — empty by default for other samples.

Without this wiring, building succeeds but the resulting DLL is missing `llama_stream_create` / `llama_graph_*` exports, and running the CUDA backend throws `EntryPointNotFoundException` at `GraphInvoke.EnsureResolved()`.

## Tests

```bash
dotnet test -c Release
```

The tests cover the tokenizer, the operators (deterministic forward pass vs. reference), and CUDA kernel conformance.

## Downloading the model

The recommended starter model is **TinyLlama-1.1B-Chat-v1.0** with **Q8_0** quantization (around 1.17 GB). Place the file at the root of the project, next to `LlamaCsharp.csproj`.

```bash
# Via wget
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf

# Or via curl
curl -L -O https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

You can also download it from the browser: [direct link](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf).

Any GGUF file using the LLaMA architecture with `F32`, `F16`, `Q8_0`, or `Q4_0` quantization should load.

## Running

```bash
dotnet run -c Release -- tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

`--` separates `dotnet run`'s own flags from program arguments. Add `--no-build` after a successful build to skip the build step on repeated runs.

### Arguments

| Position / flag      | Type   | Default              | Description |
|----------------------|--------|----------------------|-------------|
| *(positional 1)*     | path   | *(required)*         | Path to a GGUF model file. |
| *(positional 2)*     | string | `"Once upon a time"` | Prompt. Anything that doesn't start with `--`. |
| `--max-tokens N`     | int    | `1280`               | Maximum tokens to generate (stops earlier on EOS or context limit). |
| `--temperature T`    | float  | `0.7`                | Sampling temperature. `0` = greedy (argmax). |
| `--top-p P`          | float  | `0.9`                | Top-p (nucleus) sampling. `1.0` disables the filter. |
| `--backend NAME`     | enum   | `managed`            | Q8_0 matvec backend: `managed` or `cuda`. |

### Backends

- `managed` — pure C# / .NET, runs on the CPU thread pool. The reference path; no native dependencies beyond the runtime.
- `cuda` — Hybridizer-transcoded kernels compiled to a CUDA satellite (`LlamaCsharp_CUDA.dll`). Requires a working CUDA install.

### GPU-greedy fast path (CUDA Graphs)

When all three of `--backend cuda`, `--temperature 0`, and `--top-p 1` are set, the runtime takes the deferred-print path:

- the per-token forward pass is captured **once** with `cudaStreamBeginCapture` and replayed with `cudaGraphLaunch` thereafter (~378 kernel launches collapse to one);
- argmax runs on device, so there is **no** per-token D→H copy of the 125 KB logits buffer;
- generated tokens accumulate in a device ring and are drained every 200 ms for printing / EOS detection (the generator may overshoot EOS by at most one drain interval).

Stochastic CUDA and the Managed backend stay on the standard host-side argmax + per-token D→H path.

### Environment variables

- `LLAMA_DISABLE_GRAPH=1` — keep the non-default stream but skip graph capture/replay. Useful for A/B diagnostics against the per-call launch baseline.
- `LLAMA_KERNEL_PROFILE=1` — enable per-kernel timing on the CUDA backend. Forces `cudaDeviceSynchronize` after every launch (significantly slower) and prints a ranked profile at exit.

### Examples

```bash
# Default (managed backend, stochastic decoding)
dotnet run -c Release -- tinyllama-1.1b-chat-v1.0.Q8_0.gguf

# Custom prompt, longer generation
dotnet run -c Release -- tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  "Once upon a time, in a faraway land" --max-tokens 256

# Deterministic CUDA fast path (CUDA Graphs)
dotnet run -c Release -- tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  "Once upon a time" --backend cuda --temperature 0 --top-p 1

# Low-temperature creative decoding on the managed backend
dotnet run -c Release -- tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  "The meaning of life" --temperature 0.3 --top-p 0.95

# CUDA, kernel-level profile dump at exit
LLAMA_KERNEL_PROFILE=1 dotnet run -c Release -- tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --backend cuda --temperature 0 --top-p 1
```

On Windows PowerShell, set env vars with `$env:LLAMA_KERNEL_PROFILE = "1"` before the `dotnet run` invocation rather than the inline `VAR=value cmd` syntax.
