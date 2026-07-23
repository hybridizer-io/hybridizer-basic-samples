# SharedMatrix — GPU-Accelerated Dense Matrix Multiplication (Shared Memory Tiling)

This sample demonstrates **GPU-accelerated dense matrix multiplication** using Hybridizer, with a **shared-memory tiled kernel** (the classic CUDA matmul optimization) compared against a CPU-parallel reference implementation.

The program multiplies two matrices of configurable size (`512×512` by default) **10 times** on the GPU and once on the CPU, then reports the average GPU time, the CPU time, and the resulting speedup.

## Prerequisites

- .NET SDK
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version matching your Hybridizer install)
- Hybridizer runtime (`Hybridizer.Runtime.CUDAImports`, `Hybridizer.Basic.Utilities` NuGet packages)
- Visual Studio 2022+ with C++ workload (Windows) or GCC (Linux)

## Build

Always build in **Release** mode for maximum performance — Debug mode disables compiler optimizations and gives misleading benchmark results.

\`\`\`bash
# Restore NuGet packages (first time only)
dotnet restore

# Build in Release mode
dotnet build --configuration Release
\`\`\`

The build pipeline:
1. Compiles the C# project
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`Multiply`), including the shared-memory tiling and `__syncthreads` barriers
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build -- [heightA widthA heightB widthB]
\`\`\`

> **Note:** The `--` separator is required before program arguments to distinguish them from `dotnet` options.

### Options

| Argument | Description | Default |
|---|---|---|
| `heightA` | Number of rows of matrix A | `512` |
| `widthA` | Number of columns of matrix A | `512` |
| `heightB` | Number of rows of matrix B (must equal `widthA`) | `512` |
| `widthB` | Number of columns of matrix B | `512` |

### Examples

**Run with default 512×512 matrices:**
\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

**Multiply a 1024×512 matrix by a 512×256 matrix:**
\`\`\`bash
dotnet run --configuration Release --no-build -- 1024 512 512 256
\`\`\`

If `widthA != heightB`, the program throws an `ArgumentException` — the matrices are not compatible for multiplication.

## Output

The program produces:

1. **GPU total and average time** — over `redo = 10` runs
2. **CPU time** — single run of the `Parallel.For` reference implementation
3. **Speedup ratio** — CPU time divided by average GPU time

### Example Output

\`\`\`
Execution Naive matrix mul with sizes (512, 512) x (512, 512)

GPU Computation, done 10 times : 184 ms
Average GPU time : 18.4

CPU Computation : 612 ms

The average GPU time is 33.26 times faster than the CPU time

DONE
\`\`\`

## How It Works

### Shared-memory tiling

The GPU kernel splits both matrices into square tiles of size `blockDim.x × blockDim.y` and processes one tile pair per iteration of the innermost loop:

1. Each thread cooperatively loads one element of `A` and one element of `B` for the current tile into **shared memory** (`cacheA`, `cacheB`) — memory shared by all threads of a block, much faster than global memory
2. `SyncThreads()` (`__syncthreads`) ensures every thread has finished loading before any thread starts reading the tile
3. Each thread accumulates its partial dot product (`Pvalue`) using the cached tile data
4. A second `SyncThreads()` ensures the tile isn't overwritten by the next iteration before every thread is done using it
5. Once all tiles along the shared dimension have been processed, the thread writes its final result to `result[i * size + j]`

| Factor | Explanation |
|---|---|
| **Data reuse** | Each tile loaded into shared memory is reused by every thread in the block, cutting down redundant global memory reads |
| **Grid-stride loops over blocks** | `by`/`bx` loop with a `gridDim.y`/`gridDim.x` stride, so a fixed-size grid (`SetDistrib(4, 5, 32, 32, ...)`) can cover matrices larger than one launch's worth of blocks |
| **Explicit shared memory allocation** | `SharedMemoryAllocator<float>` reserves per-block scratch space sized by `blockDim.x * blockDim.y` for each cache array |
| **Repeated GPU calls** | `redo = 10` amortizes kernel launch overhead for a stable average timing measurement |

### Two execution paths

| Method | Description |
|---|---|
| **CPU parallel** | `Parallel.For` over rows of `A`, each computing a full row of the result with a triple nested loop — no tiling, no shared memory |
| **GPU (Hybridizer)** | `[EntryPoint] Multiply` — shared-memory tiled kernel, explicit grid/block distribution via `SetDistrib`, manual thread synchronization |

### Hybridizer: C# philosophy on the GPU

Unlike the naive GPU ports in other samples, this kernel is a genuine **hand-optimized CUDA algorithm** written in C#: explicit shared memory allocation, tile indexing, and `__syncthreads` barriers are all expressed directly in the method body via `SharedMemoryAllocator<float>` and the `[IntrinsicFunction("__syncthreads")]`-decorated `SyncThreads()` helper.

This demonstrates the other side of Hybridizer's value proposition: when you *do* need low-level CUDA control — shared memory, explicit synchronization, custom grid/block distribution (`SetDistrib(4, 5, 32, 32, 1, sharedMemBytes)`) — you can express it in C# with the same attributes and intrinsics mechanism used for simpler kernels, without switching languages or leaving your .NET codebase.