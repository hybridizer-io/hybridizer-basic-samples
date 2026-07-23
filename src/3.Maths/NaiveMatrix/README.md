# NaiveMatrix — GPU-Accelerated Dense Matrix Multiplication (Naive Port)

This sample demonstrates a **naive GPU port** of dense matrix multiplication using Hybridizer: the same straightforward triple-nested-loop algorithm used on CPU, ported to the GPU with a 2D grid-stride loop and **no shared memory, no tiling, and no manual pointer arithmetic**.

The program multiplies two matrices of configurable size (`512×512` by default), running the computation **10 times on the GPU** and **10 times on the CPU**, back to back.

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
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`ComputeRowsOfProduct`)
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

This sample does **not** print timing, speedup, or a CPU/GPU error comparison — it simply runs both the GPU and CPU multiplication `redo = 10` times each and prints:

\`\`\`
Execution Naive matrix mul with sizes (512, 512) x (512, 512)
DONE
\`\`\`

It's meant to be profiled externally (e.g. with `nvprof`/Nsight, or a stopwatch added around each `#region`), rather than to report numbers on its own.

## How It Works

`ComputeRowsOfProduct` is a direct, unoptimized port of the standard triple-nested-loop matrix product:

\`\`\`
result[i, j] = Σ  matrixA[i, k] * matrixB[k, j]   for k in [0, commonSize)
\`\`\`

Each thread is responsible for one or more `(i, j)` output cells, found via a **2D grid-stride loop**:

- the outer loop over rows `i` starts at `threadIdx.y + blockIdx.y * blockDim.y` and strides by `blockDim.y * gridDim.y`
- the inner loop over columns `j` starts at `threadIdx.x + blockIdx.x * blockDim.x` and strides by `blockDim.x * gridDim.x`

This lets a fixed-size grid (`SetDistrib(4, 5, 8, 32, 32, 0)`) cover a result matrix of any size, with threads simply looping around for more work if there's more than one grid's worth of cells.

Every element of `matrixA` and `matrixB` used in the inner `k` loop is re-read from global memory for every output cell — there is no reuse of loaded values across threads, unlike the shared-memory tiled version of this same problem.

| Factor | Explanation |
|---|---|
| **Simplicity over throughput** | No shared memory, no tiling, no `__syncthreads` — the whole kernel is a direct translation of the CPU loop |
| **2D grid-stride loop** | Handles matrices larger than a single kernel launch's thread grid without relaunching |
| **Redundant global memory reads** | Every thread re-fetches the same rows/columns already fetched by its neighbors — this is the main reason the tiled/shared-memory version outperforms this one |

### Two execution paths

| Method | Description |
|---|---|
| **CPU parallel** | `Parallel.For` over rows, each row computed by a call to the exact same `ComputeRowsOfProduct` method used on GPU |
| **GPU (Hybridizer)** | `[EntryPoint] ComputeRowsOfProduct` — same method, wrapped via `HybRunner.Wrap(...)` and dispatched across the configured 2D thread grid |

### Hybridizer: C# philosophy on the GPU

`ComputeRowsOfProduct` is called **unchanged** on both paths: on GPU, `blockIdx`/`threadIdx`/`blockDim`/`gridDim` resolve to real CUDA thread coordinates; on CPU, the same method is simply called with `lineFrom`/`lineTo` set by `Parallel.For`, so the row/column indexing collapses to plain sequential loops. There is no separate hand-written CUDA kernel and no separate CPU reference implementation to keep in sync — one method serves both.

This is the natural counterpart to a shared-memory tiled kernel: it shows what GPU code looks like **before** hand-optimization — a useful baseline to compare against once shared memory, tiling, or other CUDA-specific techniques are introduced.
