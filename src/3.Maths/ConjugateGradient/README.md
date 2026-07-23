# ConjugateGradient — GPU-Accelerated Iterative Linear Solver (CSR, 1D Laplacian)

This sample demonstrates a **full iterative solver running almost entirely on the GPU** using Hybridizer: the **Conjugate Gradient** method, solving `A·X = B` for a sparse **1D Laplacian** matrix in CSR format, with every vector operation (SpMV, dot product, SAXPY, copy) offloaded as its own small GPU kernel.

The program builds a `10,000 × 10,000` 1D Laplacian, solves `A·X = B` for a right-hand side of all `1.0`, and prints the residual norm every 10 iterations until convergence (or up to `1,000,000` iterations).

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
2. Runs Hybridizer to generate CUDA C++ from all five `[EntryPoint]` kernels (`ScalarProd`, `Copy`, `Saxpy`, `Fmsub`, `Multiply`)
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required — matrix size (`10,000`), max iterations (`1,000,000`), and convergence tolerance (`eps = 1e-8`) are hardcoded. As the code comments note, this problem size converges **very slowly** (or may not visibly converge within a reasonable time) without a preconditioner — this sample favors showing the algorithm's GPU structure over solver performance.

## Output

The program prints the residual norm (`√⟨R|R⟩`) every 10 iterations, until it drops below `eps`:

\`\`\`
99.9812
87.3541
...
0.0000000089
\`\`\`

The final solution `X` is refreshed back to host memory (`X.RefreshHost()`) at the end but not printed.

## How It Works

### GPU-resident memory

Unlike the other samples, vectors here are `FloatResidentArray`/`IntResidentArray` — buffers that **stay on the GPU across kernel calls**, only synchronized with the host explicitly via `RefreshDevice()`/`RefreshHost()`. Since the entire CG loop runs kernel after kernel without ever needing the intermediate vectors on the CPU, this avoids a host/device copy on every one of the potentially hundreds of thousands of iterations — only the inputs (`A`, `X`, `B`) are pushed to the GPU once at the start, and the result is pulled back once at the end.

### The five GPU kernels

Conjugate Gradient is built from a handful of basic linear-algebra operations, each implemented as its own kernel:

| Kernel | Role |
|---|---|
| `Multiply` | Sparse matrix-vector product `A·v` (CSR row-wise dot product), used to compute `AP = A·P` each iteration |
| `Fmsub` | `res = A - m·v` — computes the initial residual `R = B - A·X` |
| `Saxpy` | `res = x + alpha·y` — updates `X`, `R`, and `P` each iteration |
| `Copy` | Simple vector copy, used to initialize `P = R` |
| `ScalarProd` | Dot product `⟨r1\|r2⟩`, using the same shared-memory tree-reduction + atomic-combine pattern as the standalone reduction sample |

`Multiply`, `Fmsub`, `Saxpy`, and `Copy` are written as plain `Parallel.For` loops — Hybridizer maps each iteration to a GPU thread automatically. `ScalarProd` is the one hand-optimized kernel: shared memory, a tree reduction with `__syncthreads()`, and a single atomic add per block (`AtomicExpr.apply`) to combine each block's partial sum into the final result.

### Conjugate Gradient algorithm

The host-side loop (`ConjugateGradient`) implements the textbook CG iteration, with every vector operation dispatched to the GPU:

1. `R = B - A·X`, `P = R` (initialization)
2. Each iteration: `AP = A·P`; `α = ⟨R|R⟩ / ⟨P|AP⟩`; `X += α·P`; `R -= α·AP`; check `⟨R|R⟩ < eps²` for convergence; `β = ⟨R_new|R_new⟩ / ⟨R_old|R_old⟩`; `P = R + β·P`

Only a handful of scalars (`r`, `alpha`, `rr`, `beta`) ever touch the CPU each iteration — everything vector-sized stays resident on the GPU.

| Factor | Explanation |
|---|---|
| **Minimizing host/device traffic** | `FloatResidentArray`/`IntResidentArray` + explicit `RefreshDevice`/`RefreshHost` calls mean the GPU is only synchronized with the host twice total, not once per iteration |
| **Kernel composition** | A non-trivial iterative algorithm is built by chaining several small, reusable GPU kernels, called from an ordinary C# host loop — no single monolithic kernel |
| **Reused reduction pattern** | `ScalarProd` reuses the same shared-memory tree-reduction technique as the standalone `Reduction` sample, applied here to compute dot products instead of a plain sum |

### Hybridizer: C# philosophy on the GPU

This sample shows Hybridizer used to build a **complete numerical algorithm**, not just a single kernel: the CG loop itself is ordinary, sequential C# on the host, orchestrating GPU kernels the same way it would orchestrate calls into any other library — `wrapper.Multiply(...)`, `wrapper.Saxpy(...)`, etc. Most of those kernels are simple `Parallel.For` ports; one (`ScalarProd`) needs genuine CUDA-level optimization (shared memory, reduction, atomics), and both styles coexist naturally in the same C# codebase.