# Reduction — GPU-Accelerated Parallel Sum Reduction

This sample demonstrates a **classic CUDA parallel reduction** using Hybridizer: summing a large array of integers on the GPU with the standard grid-stride accumulation + shared-memory tree reduction + atomic combine pattern, and checking the result against a CPU-computed reference (`Aggregate`).

The program sums an array of **33,554,432** (`1024 × 1024 × 32`) integers, each randomly `1` (with 20% probability) or `0`, and prints both the GPU-computed sum and the expected sum for comparison.

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
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`ReduceAdd`), including the shared-memory allocation and `__syncthreads` barriers
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required — the input array is generated in-code with a random 20% chance of each element being `1` (otherwise `0`), so the expected sum is roughly `N × 0.2 ≈ 6,710,886`, though it varies run to run.

## Output

The program produces:

1. **`sum =`** — the value computed on the GPU
2. **`expected =`** — the same sum computed on the CPU via LINQ's `Aggregate`, as a reference

### Example Output

\`\`\`
sum =      6710498
expected = 6710498
\`\`\`

## How It Works

`ReduceAdd` combines three standard CUDA reduction techniques in a single kernel:

1. **Grid-stride accumulation** — each thread walks the array with a stride of `blockDim.x * gridDim.x`, accumulating a private partial sum (`tmp`) across however many elements fall to it. This means the kernel works regardless of how `N` compares to the total thread count.
2. **Shared-memory tree reduction** — each thread's partial sum is stored into a shared-memory array (`cache`, sized `blockDim.x`). The block then repeatedly halves the active range (`i = blockDim.x / 2`, then `i >>= 1` each round), with each active thread adding its "far" neighbor's value into its own slot, guarded by `__syncthreads()` between rounds. After `log2(blockDim.x)` rounds, `cache[0]` holds the full sum for that block.
3. **Atomic combine across blocks** — only `threadIdx.x == 0` in each block adds its block's total into the single global `result[0]`, via `Interlocked.Add`, so blocks never race on the same memory location.

| Factor | Explanation |
|---|---|
| **Two-level reduction** | Per-thread partial sums first, then a per-block tree reduction, then a cross-block atomic combine — the standard way to reduce a huge array with a bounded number of blocks/threads |
| **Shared memory sized at launch** | `SetDistrib(..., BLOCK_DIM * sizeof(int))` reserves exactly `blockDim.x` ints of shared memory per block, matching `cache`'s size |
| **Single atomic per block, not per element** | Only one `Interlocked.Add` call per block (not per thread) keeps atomic contention low even with millions of elements |
| **Grid-stride loop** | Decouples the number of GPU threads launched (`16 × multiProcessorCount` blocks of `256` threads) from the size of the input array |

### Hybridizer: C# philosophy on the GPU

This kernel is, like the shared-memory matrix multiplication and N-body samples, **hand-written CUDA logic expressed directly in C#**: explicit shared memory (`SharedMemoryAllocator<int>`), `__syncthreads()` barriers, bit-shift-based tree reduction, and an atomic add (`Interlocked.Add`, mapped straight onto CUDA's atomic operations) are all written as ordinary C# in the kernel body, with `[EntryPoint]` being the only attribute needed to turn it into a GPU-executable method.

This shows that when an algorithm genuinely needs GPU-specific tricks — like a tree reduction relying on shared memory and synchronization — Hybridizer doesn't get in the way: the same constructs a CUDA C++ programmer would reach for (shared memory, sync barriers, atomics) are available as first-class C# from inside a `[EntryPoint]` method.