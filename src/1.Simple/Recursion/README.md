# Recursion — GPU-Accelerated Recursive Factorial (Device-Side Recursion)

This sample demonstrates that **Hybridizer doesn't require flattening recursive algorithms into loops before running them on the GPU**: a simple recursive factorial (`Fact`) is marked as a `[Kernel]` device function and compiled as-is — each GPU thread evaluates its own recursive call chain, exactly like a CPU thread would.

The program computes `Fact(b[i])` for **33,554,432** (`1024 × 1024 × 32`) input values on the GPU, then re-checks every single result against the same recursive method called directly on the CPU.

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
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`Run`) and the `[Kernel]` recursive device function (`Fact`), unchanged from its CPU form
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required — the input array is generated in-code (`b[i] = i % 11`, so every factorial computed is `Fact(0)` through `Fact(10)`, repeated across all 33M elements).

## Output

The program produces:

- **`OK`** if every one of the 33,554,432 GPU results matches the CPU-computed reference
- An error line and exit code `6` at the **first** mismatch found, of the form:
  \`\`\`
  Error at <index> : <gpu_result> != <cpu_result>
  \`\`\`

### Example Output

\`\`\`
OK
\`\`\`

## How It Works

### Recursive kernel

`Fact` is a plain recursive factorial, written exactly as it would be in ordinary C#:

\`\`\`csharp
[Kernel]
public static int Fact(int N)
{
    if (N <= 1) return 1;
    return N * Fact(N - 1);
}
\`\`\`

The only constraint Hybridizer imposes on recursive `[Kernel]` functions is that **stack allocations are forbidden** inside them (no `new StackArray` or `stackalloc`) — recursion itself is fully supported.

### Explicit work distribution

`Run` calls `Fact` from inside a manual **grid-stride loop** rather than a `Parallel.For` — as the code comments note, calling a recursive function from `Parallel.For` isn't supported yet, so work distribution has to be written explicitly:

\`\`\`
for (i = threadIdx.x + blockDim.x * blockIdx.x; i < N; i += blockDim.x * gridDim.x)
    a[i] = Fact(b[i]);
\`\`\`

This is also the sample where the GPU launch configuration is left at its **default**: `HybRunner.Load()` is called without `.SetDistrib(...)`, so Hybridizer falls back to `SetDistrib(multiProcessorCount * 16, 128)` automatically.

### Validation

After the GPU run (with explicit error checking via `cuda.GetLastError()` and `cuda.DeviceSynchronize()`), the program loops over all 33M elements on the CPU, calling the **exact same `Fact` method**, and compares it against the GPU-computed value — stopping immediately at the first mismatch, if any.

| Factor | Explanation |
|---|---|
| **Per-thread call stack** | Each GPU thread gets its own CUDA call stack; recursion depth here is small and bounded (`N ≤ 10`), well within the default stack size |
| **Default launch config** | Demonstrates that a working kernel doesn't require manually tuning `SetDistrib` — Hybridizer's default grid/block sizing is enough here |
| **Full-result validation** | Every single element is checked against the CPU reference, not just a sample, since a stack-related recursion bug could show up on only a few threads |

### Hybridizer: C# philosophy on the GPU

This is one of the clearest demonstrations of Hybridizer's core value proposition: **you don't have to restructure your algorithm to fit the GPU's execution model**. A recursive method that "just works" on CPU — `Fact`, unchanged — is handed to Hybridizer with a single `[Kernel]` attribute, and it compiles down to genuine device-side recursive CUDA code. The only adjustment needed anywhere in this sample is writing the work-distribution loop explicitly, since `Parallel.For` doesn't yet support calling into recursive functions.