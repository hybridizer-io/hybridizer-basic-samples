# BlackScholes — GPU-Accelerated European Option Pricing

This sample demonstrates a **GPU-accelerated Black-Scholes pricer** using Hybridizer, computing call and put prices for a large batch of European options and validating the GPU results against a CPU-parallel reference implementation.

The program prices **`1,048,576 × CPU core count`** options in parallel on both **CPU** (`Parallel.For`) and **GPU** (Hybridizer-generated CUDA kernel), then reports the numerical error between the two.

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
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`BlackScholes`) and the `[Kernel]` device function (`CND`)
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required — option data (spot price, strike, maturity) is generated randomly at each run.

## Output

The program produces:

1. **Numerical validation** — `Linf` (max), `L2`, and `L1` error between CPU and GPU results, for both call and put prices
2. **Total execution time** — wall-clock time for the full run (data generation + GPU loop + CPU loop + validation)

### Example Output

\`\`\`
CALL ERRORS : Linf : 0.0001220703125, L2 : 1.1234567890123457E-07, L1: 3.2109876543210987E-08
PUT ERRORS  : Linf : 0.0001068115234375, L2 : 1.0456789012345678E-07, L1: 2.9876543210987654E-08

Total time for this program : 842 ms
\`\`\`

Errors on the order of `1e-4` or smaller are expected — they come purely from floating-point rounding differences between the CPU and GPU execution paths, not from a logic mismatch.

## How It Works

For each option, the pricer computes:

1. `d1` and `d2` from the Black-Scholes formula, using spot price, strike, time to maturity, risk-free rate (`RISKFREE = 0.02`), and volatility (`VOLATILITY = 0.30`)
2. The cumulative normal distribution `N(d1)` and `N(d2)` via a polynomial approximation (`CND`, Hastings' method)
3. Call and put prices from the closed-form Black-Scholes formula

| Factor | Explanation |
|---|---|
| **Massive parallelism** | Each of the ~1M+ options is priced by an independent GPU thread |
| **Embarrassingly parallel workload** | No dependency between options — ideal fit for GPU |
| **Minimal memory transfers** | Inputs uploaded once per iteration, results downloaded once |
| **Repeated GPU calls** | `NUM_ITERATIONS = 20` amortizes kernel launch and first-call JIT overhead for a stable timing measurement |

### Two execution paths

| Method | Description |
|---|---|
| **CPU parallel** | `Parallel.For` distributing options across CPU cores, calling `BlackScholes` per option |
| **GPU (Hybridizer)** | `[EntryPoint] BlackScholes` — same C# method, wrapped via `HybRunner.Wrap(...)` and dispatched across CUDA threads (`blockDim.x * blockIdx.x + threadIdx.x`, strided by `blockDim.x * gridDim.x`) |

### Hybridizer: C# philosophy on the GPU

The `BlackScholes` kernel calls `CND`, a separate `[Kernel]` device function, exactly like it would call a helper method in plain C#. There is no manual CUDA memory management, no pointer arithmetic, no explicit thread synchronization — just standard C# with a few attributes (`[EntryPoint]`, `[Kernel]`, `[IntrinsicFunction]`).

The `[IntrinsicFunction]`-decorated helpers (`fabsf`, `Expf`, `Sqrtf`, `Logf`) map directly onto CUDA's native math intrinsics (`fabsf`, `expf`, `sqrtf`, `logf`), so the generated kernel uses hardware math units instead of a naive re-implementation.

This demonstrates Hybridizer's core value proposition: **you don't have to leave C# or hand-write CUDA to get GPU-accelerated numerical code**. Write clean, well-structured C# with proper method decomposition, add the right attributes, and Hybridizer handles the CUDA generation and dispatch.