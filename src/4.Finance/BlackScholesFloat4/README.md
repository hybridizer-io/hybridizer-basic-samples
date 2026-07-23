# BlackScholes4 — GPU-Accelerated Option Pricing with float4 Vectorization

This sample is a **vectorized variant** of the Black-Scholes GPU pricer: instead of processing one option per array element, it packs **4 options per `float4`**, reducing memory transactions and letting each GPU thread do four independent option-pricing computations per iteration.

The program prices **`1,048,576 × CPU core count`** options (grouped into `float4` batches) in parallel on both **CPU** (`Parallel.For`) and **GPU** (Hybridizer-generated CUDA kernel), then reports the numerical error between the two.

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
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`BlackScholes`) and the `[Kernel]` device function (`CND`), both operating on `float4`
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required — option data (spot price, strike, maturity) is generated randomly as `float4` batches at each run.

## Output

The program produces:

1. **Numerical validation** — `Linf` (max), `L2`, and `L1` error between CPU and GPU results, for both call and put prices (computed component-wise across all four lanes of each `float4`)
2. **Total execution time** — wall-clock time for the full run (data generation + GPU loop + CPU loop + validation)

### Example Output

\`\`\`
CALL ERRORS : Linf : 0.0001220703125, L2 : 1.1234567890123457E-07, L1: 3.2109876543210987E-08
PUT ERRORS  : Linf : 0.0001068115234375, L2 : 1.0456789012345678E-07, L1: 2.9876543210987654E-08

Total time for this program : 731 ms
\`\`\`

Errors on the order of `1e-4` or smaller are expected — they come purely from floating-point rounding differences between the CPU and GPU execution paths, not from a logic mismatch.

## How It Works

For each `float4` (4 options at once), the pricer computes, lane by lane (`.x`, `.y`, `.z`, `.w`):

1. `d1` and `d2` from the Black-Scholes formula, using spot price, strike, time to maturity, risk-free rate (`RISKFREE = 0.02`), and volatility (`VOLATILITY = 0.30`)
2. The cumulative normal distribution `N(d1)` and `N(d2)` via a polynomial approximation (`CND`, Hastings' method), vectorized over `float4`
3. Call and put prices from the closed-form Black-Scholes formula

| Factor | Explanation |
|---|---|
| **Massive parallelism** | Each GPU thread processes one `float4` (4 options) per loop iteration |
| **Vectorized memory access** | `float4` loads/stores move 16 bytes per transaction instead of 4, improving memory throughput |
| **Fast math intrinsics** | `__expf`, `__logf`, `__fdividef`, `rsqrtf` map to CUDA's fast (lower-precision, higher-throughput) approximate intrinsics instead of the standard ones |
| **Repeated GPU calls** | `NUM_ITERATIONS = 20` amortizes kernel launch and first-call JIT overhead for a stable timing measurement |

### Two execution paths

| Method | Description |
|---|---|
| **CPU parallel** | `Parallel.For` distributing `float4` batches across CPU cores, calling `BlackScholes` per batch |
| **GPU (Hybridizer)** | `[EntryPoint] BlackScholes` — same C# method, wrapped via `HybRunner.Wrap(...)` and dispatched across CUDA threads (`blockDim.x * blockIdx.x + threadIdx.x`, strided by `blockDim.x * gridDim.x`) |

### Hybridizer: C# philosophy on the GPU

The `BlackScholes` kernel calls `CND`, a separate `[Kernel]` device function operating on `float4`, exactly like it would call a helper method in plain C#. There is no manual CUDA memory management, no pointer arithmetic, no explicit thread synchronization — just standard C# with a few attributes (`[EntryPoint]`, `[Kernel]`, `[IntrinsicFunction]`) and the `float4` vector type from `Hybridizer.Basic.Utilities`.

The `[IntrinsicFunction]`-decorated helpers map directly onto CUDA's fast intrinsics:

| C# method | CUDA intrinsic | Note |
|---|---|---|
| `fabsf` | `fabsf` | standard absolute value |
| `Expf` | `__expf` | fast (approximate) exponential |
| `Sqrtf` | `__sqrtf` | fast (approximate) square root |
| `rsqrtf` | `rsqrtf` | reciprocal square root |
| `Logf` | `__logf` | fast (approximate) logarithm |
| `__fdividef` | `__fdividef` | fast (approximate) division |

Compared to the scalar version, `sqrtT` here is computed as `1 / rsqrtf(x)` via `__fdividef` rather than a direct `sqrtf` call — a common CUDA trick to route the computation through the fast reciprocal-square-root unit.

This demonstrates Hybridizer's core value proposition: **you don't have to leave C# or hand-write CUDA to get GPU-accelerated, vectorized numerical code**. Write clean, well-structured C# using standard vector types and math helpers, add the right attributes, and Hybridizer handles the CUDA generation and dispatch — down to the choice of fast math intrinsics.