# Newton — GPU-Accelerated Newton Fractal (Cube Roots of Unity)

This sample demonstrates a **GPU-accelerated Newton fractal renderer** using Hybridizer: for every pixel of a `4096×4096` image, it runs Newton's method on the complex polynomial `f(z) = z³ - 1` and colors the pixel according to which of the 3 cube roots of unity it converges to, and how fast.

The program renders the fractal **4 times** on CPU and 4 times on GPU, reports the average time per image for each, and saves the final GPU-rendered image to `newton.png`.

## Prerequisites

- .NET SDK
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version matching your Hybridizer install)
- Hybridizer runtime (`Hybridizer.Runtime.CUDAImports`, `Hybridizer.Basic.Utilities` NuGet packages)
- `SixLabors.ImageSharp` NuGet package (PNG encoding)
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
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint("run")]` kernel (`Run`) and its `[Kernel]` device functions (`IterCount`, `RootFind`)
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required — the image resolution (`N = 4096`), viewing window (`fromX`, `fromY`, `size`), and iteration limit (`maxiter = 4096`) are all compile-time constants.

## Output

The program produces:

1. **CPU time per image** — average over 4 runs of the CPU (`Parallel2D.For`) implementation
2. **GPU time per image** — average over 4 runs of the Hybridizer/CUDA implementation
3. **`newton.png`** — the rendered fractal, colored by root and iteration count, written to the working directory

### Example Output

\`\`\`
C# time per image : 214 ms
CUDA time per image : 6 ms
\`\`\`

## How It Works

### Newton's method on `z³ - 1`

For each pixel `(i, j)`, mapped to a complex number `z = x + iy` in the window `[fromX, fromX + size] × [fromY, fromY + size]`, `IterCount` repeatedly applies the Newton update:

\`\`\`
z ← z - f(z) / f'(z)      where f(z) = z³ - 1
\`\`\`

expanded into real-valued arithmetic on `x` and `y` (no complex number type — the real and imaginary parts are tracked and combined by hand through the powers `xx`, `yy`, `xxy`, `xyy`, etc.), until either `maxiter` is reached or the point is close enough (`tol = 1e-7`) to one of the polynomial's 3 roots.

### Root classification and coloring

`RootFind` checks the current `(x, y)` against the three cube roots of unity:

| Root | Value | Color |
|---|---|---|
| 1 | `(1, 0)` | Red |
| 2 | `(-0.5, +√3/2)` | Blue |
| 3 | `(-0.5, -√3/2)` | Green |
| none found | — | Black |

The brightness of each pixel (`ComputeLight`) is proportional to the iteration count it took to converge, producing the characteristic banded Newton-fractal look — pixels near a root converge fast (bright), pixels near the basin boundaries take many more iterations (darker, until clamped).

| Factor | Explanation |
|---|---|
| **Per-pixel independence** | Each of the `4096 × 4096` pixels runs its own independent Newton iteration — an ideal one-thread-per-pixel GPU workload |
| **`Parallel2D.For`** | Both the CPU and GPU implementations share the exact same 2D loop body — only the execution engine differs |
| **GPU-resident memory** | `ResidentArrayGeneric<int2>` keeps the result buffer on the GPU across iterations, avoiding a host/device copy on every one of the `redo` runs; `RefreshHost()` is called once at the end, right before saving the image |
| **Repeated runs** | `redo = 4` amortizes first-call overhead (kernel load, JIT, etc.) for a more representative average timing |

### Two execution paths

| Method | Description |
|---|---|
| **CPU** | `Parallel2D.For(0, N, 0, N, ...)` — same kernel body, executed via .NET's parallel loop over rows and columns |
| **GPU (Hybridizer)** | `[EntryPoint("run")] Run` — the identical `Parallel2D.For` body, wrapped via `HybRunner.Wrap(...)` and dispatched across a `32×32` block grid of `16×16` threads (`SetDistrib(32, 32, 16, 16, 1, 0)`) |

### Hybridizer: C# philosophy on the GPU

The CPU and GPU paths call the **exact same `Run` method** — `ComputeImage` simply chooses whether to call it directly or through the Hybridizer `wrapper`. There is no separate hand-written CUDA kernel to keep in sync with the CPU version: one `Parallel2D.For` loop, decorated with `[EntryPoint("run")]`, serves as both the reference implementation and the GPU kernel.

This demonstrates Hybridizer's core value proposition: **the same idiomatic C# parallel loop can be your CPU fallback and your GPU kernel**, with correctness and iteration logic defined once, and only the `[Kernel]`/`[EntryPoint]` attributes and a `HybRunner.Wrap(...)` call needed to unlock GPU execution.