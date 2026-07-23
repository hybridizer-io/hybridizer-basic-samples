# MonteCarloHeatEquation — GPU-Accelerated Monte Carlo PDE Solver

This sample demonstrates a **Monte Carlo random-walk solver for the (steady-state) heat equation** using Hybridizer, and — unlike the other samples — wraps an **instance method on a plain object** (`MonteCarloHeatSolver.Solve()`) rather than a static method on the entry-point class, showing that Hybridizer's `[EntryPoint]`/`Wrap(...)` mechanism works on ordinary object-oriented C# as well.

The program solves the heat equation on a `128×128` grid using **512 Monte Carlo iterations per point**, then saves the resulting temperature field as a rainbow-colored heatmap, `result.png`.

> **Note:** the problem/geometry classes referenced here (`I2DProblem`, `SquareProblem<TWalker, TBoundary>`, `SimpleWalker`, `SimpleBoundaryCondition`, and the commented-out `TetrisProblem`/`TetrisBoundaryCondition`) aren't included in the files provided, so the sections below describe what's inferable from `MonteCarloHeatSolver.cs` and `Program.cs` — the exact walker/boundary logic lives in those other files.

## Prerequisites

- .NET SDK
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version matching your Hybridizer install)
- Hybridizer runtime (`Hybridizer.Runtime.CUDAImports`, `Hybridizer.Basic.Utilities` NuGet packages)
- `SixLabors.ImageSharp` NuGet package (PNG output)
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
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`MonteCarloHeatSolver.Solve`), including whatever walker/boundary logic `SquareProblem<SimpleWalker, SimpleBoundaryCondition>.Solve(ii, jj)` calls into
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required — grid size (`N = 128`) and Monte Carlo iteration count (`iterCount = 512`) are hardcoded. As the code comments note, this is fairly compute-intensive, so these constants may need adjusting depending on your GPU.

## Output

The program produces:

1. **`CUDA time :`** — wall-clock time for the full GPU solve
2. **`result.png`** — a `128×128` heatmap image of the solved temperature field, colored from black (coldest) through red, orange, yellow, green, blue, indigo, violet, to white (warmest)

### Example Output

\`\`\`
CUDA time : 1834 ms
\`\`\`

## How It Works

### Monte Carlo random-walk PDE solving

`MonteCarloHeatSolver.Solve()` loops (via `Parallel.For`, transpiled to a GPU thread grid) over every grid point of the problem (`_problem.MaxIndex()` points, mapped back to `(ii, jj)` coordinates via `_problem.Coordinates`), and calls `_problem.Solve(ii, jj)` for each one.

This is a classic setup for the **"walk on grid" Monte Carlo method**: for a steady-state heat/Laplace problem, the temperature at an interior point equals the *average* boundary temperature reached by a large number of random walks started at that point. Each of the `iterCount` walks per grid point takes random steps (governed by the `SimpleWalker` strategy) until it reaches the domain's boundary (checked by `SimpleBoundaryCondition`), and the estimated temperature is the average of the boundary values hit.

### Pluggable problem shape and walker strategy

`SquareProblem<TWalker, TBoundary>` is generic over the walker and boundary-condition types, so the same Monte Carlo driver (`MonteCarloHeatSolver`) can be reused for different domain shapes and stepping rules — the commented-out `TetrisProblem<SimpleWalker, TetrisBoundaryCondition>` line in `Main` shows an alternate domain shape (a Tetris-like boundary) plugged into the same solver unchanged.

### Heatmap coloring

`GetColor` maps a normalized temperature `[0, 1]` onto a fixed 8-color rainbow gradient (black → red → orange → yellow → green → blue → indigo → violet → white), linearly interpolating between the two nearest colors for smooth shading.

| Factor | Explanation |
|---|---|
| **Embarrassingly parallel** | Each grid point's Monte Carlo estimate is entirely independent of every other point's — ideal for one-thread-per-point GPU execution |
| **Stochastic method, deterministic parallelization** | Unlike the finite-difference/CG solvers elsewhere in the codebase, this approximates the PDE solution statistically — accuracy depends on `iterCount`, not on convergence tolerance |
| **Object-oriented `[EntryPoint]`** | The kernel is an **instance method** (`Solve()` on `MonteCarloHeatSolver`, closing over `_problem`), wrapped with `runner.Wrap(solver)` — not a static method on the top-level `Program` class like the other samples |

### Hybridizer: C# philosophy on the GPU

This sample pushes further than the others into idiomatic object-oriented design: an interface (`I2DProblem`), a generic problem class parameterized by strategy types, and an `[EntryPoint]` on an **instance method of a non-`Program` class** all compile down to a GPU kernel. There's no requirement for the GPU entry point to live on a particular class or be `static` on `Program` — Hybridizer wraps whatever object instance you give it (`runner.Wrap(solver)`) and generates CUDA from its `[EntryPoint]`-decorated method, letting the solver's design stay driven by ordinary C# abstractions (interfaces, generics) rather than by GPU-programming constraints.
