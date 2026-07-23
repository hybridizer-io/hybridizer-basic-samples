# NBody — GPU-Accelerated N-Body Simulation with OpenGL Interop

This sample demonstrates a **real-time GPU N-body gravity simulation** using Hybridizer, rendered live with OpenGL through **CUDA/OpenGL interop** (the physics buffers are shared between CUDA and the GPU renderer — no round-trip through the CPU each frame).

The program initializes a two-cluster "galaxy-like" body distribution, then runs an interactive window where each frame integrates gravity on the GPU and immediately draws the resulting point cloud.

## Prerequisites

- .NET SDK
- NVIDIA GPU with CUDA support, with OpenGL interop enabled (same GPU driving both the display and the simulation)
- CUDA Toolkit (version matching your Hybridizer install)
- Hybridizer runtime (`Hybridizer.Runtime.CUDAImports`, `Hybridizer.Basic.Utilities` NuGet packages)
- `OpenTK` NuGet packages (`OpenTK.Graphics`, `OpenTK.Windowing.Desktop`, `OpenTK.Mathematics`) for the OpenGL window and rendering
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
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`Solve`) and its `[Kernel]` device functions (`ComputeBodyAccel`, `BodyBodyInteraction`)
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required. This opens an `800×600` window titled **"Hybridizer N body simulation"** showing the bodies as gold points on a black background, animated in real time. Closing the window ends the program — there is no console benchmark output in this sample.

## How It Works

### Physics: tile-based N-body gravity

Each GPU thread owns one body. For every frame, `Solve`:

1. Calls `ComputeBodyAccel` to sum the gravitational pull of every other body on this one
2. Updates velocity from acceleration (`velocity += accel * deltaT`), applies a damping factor (`damping = 0.9995`) to slowly dissipate energy
3. Updates position from the new velocity (`position += velocity * deltaT`)

`ComputeBodyAccel` uses the classic **tile-based shared-memory algorithm** (from the *GPU Gems 3* N-body chapter): bodies are processed in blocks of `blockDim.x`, with each tile's positions staged into shared memory (`SharedMemoryAllocator<float4>`) so every thread in the block reuses the same loaded data instead of hitting global memory per pairwise interaction. `__syncthreads()` calls guard the shared buffer between the load and the compute phase.

`BodyBodyInteraction` computes the force from one other body `bj` on body `bi`, using a **softened** inverse-cube law (`softeningSquared = 0.00125`, added to `distSqr` before the `rsqrtf`) to avoid numerical blow-up when two bodies get very close.

### CUDA / OpenGL interop

Instead of copying positions from GPU to CPU and back to the renderer every frame, `RenderingWindow`:

1. Allocates two **OpenGL vertex buffers** (`_buffers`) and registers each one as a **CUDA graphics resource** (`cuda.GraphicsGLRegisterBuffer`)
2. Each frame, **maps** both buffers directly into CUDA's address space (`MapResources`) — one as the read-only previous positions, one as the write-only new positions
3. Runs the `Solve` kernel writing straight into the mapped OpenGL buffer
4. **Unmaps** the buffers (`UnMapResources`) and draws the just-written buffer directly with `GL.DrawArrays`, with **zero explicit host/device copies** in the render loop
5. `SwapPos()` swaps which buffer is "old" and which is "new" every frame — a **ping-pong double-buffering** scheme, so the kernel never reads and writes the same buffer at once

### Initial conditions

`BodyInitializer` places bodies in two symmetric half-clusters offset from the origin, each half given an initial velocity kick that scales with position — producing an initial rotation-like motion. Body masses are randomized (`RandM`, in `[0.7, 1.3]`), and after generating all velocities, the total momentum is computed and subtracted back out per-body so the whole system starts with **zero net momentum** (it doesn't drift as a whole).

| Factor | Explanation |
|---|---|
| **Per-body parallelism** | Each thread simulates exactly one body per frame — `O(numBodies)` threads, each doing an `O(numBodies)` force sum |
| **Shared-memory tiling** | Cuts down redundant global memory traffic when summing forces from every other body |
| **Zero-copy rendering** | CUDA writes new positions directly into the OpenGL vertex buffer that will be drawn — no CPU round-trip in the hot loop |
| **Softened gravity** | `softeningSquared` prevents numerical instability from near-singular forces at very small distances |

### Hybridizer: C# philosophy on the GPU

Like the shared-memory matrix multiplication sample, this kernel is **hand-optimized CUDA logic expressed in C#**: explicit shared memory (`SharedMemoryAllocator<float4>`), `__syncthreads()` barriers, and a custom `rsqrtf` intrinsic are all written directly in the method bodies with `[Kernel]`/`[EntryPoint]` attributes — no separate `.cu` file to maintain.

What's new here is that the **interop layer** (`cuda.GraphicsGLRegisterBuffer`, `GraphicsMapResources`, `GraphicsResourceGetMappedPointer`) is also called from C#, through the same `Hybridizer.Runtime.CUDAImports` namespace used for the kernel itself — so the whole pipeline, from physics to zero-copy rendering, stays in one .NET codebase without hand-written CUDA/OpenGL interop boilerplate in C++.