# Mandelbulb — GPU-Accelerated Real-Time Raymarched Fractal

This sample demonstrates a **real-time GPU raymarcher** using Hybridizer: for every pixel of an interactive OpenGL window, it sphere-traces a ray through a **Mandelbulb** fractal's distance field, shades the hit point (diffuse, specular, soft shadows), and writes the result directly into a CUDA/OpenGL-shared texture — a step up in complexity from the earlier N-body OpenGL interop sample, using **surface objects** instead of vertex buffers.

The program opens an `800×600` animated window (camera and light slowly orbiting) and displays a live FPS / kernel-time counter in the console.

## Prerequisites

- .NET SDK
- NVIDIA GPU with CUDA support, with OpenGL interop enabled (same GPU driving both the display and the simulation) — **make sure your preferred rendering GPU is CUDA-compatible, not an integrated Intel GPU** (the program prints a reminder about this at startup: check NVIDIA Control Panel → Manage 3D Settings → Preferred Graphics Processor)
- CUDA Toolkit (version matching your Hybridizer install)
- Hybridizer runtime (`Hybridizer.Runtime.CUDAImports`, `Hybridizer.Basic.Utilities` NuGet packages)
- `OpenTK` NuGet packages (`OpenTK.Graphics`, `OpenTK.Windowing.Desktop`) for the OpenGL window and rendering
- `vertex.glsl` and `fragment.glsl` shader files, present next to the built executable (loaded at runtime via `File.ReadAllText`) — a minimal textured-quad shader pair
- Visual Studio 2022+ with C++ workload (Windows) or GCC (Linux)

## Build

Always build in **Release** mode for maximum performance — Debug mode disables compiler optimizations and gives misleading benchmark results.

\`\`\`bash
# Restore NuGet packages (first time only)
dotnet restore

# Build in Release mode
dotnet build --configuration Release
\`\`\`

Make sure `vertex.glsl` and `fragment.glsl` are copied to the output directory — the window fails to load (with a caught exception logged to console) without them.

The build pipeline:
1. Compiles the C# project
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`Render`) and its helper functions (`Distance`, `Shadow`, and the `MathFunctions`/`TextureHelpers` intrinsics)
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required. This opens an `800×600`, non-resizable window titled **"Mandelbulb"**, rendering the fractal live as the camera and light both slowly orbit around it. Every second, the console prints the average GPU kernel time and FPS. Closing the window ends the program.

## How It Works

### Distance-estimated raymarching (sphere tracing)

Rather than intersecting explicit geometry, the Mandelbulb is defined by a **distance estimator** (`Distance`): given a 3D point, it iterates the Mandelbulb's power-8 formula in spherical coordinates (`theta`, `phi`, `r`) up to `iterations` times (or until the point escapes a radius of `DEPTH_OF_FIELD`), tracking a running derivative (`dr`) used to convert the escape behavior into an approximate **distance to the fractal surface**.

For each pixel, `Render`:

1. Computes a camera ray from the pixel's position on a near-field plane through the eye location
2. **Sphere-traces** the ray: repeatedly evaluates `Distance` at the current position and steps forward by that distance (guaranteed not to overshoot the surface), until the estimated distance is below half a pixel's width (a hit) or the ray leaves the depth of field (a miss)
3. On a hit, estimates the **surface normal** by evaluating `Distance` at six points offset slightly along each axis (a numerical gradient / finite-difference normal)
4. Shades the point with a **Blinn-Phong**-style model: diffuse light from `dotNL`, a sharp specular term (`s^35`), and a **soft shadow** (`Shadow`, itself another short raymarch toward the light source) — combined into an RGB color written to the pixel
5. On a miss, writes a simple sky-gradient color instead, based on how many raymarching steps were taken

### CUDA / OpenGL surface interop

Instead of a vertex buffer (as in the N-body sample), this sample shares a **2D texture**: `cuda.GraphicsGLRegisterImage` registers the OpenGL texture as a CUDA graphics resource, `GraphicsMapResources`/`GraphicsSubResourceGetMappedArray` retrieve the underlying CUDA array, and `cuda.CreateSurfaceObject` wraps it as a `cudaSurfaceObject_t`. Each frame, the `Render` kernel writes pixel colors directly into that surface via `surf2Dwrite` (an `[IntrinsicFunction]`), and the same texture is immediately drawn on a full-screen quad by OpenGL — no host round-trip for the rendered image at all.

| Factor | Explanation |
|---|---|
| **Per-pixel parallelism** | Each of the `800 × 600` pixels independently sphere-traces and shades its own ray — an ideal one-thread-per-pixel GPU workload |
| **Distance-field rendering** | Sphere tracing needs far fewer steps than naive ray-sampling, since each step safely advances by the estimated distance to the nearest surface |
| **Zero-copy display** | The GPU writes shading results straight into the texture OpenGL renders from, via CUDA surface objects — the same zero-copy idea as the N-body sample's vertex-buffer interop, applied to a 2D image instead of point data |
| **Live performance readout** | `runner.LastKernelDuration` exposes the GPU kernel's execution time directly from C#, averaged and printed alongside FPS every second |

### Hybridizer: C# philosophy on the GPU

This is the most elaborate hand-written CUDA logic among the samples so far: a distance-estimator fractal formula, sphere tracing, finite-difference normals, a full shading model, and low-level CUDA surface/texture interop calls (`cudaCreateChannelDesc`, `CreateSurfaceObject`, `surf2Dwrite`) are all expressed as ordinary C# methods and structs (`float3`, `uchar4`), decorated with `[Kernel]`/`[EntryPoint]`/`[IntrinsicFunction]` where CUDA-specific behavior is needed. `MathFunctions` mirrors CUDA's math intrinsics (`sincosf`, `atan2f`, `rsqrtf`, etc.) as thin C# wrappers, and `TextureHelpers` does the same for the surface/texture API — keeping even fairly advanced CUDA features accessible from a single .NET codebase, with no separate `.cu` files to maintain.