# Sobel2D — GPU-Accelerated Sobel Edge Detection (2D Arrays)

This sample demonstrates **GPU-accelerated image processing** using Hybridizer, and specifically that Hybridizer supports **true 2D arrays** (`byte[,]`) as kernel arguments — not just flattened 1D buffers with manual index arithmetic.

The program loads `lena512.bmp` (512×512), converts it to grayscale, computes the Sobel gradient magnitude for every interior pixel on the GPU using natural `[i, j]` indexing, saves the result to `lena-sobel.bmp`, and opens it automatically.

## Prerequisites

- .NET SDK
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version matching your Hybridizer install)
- Hybridizer runtime (`Hybridizer.Runtime.CUDAImports`, `Hybridizer.Basic.Utilities` NuGet packages)
- `SixLabors.ImageSharp` NuGet package (image loading/encoding)
- Visual Studio 2022+ with C++ workload (Windows) or GCC (Linux)
- An input image named `lena512.bmp`, placed next to the built executable (`AppContext.BaseDirectory`)

## Build

Always build in **Release** mode for maximum performance — Debug mode disables compiler optimizations and gives misleading benchmark results.

\`\`\`bash
# Restore NuGet packages (first time only)
dotnet restore

# Build in Release mode
dotnet build --configuration Release
\`\`\`

Make sure `lena512.bmp` is copied to the output directory (e.g. set it to **Copy to Output Directory** in the project, or copy it manually next to the built binary) — the program looks for it at `AppContext.BaseDirectory`.

The build pipeline:
1. Compiles the C# project
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`ComputeSobel`), including the 2D array marshaling for `byte[,]` parameters
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required — the input filename (`lena512.bmp`), output filename (`lena-sobel.bmp`), and image size (`512`) are hardcoded.

## Output

The program produces:

- **`lena-sobel.bmp`** — a 512×512 grayscale image where each pixel's brightness represents the strength of the detected edge at that location (bright = strong edge, dark = flat/uniform area)
- The output image is **opened automatically** after processing (`Process.Start`), wrapped in a `try/catch` so the program doesn't fail on headless/non-interactive machines

There is no console output, timing, or CPU/GPU comparison in this sample — it's a pure image-processing demo.

## How It Works

### Grayscale conversion (CPU)

`ReadImage` converts the loaded RGBA image into a `byte[512, 512]` grayscale buffer using the standard luminance formula:

\`\`\`
grey = 0.2126 * R + 0.7152 * G + 0.0722 * B
\`\`\`

This runs on the CPU, before any GPU work — it's cheap relative to the convolution itself and only needs to happen once.

### Sobel filter (GPU)

`ComputeSobel` assigns each GPU thread one or more pixels via a **2D grid-stride loop** (rows via `threadIdx.y`/`blockIdx.y`, columns via `threadIdx.x`/`blockIdx.x`), reading the image size directly from the array with `inputPixel.GetLength(0)`. For every interior pixel (borders are skipped, since the 3×3 neighborhood would go out of bounds), it:

1. Reads the 8 surrounding pixels using natural 2D indexing (`inputPixel[i - 1, j - 1]`, etc. — no manual `row * width + col` arithmetic)
2. Computes the horizontal gradient `sobelx` and vertical gradient `sobely` using the standard 3×3 Sobel kernels
3. Combines them into a gradient magnitude, `sqrt(sobelx² + sobely²)`, clamped to `[0, 255]` to fit back into a byte

| Factor | Explanation |
|---|---|
| **True 2D arrays on the GPU** | `byte[,]` parameters are passed directly to the kernel — Hybridizer handles the underlying memory layout and index translation, so kernel code reads like ordinary 2D-array C# |
| **Per-pixel parallelism** | Each output pixel's Sobel value is independent of every other — a textbook embarrassingly-parallel image filter |
| **Stencil pattern** | Each thread reads a fixed 3×3 neighborhood from `inputPixel`, a classic GPU stencil-computation pattern |
| **Border handling** | Pixels on the image edge are left untouched (`output` stays `0`) rather than reading out of bounds |
| **2D grid-stride loop** | `SetDistrib(32, 32, 16, 16, 1, 0)` launches a fixed grid that covers images larger than one launch's worth of threads by looping |

### Hybridizer: C# philosophy on the GPU

Compared to a flat-buffer Sobel kernel, this version reads noticeably closer to plain, idiomatic C#: `inputPixel[i - 1, j - 1]` instead of `inputPixel[(i-1) * width + (j-1)]`. There is still no shared memory or manual CUDA memory management — reads go straight to the 2D input buffer, one 3×3 neighborhood per pixel — but the multi-dimensional array support removes an entire class of manual index-arithmetic bugs (like row/column stride mix-ups) that a flattened-array version has to get right by hand.

This demonstrates Hybridizer's core value proposition applied to image processing: **you can keep using .NET's native multi-dimensional arrays on the GPU**, with the parts of the pipeline that benefit from massive parallelism (the per-pixel convolution) offloaded with a single attribute, while file I/O, color conversion, and encoding stay as plain, familiar C#.