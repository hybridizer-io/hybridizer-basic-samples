# Sobel — GPU-Accelerated Sobel Edge Detection

This sample demonstrates **GPU-accelerated image processing** using Hybridizer: it loads a grayscale-converted image, runs a classic **Sobel edge detection** filter on the GPU, and saves the result to disk.

The program loads `lena512.bmp` (512×512), converts it to grayscale, computes the Sobel gradient magnitude for every interior pixel on the GPU, and writes the edge map to `lena-sobel.bmp`.

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
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`ComputeSobel`)
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required — the input filename (`lena512.bmp`) and output filename (`lena-sobel.bmp`) are hardcoded.

## Output

The program produces:

- **`lena-sobel.bmp`** — a grayscale image the same size as the input, where each pixel's brightness represents the strength of the detected edge at that location (bright = strong edge, dark = flat/uniform area)

There is no console output, timing, or CPU/GPU comparison in this sample — it's a pure image-processing demo.

## How It Works

### Grayscale conversion (CPU)

`ReadImage` converts the loaded RGBA image to a single-byte-per-pixel grayscale buffer using the standard luminance formula:

\`\`\`
grey = 0.2126 * R + 0.7152 * G + 0.0722 * B
\`\`\`

This runs on the CPU, before any GPU work — it's cheap relative to the convolution itself and only needs to happen once.

### Sobel filter (GPU)

`ComputeSobel` assigns each GPU thread one or more pixels via a **2D grid-stride loop** (rows via `threadIdx.y`/`blockIdx.y`, columns via `threadIdx.x`/`blockIdx.x`). For every interior pixel (borders are skipped, since the 3×3 neighborhood would go out of bounds), it:

1. Reads the 8 surrounding pixels (`topl`, `top`, `topr`, `l`, `r`, `botl`, `bot`, `botr`)
2. Computes the horizontal gradient `sobelx` and vertical gradient `sobely` using the standard 3×3 Sobel kernels
3. Combines them into a gradient magnitude, `sqrt(sobelx² + sobely²)`, clamped to `[0, 255]` to fit back into a byte

| Factor | Explanation |
|---|---|
| **Per-pixel parallelism** | Each output pixel's Sobel value is independent of every other — a textbook embarrassingly-parallel image filter |
| **Stencil pattern** | Each thread reads a fixed 3×3 neighborhood from `inputPixel`, a classic GPU stencil-computation pattern (related to the constant-memory stencil samples elsewhere in the codebase) |
| **Border handling** | Pixels on the image edge are left untouched (`output` stays `0`) rather than reading out of bounds |
| **2D grid-stride loop** | `SetDistrib(32, 32, 16, 16, 1, 0)` launches a fixed grid that covers images larger than one launch's worth of threads by looping |

> **Note:** pixel indexing throughout (`i * height + j`, and `i * width + j` for the Sobel neighborhood) mixes `width` and `height` as the row stride. This is only correct because `lena512.bmp` is square (`width == height`) — using this code as-is on a non-square image would misalign rows.

### Hybridizer: C# philosophy on the GPU

`ComputeSobel` is a direct, idiomatic port of a CPU-style Sobel filter: explicit 2D thread indexing, but no shared memory or manual CUDA memory management — reads go straight to the input buffer, one 3×3 neighborhood per pixel. The image I/O (`ImageSharp` loading, grayscale conversion, PNG encoding) stays in ordinary, unattributed C# and runs on the CPU; only `ComputeSobel` is marked `[EntryPoint]` and offloaded to the GPU.

This demonstrates Hybridizer's core value proposition applied to image processing: **the parts of the pipeline that benefit from massive parallelism (the per-pixel convolution) go to the GPU with a single attribute, while everything else — file I/O, color conversion, encoding — stays as plain, familiar C#**.