# WmmaGemm — half-precision GEMM with tensor cores

A progression of half-precision matrix-multiply kernels written in C# and
transcoded by Hybridizer, walking from a naive textbook matmul up to a
CUTLASS-style hand-tuned tensor-core kernel that runs at **~93% of cuBLAS**
at n=4096 on consumer Blackwell (sm_120).

The sample doubles as a tutorial on the new bindings shipping in
Hybridizer.Runtime.CUDAImports:

* `pipeline` — `cp.async` producer/consumer staging primitives.
* `ldmatrix` — warp-collective shared-memory loads in the layout
  `mma.sync.m16n8k16` expects.
* `mma` — Ampere/Blackwell tensor-core multiply-accumulate as a PTX
  intrinsic.
* `wmma` (already shipped) — the older `nvcuda::wmma::*` fragment API.

## Kernels (best-of-10 GFLOPS at n=4096, RTX 5070 Laptop)

| Kernel       | Approach                                                  | GFLOPS | vs cuBLAS |
|--------------|-----------------------------------------------------------|-------:|----------:|
| `naive`      | One-thread-per-output, no tiling                          |   1035 |     0.025 |
| `tiled`      | Classic block-shmem-tile (16×16, float)                   |   1150 |     0.028 |
| `regtiled`   | Block-tile + per-thread register tile (64×64 / 4×4)       |   4500 |     0.108 |
| `big`        | 128×128 block, 8×8 per-thread register tile               |   7440 |     0.179 |
| `wmma`       | One-warp-per-tile, `nvcuda::wmma::fragment` + half input  |   9900 |     0.238 |
| `wmma-sh`    | 64×64 block, 4 warps cooperate, bank-conflict-free shmem  |  23200 |     0.557 |
| `wmma-as`    | + `cp.async` double-buffered global→shmem                 |  28200 |     0.677 |
| `wmma-ptx`   | Same shape, raw PTX (`ldmatrix.x4` + `mma.sync.m16n8k16`) |  28600 |     0.687 |
| `wmma-big`   | 128×128 / 8 warps / 3-stage, `wmma::load_matrix_sync`     |  35900 |     0.862 |
| `wmma-bigp`  | 128×128 / 8 warps / 4-stage, PTX + block swizzle          | **38500** | **0.925** |
| `cublas`     | `cublasGemmEx` (`CUDA_R_16F` × `CUDA_R_16F` → `CUDA_R_32F`) |  41400 |     1.000 |

The "bigp" kernel is the one to study — it's the canonical CUTLASS-Mma_sm80
shape (mma.sync.m16n8k16 driven from ldmatrix-staged shmem buffers filled by
cp.async). The bindings it uses are all upstream-ready primitives.

## Prerequisites

- .NET 8.0 SDK
- NVIDIA GPU, sm_70+ for wmma, sm_75+ for ldmatrix, sm_80+ for cp.async + mma.sync
- CUDA Toolkit (12.x or 13.x; tested on 13.0/13.2)
- The licensed **Hybridizer suite** checked out somewhere — by default the
  build looks for it under `D:\hybridizer-software-suite` (Windows) or
  `/mnt/d/hybridizer-software-suite` (Linux/WSL); override with
  `-p:HybridizerSuiteRoot=...` if it lives elsewhere.

Unlike the other samples here (which use the free `hybridizer` CLI's BASIC
JIT mode), WmmaGemm runs `Hybridizer.Application` in standalone mode so the
kernel's inline-asm helpers (in `hybridizer.cuda.cuh`) are visible to nvcc.

## Build

```bash
dotnet build -c Release -p:Platform=x64
```

## Run

```bash
dotnet bin/x64/Release/net8.0/WmmaGemm.dll [N]
```

`N` defaults to 512; must be a multiple of 128. Set `HYB_PROFILE=1` to bench
each kernel for 10 iterations after a 2-iter warmup:

```bash
HYB_PROFILE=1 dotnet bin/x64/Release/net8.0/WmmaGemm.dll 4096
```

## Files

| File | What it is |
|---|---|
| `Program.cs` | All 11 kernels + benchmark harness. |
| `wmma_helpers.cuh` | Project-local C++ glue — folds the `(array, offset)` shape Hybridizer hands kernel authors into the raw-pointer shape the upstream PTX wrappers take. |
| `Pipeline.cs`, `Ldmatrix.cs`, `Mma.cs` | Local copies of the new CUDAImports bindings, kept here until the next CUDAImports release picks them up. |
| `Cublas.cs` | Minimal P/Invoke to `cublasGemmEx` for the reference comparison. |
| `SatelliteLoader.cs` | Standard `HybRunner.Cuda` loader. |
| `WmmaGemm.csproj` + `Directory.Build.targets` | Standalone-mode build wiring (nvcc invocation distinct from the BASIC-mode siblings). |
