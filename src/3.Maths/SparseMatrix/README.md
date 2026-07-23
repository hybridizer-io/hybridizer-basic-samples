# SpMV — GPU-Accelerated Sparse Matrix-Vector Multiplication (CSR, 1D Laplacian)

This sample demonstrates **GPU-accelerated sparse matrix-vector multiplication (SpMV)** using Hybridizer, on a **1D Laplacian matrix** stored in **CSR (Compressed Sparse Row)** format.

The program first validates the matrix construction on a small **10×10** Laplacian (printed as a dense matrix for visual inspection), then builds and multiplies a **10,000,000-row** Laplacian matrix by a constant vector on the GPU, repeating the multiplication `redo = 2` times.

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
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernel (`Multiply`), transpiling its internal `Parallel.For` loop into a CUDA thread grid
3. Compiles the generated CUDA code with `nvcc` into a native GPU satellite DLL

## Run

\`\`\`bash
dotnet run --configuration Release --no-build
\`\`\`

No arguments are required — the matrix is generated in-code (`SparseMatrix.Laplacian_1D`), it isn't read from a file. A `SparseMatrixReader` is included for loading real sparse matrices in **Matrix Market** format (`.mtx`, general / symmetric / skew-symmetric), but it is not currently wired into `Main`.

## Output

The program produces:

1. **Small-matrix correctness test** — builds and prints the dense 10×10 form of the 1D Laplacian, so the CSR construction can be checked by eye
2. **Matrix generation time** — wall-clock time to build the 10M-row Laplacian and allocate the result vector
3. The GPU multiply itself currently runs silently (no timing or result validation is printed after the CUDA loop)

### Example Output

\`\`\`
=== TEST 1D Laplacian  (n = 10) ===
Calculated Matrix :
2 -1 0 0 0 0 0 0 0 0 |
-1 2 -1 0 0 0 0 0 0 0 |
0 -1 2 -1 0 0 0 0 0 0 |
0 0 -1 2 -1 0 0 0 0 0 |
0 0 0 -1 2 -1 0 0 0 0 |
0 0 0 0 -1 2 -1 0 0 0 |
0 0 0 0 0 -1 2 -1 0 0 |
0 0 0 0 0 0 -1 2 -1 0 |
0 0 0 0 0 0 0 -1 2 -1 |
0 0 0 0 0 0 0 0 -1 2 |
=== END OF TEST ===

matrix read --- starting computations
Computing time  : 187 ms
\`\`\`

## How It Works

### CSR (Compressed Sparse Row) format

`SparseMatrix` stores only the non-zero coefficients, in three flat arrays:

| Array | Role |
|---|---|
| `data` | non-zero values, row by row |
| `indices` | column index of each entry in `data` |
| `rows` | row pointer — `rows[i]` is the offset into `data`/`indices` where row `i` starts, so row `i` spans `[rows[i], rows[i+1])` |

### 1D Laplacian construction

`SparseMatrix.Laplacian_1D(n)` builds the classic **tridiagonal** discretization of the 1D Laplacian operator: `2` on the diagonal, `-1` on the two adjacent off-diagonals, and only 2 non-zeros on the first and last rows (Dirichlet-style boundary rows). This gives a matrix with `3n - 2` non-zero entries.

### SpMV kernel

For each row `i`, `Multiply` computes the dot product between that row's non-zero values and the corresponding entries of the input vector:

\`\`\`
res[i] = Σ  data[j] * v[indices[j]]   for j in [rows[i], rows[i+1])
\`\`\`

| Factor | Explanation |
|---|---|
| **Row-parallelism** | Each row's dot product is independent — one GPU thread per row, ideal for SpMV |
| **Irregular memory access** | Unlike dense linear algebra, each row can have a different number of non-zeros and scattered column indices — this is what makes SpMV harder to optimize than dense matrix-vector products |
| **Repeated GPU calls** | `redo = 2` runs the multiply twice, so the loop overhead / first-call cost is amortized |

### Hybridizer: C# philosophy on the GPU

The `Multiply` kernel is written as a plain **`Parallel.For`** loop — exactly the same code a developer would write for a CPU-parallel SpMV. There is no manual CUDA memory management, no pointer arithmetic, no explicit thread indexing (`blockIdx`/`threadIdx`) in this version: Hybridizer recognizes the `Parallel.For` pattern inside the `[EntryPoint]` method and maps each loop iteration onto a CUDA thread automatically.

This demonstrates Hybridizer's core value proposition: **you don't have to rewrite your parallel CPU code in CUDA to run it on the GPU**. A `Parallel.For` loop over independent rows of a sparse matrix — a pattern most .NET developers already know — becomes a GPU kernel with a single `[EntryPoint]` attribute and a call through `HybRunner.Wrap(...)`.