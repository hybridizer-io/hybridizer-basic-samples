# StrategyBacktest — GPU-Accelerated Trading Strategy Parameter Sweep

This sample demonstrates a **x400+ speedup** using Hybridizer by performing an exhaustive parameter sweep of Moving Average crossover strategies on real market data.

The program tests **78,804 combinations** of (shortMA, longMA) parameters and compares execution times across 4 methods: CPU sequential, CPU parallel, GPU naive port, and GPU native kernel.

## Prerequisites

- .NET 8.0 SDK
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version matching `Directory.Build.props`, currently 13.0)
- Hybridizer CLI tool (`dotnet tool install -g hybridizer`)
- Visual Studio 2022+ with C++ workload (Windows) or GCC (Linux)

## Build

Always build in **Release** mode for maximum performance — Debug mode disables compiler optimizations and gives misleading benchmark results.

```bash
# Restore NuGet packages (first time only)
dotnet restore

# Build in Release mode
dotnet build --configuration Release
```

The build pipeline:
1. Compiles the C# project
2. Runs Hybridizer to generate CUDA C++ from the `[EntryPoint]` kernels
3. Compiles the generated CUDA code with `nvcc` into a native GPU DLL

## Run

```bash
dotnet run --configuration Release --no-build -- [options]
```

> **Note:** The `--` separator is required before program arguments to distinguish them from `dotnet` options.

### Options

| Option | Description | Default |
|---|---|---|
| `--file <path>` | Load price data from a local CSV file | *(downloads from internet)* |
| `--symbol <ticker>` | Yahoo Finance ticker symbol to download | `BTC-USD` |
| `--days <count>` | Number of days of history to download | `3650` (10 years) |

### Examples

**Download BTC-USD data automatically (default):**
```bash
dotnet run --configuration Release --no-build
```

**Use a local CSV file:**
```bash
dotnet run --configuration Release --no-build -- --file btc_history.csv
```

**Download a different asset (e.g. Apple stock):**
```bash
dotnet run --configuration Release --no-build -- --symbol AAPL
```

**Download 5 years of Ethereum data:**
```bash
dotnet run --configuration Release --no-build -- --symbol ETH-USD --days 1825
```

### CSV File Format

The program accepts standard Yahoo Finance CSV format:

```csv
Date,Open,High,Low,Close,Adj Close,Volume
2024-01-02,42681.37,44733.25,42526.00,44156.78,44156.78,6543567891
```

It looks for a `Close` column in the header (falls back to `Adj Close`, then second column).

To download a CSV manually:
1. Go to https://finance.yahoo.com/quote/BTC-USD/history/
2. Set the time period (at least 2 years recommended)
3. Click **Download**
4. Run with `--file downloaded.csv`

## Output

The program produces:

1. **Performance comparison table** — CPU sequential, CPU parallel, GPU naive, GPU native kernel
2. **Validation** — Verifies CPU and GPU results match (max/avg error)
3. **Strategy analysis** — Best/worst parameter combination and profitability stats
4. **Heatmap PNG** (`strategy_heatmap.png`) — Visual map of returns by parameter combination
5. **Console heatmap** — ANSI-colored terminal visualization (requires modern terminal)

### Example Output

```
╔══════════════════════════════════════════════════════════════╗
║   Strategy Backtest — Hybridizer GPU Parameter Sweep        ║
╠══════════════════════════════════════════════════════════════╣
║  Price data points :       3651                             ║
║  Short MA range    : [2..200]                               ║
║  Long  MA range    : [5..400]                               ║
║  Total combinations:      78804                             ║
╚══════════════════════════════════════════════════════════════╝

Running CPU sequential benchmark...       36825 ms
Running CPU parallel (24 cores)...          2422 ms
GPU detected: NVIDIA GeForce RTX 4070 Laptop GPU
Running GPU naive (from Parallel)...          91 ms
Running GPU native kernel...                  84 ms

┌──────────────────────────────────────────────────────────┐
│                PERFORMANCE COMPARISON                    │
├──────────────────────┬──────────┬────────────────────────┤
│ Method               │ Time(ms) │ Speedup vs seq. CPU    │
├──────────────────────┼──────────┼────────────────────────┤
│ CPU sequential       │    36825 │             (baseline) │
│ CPU parallel (24c)   │     2422 │                  x15.2 │
│ GPU naive (from CPU) │       91 │                 x404.7 │
│ GPU native kernel    │       84 │                 x438.4 │
└──────────────────────┴──────────┴────────────────────────┘
```

## How It Works

The demo performs a **parameter sweep** over all combinations of short and long moving average periods:

- **Short MA**: 2 to 200 (199 values)
- **Long MA**: 5 to 400 (396 values)
- **Total**: 78,804 combinations (~59,500 valid where short < long)

For each combination, a complete backtest is run:
1. Compute both moving averages at each time step (naive O(window) sum)
2. Detect **golden cross** (short MA crosses above long MA) → buy
3. Detect **death cross** (short MA crosses below long MA) → sell
4. Track equity and compute total return

### Why is the GPU so much faster?

| Factor | Explanation |
|---|---|
| **Massive parallelism** | Each of the 78,804 combinations runs as an independent GPU thread |
| **High arithmetic intensity** | Naive SMA computation = O(window_size) per data point per combination |
| **Read-only shared data** | Price array (~15 KB) fits entirely in GPU L2 cache |
| **Minimal memory transfers** | Prices uploaded once, results downloaded once |

### Four benchmark methods

| Method | Description |
|---|---|
| **CPU sequential** | Single-threaded `for` loop over all combinations |
| **CPU parallel** | `Parallel.For` distributing combinations across CPU cores |
| **GPU naive** | `[EntryPoint]` calling a `[Kernel]` device function — minimal port from CPU code |
| **GPU native** | `[EntryPoint]` with fully inlined backtest logic — hand-optimized kernel |

### Hybridizer: C# philosophy on the GPU

A key takeaway from this demo is the comparison between the **GPU naive** and **GPU native** approaches:

| Approach | Time | Style |
|---|---|---|
| GPU naive (from CPU) | 91 ms | Clean C# — method calls, separation of concerns |
| GPU native kernel | 84 ms | Fully inlined — all logic in a single `[EntryPoint]` method |
| **Difference** | **~8%** | |

The "naive" GPU version keeps **idiomatic C# code structure**: the backtest logic lives in a separate `[Kernel]` method (`RunSingleBacktest`) that is called from the `[EntryPoint]`, exactly like how `Parallel.For` calls a method on CPU. There is no manual CUDA memory management, no pointer arithmetic, no explicit thread synchronization — just standard C# with a couple of attributes.

Despite this clean, maintainable code style, the naive port is **only ~8% slower** than the fully hand-optimized kernel where everything is inlined. The CUDA compiler (`nvcc`) is smart enough to inline the device function automatically, making the performance difference negligible.

This demonstrates Hybridizer's core value proposition: **you don't have to sacrifice C# code quality and maintainability to get GPU performance**. Write clean, well-structured C# code with proper method decomposition, add `[EntryPoint]` and `[Kernel]` attributes, and Hybridizer handles the rest.
