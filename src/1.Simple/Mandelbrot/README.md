MANDELBROT EXAMPLE
=====================

What this sample shows
-----------------------
This sample renders the Mandelbrot set — a well-known fractal image — 
and uses it to compare performance between running the same C# code 
on the CPU versus on the GPU through Hybridizer, on a computation that 
is actually visual and easy to understand.

The Mandelbrot set is computed by testing, for every pixel of the 
image, how many iterations of a simple mathematical formula it takes 
before the result "escapes" past a certain threshold (or reaching a 
maximum of 32 iterations, meaning that pixel is considered part of the 
set). Pixels that escape quickly are colored differently from pixels 
that never escape, which is what produces the fractal's iconic pattern.

The example
-----------
The image is a 4096 x 4096 grid of pixels. For each pixel, the 
"IterCount" function runs the iteration formula and returns how many 
steps it took (from 0 to 32).

This whole computation is run twice, 20 times each (to get a stable 
average), for a fair comparison:
1. On the CPU, using Parallel.For to spread the work across CPU cores.
2. On the GPU, using the exact same "Run" method through Hybridizer.

Once computed, the result is turned into colors (each iteration count 
maps to a distinct RGB color, black meaning the pixel never escaped) 
and saved as "mandelbrot.png".

Expected output
----------------
elapsed time per image (C#) : ~180 ms
elapsed time per image (CUDA) : ~4 ms
Speedup : 45.00x
CPU/GPU results match : OK
Image saved to mandelbrot.png

(exact timings depend heavily on your CPU and GPU; the speedup is 
usually very significant here, since this is an "embarrassingly 
parallel" workload — every pixel is fully independent from every 
other pixel, which is exactly the kind of problem GPUs excel at)