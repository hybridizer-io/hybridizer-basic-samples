CONSTANT MEMORY EXAMPLE
==========================

What this sample shows
-----------------------
GPUs offer a small, cached, read-only memory space called "constant 
memory", optimized for values that are the same across all threads — 
typically fixed coefficients or stencil weights read repeatedly during 
a computation. This sample shows how to declare and use constant 
memory in Hybridizer, and applies it to a stencil computation, where 
each output element is computed from a fixed set of neighboring input 
elements weighted by coefficients stored in constant memory.

The example
-----------
The same stencil computation is run using two almost identical 
approaches:

1. "Global memory": the stencil coefficients are stored in a regular 
   (global memory) array, and each thread re-reads them from global 
   memory for every output element it computes — used here as a 
   baseline.

2. "Constant memory": the exact same coefficients are instead placed 
   in a constant-memory buffer, declared and accessed via 
   Hybridizer's constant memory support, letting every thread read the 
   coefficients through the constant cache instead of global memory.

The output produced by the constant-memory version is compared 
element by element against the global-memory reference to confirm 
correctness, and a Stopwatch measures the execution time of each 
version separately.

Expected output
----------------
Selecting device <GPU name> with compute capability <XX>
running stencil (global memory)
global memory time : <X> ms
running stencil (constant memory)
constant memory time : <Y> ms
comparing results...
OK

(the exact GPU name and timings depend on your hardware, the stencil 
size, and the number of coefficients used; the constant-memory 
version is expected to perform at least as well as, and often better 
than, the global-memory version since coefficients are broadcast 
efficiently to all threads from the constant cache; watch out for 
indexing mismatches between the stencil coefficients and neighbor 
offsets — a swapped array or an off-by-one in the indexing is a common 
source of incorrect results rather than a CUDA error; the program 
prints an error and stops instead of "OK" if a mismatch or a CUDA 
error is detected)