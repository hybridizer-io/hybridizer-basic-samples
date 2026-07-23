MALLOC EXAMPLE
=================

What this sample shows
-----------------------
Normally, you cannot use .NET's "new" keyword to allocate objects from 
code that runs on the GPU — GPU threads don't have the same kind of 
memory management as the CPU. There is one exception though: arrays. 
Each GPU thread is allowed to allocate its own small, local array with 
"new", which gets automatically freed once that thread is done with it.

This sample is a toy example (the author's own comment says "no 
physical meaning at all") whose only purpose is to demonstrate that 
this thread-local array allocation works — not to compute anything 
meaningful.

The example
-----------
Every GPU thread computes one output value by:
1. Allocating its own small array of 9 numbers ("stencil"), filled with 
   the values -4, -3, -2, ... up to 4. This "new double[9]" happens 
   individually inside each thread — it is not shared or precomputed 
   on the CPU.
2. Using that array as a set of 9 weights, applied to 9 neighboring 
   values around its position in the "src" array (4 values before, 
   itself, and 4 values after) — a classic "stencil" pattern used in 
   things like signal processing or numerical simulations.
3. Storing the weighted sum into "dest".

This is repeated for 33,554,432 (32 * 1024 * 1024) values, each 
computed independently and in parallel, with every thread creating and 
discarding its own little array along the way.

Expected output
----------------
GPU time for 33 554 432 elements : ~450 ms
Results Samples (from i = 4 to 13) : 
-1.2345, 0.8821, ...

(exact timing and values depend on your GPU and the random seed; since 
"src" is filled with random numbers between 0 and 1, the sample output 
values will differ on every run — this sample doesn't verify 
correctness against a CPU reference, so the numbers are only shown to 
confirm the GPU computation actually produced something)