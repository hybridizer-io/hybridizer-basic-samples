HELLOWORLD EXAMPLE
====================

What this sample shows
-----------------------
This is the most basic Hybridizer sample: it takes a piece of ordinary 
C# code, runs it on the GPU, then runs the exact same code on the CPU 
(through plain .NET), and checks that both give the same result.

The idea is to prove that Hybridizer really does what it promises: you 
write your logic once in C#, and it works identically whether it 
executes on the GPU or on the CPU — no separate GPU-specific code 
needed.

The example
-----------
The method "Run" simply adds two big arrays of random numbers together, 
element by element (a[i] += b[i]), using Parallel.For so each element 
can be computed independently and in parallel. Two arrays are used: 
16 million doubles each (about 268 MB), which is a size that fits on 
basically any CUDA-compatible GPU.

The same "Run" method is called twice:
1. Once through "wrapped.Run(...)", which sends it to the GPU via 
   Hybridizer.
2. Once as a normal, unmodified C# method call, which runs on the CPU 
   as regular .NET code.

The program then compares the two resulting arrays value by value. If 
every single value matches, it means the GPU computed exactly the same 
thing as the CPU would have — confirming the translation from C# to 
GPU code was correct.

Expected output
----------------
Expected output
----------------
GPU Results : 
0.732961, 1.245..., 0.891..., (... 16,777,216 comma-separated values ...)
CPU Results : 
0.732961, 1.245..., 0.891..., (... 16,777,216 comma-separated values ...)
DONE

(the two lists should be identical, value by value; if any value 
differs, the program prints "ERROR !" and stops instead of reaching 
"DONE")