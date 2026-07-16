BUILTIN FUNCTION EXAMPLE
=========================

What this sample shows
-----------------------
Normally, when Hybridizer compiles your C# code for the GPU or CPU, it 
translates every instruction itself. But sometimes you want to tell 
Hybridizer: "for this specific .NET method, don't translate it yourself — 
just replace it directly with this native function instead." 

This is what a "builtin" is: a manual mapping between an existing .NET 
method and a native function that already exists in CUDA (or AVX/OMP).

The example
-----------
This sample sums an array of 1024 integers (0, 1, 2, ... 1023) in 
parallel, using System.Threading.Interlocked.Add to safely add each 
value into a shared result variable without threads overwriting each 
other.

Normally, Interlocked.Add doesn't mean anything on a GPU. To make it 
work, the file "sample.builtins" tells Hybridizer: "whenever you see 
Interlocked.Add being called, replace it with CUDA's native atomicAdd 
function instead." atomicAdd is a function built into CUDA that does 
exactly the same thing — it lets many threads add to the same variable 
at once without conflicts.

So this sample is really about showing that you can reuse familiar 
.NET methods (like Interlocked.Add) and have Hybridizer swap them out 
for the right native equivalent on the target hardware — without you 
having to rewrite your code.

Expected output
----------------
sum = 523776

(this is the sum of all integers from 0 to 1023)