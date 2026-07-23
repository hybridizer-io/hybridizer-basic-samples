PRINTF EXAMPLE
=================

What this sample shows
-----------------------
Normally, when you write to the console in C#, that code runs on the 
CPU. This sample shows that Hybridizer also lets GPU threads print 
directly to the console themselves, using the same Console.Out.Write 
syntax you'd use in ordinary C# — no special GPU-specific print 
function needed.

Each GPU thread prints its own identity (which thread it is, which 
block it belongs to) along with the value it's working on, which is a 
simple way to see the GPU's parallelism directly, instead of just 
trusting that "it ran on many threads at once".

The example
-----------
The array "a" contains 13 integers (1 through 13). The kernel is 
launched with 4 blocks of 4 threads each — 16 threads in total, more 
threads than there are elements in the array.

Each thread checks its own global index (based on threadIdx and 
blockIdx) and, if that index falls within the array (index < 13), it 
prints a line with its thread number, its block number, and the value 
of a[i] at that index. The 3 extra threads (16 - 13) simply have 
nothing to do and print nothing.

Because all 16 threads run independently and in parallel on the GPU, 
the order in which their print lines actually appear in the console is 
not guaranteed — it depends on how the GPU schedules the threads, and 
can vary from one run to another.

Expected output
----------------
hello from thread = 0 in block = 0 a[i] = 1
hello from thread = 1 in block = 0 a[i] = 2
hello from thread = 2 in block = 0 a[i] = 3
hello from thread = 3 in block = 0 a[i] = 4
hello from thread = 0 in block = 1 a[i] = 5
... (13 lines total, one per array element)
---- Done ----

(the exact order of the "hello from thread..." lines can differ 
between runs — this is expected, since GPU threads execute 
concurrently and are not printed in a fixed sequence)