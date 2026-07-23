INOUT EXAMPLE
===============

What this sample shows
-----------------------
When Hybridizer sends an array to the GPU, by default it has to copy it 
in both directions: from CPU to GPU before the kernel runs, and from 
GPU back to CPU after — just in case the array was both read and 
modified. But often, an array is only ever read (input) or only ever 
written (output), and copying it in the unused direction is wasted 
time.

The [In] and [Out] attributes let you tell Hybridizer exactly how an 
array is used, so it can skip the unnecessary copy. This sample 
measures how much time that actually saves.

The example
-----------
The same computation, "dst[i] = src[i] + i", is run twice, using two 
almost identical methods:

1. "NoAttributes": takes "dst" and "src" as plain arrays, with no hint 
   about how they're used. Hybridizer copies both arrays in both 
   directions to be safe.

2. "Attributes": the exact same computation, but "dst" is marked [Out] 
   (only ever written to, never read) and "src" is marked [In] (only 
   ever read, never written). This tells Hybridizer it can skip copying 
   "dst" to the GPU beforehand, and skip copying "src" back afterward.

Both versions process 16,777,216 (2^24) random integers, and a 
Stopwatch measures the execution time of each version separately.

Expected output
----------------
Selecting device <GPU name> with compute capability <XX>
running generated CUDA (no attributes)
no in/out attribute time : <X> ms
running generated CUDA (attributes)
in/out attributes time : <Y> ms
OK

(the exact GPU name and timings depend on your hardware, but "Y" 
should generally be lower than "X", showing the benefit of the [In]/
[Out] attributes; the program prints an error and stops instead of 
"OK" if a CUDA error is detected)