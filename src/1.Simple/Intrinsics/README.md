INTRINSICS EXAMPLE
=====================

What this sample shows
-----------------------
GPUs can compute using different levels of numeric precision. Standard 
precision uses 32-bit floats (float), but modern GPUs also support 
16-bit "half precision" floats, which take half the memory and can be 
computed faster — at the cost of a much smaller range of representable 
values (roughly up to 65504) and less numeric accuracy.

This sample demonstrates "mixed precision" arithmetic: using the half2 
type, which packs two 16-bit half-precision numbers together, and 
performing a custom exponential function on them entirely in half 
precision.

Hardware requirement
---------------------
Mixed precision (half2) requires a GPU with compute capability 5.3 or 
higher — this covers Volta, Pascal, and Jetson TX1 GPUs onwards. On 
older GPUs (Maxwell, Kepler, Fermi), the generated code won't even 
compile. Note that Pascal itself is now old enough that NVIDIA has 
dropped it from the latest drivers.

The example
-----------
The sample defines its own approximation of the exponential function 
("exp"), computed as a 14-term polynomial (a Taylor series 
approximation), operating entirely on half2 values. This custom "exp" 
is then applied 12 times in a row ("exp12") to every element of a large 
array (33,554,432 elements), fully in parallel on the GPU.

Because "exp12" applies the exponential 12 times in a row, even a tiny 
starting value grows extremely fast — well beyond what 16-bit half 
precision can represent (which tops out around 65504). This makes the 
sample a good illustration of a real limitation of mixed precision: 
values can silently overflow into Infinity or NaN (Not a Number) if 
you're not careful about the range of numbers you're working with.

Expected output
----------------
INTRINSICS EXAMPLE
=====================

What this sample shows
-----------------------
GPUs can compute using different levels of numeric precision. Standard 
precision uses 32-bit floats (float), but modern GPUs also support 
16-bit "half precision" floats, which take half the memory and can be 
computed faster — at the cost of a much smaller range of representable 
values (roughly up to 65504) and less numeric accuracy.

This sample demonstrates "mixed precision" arithmetic: using the half2 
type, which packs two 16-bit half-precision numbers together, and 
performing a custom exponential function on them entirely in half 
precision.

Hardware requirement
---------------------
Mixed precision (half2) requires a GPU with compute capability 5.3 or 
higher — this covers Volta, Pascal, and Jetson TX1 GPUs onwards. On 
older GPUs (Maxwell, Kepler, Fermi), the generated code won't even 
compile. Note that Pascal itself is now old enough that NVIDIA has 
dropped it from the latest drivers.

The example
-----------
The sample defines its own approximation of the exponential function 
("exp"), computed as a 14-term polynomial (a Taylor series 
approximation), operating entirely on half2 values. This custom "exp" 
is then applied 12 times in a row ("exp12") to every element of a large 
array (33,554,432 elements), fully in parallel on the GPU.

Because "exp12" applies the exponential 12 times in a row, even a tiny 
starting value grows extremely fast — well beyond what 16-bit half 
precision can represent (which tops out around 65504). This makes the 
sample a good illustration of a real limitation of mixed precision: 
values can silently overflow into Infinity or NaN (Not a Number) if 
you're not careful about the range of numbers you're working with.

Expected output
----------------
Value before : Hybridizer.Runtime.CUDAImports.half2
Value after exp12 : Hybridizer.Runtime.CUDAImports.half2
GPU time for 33 554 432 elements : ~400 ms

(the exact timing depends on your GPU; starting from a small value like 
0.001, the repeated exponential is expected to overflow to Infinity for 
every element well before the 12th iteration — this is the expected 
and instructive behavior of this sample, not a bug)


(the exact timing depends on your GPU; starting from a small value like 
0.001, the repeated exponential is expected to overflow to Infinity for 
every element well before the 12th iteration — this is the expected 
and instructive behavior of this sample, not a bug)