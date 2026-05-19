// Static-build TU for the cudaStream_*/cudaGraph_* host wrappers in
// intrinsics.cuh. Hybridizer's auto-generated wrappers.cu does not include
// intrinsics.cuh (the JIT-include flag only feeds NVRTC), so the wrappers
// would otherwise be absent from LlamaCsharp_CUDA.dll. Loaded from C# via
// NativeLibrary.GetExport in Utils/GraphInvoke.cs.
#include "intrinsics.cuh"
