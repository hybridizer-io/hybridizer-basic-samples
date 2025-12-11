#ifndef __DEF_INTRINSIC_GENERIC_RED__
#define __DEF_INTRINSIC_GENERIC_RED__

__device__ __forceinline__ float atomicMax(float* address, float val)
{
    int* addr_as_i = reinterpret_cast<int*>(address);
    int old = *addr_as_i;
    int assumed;

    while (true) {
        assumed = old;
        float old_val = __int_as_float(assumed);

        if (old_val >= val)
            break;  // nothing to do

        int new_bits = __float_as_int(val);
        old = atomicCAS(addr_as_i, assumed, new_bits);

        if (old == assumed)
            break;  // CAS succeeded
    }

    return __int_as_float(old);
}

#endif