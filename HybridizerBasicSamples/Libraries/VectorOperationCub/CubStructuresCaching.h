#pragma once
#ifndef __CUB_STRUCTURES_CACHING__
#define __CUB_STRUCTURES_CACHING__

enum ReduceOperation {
	Sum
};

struct ReduceTmpStorage {
	void* temp_storage;
	size_t temp_storage_bytes;
	void* d_out;

	inline ReduceTmpStorage() : temp_storage(NULL), temp_storage_bytes(0), d_out(NULL) {}
};

#pragma region SingleVectorCaching
struct DeviceSingleVectorReduceOp {
	size_t ptr;
	size_t N;
	ReduceOperation op;
};


struct DeviceSingleVectorReduceOp_lt {
	bool operator()(const DeviceSingleVectorReduceOp & D, const DeviceSingleVectorReduceOp & R) const
	{
		if (D.ptr < R.ptr)
			return true;
		else if (D.ptr == R.ptr && D.N < R.N)
			return true;
		else if (D.ptr == R.ptr && D.N == R.N && D.op < R.op)
			return true;
		return false;
	}
};

static std::map<DeviceSingleVectorReduceOp, ReduceTmpStorage, DeviceSingleVectorReduceOp_lt> cacheDeviceOneVector;

#pragma endregion

#pragma region TwoVectorsCaching
struct DeviceTwoVectorsReduceOp {
	size_t ptr1;
	size_t ptr2;
	size_t N;
	ReduceOperation op;
};

struct DeviceTwoVectorsReduceOp_lt {
	bool operator()(const DeviceTwoVectorsReduceOp & D, const DeviceTwoVectorsReduceOp & R) const
	{
		if (D.ptr1 < R.ptr1)
			return true;
		else if (D.ptr1 == R.ptr1 && D.ptr2 < R.ptr2)
			return true;
		else if (D.ptr1 == R.ptr1 && D.ptr2 == R.ptr2 && D.N < R.N)
			return true;
		else if (D.ptr1 == R.ptr1 && D.ptr2 == R.ptr2 && D.N == R.N && D.op < R.op)
			return true;
		return false;
	}
};

static std::map<DeviceTwoVectorsReduceOp, ReduceTmpStorage, DeviceTwoVectorsReduceOp_lt> cacheDeviceTwoVectors;

#pragma endregion


cub::CachingDeviceAllocator g_allocator(true);

#endif //__CUB_STRUCTURES_CACHING__