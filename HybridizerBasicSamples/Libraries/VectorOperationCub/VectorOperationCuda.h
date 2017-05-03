#define CUB_STDERR

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cub/util_allocator.cuh"
#include "cub/device/device_reduce.cuh"
#include "test/test_util.h"
#include "cub/iterator/transform_input_iterator.cuh"
#include <map>

#include "CubStructuresCaching.h"

namespace hybridizer {
	template<typename T> struct complex_data {};
	template<> struct complex_data<float> { typedef float2 inner_type;	};
	template<> struct complex_data<double> { typedef double2 inner_type; };

	template<typename T>
	struct complex {
		complex_data<T>::inner_type data;
		__host__ __device__ __forceinline__ complex& operator=(const complex& r)
		{
			data.x = r.data.x;
			data.y = r.data.y;
			return *this;
		}
	};

	template<typename T>
	__host__ __device__ __forceinline__ complex<T> operator*(const complex<T>& l, const complex<T>& r) {
		return{ l.data.x * r.data.x - l.data.y * r.data.y, l.data.x * r.data.y + l.data.y * r.data.x };
	}

	template<typename T>
	__host__ __device__ __forceinline__ complex<T> operator*(const complex<T>& l, const T& r) {
		return{ l.data.x * r, l.data.y * r };
	}

	template<typename T>
	__host__ __device__ __forceinline__ complex<T> operator*(const T& l, const complex<T>& r) {
		return{ l * r.data.x, l * r.data.y };
	}

	template<typename T>
	__host__ __device__ __forceinline__ complex<T> operator+(const complex<T>& l, const complex<T>& r) {
		return{ l.data.x + r.data.x, l.data.y + r.data.y };
	}

	template<typename T>
	__host__ __device__ __forceinline__ complex<T> operator+(const complex<T>& l, const T& r) {
		return{ l.data.x + r, l.data.y };
	}

	template<typename T>
	__host__ __device__ __forceinline__ complex<T> operator+(const T& l, const complex<T>& r) {
		return { l + r.data.x, r.data.y };
	}

	template<typename T>
	struct Pair {
		T a, b;
		__host__ __device__ __forceinline__ Pair(const T& aa, const T& bb) : a(aa), b(bb) {}

	};

	template<typename T>
	struct PairIterator {
		T* a;
		T* b;
		__host__ __device__ __forceinline__ PairIterator(T* itr1, T* itr2): a(itr1), b(itr2) {}
		__host__ __device__ __forceinline__ Pair<T> operator[](const int& i) const
		{
			return{ a[i], b[i] };
		}
		__host__ __device__ __forceinline__ PairIterator<T> operator+(const int& i) const
		{
			return{ a + i, b + i };
		}
	};

	template<typename T>
	struct PairIterator_lt {
		bool operator()(const PairIterator<T> & D, const PairIterator<T> & R) const
		{
		
			return false;
		}
	};

	template<typename T>
	struct Product {
		__host__ __device__ __forceinline__ T operator()(const Pair<T>& p) const {
			return p.a * p.b;
		}
	};

	template<typename T>
	T ReduceSum(T * input, size_t N)
	{
		printf("%d\n", iscomplex);
		T h_out = 0;
		DeviceSingleVectorReduceOp key = { (size_t)input, N, ReduceOperation::Sum };
		ReduceTmpStorage storage;

		auto it = cacheDeviceOneVector.find(key);
		if (it == cacheDeviceOneVector.end()) {
			CubDebugExit(cub::DeviceReduce::Sum(storage.temp_storage, storage.temp_storage_bytes, input, (T*)storage.d_out, N));
			CubDebugExit(g_allocator.DeviceAllocate(&storage.temp_storage, storage.temp_storage_bytes));
			CubDebugExit(g_allocator.DeviceAllocate((void**)&storage.d_out, sizeof(T)));
			cacheDeviceOneVector[key] = storage;
			printf("not found\n");
		}
		else {
			storage = it->second;
			printf("found\n");
		}

		CubDebugExit(cub::DeviceReduce::Sum(storage.temp_storage, storage.temp_storage_bytes, input, (T*)storage.d_out, N));
		CubDebugExit(cudaMemcpy(&h_out, storage.d_out, sizeof(T), cudaMemcpyDeviceToHost));

		return h_out;
	}

	template<typename T>
	T ScalarProduct(T* vec1, T* vec2, size_t N)
	{
		T h_res;

		PairIterator<T> p(vec1, vec2);
		Product<T> conversion;
		DeviceTwoVectorsReduceOp key = { (size_t) vec1, (size_t) vec2, N, ReduceOperation::Sum };
		ReduceTmpStorage storage;
		cub::TransformInputIterator<T, Product<T>, PairIterator<T>> new_iter(p, conversion);

		auto it = cacheDeviceTwoVectors.find(key);
		if (it == cacheDeviceTwoVectors.end()) {

			CubDebugExit(cub::DeviceReduce::Sum(storage.temp_storage, storage.temp_storage_bytes, new_iter, (T*)storage.d_out, (int) N));
			CubDebugExit(g_allocator.DeviceAllocate(&storage.temp_storage, storage.temp_storage_bytes));
			CubDebugExit(g_allocator.DeviceAllocate((void**)&storage.d_out, sizeof(T)));
			cacheDeviceTwoVectors[key] = storage;
			printf("not found\n");
		}
		else {
			storage = it->second;
			printf("found\n");
		}

		CubDebugExit(cub::DeviceReduce::Sum(storage.temp_storage, storage.temp_storage_bytes, new_iter, (T*) storage.d_out, (int)N));
		CubDebugExit(cudaMemcpy(&h_res, storage.d_out, sizeof(T), cudaMemcpyDeviceToHost));

		return h_res;
	}
}





