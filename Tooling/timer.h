#ifndef __Timer_H__
#define __Timer_H__

#pragma region abstract timer

#pragma once
class Timer
{
public:
	Timer();
	virtual ~Timer() = 0;
	virtual void start() = 0;
	virtual void stop() = 0;
	virtual double seconds() const;
};

#pragma endregion abstract timer

#endif // !__Timer_H__

#ifndef __Cuda_Timer_H__
#define __Cuda_Timer_H__

#pragma region cuda timer

#pragma once

#ifdef CUDA_TIMER
#include <cuda_runtime.h>
#include <include\cuda.h>

class CudaTimer :
	public Timer
{
private:
	cudaEvent_t startingTime, endingTime;

public:
	CudaTimer()
	{
		cudaEventCreate(&startingTime);
		cudaEventCreate(&endingTime);
	}

	~CudaTimer()
	{
		cudaEventDestroy(startingTime);
		cudaEventDestroy(endingTime);
	}

	void start()
	{
		cudaEventRecord(startingTime, 0);
		cudaEventSynchronize(startingTime);
	}

	void stop()
	{
		cudaDeviceSynchronize();
		cudaEventRecord(endingTime, 0);
		cudaEventSynchronize(endingTime);
	}

	void start(cudaStream_t stream)
	{
		cudaEventRecord(startingTime, stream);
		cudaEventSynchronize(startingTime);
	}

	void stop(cudaStream_t stream)
	{
		cudaDeviceSynchronize();
		cudaEventRecord(endingTime, stream);
		cudaEventSynchronize(endingTime);
	}

	double seconds() const
	{
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, startingTime, endingTime);
		return (double)(elapsedTime / 1000);
	}
};
#endif

#pragma endregion cuda timer

#endif // !__Cuda_Timer_H__

#ifndef __Omp_Timer_H__
#define __Omp_Timer_H__

#pragma region omp timer

#pragma once
#include <omp.h>

class OmpTimer :
	public Timer
{

private:
	double startingTime, endingTime;

public:

	OmpTimer() : startingTime(0), endingTime(0) {}
	~OmpTimer() {}

	void start()
	{
		this->startingTime = omp_get_wtime();
	}

	void stop()
	{
		this->endingTime = omp_get_wtime();
	}

	double seconds() const
	{
		return (double)(this->endingTime - this->startingTime);
	}

};

#pragma endregion omp timer

#endif // !__Timer_H__

#ifndef __System_Timer_H__
#define __System_Timer_H__

#pragma once

#pragma region windows timer
#if defined(WIN32) || defined(_WIN64)

#define NOMINMAX
#include <Windows.h>

class SystemTimer :
	public Timer
{
private:
	LARGE_INTEGER startingTime, endingTime, frequency;

public:
	SystemTimer() :startingTime({ 0 }), endingTime({ 0 }), frequency({ 0 }) {}
	~SystemTimer() {}
	void start()
	{
		QueryPerformanceFrequency(&(this->frequency));
		QueryPerformanceCounter(&(this->startingTime));
	}

	void stop()
	{
		QueryPerformanceCounter(&(this->endingTime));
	}

	double seconds() const 
	{
		return ((double)((this->endingTime.QuadPart - this->startingTime.QuadPart) * 1000000 / frequency.QuadPart)) / 1000000;
	}
};

#endif

#pragma endregion windows timer

#pragma region linux timer

#ifdef __unix__   

//TODO : implement Timer based on unix system

class SystemTimer :
	public Timer
{
public:
	SystemTimer();
	~SystemTimer();
	void start();
	void stop();
	double seconds() const;
}

#endif

#pragma endregion linux timer

#endif // !__System_Timer_H__

