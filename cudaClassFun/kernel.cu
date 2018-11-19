
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include "curand.h"
#include  <iostream>
#include <chrono>
#include <stdio.h>
class Managed {
public:
	void *operator new(size_t len) {
		void *ptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	}
	void *operator new[](size_t len) {
		void *ptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	}

	void operator delete(void *ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
	void operator delete[](void *ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};

class Dummy : public Managed
{
public:
	int counter;
	double magicNumber;
	Dummy() : counter(0) { ; }
	__device__ void incrementCounter()
	{
		++counter;
		magicNumber = counter * 5 * pow((double)3, __double2int_rd(magicNumber) % 100) + counter * magicNumber + counter * 2 * pow((double)2, __double2int_rd(magicNumber) % 50) + counter * magicNumber;
	}
	void incrementHost()
	{
		++counter;
		magicNumber = counter * 5 * pow(3, ((int)magicNumber % 100)) + counter * magicNumber + counter * 2 * pow(2, ((int)magicNumber % 50)) + counter * magicNumber;
	}
};

__global__ void incrementDummy(Dummy *a)
{
	auto i = threadIdx.x + blockIdx.x * blockDim.x;
	a[i].incrementCounter();
}
int main()
{
    auto constexpr arraySize = 10000000;
	Dummy *arrPtr = new Dummy[arraySize];
	auto start = std::chrono::high_resolution_clock::now();
	incrementDummy <<< (arraySize + 255) / 256, 256 >>> (arrPtr);
	cudaDeviceSynchronize();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed1 = finish - start;
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		std::cout << cudaGetErrorString(err) << std::endl;
	auto start2 = std::chrono::high_resolution_clock::now();
	for (auto i = 0; i < arraySize; ++i)
	{
		arrPtr[i].incrementHost();
	}
	auto finish2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed2 = finish2 - start2;
	for(auto i = 0; i < arraySize; ++i)
	{
		if(arrPtr[i].counter != 2)
		{
			std::cout << arrPtr[i].counter << std::endl;
		}
	}
	std::cout << "Elapsed time: " << elapsed1.count() << " s\n";
	std::cout << "Elapsed time: " << elapsed2.count() << " s\n";
	delete arrPtr;
    return 0;
}