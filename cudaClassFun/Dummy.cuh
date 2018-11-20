#include <cuda_runtime_api.h>

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
	__host__ __device__ Dummy();
	__device__ void incrementCounterDevice();
	__host__ void incrementCounterHostHost();
};