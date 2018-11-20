#include "Dummy.cuh"
#include <cmath>
__host__ __device__ Dummy::Dummy() : counter(0) { ; }
__device__ void Dummy::incrementCounterDevice()
{
	++counter;
	magicNumber = counter * 5 * pow((double)3, __double2int_rd(magicNumber) % 100) + counter * magicNumber + counter * 2 * pow((double)2, __double2int_rd(magicNumber) % 50) + counter * magicNumber;
}
__host__ void Dummy::incrementCounterHostHost()
{
	++counter;
	magicNumber = counter * 5 * pow(3, ((int)magicNumber % 100)) + counter * magicNumber + counter * 2 * pow(2, ((int)magicNumber % 50)) + counter * magicNumber;
}