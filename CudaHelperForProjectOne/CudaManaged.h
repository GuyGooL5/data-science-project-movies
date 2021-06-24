#pragma once
#include <cuda_runtime.h>
template <typename T>
class CudaManaged
{
public:
	CudaManaged(T* pointerToDeviceMemory, size_t size);
	size_t size();

	void* copyToHost();

	operator T* ();
	~CudaManaged();
private:
	T* device_ptr;
	const size_t _size;
};
