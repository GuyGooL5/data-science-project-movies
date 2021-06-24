#include "CudaManaged.h"
#include "Types.h"

template<typename T>
inline CudaManaged<T>::CudaManaged(T* pointerToDeviceMemory, size_t size) :_size(size)
{
	device_ptr = pointerToDeviceMemory;
}

template<typename T>
inline size_t CudaManaged<T>::size()
{
	return _size;
}

template<typename T>
inline void* CudaManaged<T>::copyToHost()
{
	T* hostPtr = new T[_size];
	CUDA_ASSERT(cudaMemcpy(hostPtr, device_ptr, _size * sizeof(T), cudaMemcpyDeviceToHost), "Failed to copy CudaManaged<T> to host");

	return hostPtr;
}




template<typename T>
inline CudaManaged<T>::operator T* ()
{
	return device_ptr;
}

template<typename T>
inline CudaManaged<T>::~CudaManaged()
{
	if (device_ptr == nullptr) return;
	CUDA_ASSERT(cudaFree(device_ptr), "Failed to free CudaManaged<T>");
}

template CudaManaged<d_int>;
template CudaManaged<d_char>;
