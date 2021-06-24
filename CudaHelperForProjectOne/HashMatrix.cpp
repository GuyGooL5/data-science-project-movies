#include "HashMatrix.h"
#include <cuda_runtime.h>
#include <iostream>

HashMatrix::HashMatrix(d_char* hash, int size,int bufferSize):_size(size),_bufSize(bufferSize)
{
	_hashMatrix = hash;
}

void HashMatrix::printHash(int i) const
{
	if (i >= _size) {
		std::cout << "Illegal array index " << i << std::endl;
		return;
	}


	char* buf = new char[_bufSize+1];
	cudaMemcpy(buf, _hashMatrix + (i * _bufSize), _bufSize * sizeof(char), cudaMemcpyDeviceToHost);
	buf[_bufSize] = '\0';
	std::cout << buf << std::endl;
	delete[] buf;
}

d_char* HashMatrix::getMatrix() const
{
	return _hashMatrix;
}


HashMatrix::~HashMatrix()
{
	cudaFree(_hashMatrix);
}
