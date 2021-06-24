#pragma once

#include "Types.h"

class HashMatrix
{
public:
	HashMatrix(d_char* hash, int size,int bufferSize);


	void printHash(int index) const;
	d_char* getMatrix() const;
	int getSize() const { return _size; };
	~HashMatrix();
private:
	const int _size;
	const int _bufSize;
	d_char* _hashMatrix;
};

