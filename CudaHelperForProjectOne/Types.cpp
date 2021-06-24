#pragma once
#include "Types.h"
#include <iostream>

void CUDA_ASSERT(cudaError_t err, const char* msg) {
	if (err == 0) return;
	std::cout << msg << std::endl;
	exit(err);
}
