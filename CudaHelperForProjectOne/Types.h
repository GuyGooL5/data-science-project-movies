#pragma once
#include <cuda_runtime.h>

typedef unsigned int uint;
typedef char d_char;
typedef int d_int;


void CUDA_ASSERT(cudaError_t err, const char* msg);