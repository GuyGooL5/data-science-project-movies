
#include "FileLoader.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


FileHandler::FileHandler (FileHandler&& ref) noexcept
{
	file = ref.file;
	ref.file = nullptr;
}

FileHandler FileHandler::load(const char* filename)
{
	FileHandler fl;
	fl.file = fopen(filename, "r");
	return fl;
}

FileHandler FileHandler::write(const char* filename)
{
	FileHandler fl;
	fl.file = fopen(filename, "w");
	return fl;
}

d_char* FileHandler::readFile(size_t lines, size_t lineLength) const {
	int buf = lines * lineLength;

	char* hash = new char[buf];
	fread(hash, sizeof(char), buf, file);
	getc(file);

	d_char* cuda_hash;
	cudaMalloc(&cuda_hash, sizeof(char) * buf);
	cudaMemcpy(cuda_hash, hash, sizeof(char) * buf, cudaMemcpyHostToDevice);

	delete[] hash;
	return cuda_hash;
}

size_t FileHandler::writeFile(d_int* linesArray, size_t size) const
{
	int* tmp = new int[size];
	cudaMemcpy(tmp, linesArray, sizeof(int) * size, cudaMemcpyDeviceToHost);

	size_t blocks = fwrite(tmp, sizeof(int), size, file);
	delete[] tmp;
	return blocks;
}

FileHandler::~FileHandler()
{
	if (file == nullptr) return;
	fclose(file);
}