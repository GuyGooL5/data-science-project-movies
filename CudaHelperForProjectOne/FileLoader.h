#pragma once
#include <stdio.h>
#include "Types.h"



class FileHandler

{
public:

	FileHandler(FileHandler&& ref) noexcept;
	static FileHandler load(const char* filename);
	static FileHandler write(const char* filename);
	/** Returns a line (hash/tconst) in a new DEVICE memory block*/
	d_char* readFile(size_t lines, size_t lineLength) const;

	size_t writeFile(d_int* linesArray, size_t size) const;

	~FileHandler();

private:
	FileHandler() {};
	FILE *file=nullptr;
};

