
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_atomic_functions.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include "FileLoader.h"
//#include "HashMatrix.h"
#include "CudaManaged.h"


#define TCONST "tconst.txt"
#define PRIMARY "D:\\Programming\\ComputerStuff\\Python\\hit-data-science-class\\projectOne\\cuda_data\\primaryTitle"
#define ORIGINAL "D:\\Programming\\ComputerStuff\\Python\\hit-data-science-class\\projectOne\\cuda_data\\originalTitle"
#define MATCHING "D:\\Programming\\ComputerStuff\\Python\\hit-data-science-class\\projectOne\\cuda_data\\the_numbers"

#define NO_MATCH -1

#define MATCHING_LEN 13625
#define PRIMARY_LEN 329887
#define BUFFER_SIZE 32
#define BUFFER_COUNT 35
#define IJ(i,j) i*j

#ifdef _INTELLISENSE_
void atomicAdd(int* ptr, int count);
#endif

__device__ int get_next_block(int* executed)
{
	return atomicAdd(executed, 1);
}


__device__ bool compareStringsDeviceFn(d_char* a, d_char* b, int buf) {
	while (--buf)
		if (a[buf] != b[buf])
			return false;
	return true;
}

__global__ void compareStringsKernel(
	d_char* sourceArray, d_char* referenceArray, d_int* matchingMatrix, d_int* matchingMatrixCount,
	int source_size, int reference_size, int count_buffer, int buffer_size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x,
		y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < source_size && y < reference_size) {
		if (compareStringsDeviceFn(sourceArray + (x * buffer_size), referenceArray + (y * buffer_size), buffer_size)) {
			int j = get_next_block(matchingMatrixCount + x);
			matchingMatrix[x * count_buffer + j] = y;
		}
	}
}


struct MatrixTemp {
	d_int* matrix;
	d_int* count;
	int count_size;
	int buffer_size;
};

/**
 * This function will compare each element in 'a' to each element in 'b' and put the resulting index of a match<br>
 * in a new array that stores the index of 'b' in 'a's matching index for a no match it will put -1.
 *
 * \param a - device char 2D array where each row is of size buffer_size and the length is a_len
 * \param a_len - the length of 'a'
 * \param b - device char 2D array where each row is of size buffer_size and the length is b_len
 * \param b_len - the length of 'b'
 * \param buffer_size - the size of each row in both arrays.
 * \return a device int array of matches.
 */
MatrixTemp findAinB(d_char* a, int a_len, d_char* b, int b_len, int buffer_count, int buffer_size) {
	dim3 threadDim(32, 32);
	dim3 blockDim(a_len / 32 + 1, b_len / 32 + 1);
	d_int* matchingMatrix;
	d_int* matchingMatrixCount;

	CUDA_ASSERT(cudaMalloc(&matchingMatrix, a_len * buffer_count * sizeof(int)), "Failed to allocate memory for matchingMatrix");
	std::cout << "Created Matrix for A [" << a_len * buffer_count << "]" << std::endl;
	CUDA_ASSERT(cudaMemset(matchingMatrix, NO_MATCH, a_len * buffer_count * sizeof(int)), "Failed to set memory of matchingMatrix");


	CUDA_ASSERT(cudaMalloc(&matchingMatrixCount, a_len * sizeof(int)), "Failed to allocate memory to matchingMatrixCount");
	std::cout << "Created Counter Vector for A [" << a_len << "]" << std::endl;

	CUDA_ASSERT(cudaMemset(matchingMatrixCount, 0, a_len * sizeof(int)), "Failed to set memory to matchingMatrixCount");

	compareStringsKernel << <blockDim, threadDim >> > (a, b, matchingMatrix, matchingMatrixCount, a_len, b_len, buffer_count, buffer_size);

	//CUDA_ASSERT(cudaFree(matchingMatrixCount), "Failed to free matchingMatrixCount");

	return { matchingMatrix,matchingMatrixCount,a_len,buffer_count };
}



struct MovieList {
	int size;
	int* items;
};

/**
 * @return -1 for an error, positive integer for a result.
 */
uint getArgNumber(const char* c_str) {
	std::string str = c_str;
	try {
		return std::stoi(str);
	}
	catch (const std::exception& e) {
		std::cout << "Exception occoured while parsing a number: " << std::endl
			<< e.what() << std::endl
			<< "[INPUT]: " << c_str << std::endl
			<< "[rows] argument must be an integer. use -h for help" << std::endl;
		exit(-1);
	}
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		std::cout << "Please supply argumets, use -h for help" << std::endl;
		return 0;
	};


	if (argc == 2 && strcmp(argv[1], "-h")) {
		std::cout << "Invalid argument, use -h for help." << std::endl<<argv[1];
		return 0;
	}
	else if (argc == 2) {
		std::cout
			<< "cudamerge [source] [source_length] [target] [taget_length] [output] [rows=32]" << std::endl
			/*2 */ << "[source] - the source table to match" << std::endl
			/*3 */ << "[source_length] - the length of the source table" << std::endl
			/*4 */ << "[target] - the target table to compare to" << std::endl
			/*5 */ << "[taget_length] - the length of the target table" << std::endl
			/*6 */ << "[output] - the destination file of the matches" << std::endl
			/*7 */ << "[rows] - the number of potential duplicate matches <minimum = 1, maximum = 100> (default is 32)" << std::endl;
		return 0;
	}

	if (argc!=6 && argc!=7) {
		std::cout << "Invalid argumets supplied, use -h for help" << std::endl;
		return 0;
	}


	const char* source = argv[1];
	const char* target = argv[3];
	const char* output = argv[5];

	uint source_length = getArgNumber(argv[2]);
	uint target_length = getArgNumber(argv[4]);
	uint rows = 32;

	if (argc == 7) {
		uint result = getArgNumber(argv[6]);
		if (result < 1 || result>100) {
			std::cout << "[rows] is invalid, use -h for help" << std::endl;
			return 0;
		}
		rows = result;
	}





	FileHandler source_file = FileHandler::load(source);
	CudaManaged<d_char> primaryHashArray(source_file.readFile(source_length, BUFFER_SIZE), source_length);
	d_char* hash = source_file.readFile(source_length, BUFFER_SIZE);

	FileHandler target_file = FileHandler::load(target);
	CudaManaged<d_char> matchingHashArray(target_file.readFile(target_length, BUFFER_SIZE), target_length);

	MatrixTemp matTemp = findAinB(matchingHashArray, matchingHashArray.size(),
		primaryHashArray, primaryHashArray.size(), BUFFER_COUNT, BUFFER_SIZE);

	CudaManaged<d_int> matchingMatrix(matTemp.matrix, matTemp.buffer_size * matTemp.count_size);
	CudaManaged<d_int> matchingMatrixCount(matTemp.count, matTemp.count_size);


	int* hostMatchingMatrix = (int*)matchingMatrix.copyToHost();
	int* hostMatchingMatrixCount = (int*)matchingMatrixCount.copyToHost();

	MovieList* list = new MovieList[matTemp.count_size];

	int max = 0;
	int index = 0;
	for (int i = 0; i < matTemp.count_size; i++) {
		list[i] = { hostMatchingMatrixCount[i],hostMatchingMatrix + i * matTemp.buffer_size };
	}

	FILE* fp = fopen(output, "w");
	for (int i = 0; i < matTemp.count_size; i++) {
		for(int j = 0;j<list[i].size;j++)
			fprintf(fp,"%d ",list[i].items[j]);
		fprintf(fp,"\n");
	}
	fclose(fp);


	//std::cout << "Maximum count,index: ["<<max<<","<<index<<"]"<<std::endl;


	//std::cout << "starting indexing" << std::endl;
	//for (int i = 0; i < MATCHING_LEN; i++) {
	//	std::cout << "Row\t" << i << " count: "<<hostMatchingMatrixCount[i]<<": [";
	//	for (int j = 0; j < BUFFER_COUNT; j++)
	//		std::cout << hostMatchingMatrix[i * BUFFER_COUNT + j] << ",";
	//	std::cout << "]" << std::endl;
	//}

	delete[] hostMatchingMatrix;
	delete[] hostMatchingMatrixCount;

	//FileHandler write1 = FileHandler::write("output");
	//size_t successful_blocks = write1.writeFile(matchingMatrix, matchingMatrix.size());

	//std::cout << "successfully written " << successful_blocks << "/" << matchingMatrix.size() << " blocks";



	return 0;
}