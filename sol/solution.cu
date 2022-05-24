#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include "helper.h"

using namespace std;

__global__ void countPopulation(unsigned int *device_result, float *lat, float *lon,
    int *pop, float kmRange, int nr_of_cities) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    for (int j = 0; j < nr_of_cities; j ++) {
        float phi1 = (90.f - lat[i]) * DEGREE_TO_RADIANS;
        float phi2 = (90.f - lat[j]) * DEGREE_TO_RADIANS;

        float theta1 = lon[i] * DEGREE_TO_RADIANS;
        float theta2 = lon[j] * DEGREE_TO_RADIANS;

        float cs = sin(phi1) * sin(phi2) * cos(theta1 - theta2) + cos(phi1) * cos(phi2);
        if (cs > 1) {
            cs = 1;
        } else if (cs < -1) {
            cs = -1;
        }

        float rez =  6371.f * acos(cs);
        if (rez <= kmRange) {
            atomicAdd(&device_result[i], pop[j]);
        }
    }
   
}

// sampleFileIO demos reading test files and writing output
void my_sampleFileIO(float kmRange, const char* fileIn, const char* fileOut)
{
    string geon;
    float lat;
    float lon;
    int pop;
    int nr_of_cities = 0;

    float *host_lat_array = 0;
    float *host_lon_array = 0;
    int *host_pop_array = 0;
    unsigned int *host_result = 0;

    host_lat_array = (float *) malloc(sizeof(float) * 1);
    host_lon_array = (float *) malloc(sizeof(float) * 1);
    host_pop_array = (int *) malloc(sizeof(int) * 1);

    float *device_lat_array = 0;
    float *device_lon_array = 0;
    int *device_pop_array = 0;
    unsigned int *device_result = 0;

    ifstream ifs(fileIn);
    ofstream ofs(fileOut);

    while(ifs >> geon >> lat >> lon >> pop)
    {
        host_lat_array[nr_of_cities] = lat;
        host_lon_array[nr_of_cities] = lon;
        host_pop_array[nr_of_cities] = pop;
        nr_of_cities++;

        host_lat_array  = (float *) realloc(host_lat_array, sizeof(float) * (nr_of_cities + 1));
        host_lon_array  = (float *) realloc(host_lon_array, sizeof(float) * (nr_of_cities + 1));
        host_pop_array  = (int *) realloc(host_pop_array, sizeof(int) * (nr_of_cities + 1));
    }

    
    cudaMalloc(&device_lat_array, nr_of_cities * sizeof(float));
    cudaMalloc(&device_lon_array, nr_of_cities * sizeof(float));
    cudaMalloc(&device_pop_array, nr_of_cities * sizeof(int));
    cudaMalloc(&device_result, nr_of_cities * sizeof(unsigned int));
    
    if (device_lat_array == 0 || device_lon_array == 0 || device_result == 0
        || device_pop_array == 0) {
        cout << "[HOST] Couldn't allocate memory\n";
        exit(-1);
    }
    
    cudaMemset(device_result, 0, nr_of_cities);
    cudaMemcpy(device_lat_array, host_lat_array, nr_of_cities * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_lon_array, host_lon_array, nr_of_cities * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_pop_array, host_pop_array, nr_of_cities * sizeof(int), cudaMemcpyHostToDevice);

    const size_t block_size = 1024;
    size_t grid_size = nr_of_cities / block_size;
  
    if (nr_of_cities % block_size) 
        ++grid_size;

    countPopulation<<<grid_size, block_size>>>(device_result, device_lat_array,
        device_lon_array, device_pop_array, kmRange, nr_of_cities);
    host_result = (unsigned int *) malloc (sizeof(unsigned int) * nr_of_cities);
    cudaDeviceSynchronize();

    cudaMemcpy(host_result, device_result, nr_of_cities * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
    for(int i = 0; i < nr_of_cities; i++) {
        ofs << host_result[i] << endl;
    }

    ifs.close();
    ofs.close();

    free(host_lat_array);
    free(host_lon_array);
    free(host_pop_array);
    free(host_result);

    cudaFree(device_lat_array);
    cudaFree(device_lon_array);
    cudaFree(device_pop_array);
    cudaFree(device_result);
}


int main(int argc, char *argv[]) {
    DIE( argc == 1,
         "./accpop <kmrange1> <file1in> <file1out> ...");
    DIE( (argc - 1) % 3 != 0,
         "./accpop <kmrange1> <file1in> <file1out> ...");

    for(int argcID = 1; argcID < argc; argcID += 3) {
        // uncomment this if to check all tests
        if (strcmp(argv[argcID + 1], "../tests/H1.in") != 0) {
            float kmRange = atof(argv[argcID]);
            my_sampleFileIO(kmRange, argv[argcID + 1], argv[argcID + 2]);
        }
    }
}