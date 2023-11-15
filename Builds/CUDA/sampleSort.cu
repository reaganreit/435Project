#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <algorithm>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

int THREADS;
int BLOCKS;
int NUM_VALS;

/* Define Caliper region names */
const char* main_region = "main";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";

using namespace std;

__global__ void sampleSort(int* data, int* out, int *splitters, int num_buckets) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    // only smallest thread sorts block
    if (threadIdx.x == 0) {
      // sort each block
      for (int i = 0; i < blockDim.x - 1; ++i) {
        for (int j = 0; j < blockDim.x - i - 1; ++j) {
            if (data[index + j] > data[index + j + 1]) {
                // Swap elements if they are in the wrong order
                int temp = data[index + j];
                data[index + j] = data[index + j + 1];
                data[index + j + 1] = temp;
            }
        }
      }

      // Write the sorted data back to the global memory
      for (int i = 0; i < blockDim.x; ++i) {
          out[blockDim.x * blockIdx.x + i] = data[index + i];
      }
    }
    
}

int main(int argc, char *argv[])
{
    int inputType;
    inputType = atoi(argv[3]);
    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);
    printf("Input type: %d\n", inputType);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();
    
    int *out = (int*)malloc(sizeof(int) * NUM_VALS);

    // initialize data according to inputType
    int* hostData = new int[NUM_VALS];
    for (int i = 0; i < NUM_VALS; ++i) {
        hostData[i] = NUM_VALS-i;
    }
    
    cout << "original arr" << endl;  
    for (int i = 0; i < NUM_VALS; ++i) {
        cout << hostData[i] << " ";
    }
    cout << endl;  
    
    // NUM SPLITTERS should equal # of blocks-1
    int* splitters = new int[3];
    for (int i = 1; i <= 3; ++i) {
        splitters[i-1] = 10 * i;
    }
    
    cout << "splitters" << endl;  
    for (int i = 0; i < 3; ++i) {
        cout << splitters[i] << " ";
    }
    cout << endl;
    // device data
    int* devData, *dout, *dsplitters;
    cudaMalloc((void**)&devData, NUM_VALS * sizeof(int));
    cudaMalloc((void**)&dout, NUM_VALS * sizeof(int));
    cudaMalloc((void**)&dsplitters, 3 * sizeof(int));
    
    // send chunks to device
    cudaMemcpy(devData, hostData, NUM_VALS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dsplitters, splitters, 3 * sizeof(int), cudaMemcpyHostToDevice);
    
    // have device sort and send back samples
    sampleSort<<<BLOCKS, THREADS>>>(devData, dout, dsplitters, 10);
    
    // test send back
    cudaMemcpy(out, dout, sizeof(int) * NUM_VALS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_VALS; ++i) {
        cout << out[i] << " ";
    }
    cout << endl;
    
    // sort samples in device and choose splitters
    
    // have 2d arr. each row is a diff bucket
    
    // send chunks to device w/ splitters
    // each thread puts values in buckets
    
    // send buckets back from device to host and append to global 2d buckets arr
    
    // final sort each row
    
    // append to one array and done!


    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
};

