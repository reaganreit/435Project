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

void chooseSplitters(int *splitters, int *samples) {
    // samples
    int samplesSize = BLOCKS * (BLOCKS-1);
    for (int i = 0; i < samplesSize - 1; ++i) {
      for (int j = 0; j < samplesSize - i - 1; ++j) {
          if (samples[j] > samples[j + 1]) {
              // Swap elements if they are in the wrong order
              int temp = samples[j];
              samples[j] = samples[j + 1];
              samples[j + 1] = temp;
          }
      }
    }
    
    cout << "sorted samples" << endl;
    for (int i = 0; i < samplesSize; ++i) {
        cout << samples[i] << " ";
    }
    cout << endl;
    
    // choose splitters
    int spacing = std::ceil((float)samplesSize/(float)BLOCKS);
    int splitterIndex = spacing-1;
    
    for (int i = 0; i < BLOCKS-1; i++) {
      splitters[i] = samples[splitterIndex];
      splitterIndex += spacing;
    }
}


__global__ void chooseSamples(int* data, int* out, int *samples, int numBlocks) {
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
      
      // choose samples from sorted block
      int spacing = blockDim.x /(numBlocks-1);
      int sampleIndex = spacing-1;
      
      for (int i = 0; i < numBlocks-1; i++) {
        samples[blockIdx.x * (numBlocks-1) + i] = data[index+sampleIndex];
        printf("sample index: %d ", blockIdx.x * (numBlocks-1) + i);
        //printf("index: %d\n", index);
        sampleIndex += spacing;
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

    // host data
    int* hostData = new int[NUM_VALS];
    int *out = (int*)malloc(sizeof(int) * NUM_VALS);
    int *splitters = new int[BLOCKS-1]; 
    int *samples = (int*)malloc(sizeof(int) * (BLOCKS-1)*BLOCKS);  // each block picks out potential splitter candidates
    
    // initialize data according to inputType
    for (int i = 0; i < NUM_VALS; ++i) {
        hostData[i] = NUM_VALS-i;
    }
    cout << "original arr" << endl;  
    for (int i = 0; i < NUM_VALS; ++i) {
        cout << hostData[i] << " ";
    }
    cout << endl;  

    // device data
    int* devData, *dout, *dsplitters, *dsamples;
    cudaMalloc((void**)&devData, NUM_VALS * sizeof(int));
    cudaMalloc((void**)&dout, NUM_VALS * sizeof(int));
    cudaMalloc((void**)&dsplitters, (BLOCKS-1) * sizeof(int));
    cudaMalloc((void**)&dsamples, (BLOCKS-1)*BLOCKS * sizeof(int));
    
    // send chunks to device
    cudaMemcpy(devData, hostData, NUM_VALS * sizeof(int), cudaMemcpyHostToDevice);
    
    // have device sort and send back samples
    chooseSamples<<<BLOCKS, THREADS>>>(devData, dout, dsamples, BLOCKS);
    
    // receive samples from device
    cudaMemcpy(out, dout, sizeof(int) * NUM_VALS, cudaMemcpyDeviceToHost);
    cudaMemcpy(samples, dsamples, (BLOCKS-1) * BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_VALS; ++i) {
        cout << out[i] << " ";
    }
    cout << endl;
    cout << "SAMPLES" << endl;
    for (int i = 0; i < (BLOCKS-1)*BLOCKS; ++i) {
        cout << samples[i] << " ";
    }
    cout << endl;

    // sort samples and choose splitters
    chooseSplitters(splitters, samples);
    cout << "chosen splitters" << endl;
    for (int i = 0; i < BLOCKS-1; ++i) {
        cout << splitters[i] << " ";
    }
    cout << endl;
    
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

