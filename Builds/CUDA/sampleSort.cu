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

__global__ void sampleSort(int* data, int** buckets, int* splitters, int* flattenedArr, int numSplitters, int numVals) {
    
    int index = blockDim.x * blockIdx.x + threadIdx.x;
       
    // each thread checks which bucket they fall into
    int j = 0;
    while(j < numSplitters) {  // j being which bucket it should belong to
  			if (j == numSplitters-1) {
          // means it should go in last bucket
          // makes sure that we don't try to access splitters[buckets.size()-1]. will go out of range
          buckets[j][index] = data[index];
          break;
        }
        if(data[index] < splitters[j]) {
  				buckets[j][index] = data[index];
          break;
  			}
  			j++;
    }
    
    // store bucket values in flattened array
    int arrIndex = 0;
    for (int i = 0; i < numSplitters; ++i) {
        for (int j = 0; j < numVals; ++j) {
            flattenedArr[arrIndex++] = buckets[i][j];
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
    
    // allocate memory for host and device 2d bucket arrays
    int rows = BLOCKS-1;
    int** buckets = new int*[rows];
    int** dbuckets;
    int* dflattenedArr;
    cudaMalloc((void**)&dflattenedArr, rows * NUM_VALS * sizeof(int));
    cudaMalloc((void**)&dbuckets, rows * sizeof(int*));
    for (int i = 0; i < rows; ++i) {
        cudaMalloc((void**)&buckets[i], NUM_VALS * sizeof(int));
        cudaMemcpy(dbuckets + i, &(buckets[i]), sizeof(int*), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dbuckets, buckets, rows * sizeof(int*), cudaMemcpyHostToDevice);  // Copy the row pointers to the device
    
    // send chunks to device w/ splitters
    cudaMemcpy(dsplitters, splitters, sizeof(int) * (BLOCKS-1), cudaMemcpyHostToDevice);
    sampleSort<<<BLOCKS, THREADS>>>(devData, dbuckets, dsplitters, dflattenedArr, BLOCKS-1, NUM_VALS);
    
    int *flattenedArr = (int*)malloc(sizeof(int) * (BLOCKS-1)*NUM_VALS);
    cudaMemcpy(flattenedArr, dflattenedArr, (BLOCKS-1) * NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "flattened" << endl;
    for (int i = 0; i < NUM_VALS * (BLOCKS-1); i++) {
      cout << flattenedArr[i] << " ";
    }
    
    int** unflattened = new int*[rows];
    for (int i = 0; i < rows; ++i) {
        unflattened[i] = new int[NUM_VALS];
    }
    
    // unflatten the arr
    int index = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < NUM_VALS; ++j) {
            unflattened[i][j] = flattenedArr[index++];
        }
    }
    
    cout << "unflattened" << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < NUM_VALS; ++j) {
            cout << unflattened[i][j] << " ";
        }
        cout << endl;
    }
    
    
    
    // cudaMemcpy(buckets, dbuckets, rows * sizeof(int*), cudaMemcpyDeviceToHost);
    
    /*
    cout << "buckets" << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < NUM_VALS; j++) {
          cout << buckets[i][j] << " ";
        }
    }
    cout << endl;
    */
    
    // send buckets back from device to host and append to global 2d buckets arr
    
    // final sort each row
    
    // append to one array and done!


    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
};

