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

// Cali Regions
const char* main_region = "main";
const char* data_init = "data_init";
const char* comp = "comp";
const char* comm = "comm";
const char* comp_small = "comp_small";
const char* comm_small = "comm_small";
const char* comp_large = "comp_large";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";

using namespace std;

int correctnessCheck(int *arr, int size) {
  CALI_MARK_BEGIN(correctness_check);
  for (int i=0; i<size-1; i++) {
    if (arr[i+1] < arr[i])
      return 0;  // means it's not ordered correctly
  }
  CALI_MARK_END(correctness_check);

  return 1;
}

void dataInit(int *arr, int size, int inputType) {
  int numToSwitch = size / 100;
  int firstIndex, secondIndex;
  switch (inputType) {
    case 1:
      // sorted
      for (int i=0; i<size; i++) {
        arr[i] = i;
      }
      break;
    case 2:
      // reverse sorted
      for (int i=0; i<size; i++) {
        arr[i] = size-i;
      }
      break;
    case 3:
      // randomized
      for (int i=0; i<size; i++) {
        arr[i] = rand() % RAND_MAX;
      }
      break;
    case 4:
      // 1% perturbed
      for (int i=0; i<size; i++) {
        arr[i] = i;
      }
      if (numToSwitch == 0)  // at the very least one value should be switched
        numToSwitch = 1;
      
      for (int i=0; i<numToSwitch; i++) {
        firstIndex = rand() % size;
        secondIndex = rand() % size;
        //printf("first index: %d, second index: %d\n", firstIndex, secondIndex);
        while (firstIndex == secondIndex) {
          secondIndex = rand() % size;
        } 
        std::swap(arr[firstIndex], arr[secondIndex]); 
      }
      break;
    default:
      printf("THAT'S NOT A VALID INPUT TYPE");
      break;
  }
}

void finalSort(int** buckets, int rows) {
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < NUM_VALS - 1; ++i) {
          for (int j = 0; j < NUM_VALS - i - 1; ++j) {
              if (buckets[r][j] > buckets[r][j + 1]) {
                  // Swap elements if they are in the wrong order
                  int temp = buckets[r][j];
                  buckets[r][j] = buckets[r][j + 1];
                  buckets[r][j + 1] = temp;
              }
          }
        }
    }
}

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
    
    // choose splitters
    int spacing = std::ceil((float)samplesSize/(float)BLOCKS);
    int splitterIndex = spacing-1;
    
    for (int i = 0; i < BLOCKS-1; i++) {
      splitters[i] = samples[splitterIndex];
      splitterIndex += spacing;
    }
}


__global__ void chooseSamples(int* data, int *samples, int numBlocks) {
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
        sampleIndex += spacing;
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
    int *splitters = new int[BLOCKS-1]; 
    int *samples = (int*)malloc(sizeof(int) * (BLOCKS-1)*BLOCKS);  // each block picks out potential splitter candidates
    
    // initialize data according to inputType
    dataInit(hostData, NUM_VALS, inputType);
    
    cout << "original arr" << endl;  
    for (int i = 0; i < NUM_VALS; ++i) {
        cout << hostData[i] << " ";
    }
    cout << endl;  

    // device data
    int* devData, *dsplitters, *dsamples;
    cudaMalloc((void**)&devData, NUM_VALS * sizeof(int));
    cudaMalloc((void**)&dsplitters, (BLOCKS-1) * sizeof(int));
    cudaMalloc((void**)&dsamples, (BLOCKS-1)*BLOCKS * sizeof(int));
    
    // send chunks to device
    cudaMemcpy(devData, hostData, NUM_VALS * sizeof(int), cudaMemcpyHostToDevice);
    
    // have device sort and send back samples
    chooseSamples<<<BLOCKS, THREADS>>>(devData, dsamples, BLOCKS);
    
    // receive samples from device
    cudaMemcpy(samples, dsamples, (BLOCKS-1) * BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    // sort samples and choose splitters
    chooseSplitters(splitters, samples);
    
    // allocate memory for host and device 2d bucket arrays
    int rows = BLOCKS-1;
    int** buckets = new int*[rows];
    int** dbuckets;
    int* dflattenedArr;
    cudaMalloc((void**)&dflattenedArr, rows * NUM_VALS * sizeof(int));
    for (int i = 0; i < rows; ++i) {
        buckets[i] = new int[NUM_VALS];
    }
    
    // initalize buckets with -1 so we know what to remove later
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < NUM_VALS; j++) {
        buckets[i][j] = -1;
      }
    }
    
    // Allocate device memory for the 2D array
    cudaMalloc((void**)&dbuckets, rows * sizeof(int*));
    for (int i = 0; i < rows; ++i) {
        int* d_row;
        cudaMalloc((void**)&d_row, NUM_VALS * sizeof(int));
        cudaMemcpy(d_row, buckets[i], NUM_VALS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dbuckets + i, &d_row, sizeof(int*), cudaMemcpyHostToDevice);
    }
    
    // send chunks to device w/ splitters
    cudaMemcpy(dsplitters, splitters, sizeof(int) * (BLOCKS-1), cudaMemcpyHostToDevice);
    sampleSort<<<BLOCKS, THREADS>>>(devData, dbuckets, dsplitters, dflattenedArr, BLOCKS-1, NUM_VALS);
    
    int *flattenedArr = (int*)malloc(sizeof(int) * (BLOCKS-1)*NUM_VALS);
    cudaMemcpy(flattenedArr, dflattenedArr, (BLOCKS-1) * NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // initializing unflattened arr
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
    
    // final sort each row
    finalSort(unflattened, rows);
    
    // append to one array and done!
    int* finalArr = new int[NUM_VALS];
    int finalArrIndex = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < NUM_VALS; ++j) {
            if (unflattened[i][j] != -1) {
              finalArr[finalArrIndex++] = unflattened[i][j];
            }
        }
        
    }
    
    cout << "FINAL ARRAY" << endl;
    for (int i = 0; i < NUM_VALS; i++) {
      cout << finalArr[i] << " ";
    }
    
    if (correctnessCheck(finalArr, NUM_VALS)) {
      printf("\nCORRECT");
    } else {
      printf("\nINCORRECT");
    }

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
};

