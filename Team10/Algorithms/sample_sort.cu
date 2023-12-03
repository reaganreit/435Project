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
  CALI_MARK_BEGIN(data_init);
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
  
  CALI_MARK_END(data_init);
}

void finalSort(int** buckets, int rows) {
    printf("entered final sort\n");
    for (int r = 0; r < rows; ++r) {
        std::sort(buckets[r], buckets[r] + NUM_VALS);
    }
}

void chooseSplitters(int *splitters, int *samples) {
    cout << "entered choose splitters" << endl;
    // sort samples
    cout << "BLOCKS: " << BLOCKS << endl;
    int samplesSize = 5 * BLOCKS;
    cout << "samplesSize: " << samplesSize << endl;
    std::sort(samples, samples + samplesSize);
    
    // choose splitters
    int spacing = std::ceil((float)samplesSize/(float)THREADS);
    int splitterIndex = spacing-1;
    
    for (int i = 0; i < THREADS; i++) {
      if (splitterIndex > samplesSize-1) {
        cout << "BROKE CHOOSE SPLITTERS EARLY AT i= " << i << endl;
        break;     
      }
      // cout << "i: " << i << "index: " << splitterIndex << " samples[index]: " << samples[splitterIndex] << endl; 
      splitters[i] = samples[splitterIndex];
      splitterIndex += spacing;
    }
}


__global__ void chooseSamples(int* data, int *samples, int numBlocks) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    // only smallest thread sorts block
    if (threadIdx.x == 0) {
      // sort each block
      //printf("choose samples data[%d]: %d\n", index, data[index]);
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
      int spacing = blockDim.x / 5;
      int sampleIndex = spacing-1;
      
      for (int i = 0; i < 5; i++) {
        if (index+sampleIndex > blockDim.x * numBlocks) {
          printf("BROKE CHOOSE SAMPLES EARLY AT i=%d\n", i);
          break;
        }
        samples[blockIdx.x * 5 + i] = data[index+sampleIndex];
        // printf("index: %d, chosen sample: %d\n", blockIdx.x * (numBlocks-1) +i, data[index+sampleIndex]);
        sampleIndex += spacing;
      }
    }

    
}

__global__ void sampleSort(int* data, int** buckets, int* splitters, int* flattenedArr, int numSplitters, int numVals) {
    
    // printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index == 0)
      printf("numSplitters: %d", numSplitters);
      
    // each thread checks which bucket they fall into
    int j = 0;
    //printf("index: %d \n", index, j);
    while(j < numSplitters) {  // j being which bucket it should belong to
  			if (j == numSplitters-1) {
          // printf("data[%d]: %d went into buckets[%d][%d]\n", index, data[index], j, index);
          // means it should go in last bucket
          // makes sure that we don't try to access splitters[buckets.size()-1]. will go out of range
          buckets[j][index] = data[index];
          break;
        }
        if(data[index] < splitters[j]) {
          // printf("data[%d]: %d went into buckets[%d][%d]\n", index, data[index], j, index);  
          __syncthreads();      
  				buckets[j][index] = data[index];
          break;
  			}
  			j++;
    }
    
    __syncthreads();
    
    // store bucket values in flattened array. Only have one thread do it
    if (index == 0) {
      printf("entering bucket values\n");
      int arrIndex = 0;
      for (int i = 0; i < numSplitters; ++i) {
          for (int j = 0; j < numVals; ++j) {
              // printf("arrIndex: %d, value: %d\n", arrIndex, buckets[i][j]);
              flattenedArr[arrIndex++] = buckets[i][j];
          }
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
    
    int device = 0;  // Assuming device 0, change if needed

    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes" << std::endl;
    
    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();
    
    CALI_MARK_BEGIN(main_region);

    // host data
    int* hostData = new int[NUM_VALS];
    int *splitters = new int[THREADS]; 
    int *samples = (int*)malloc(sizeof(int) * 5*BLOCKS);  // each block picks out 5 potential splitter candidates
    
    // initialize data according to inputType
    dataInit(hostData, NUM_VALS, inputType);
    
    /*
    cout << "original arr" << endl;  
    for (int i = 0; i < NUM_VALS; ++i) {
        cout << hostData[i] << " ";
    }
    cout << endl;  
    */

    // device data
    int* devData, *dsplitters, *dsamples;
    cudaMalloc((void**)&devData, NUM_VALS * sizeof(int));
    cudaMalloc((void**)&dsplitters, THREADS * sizeof(int));
    cudaMalloc((void**)&dsamples, 5*BLOCKS * sizeof(int));
    cout << "device data" << endl;
    
    // send chunks to device
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(devData, hostData, NUM_VALS * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    cout << "send chunks" << endl;
    
    // have device sort and send back samples
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    chooseSamples<<<BLOCKS, THREADS>>>(devData, dsamples, BLOCKS);
    cudaDeviceSynchronize();
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    cout << "choose samples" << endl;
    
    // receive samples from device
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(samples, dsamples, 5 * BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);
    cout << "receive samples" << endl;
    cudaFree(dsamples);
    /*
    for (int i=0; i<5*BLOCKS; i++) {
      cout << samples[i] << " ";
    }
    cout << endl;
    */
    
    // sort samples and choose splitters
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    chooseSplitters(splitters, samples);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    cout << "choose splitters" << endl;
    
    
    for (int i=0; i<THREADS; i++) {
      cout << splitters[i] << " ";
    }
    cout << endl;
    
    
    // allocate memory for host and device 2d bucket arrays
    int rows = THREADS;
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
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMalloc((void**)&dbuckets, rows * sizeof(int*));
    for (int i = 0; i < rows; ++i) {
        int* d_row;
        cudaMalloc((void**)&d_row, NUM_VALS * sizeof(int));
        cudaMemcpy(d_row, buckets[i], NUM_VALS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dbuckets + i, &d_row, sizeof(int*), cudaMemcpyHostToDevice);
    }
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    // send chunks to device w/ splitters
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(dsplitters, splitters, sizeof(int) * THREADS, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);
    cout << "send chunks" << endl;
    
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    sampleSort<<<BLOCKS, THREADS>>>(devData, dbuckets, dsplitters, dflattenedArr, THREADS, NUM_VALS);
    cudaDeviceSynchronize();
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        // Handle the error appropriately
    }
    printf("cudaErr: \n", cudaErr);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    cout << "sample sort" << endl;
    
    cout << "before malloc" << endl;
    // int *flattenedArr = (int*)malloc(sizeof(int) * (BLOCKS-1)*NUM_VALS);
    int *flattenedArr;
    cudaMallocHost((void**)&flattenedArr, THREADS * NUM_VALS * sizeof(int));
    cout << "after malloc" << endl;
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cout << "before cudaMemcpy" << endl;
    cudaMemcpy(flattenedArr, dflattenedArr, THREADS * NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    cout << "before initialize" << endl;
    
    // initializing unflattened arr
    int** unflattened = new int*[rows];
    for (int i = 0; i < rows; ++i) {
        unflattened[i] = new int[NUM_VALS];
    }
    
    /*
    for (int i = 0; i < THREADS; i++) {
      cout << "i: " << i << endl;
      cout << flattenedArr[i] << endl;
    }
    */
    
    
    cout << "before unflatten" << endl;
    
    // unflatten the arr
    int index = 0;
    for (int i = 0; i < rows; ++i) {
        // cout << "ROW: " << i << endl;
        for (int j = 0; j < NUM_VALS; ++j) {
            //cout << "j: " << j << "index: " << index << endl;
            unflattened[i][j] = flattenedArr[index++];
        }
    }
    
    cout << "finished unflattened" << endl;
    
    // final sort each row
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    finalSort(unflattened, rows);
    cout << "final sort" << endl;
    
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
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    
    cout << "FINAL ARRAY" << endl;
    for (int i = 0; i < NUM_VALS; i++) {
      cout << finalArr[i] << " ";
    }
    
    if (correctnessCheck(finalArr, NUM_VALS)) {
      printf("\nCORRECT");
    } else {
      printf("\nINCORRECT");
    }
    
    
    CALI_MARK_END(main_region);
    
    const char* algorithm = "Sample Sort";
    const char* programmingModel = "CUDA";
    const char* datatype = "int";
    const char* inputTypeStr;
    switch (inputType) {
      case 1:
        inputTypeStr = "Sorted";
        break;
      case 2:
        inputTypeStr = "Reverse Sorted";
        break;
      case 3:
        inputTypeStr = "Random";
        break;
      case 4:
        inputTypeStr = "1% Perturbed";
        break;
      default:
        inputTypeStr = "No input type. Invalid input argument entered";
        break;
    }
    
    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();   
    adiak::clustername();  
    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputTypeStr); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 10); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten");
  
    adiak::value("main", main_region);
    adiak::value("data_init", data_init);
    adiak::value("comm", comm);
    adiak::value("comp", comp);
    adiak::value("comm_large", comm_large);
    adiak::value("comm_small", comm_small);
    adiak::value("comp_large", comp_large);
    adiak::value("comp_small", comp_small);
    adiak::value("correctness_check", correctness_check);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
};

