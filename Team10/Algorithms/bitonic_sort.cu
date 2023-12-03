/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <random>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;
int inputType;
int iterator = 0;

// Define Caliper regions
const char* main_region = "main";
const char* data_init = "data_init";
const char* comp = "comp";
const char* comm = "comm";
const char* comp_large = "comp_large";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";

const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";
cudaEvent_t biotonic_start, biotonic_end;
cudaEvent_t host_to_device_start, host_to_device_end;
cudaEvent_t device_to_host_start, device_to_host_end;


void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

float random_float_in_range(float minValue, float maxValue) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(minValue, maxValue);

    return dis(gen);
}

void array_fill(float *arr, int length, int inputType)
{
  srand(time(NULL));
  int i;
  switch (inputType) {
    case 1: // Sorted array
      for (i = 0; i < length; ++i) {
        arr[i] = (float)i; // Assuming float array, filling with ascending values
      }
      break;
    case 2: // Reversed sorted
      for (i = 0; i < length; ++i) {
        arr[i] = (float)(length - 1 - i); // Filling with descending values
      }
      break;
    case 3: // Random
      for (i = 0; i < length; ++i) {
        arr[i] = random_float(); // Filling with random float values
      }
      break;
    case 4:
      for (int i = 0; i < length; ++i) {
        arr[i] = random_float_in_range(0.0f, 100.0f);  // Filling with random float values within a range
      }
      break;
      
    default:
      // Handle invalid inputType, maybe fill it randomly by default
      printf("THAT'S NOT A VALID INPUT TYPE");
      break;
  }
}

void is_sorted(float *arr, int length) {
    for (int i = 0; i < length - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            printf("Array is not sorted");
            break; // Array is not sorted
        }
    }
    printf("Array is sorted. Sorted YAY!");
    // Array is sorted
}


__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
  
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  
  //MEM COPY FROM HOST TO DEVICE
  
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  cudaEventRecord(host_to_device_start,0);

  

  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  

  cudaEventRecord(host_to_device_end,0);
  CALI_MARK_END(cudaMemcpy_host_to_device);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);
  cudaEventSynchronize(host_to_device_end);
  

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  
  // CALI_MARK_BEGIN(bitonic_sort_step_region);
  cudaEventRecord(biotonic_start,0);

  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);
  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      iterator++;
    }
  }
  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);
  
  cudaDeviceSynchronize();
  cudaEventRecord(biotonic_end,0);
  // CALI_MARK_END(bitonic_sort_step_region);
  cudaEventSynchronize(biotonic_end);
  
  
  //MEM COPY FROM DEVICE TO HOST
  cudaEventRecord(device_to_host_start,0);
  
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);

  CALI_MARK_END(cudaMemcpy_device_to_host);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);
  
  cudaEventRecord(device_to_host_end,0);
  cudaEventSynchronize(device_to_host_end);

  cudaFree(dev_values);
}

int main(int argc, char *argv[])
{
  CALI_MARK_BEGIN(main_region);
  if (argc < 4) {
    printf("Usage: %s <num_threads> <num_vals> <inputType>\n", argv[3]);
    return -1;
  }
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;
  int inputType = atoi(argv[3]);


  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);
  printf("Input Type %d\n", inputType);
  // bitonic_sort_step_region

  cudaEventCreate(&biotonic_start);
  cudaEventCreate(&biotonic_end);
  cudaEventCreate(&host_to_device_start);
  cudaEventCreate(&host_to_device_end);
  cudaEventCreate(&device_to_host_start);
  cudaEventCreate(&device_to_host_end);
  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  clock_t start, stop;

  CALI_MARK_BEGIN(data_init);
  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS, inputType);
  CALI_MARK_END(data_init);

  printf("Before:  ");
  // array_print(values, NUM_VALS);


  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);
  printf("After:  ");
  // array_print(values, NUM_VALS);

  CALI_MARK_BEGIN(correctness_check);
  is_sorted(values, NUM_VALS);
  CALI_MARK_END(correctness_check);


  // Store results in these variables.
  float effective_bandwidth_gb_s;
  float bitonic_sort_step_time;
  float cudaMemcpy_host_to_device_time;
  float cudaMemcpy_device_to_host_time;

  cudaEventElapsedTime(&bitonic_sort_step_time, biotonic_start, biotonic_end);
  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, host_to_device_start, host_to_device_end);
  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, device_to_host_start, device_to_host_end);
  effective_bandwidth_gb_s = NUM_VALS*4*iterator*sizeof(float)/bitonic_sort_step_time/1e6;
  printf("Effective Bandwidth: %fn", effective_bandwidth_gb_s);
  
  CALI_MARK_END(main_region);

  const char* algorithm ="BitonicSort";
  const char* programmingModel = "CUDA"; 
  const char* datatype = "int"; 
  int inputSize = NUM_VALS; 
  int group_number =10;
  const char* implementation_source = "Online"; 
  
  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
  adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
  adiak::value("num_threads", THREADS);
  adiak::value("num_blocks", BLOCKS);
  adiak::value("num_vals", NUM_VALS);
  adiak::value("program_name", "cuda_bitonic_sort");
  adiak::value("inputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("datatype_size", sizeof(float));
  adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
  adiak::value("effective_bandwidth (GB/s)", effective_bandwidth_gb_s);
  adiak::value("bitonic_sort_step_time", bitonic_sort_step_time);
  adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
  adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);
  adiak::value("data_init", data_init);
  adiak::value("comm", comm);
  adiak::value("comp", comp);
  adiak::value("comm_large", comm_large);
  adiak::value("comp_large", comp_large);
  adiak::value("correctness_check", correctness_check);

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}