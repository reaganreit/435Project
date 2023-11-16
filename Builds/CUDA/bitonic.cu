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

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;
int iterator = 0;


const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";
cudaEvent_t biotonic_start, biotonic_end;
cudaEvent_t host_to_device_start, host_to_device_end;
cudaEvent_t device_to_host_start, device_to_host_end;

// Define Caliper regions
const char* main_region = "main";
const char* data_init_region = "data_init";
const char* comm_region = "comm";
const char* comm_large_region = "comm_large";
const char* comp_region = "comp";
const char* comp_small_region = "comp_small";
const char* correctness_check_region = "correctness_check";


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

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
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
      CALI_MARK_BEGIN(comp_small_region); // Start of comp_small region
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
      CALI_MARK_END(comp_small_region); // End of comp_small region

    }
    if ((i&k)!=0) {
      /* Sort descending */
      CALI_MARK_BEGIN(comp_large_region); // Start of comp_large region

      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
      CALI_MARK_END(comp_large_region); // End of comp_large region

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
  
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  cudaEventRecord(host_to_device_start,0);

  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  cudaEventRecord(host_to_device_end,0);
  CALI_MARK_END(cudaMemcpy_host_to_device);
  cudaEventSynchronize(host_to_device_end);
  



  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  CALI_MARK_BEGIN(comm_region);

  
  CALI_MARK_BEGIN(bitonic_sort_step_region);
  CALI_MARK_BEGIN(comp_region);
  cudaEventRecord(biotonic_start,0);
  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {

      if (k <= NUM_VALS / 4) {
        CALI_MARK_BEGIN(comm_small_region); // Communication for small data
      } else {
        CALI_MARK_BEGIN(comm_large_region); // Communication for large data
      }

      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      iterator++;

      if (k <= NUM_VALS / 4) {
        CALI_MARK_END(comm_small_region); // End of comm_small
      } else {
        CALI_MARK_END(comm_large_region); // End of comm_large
      }
    }
  }
  cudaDeviceSynchronize();
  cudaEventRecord(biotonic_end,0);
  CALI_MARK_END(comp_region);
  CALI_MARK_END(bitonic_sort_step_region);
  CALI_MARK_END(comm_region);
  cudaEventSynchronize(biotonic_end);
  
  
  //MEM COPY FROM DEVICE TO HOST
  cudaEventRecord(device_to_host_start,0);

  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  CALI_MARK_END(cudaMemcpy_device_to_host);
  
  cudaEventRecord(device_to_host_end,0);
  cudaEventSynchronize(device_to_host_end);

  cudaFree(dev_values);
}

int main(int argc, char *argv[])
{
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);
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

  //data initilization
  CALI_MARK_BEGIN(data_init_region);
  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);
  CALI_MARK_END(data_init_region);

  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();



  print_elapsed(start, stop);



  // Store results in these variables.
  float effective_bandwidth_gb_s;
  float bitonic_sort_step_time;
  float cudaMemcpy_host_to_device_time;
  float cudaMemcpy_device_to_host_time;

  CALI_MARK_BEGIN(comp_region);
  cudaEventElapsedTime(&bitonic_sort_step_time, biotonic_start, biotonic_end);
  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, host_to_device_start, host_to_device_end);
  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, device_to_host_start, device_to_host_end);
  effective_bandwidth_gb_s = NUM_VALS*4*iterator*sizeof(float)/bitonic_sort_step_time/1e6;
  printf("Effective Bandwidth: %fn", effective_bandwidth_gb_s);
  CALI_MARK_END(comp_region);




  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("num_threads", THREADS);
  adiak::value("num_blocks", BLOCKS);
  adiak::value("num_vals", NUM_VALS);
  adiak::value("program_name", "cuda_bitonic_sort");
  adiak::value("datatype_size", sizeof(float));
  adiak::value("effective_bandwidth (GB/s)", effective_bandwidth_gb_s);
  adiak::value("bitonic_sort_step_time", bitonic_sort_step_time);
  adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
  adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);
  adiak::value("main", main_region);
  adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);


  const char* main_region = "main";
const char* data_init_region = "data_init";
const char* comm_region = "comm";
const char* comm_large_region = "comm_large";
const char* comp_region = "comp";
const char* comp_small_region = "comp_small";
const char* correctness_check_region = "correctness_check";

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();