#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <time.h>
#include <stdlib.h>

int SIZE;
int THREADSIZE;
int BLOCKSIZE; // ((SIZE-1)/THREADSIZE + 1) 
#define RADIX 10

//Caliper regions
const char* main_region = "main";
const char* data_init = "data_init";
const char* comp = "comp";
const char* comm = "comm";
const char* comp_large = "comp_large";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";

int correctness_check(int arr[], int size) {
  CALI_MARK_BEGIN(correctness_check);
  for (int i=0; i<size-1; i++) {
    if (arr[i+1] < arr[i])
      return 0;  // means it's not ordered correctly
  }
  CALI_MARK_END(correctness_check);

  return 1;
}

__global__ void copyKernel(int * inArray, int * semiSortArray, int arrayLength){

    int index   = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < arrayLength){
        inArray[index]      = semiSortArray[index];
    }
}

__global__ void histogramKernel(int * inArray, int * outArray, int * radixArray, int arrayLength, int significantDigit){
    extern __shared__ int sharedData[]; 

    int* inArrayShared = sharedData;
    int* radixArrayShared = inArrayShared + blockDim.x;
    //__shared__ int inArrayShared[THREADSIZE];
    __shared__ int outArrayShared[RADIX];
    //__shared__ int radixArrayShared[THREADSIZE];

    int index   = blockIdx.x * blockDim.x + threadIdx.x;
    int thread  = threadIdx.x;
    int blockIndex  = blockIdx.x * RADIX;

    int radix;
    int arrayElement;
    int i;

    if(thread ==  0){
        for(i =0; i < RADIX; i ++){
            outArrayShared[i]       = 0;
        }
    }

    if(index < arrayLength){
        inArrayShared[thread]       = inArray[index];
    }

    __syncthreads();

    if(index < arrayLength)
    {   
        arrayElement            = inArrayShared[thread];
        radix               = ((arrayElement/significantDigit) % 10);
        radixArrayShared[thread]    = radix;

        atomicAdd(&outArrayShared[radix], 1);
    }

    if(index < arrayLength){
        radixArray[index]       = radixArrayShared[thread];
    }

    if(thread == 0){
        for(i =0; i < RADIX; i ++){
            outArray[blockIndex + i]        = outArrayShared[i];
        }
    }
}

__global__ void combineBucket(int * blockBucketArray, int * bucketArray, int BLOCKSIZE){

    __shared__ int bucketArrayShared[RADIX];

    int index   = blockIdx.x * blockDim.x + threadIdx.x;

    int i;

    bucketArrayShared[index]    = 0;

    for(i = index; i < RADIX*BLOCKSIZE; i=i+RADIX){
        atomicAdd(&bucketArrayShared[index], blockBucketArray[i]);      
    } 

    bucketArray[index]      = bucketArrayShared[index];
}


__global__ void indexArrayKernel(int * radixArray,  int * bucketArray, int * indexArray, int arrayLength, int significantDigit){

    int index   = blockIdx.x * blockDim.x + threadIdx.x;

    int i;
    int radix;
    int pocket;

    if(index < RADIX){

        for(i = 0; i < arrayLength; i++){

            radix           = radixArray[arrayLength -i -1];
            if(radix == index){
                pocket              = --bucketArray[radix];
                indexArray[arrayLength -i -1]   = pocket;       
            }
        }
    }
}

__global__ void semiSortKernel(int * inArray, int * outArray, int* indexArray, int arrayLength, int significantDigit){

    int index   = blockIdx.x * blockDim.x + threadIdx.x;

    int arrayElement;
    int arrayIndex;

    if(index < arrayLength){
        arrayElement            = inArray[index];
        arrayIndex          = indexArray[index];
        outArray[arrayIndex]        = arrayElement;
    }



}

void printArray(int * array, int size){
    int i;
    printf("[ ");
    for (i = 0; i < size; i++)
        printf("%d ", array[i]);
    printf("]\n");
}

int findLargestNum(int * array, int size){
    int i;
    int largestNum = -1;
    for(i = 0; i < size; i++){
        if(array[i] > largestNum)
            largestNum = array[i];
    }
    return largestNum;
}


void cudaScanThrust(int* inarray, int arr_length, int* resultarray) {

        int length = arr_length;

    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
        thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

        cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

        thrust::inclusive_scan(d_input, d_input + length, d_output);

        cudaThreadSynchronize();

        cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

        thrust::device_free(d_input);
        thrust::device_free(d_output);
}

void radixSort(int * array, int size, int BLOCKSIZE){

    double startTime;
    double endTime;
    double duration;

    int significantDigit    = 1;

    int threadCount;
    int blockCount;

    threadCount             = THREADSIZE;
    blockCount          = BLOCKSIZE;

    int * outputArray;
    int * inputArray;
    int * radixArray;
    int * bucketArray;
    int * indexArray;
    int * semiSortArray;
    int * blockBucketArray;

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    cudaMalloc((void **)& inputArray, sizeof(int)*size);
    cudaMalloc((void **)& indexArray, sizeof(int)*size);
    cudaMalloc((void **)& radixArray, sizeof(int)*size);
    cudaMalloc((void **)& outputArray, sizeof(int)*size);
    cudaMalloc((void **)& semiSortArray, sizeof(int)*size);
    cudaMalloc((void **)& bucketArray, sizeof(int)*RADIX);
    cudaMalloc((void **)& blockBucketArray, sizeof(int)*RADIX*BLOCKSIZE);   

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(inputArray, array, sizeof(int)*size, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    int largestNum;
    thrust::device_ptr<int>d_in     = thrust::device_pointer_cast(inputArray);
    thrust::device_ptr<int>d_out;
    d_out = thrust::max_element(d_in, d_in + size);
    largestNum      = *d_out;   
    printf("\tLargestNumThrust : %d\n", largestNum);

    //startTime   = CycleTimer::currentSeconds(); 

    while (largestNum / significantDigit > 0){

        int bucket[RADIX] = { 0 };
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);

        cudaMemcpy(bucketArray, bucket, sizeof(int)*RADIX, cudaMemcpyHostToDevice);

        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        histogramKernel<<<blockCount, threadCount>>>(inputArray, blockBucketArray, radixArray, size, significantDigit);  
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);   
        cudaThreadSynchronize();

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        combineBucket<<<1, RADIX>>>(blockBucketArray,bucketArray, BLOCKSIZE);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp); 
        cudaThreadSynchronize();            

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        cudaScanThrust(bucketArray, RADIX, bucketArray); 
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);   
        cudaThreadSynchronize();

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        indexArrayKernel<<<blockCount, threadCount>>>(radixArray, bucketArray, indexArray, size, significantDigit);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
        cudaThreadSynchronize();

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        semiSortKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, indexArray, size, significantDigit);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
        cudaThreadSynchronize();

        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        copyKernel<<<blockCount, threadCount>>>(inputArray, semiSortArray, size);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
        cudaThreadSynchronize();


        significantDigit *= RADIX;

    }

    //endTime     = CycleTimer::currentSeconds();
    //duration    = endTime - startTime;
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(array, semiSortArray, sizeof(int)*size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    printf("Duration : %.3f ms\n", 1000.f * duration);

    cudaFree(inputArray);
    cudaFree(indexArray);
    cudaFree(radixArray);
    cudaFree(bucketArray);
    cudaFree(blockBucketArray);
    cudaFree(outputArray);
    cudaFree(semiSortArray);
}

int main(int argc, char **argv){

    printf("\n\nRunning Radix Sort Example in C!\n");
    printf("----------------------------------\n");

    CALI_MARK_BEGIN(main_region);

    THREADSIZE = atoi(argv[1]);
    SIZE = atoi(argv[2]);
    BLOCKSIZE = ((SIZE-1)/THREADSIZE + 1); //SIZE / THREADSIZE;
    int size = SIZE;
    int* array;
    int list[size];

    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);
    srand(time(NULL));
    //srand(time(NULL) * (id + 1));

    /*for(int i =0; i < size; i++){    //this is decreasing for later
        list[i]     = SIZE -i;
    }*/ 
    
    for(int i =0; i < size; i++){    
        list[i]     = rand() % 1000;
    }
    CALI_MARK_END(data_init);

    array = &list[0];
    printf("\nUnsorted List: ");
    printArray(array, size);

    radixSort(array, size, BLOCKSIZE);

    printf("\nSorted List:");
    printArray(array, size);

    correctness_check(array, size)

    printf("\n");

    return 0;

    CALI_MARK_END(main_region);

    const char* algorithm ="RadixSort";
    const char* programmingModel = "CUDA"; 
    const char* datatype = "int"; 
    int sizeOfDatatype =4;
    int inputSize =1000; 
    const char* inputType= "Random";
    int num_threads = size; 
    int num_blocks = BLOCKSIZE;
    int group_number =10;
    const char* implementation_source = "Online";

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    adiak::value("data_init", data_init);
    adiak::value("comm", comm);
    adiak::value("comp", comp);
    adiak::value("comm_large", comm_large);
    adiak::value("comp_large", comp_large);
    adiak::value("correctness_check", correctness_check); 

    mgr.stop();
    mgr.flush();
}