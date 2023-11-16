#include <stdio.h>
#include <stdlib.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <algorithm>
using namespace std;

/* Define Caliper region names */
const char* mainCali = "main";
const char* dataInit = "data_init";
const char* correctnessCheck = "correctness_check ";
const char* comm = "comm";
const char* commLarge = "comm_large";
const char* comp = "comp";
const char* compLarge = "comp_large";

void data_initialization(int arr[], int size, int inputType) {
    
    int numToSwitch = size / 100;
    int firstIndex, secondIndex;

    switch (inputType)
    {
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
            if (numToSwitch == 0) {  // at the very least one value should be switched
                numToSwitch = 1;
            }
            
            for (int i=0; i<numToSwitch; i++) {
                firstIndex = rand() % size;
                secondIndex = rand() % size;
                while (firstIndex == secondIndex) {
                    secondIndex = rand() % size;
                } 
                swap(arr[firstIndex], arr[secondIndex]); 
            }
            break;
        default:
            printf("THAT'S NOT A VALID INPUT TYPE\n");
            break;
    }
}

//Device function for recursive Merge
__device__ void Merge(int *arr, int *temp, int left, int mid, int right) {
    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right) 
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < mid)
        temp[k++] = arr[i++];
    while (j < right)
        temp[k++] = arr[j++];

    for (int x = left; x < right; x++)
        arr[x] = temp[x];
}

//GPU Kernel for Merge Sort
__global__ void MergeSortGPU(int* arr, int* temp, int n, int width) 
{    
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int left = tid * width;
    int middle = min(left + width / 2, n);  // Ensure middle is within array bounds
    int right = min(left + width, n);      // Ensure right is within array bounds

    if (left < n && middle < n) 
    {
        Merge(arr, temp, left, middle, right);
        __syncthreads();
    }
}

void printArray(int *arr, int size)
{
    printf("Array: ");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

bool isSorted(int *arr, int size)
{
    CALI_MARK_BEGIN(correctnessCheck);
    for (int i = 0; i < size - 1; i++)
    {
        if (arr[i] > arr[i + 1])
        {
            CALI_MARK_END(correctnessCheck);
            return false;
        }
    }
    CALI_MARK_END(correctnessCheck);
    return true;
}

int main(int argc, char **argv)
{
    CALI_MARK_BEGIN(mainCali);
    int numThreads = atoi(argv[1]);
    int SIZE = atoi(argv[2]);
    int sortType = atoi(argv[3]);
    int numBlocks = (SIZE + numThreads - 1) / numThreads;

    const char* sortName;
    switch (sortType)
    {
    case 0:
        sortName = "Random";
        break;
    case 1:
        sortName = "Sorted";
        break;
    case 2:
        sortName = "ReverseSorted";
        break;
    case 3:
        sortName = "1%Perturbed";
        break;
    }
    

    int *h_arr = new int[SIZE]; //store and manipulate data on CPU
    int *d_arr; //store data on GPU
    int initialize_arr[SIZE]; 
    int *temp; //temp storage during Merge sort on GPU
    
    cudaMalloc((void**)&d_arr, sizeof(int) * SIZE);
    cudaMalloc((void **)&temp, sizeof(int) * SIZE);

    CALI_MARK_BEGIN(mainCali);
    cali::ConfigManager mgr;
    mgr.start();
    
    // Initialize array
    CALI_MARK_BEGIN(dataInit);
    data_initialization(initialize_arr, SIZE, sortType);
    CALI_MARK_END(dataInit);
 
    // copy initalized arr onto d_arr
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commLarge);
    cudaMemcpy(d_arr, initialize_arr, SIZE * sizeof(int), cudaMemcpyHostToDevice);    
    cudaMemcpy(h_arr, d_arr, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commLarge);
    cudaMemcpy(d_arr, h_arr, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);
    
    // Merge sort
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(compLarge);
    for (int wid = 1; wid < SIZE; wid *= 2) {
        MergeSortGPU<<<numBlocks, numThreads>>>(d_arr, temp, SIZE, wid * 2);
    }
    cudaDeviceSynchronize();
    CALI_MARK_END(compLarge);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commLarge);
    cudaMemcpy(h_arr, d_arr, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);

    // Correctness Check
    printArray(h_arr, SIZE);
    
    bool sorted = isSorted(h_arr, SIZE);
    if (sorted)
    {
        printf("Array is sorted.\n");
    }
    else
    {
        printf("Array is not sorted.\n");
    }

    delete[] h_arr;
    cudaFree(d_arr);
    cudaFree(temp);
    
    CALI_MARK_END(mainCali);


    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Merge Sort");
    adiak::value("ProgrammingModel", "CUDA");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", SIZE);
    adiak::value("InputType", sortName);
    adiak::value("num_processors", numThreads);
    adiak::value("group_num", 10);
    adiak::value("implementation_source", "AI & Handwritten & Online");
    
    adiak::value("main", mainCali);
    adiak::value("data_init", dataInit);
    adiak::value("correctness_check", correctnessCheck);
    adiak::value("comm", comm);
    adiak::value("comp", comp);
    adiak::value("comp_large", compLarge);
    adiak::value("comm_large", commLarge);
    
    // Flush Caliper output
    mgr.stop();
    mgr.flush();

    return 0;
}