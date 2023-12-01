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

int numThreads;
int numBlocks;
int SIZE;

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
            
            for (int i = 0; i < numToSwitch; ++i) {
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
            fflush(stdout);
            break;
    }
}

__global__ void merge_sort_step(int *device_vals, int *temp, int n, unsigned int width)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start = 2 * width * idx;

    if (start < n)
    {
        unsigned int middle = min(start + width, n);
        unsigned int end = min(start + 2 * width, n);
        unsigned int i = start;
        unsigned int j = middle;
        unsigned int k = start;

        while (i < middle && j < end)
        {
            if (device_vals[i] < device_vals[j])
            {
                temp[k++] = device_vals[i++];
            }
            else
            {
                temp[k++] = device_vals[j++];
            }
        }
        while (i < middle)
            temp[k++] = device_vals[i++];
        while (j < end)
            temp[k++] = device_vals[j++];

        for (i = start; i < end; i++)
        {
            device_vals[i] = temp[i];
        }
    }
}

void merge_sort(int *initial_arr, int length)
{
    int *device_vals, *temp;
    size_t bytes = length * sizeof(int);
    cudaMalloc((void **)&device_vals, bytes);
    cudaMalloc((void **)&temp, bytes);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commLarge);
    cudaMemcpy(device_vals, initial_arr, bytes, cudaMemcpyHostToDevice);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);

    dim3 threadsPerBlock(numThreads, 1);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(compLarge);
    for (int width = 1; width < length; width *= 2)
    {
        long long totalThreads = (long long)length / (2 * width);
        int numBlocks = (totalThreads + threadsPerBlock.x - 1) / threadsPerBlock.x;

        merge_sort_step<<<numBlocks, threadsPerBlock>>>(device_vals, temp, length, width);
        cudaDeviceSynchronize();
    }
    CALI_MARK_END(compLarge);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commLarge);
    cudaMemcpy(initial_arr, device_vals, bytes, cudaMemcpyDeviceToHost);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);

    cudaFree(device_vals);
    cudaFree(temp);
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
    for (int i = 0; i < size - 1; i++)
    {
        if (arr[i] > arr[i + 1])
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    CALI_MARK_BEGIN(mainCali);
    numThreads = atoi(argv[1]);
    SIZE = atoi(argv[2]);
    int sortType = atoi(argv[3]);
    numBlocks = SIZE / numThreads;

    const char* sortName;
    switch (sortType)
    {
    case 1:
        sortName = "Random";
        break;
    case 2:
        sortName = "Sorted";
        break;
    case 3:
        sortName = "ReverseSorted";
        break;
    case 4:
        sortName = "1%Perturbed";
        break;
    }
    

    int *h_arr = new int[SIZE]; //store and manipulate data on CPU
    int *d_arr; //store data on GPU
    int initialize_arr[SIZE]; 
    
    cudaMalloc((void**)&d_arr, sizeof(int) * SIZE);

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
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);
    
    
    // Merge sort
    merge_sort(d_arr, SIZE);

    //Copy array from device to host
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commLarge);
    cudaMemcpy(h_arr, d_arr, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);
    
    // Correctness Check
    //printArray(h_arr, SIZE);
    
    CALI_MARK_BEGIN(correctnessCheck);
    bool sorted = isSorted(h_arr, SIZE);
    CALI_MARK_END(correctnessCheck);
    
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
    adiak::value("num_threads", numThreads);
    adiak::value("num_blocks", numBlocks);
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