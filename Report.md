# CSCE 435 Group project

## 1. Group members:
1. Reagan Reitmeyer
2. Lasyasri Shilpi
3. Emily Ha
4. Sry Hak

---

## 2. _due 10/25_ Project topic
Performance comparison of different parallel sorting algorithms

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

For example:
- Radix sort (CUDA)
- Radix sort (MPI)
- Merge sort (CUDA)
- Merge sort (MPI)
- Quick sort (CUDA)
- Quick sort (MPI)
- Bitonic sort (CUDA)
- Bitonic sort (MPI)

We will communicate over text and in person. 
```
**Radix MPI Pseudocode**
https://andreask.cs.illinois.edu/Teaching/HPCFall2012/Projects/yourii-report.pdf 

Initialize MPI
Get process rank and number of processes

// Determine how to split the data among processors
elements_per_processor = total_elements / num_processes
my_start_index = process_rank * elements_per_processor
my_end_index = (process_rank + 1) * elements_per_processor

// Scatter the data to different processors
MPI_Scatter(data, elements_per_processor, MPI_INT, my_data, elements_per_processor, MPI_INT, 0, MPI_COMM_WORLD)

// Initialize local data structures
local_data = my_data
local_buckets = new array of size NUM_BUCKETS
local_prefix_sum = new array of size NUM_BUCKETS

for pass = 0 to MAX_PASSES:
    // Count the number of keys per bucket (local operation)
    for i = my_start_index to my_end_index:
        digit_value = get_digit(local_data[i], pass)
        local_buckets[digit_value]++

    // Perform local exclusive prefix sum to get local bucket offsets
    local_prefix_sum[0] = 0
    for i = 1 to NUM_BUCKETS:
        local_prefix_sum[i] = local_prefix_sum[i-1] + local_buckets[i-1]

    // Move keys within each processor to appropriate buckets (local operation)
    for i = my_start_index to my_end_index:
        digit_value = get_digit(local_data[i], pass)
        new_index = my_start_index + local_prefix_sum[digit_value]
        swap(local_data[i], local_data[new_index])
        local_prefix_sum[digit_value]++

    // 1-to-all transpose buckets across processors to find global prefix sum (global operation)
    MPI_Allreduce(local_buckets, global_bucket_sum, NUM_BUCKETS, MPI_INT, MPI_SUM,            MPI_COMM_WORLD)

    // Calculate global prefix sum
    global_prefix_sum[0] = 0
    for i = 1 to NUM_BUCKETS:
        global_prefix_sum[i] = global_prefix_sum[i-1] + global_bucket_sum[i-1]

    // Send/receive keys between the processors (global operation)
    MPI_Alltoallv(local_data, local_buckets, local_prefix_sum, MPI_INT, my_data, global_bucket_sum, global_prefix_sum, MPI_INT, MPI_COMM_WORLD)

// Gather the sorted data back to the root process
MPI_Gather(my_data, elements_per_processor, MPI_INT, data, elements_per_processor, MPI_INT, 0, MPI_COMM_WORLD)

// Finalize MPI
Finalize MPI




**Radix CUDA Pseudocode**
https://forums.developer.nvidia.com/t/radix-sort-implementation-in-cuda/46581

function radixSortCUDA(data, numElements)
    for bit = 0 to 31
        mask = 1 << bit
        numBlocks = calculateNumberofBlocks(numElements)
        threadsPerBlock = calculateThreadsPerBlock(numElements)

        // Kernel invocation for counting and reordering
        launch radixSortKernel(data, temp, numElements, bit, mask, numBlocks, threadsPerBlock)

        // Swap data and temp arrays for the next iteration
        swap(data, temp)

function radixSortKernel(data, temp, numElements, bit, mask, numBlocks, threadsPerBlock)
    threadId = threadIdx in the block
    blockId = blockIdx
    blockSize = blockDim
    globalId = threadId + blockId * blockSize

    local count[2] = {0}

    // Count the number of 0s and 1s for the current bit
    for i = globalId to numElements with a stride of blockSize
        bitValue = (data[i] & mask) >> bit
        count[bitValue]++

    // Synchronize threads within the block
    synchronizeThreadsInBlock

    if threadId < 2
        // Prefix sum (scan) on the counts
        for offset = 1 to 2
            index = threadId + offset
            if index < 2
                count[index] += count[index - 1]

    // Synchronize threads within the block
    synchronizeThreadsInBlock

    // Reorder the elements based on counts
    for i = numElements - 1 down to 0 with a stride of blockSize
        bitValue = (data[i] & mask) >> bit
        temp[count[bitValue] - 1] = data[i]
        count[bitValue]--

    // Synchronize threads within the block
    synchronizeThreadsInBlock

    // Copy sorted data back to the original array
    for i = globalId to numElements with a stride of blockSize
        data[i] = temp[i]

function main()
    numElements = 1000  // Number of elements to sort
    h_data = new int[numElements]  // Host array
    d_data, d_temp  // Device arrays
    numBlocks = 32
    threadsPerBlock = 128

    // Initialize data with random values
    for i = 0 to numElements - 1
        h_data[i] = randomInt(0, 999)  // Assuming integers in the range [0, 999]

    // Allocate memory on the GPU
    cudaMalloc(d_data, numElements * sizeof(int))
    cudaMalloc(d_temp, numElements * sizeof(int))

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, numElements * sizeof(int), cudaMemcpyHostToDevice)

    // Perform radix sort for 32 bits (assuming integers are 32-bit)
    radixSortCUDA(d_data, numElements)

    // Copy the sorted data back to the host
    cudaMemcpy(h_data, d_data, numElements * sizeof(int), cudaMemcpyDeviceToHost)

    // Free device memory
    cudaFree(d_data)
    cudaFree(d_temp)

    // Display sorted data
    for i = 0 to numElements - 1
        print h_data[i] + " "

    delete[] h_data


```
