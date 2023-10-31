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



**Merge Sort MPI Pseudo**
function merge(arr, aux, low, mid, high)
    h := low
    i := low
    j := mid + 1

    while (h <= mid) and (j <= high)
        if arr[h] <= arr[j]
            aux[i] := arr[h]
            h := h + 1
        else
            aux[i] := arr[j]
            j := j + 1
        i := i + 1

    if mid < h
        for k from j to high
            aux[i] := arr[k]
            i := i + 1
    else
        for k from h to mid
            aux[i] := arr[k]
            i := i + 1

    for k from low to high
        arr[k] := aux[k]

end function

function mergeSort(arr, aux, low, high)
    if low < high
        mid := (low + high) / 2

        mergeSort(arr, aux, low, mid)
        mergeSort(arr, aux, mid + 1, high)
        merge(arr, aux, low, mid, high)
    end if
end function

function parallelMergeSort(arr, n)
    Initialize MPI
    world_rank := current process rank
    world_size := total number of processes

    size := n / world_size
    sub_arr := allocate memory for a subarray of size

    Scatter original_array into sub_arr

    tmp_arr := allocate memory for a temporary array of size

    mergeSort(sub_arr, tmp_arr, 0, size - 1)

    sorted := allocate memory for the final sorted array

    Gather sub_arr into sorted

    if world_rank is 0
        other_arr := allocate memory for another temporary array of size
        mergeSort(sorted, other_arr, 0, n - 1)

        Display "This is the sorted array: "
        for each element in sorted
            Display the element
        Display a newline

        Clean up root process
        free sorted
        free other_arr
    end if

    Clean up other processes
    free original_array
    free sub_arr
    free tmp_arr

    Finalize MPI
end function

function main(argc, argv)
    n := convert argv[1] to integer
    original_array := allocate memory for an array of n integers

    Initialize random number generator
    Display "This is the unsorted array: "
    for c from 0 to n - 1
        original_array[c] := generate a random integer between 0 and n - 1
        Display original_array[c]
    Display a newline

    parallelMergeSort(original_array, n)
end function

**Merge Sort CUDA Implementation**
function mergesort(data)
    Initialize threadsPerBlock and blocksPerGrid
    size = readList(data)
    
    D_data = allocateDeviceMemory(size)
    D_swp = allocateDeviceMemory(size)
    D_threads = allocateDeviceMemory(threadsPerBlock)
    D_blocks = allocateDeviceMemory(blocksPerGrid)
    
    copyDataToDevice(data, D_data)
    
    A = D_data
    B = D_swp
    nThreads = calculateTotalThreads(threadsPerBlock, blocksPerGrid)
    
    for width = 2 to (size * 2) step width * 2
        slices = size / (nThreads * width) + 1
        call gpu_mergesort_kernel(A, B, size, width, slices, D_threads, D_blocks)
        swap(A, B)

    copyDataToHost(A, data)
    freeDeviceMemory(D_data, D_swp, D_threads, D_blocks)

function gpu_mergesort_kernel(source, dest, size, width, slices, threads, blocks)
    idx = getIdx(threads, blocks)
    start = width * idx * slices
    
    for slice = 0 to slices
        if start >= size
            break
        middle = min(start + (width / 2), size)
        end = min(start + width, size)
        gpu_bottomUpMerge(source, dest, start, middle, end)
        start += width

function gpu_bottomUpMerge(source, dest, start, middle, end)
    i = start
    j = middle
    for k = start to end
        if i < middle and (j >= end or source[i] < source[j])
            dest[k] = source[i]
            i++
        else
            dest[k] = source[j]
            j++

function readList(list)
    size = 0
    first = None
    node = None
    while read a value from stdin
        create a new LinkNode with the value
        if node is not None
            set node.next to the new node
        else
            set first to the new node
        set node to the new node
        size++
    
    if size > 0
        allocate memory for list of size
        node = first
        i = 0
        while node is not None
            set list[i] to node.value
            move to the next node
            increment i
    
    return size

function tm()
    current_time = get current time in microseconds
    elapsed_time = current_time - previous_time
    previous_time = current_time
    return elapsed_time

**Quicksort CUDA Pseudo**
https://forums.developer.nvidia.com/t/quick-sort-depth/35100

unsigned int *lptr = data + left;
unsigned int *rptr = data + right;
unsigned int  pivot = data[(left + right) / 2];

// Do the partitioning.
while (lptr <= rptr)
{
	// Find the next left- and right-hand values to swap
	unsigned int lval = *lptr;
	unsigned int rval = *rptr;

	// Move the left pointer as long as the pointed element is smaller than the pivot.
	while (lval < pivot)
	{
		lptr++;
		lval = *lptr;
	}

	// Move the right pointer as long as the pointed element is larger than the pivot.
	while (rval > pivot)
	{
		rptr--;
		rval = *rptr;
	}

	// If the swap points are valid, do the swap!
	if (lptr <= rptr)
	{
		*lptr++ = rval;
		*rptr-- = lval;
	}
}

// Now the recursive part
int nright = rptr - data;
int nleft = lptr - data;

// Launch a new block to sort the left part.
if (left < (rptr - data))
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	cdp_simple_quicksort << < 1, 1, 0, s >> >(data, left, nright, depth + 1);
	cudaStreamDestroy(s);
}

// Launch a new block to sort the right part.
if ((lptr - data) < right)
{
	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
	cdp_simple_quicksort << < 1, 1, 0, s1 >> >(data, nleft, right, depth + 1);
	cudaStreamDestroy(s1);
}


**Quicksort MPI Pseudo**
https://www.codeproject.com/KB/threads/Parallel_Quicksort/Parallel_Quick_sort_without_merge.pdf 

function sort_recursive(arr, size, pr_rank, max_rank, rank_index)
    dtIn := MPI_Status

    share_pr := pr_rank + 2^rank_index // Calculate the rank of the sharing process 
    rank_index := rank_index + 1 // Increment the count index 

    if share_pr > max_rank // If no process to share, sort recursively sequentially 
        sort_rec_seq(arr, size)
        return 0
    end if

    pivot := arr[size / 2] // Select the pivot 
    partition_pt := sequential_quicksort(arr, pivot, size, (size / 2) - 1) // Partition array 
    offset := partition_pt + 1

    if offset > size - offset
        MPI_Send((arr + offset), size - offset, MPI_INT, share_pr, offset, MPI_COMM_WORLD)
        sort_recursive(arr, offset, pr_rank, max_rank, rank_index)
        MPI_Recv((arr + offset), size - offset, MPI_INT, share_pr, MPI_ANY_TAG, MPI_COMM_WORLD, dtIn)
    else
        MPI_Send(arr, offset, MPI_INT, share_pr, tag, MPI_COMM_WORLD)
        sort_recursive((arr + offset), size - offset, pr_rank, max_rank, rank_index)
        MPI_Recv(arr, offset, MPI_INT, share_pr, MPI_ANY_TAG, MPI_COMM_WORLD, dtIn)
    end if
end function



```
