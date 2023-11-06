# CSCE 435 Group project

## 0. Group number: 10

## 1. Group members:
1. Reagan Reitmeyer
2. Lasyasri Shilpi
3. Emily Ha
4. Sry Hak

## 2. Project topic (e.g., parallel sorting algorithms)
Parallel sorting algorithms

### 2a. Brief project description (what algorithms will you be comparing and on what architectures)

- Radix sort (CUDA)
- Radix sort (MPI)
- Merge sort (CUDA)
- Merge sort (MPI)
- Sample sort (CUDA)
- Sample sort (MPI)
- Bitonic sort (CUDA)
- Bitonic sort (MPI)

We will communicate over text and in person. 

### 2b. Pseudocode for each parallel algorithm
- For MPI programs, include MPI calls you will use to coordinate between processes
- For CUDA programs, indicate which computation will be performed in a CUDA kernel,
  and where you will transfer data to/from GPU

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
https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c 
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

**Merge Sort CUDA Pseudo**
https://feels2019.pdn.ac.lk/login/index.php

# Function to check if a number is a power of 2
function isPowerOfTwo(num):
    i = 0
    val = 1
    while val <= num:
        if val == num:
            return true
        i = i + 1
        val = pow(2, i)
    return false

# Main function
function main(argc, argv):
    SIZE = convert_to_integer(argv[2])
    
    if not isPowerOfTwo(SIZE):
        print("This implementation needs the list size to be a power of two")
        exit(1)
    
    # Allocate a list of floats
    list = allocate_memory(SIZE * sizeof(float))
    
    if list is null:
        print("Memory full")
        exit(1)
    
    # Generate random float values and populate the list
    for i in range(SIZE):
        list[i] = random_float()  # Generate a random float value
    
    # Print the input list
    print("The input list is:")
    for i in range(SIZE):
        print("%.2f ", list[i])
    print("\n")
    
    # CUDA stuff begins here
    start = start_timer()
    
    # Allocate memory in CUDA
    list_cuda = allocate_cuda_memory(SIZE * sizeof(float))
    
    # Copy memory from RAM to CUDA
    copy_memory_to_cuda(list_cuda, list, SIZE * sizeof(float))
    
    # Thread configurations
    threadsPerBlock = convert_to_integer(argv[1])
    numBlocks = ceil(SIZE / (256 * 2))
    
    # Start measuring time for the CUDA kernel only
    start_kernel = start_timer()
    
    # Perform merge sort using CUDA
    mergesort<<<numBlocks, threadsPerBlock>>>(list_cuda, SIZE)
    check_for_cuda_errors()
    
    # End measuring time for the CUDA kernel
    stop_kernel = stop_timer()
    
    # Copy the answer back from CUDA to RAM
    copy_memory_to_ram(list, list_cuda, SIZE * sizeof(float))
    
    # Free CUDA memory
    free_cuda_memory(list_cuda)
    
    # End measuring time
    stop = stop_timer()
    
    # CUDA stuff ends here
    
    # Print the sorted list
    print("The sorted list is:")
    for i in range(SIZE):
        print("%.2f ", list[i])
    print("\n")
    
    # Print the time spent to stderr
    print("Time spent for CUDA kernel is %.5f seconds" % (elapsed_time_kernel / 1000))
    print("Time spent for the CUDA operation (including memory allocation and copying) is %.5f seconds" % (elapsed_time / 1000))
    
# Function to merge two lists while sorting them in ascending order
function merge(list, left, middle, right):
    n = right - left + 1
    temp = allocate_memory(n * sizeof(float))
    assert(temp is not null, "Memory allocation failed")
    
    i = left
    j = middle
    k = 0
    
    while i < middle and j <= right:
        if list[i] < list[j]:
            temp[k] = list[i]
            i = i + 1
        else:
            temp[k] = list[j]
            j = j + 1
        k = k + 1
    
    while i < middle:
        temp[k] = list[i]
        i = i + 1
        k = k + 1
    
    while j <= right:
        temp[k] = list[j]
        j = j + 1
        k = k + 1
    
    for i = left, k = 0, i <= right:
        list[i] = temp[k]
        i = i + 1
        k = k + 1
    
    free_memory(temp)

# Function to perform merge sort in ascending order
function mergesort(list, SIZE):
    tid = get_thread_index()
    step = 1
    
    while step < SIZE - 1:
        if tid % step == 0 and tid * 2 < SIZE:
            left = 2 * tid
            middle = 2 * tid + step
            right = 2 * tid + 2 * step - 1
            merge(list, left, middle, right)
        
        step = step * 2
        synchronize_threads()

# Define other required functions for CUDA memory allocation, copying, timing, etc.

# Call the main function
main(argc, argv)

**Sample Sort CUDA Pseudo**
https://github.com/SwayambhuNathRay/Sample-Sort-CUDA/blob/master/sample_sort.cu 

// Constants
n: total number of elements
per_block: number of elements processed per block

// CUDA kernel for initial local sorting
__global__ void sample_sort(int *A)
{
    loc[per_block]: shared memory array accessible to all threads within same thread block
    int i: index to traverse the array A
    int k: index to traverse elements in the shared memory array loc
    loc[k] = A[i] //copying an element from A to shared memory array
    __syncthreads();
    int j;
    
    //general sorting alg within a block to perform pairwise comparisons of adjacent elements in loc to sort
        //first phase: threads with even k values compare and potentially swap
        //sec phase: threads with odd k values compare and potentially swap
        //after each phase, __syncthreads()
        //after each iteration of loop, sorted elements are written back to global mem A. Each thread updates value in A based on the sorted value in loc
}

// CUDA kernel for final merge and sorting
__global__ void final_merge(int *A, int* S) //S is the info about the splitters
{
    //splitter range
    int lower_limit
    int upper_limit

    //redistributes and sorts the locally stored data into final sorted order, using info from S to partition and sort the data. Data from each thread block is correctly ordered across entire dataset

}

// CPU based sorting, not GPU
void merge(int *arr, int l, int m, int r)
{
    //merge two sorted subarrays into single sorted array
}

// CPU based sorting, not GPU bc efficient for large datasets
void mergeSort(int *arr, int left, int right)
{
    // recursive merge sort
}

int main()
{
    // Host (CPU) memory allocation and data generation
    // ...

    // Device (GPU) memory allocation
    // ...

    // Measure time for serial merge sort on m_A
    // ...

    // Set up CUDA grid and block dimensions
    // ...

    // Measure time for parallel sorting using sample_sort kernel on d_A
    // ...

    // Sort the splitter array h_S on the CPU
    // ...

    // Generate array h_F to store final splitters
    // ...

    // Launch final_merge kernel on d_A, sending splitters from d_S
    // ...

    // Copy sorted data from device to host
    // ...

    // Free device memory
    // ...

    // Print the time taken for serial and parallel sorting
    // ...

    return 0;
}



**Sample Sort MPI Pseudo**

int sampleSort(int argc, char *argv[]) {
  int numValues, numProcs;
  int taskid;

	// get num of processors and values
  if (argc == 2)
  {
      numValues = atoi(argv[1]);
  }
  else
  {
      printf("\n Please provide the size of the matrix");
  }
  
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
  MPI_Comm_size(MPI_COMM_WORLD,&numProcs);
  int numWorkers = numProcs - 1;
  // initialize master process and generate array values
  mainArr[numValues]
  
  // distribute numValues equally to each worker
  if (MASTER) {
    values = numValues/(numWorkers)
    offset = 0;
    MPI_Send {values} number of vals from mainArr to each worker
    increment offset by vals
    // receive values by workers
    for (i=1; i<=numWorkers; i++)
        {
           MPI_Recv values
    }
  }
  
  if (worker) {
    //create array of size numWorkers-1 to store chosen samples
    sampleArr[numWorkers-1]
    /*locally sort and pick samples from each worker
      It chooses p-1 evenly spaced elements from the sorted block
        So say for example there are 3 processes and 24 elements:
          each process gets 8 elements and returns 2 (3-1) samples
    */
    choose samples and add to sampleArr
    // All workers send their sample elements to master process
    MPI_Send sampleArr to MASTER
  }
  
  if (MASTER) {
    // receive samples from workers
    totalSamples[(numWorkers-1)(numWorkers)]
    for (i=1; i<=numWorkers; i++)
    {
           MPI_Recv samples from WORKER
           add received values to totalSamples
    }
    // Master process sequentially sorts the p(p-1) sample elements and selects p-1 splitters
    splitters[numWorkers-1]
    sort totalSamples
    choose p-1 splitters and store in splitters array
    // Master process broadcasts the splitters to all other processes
    // splitters dictate what the start/end of each subarr should be
    // make sure to maintain offset so process can write to main arr correctly
    currSplitter = 0
    for (i=1; i<=numWorkers; i++)
    {
           MPI_Send entire arr, and [start, splitters[currSplitter]) to each worker
           MPI_Send offset to each worker
           offset += values
           start = splitters[currSplitter]
           currSplitter++
    }
  }
  
  if (WORKER) {
    // Receive arr, start, splitter, and offset from master
    MPI_Recv arr from MASTER
    MPI_Recv start from MASTER
    MPI_Recv splitter from MASTER
    MPI_Recv offset from MASTER
    
    localArr[values]
    find values from arr that fall in between [start, splitter) and add to localArr
    sort localArr
    send localArr back to MASTER along with offset so it knows where to insert values
    MPI_Send localArr to MASTER
    MPI_Send offset to MASTER
  }
  
  finalArr[numValues]
  if (MASTER) {
    for (i=1 ; i<=numWorkers; i++) {
      MPI_Recv localArr
      MPI_offset
      insert localArr starting at finalArr[offset]
    }
  }
    
	return 0;
}



**Bitonic Sort CUDA Pseudo**
https://gist.github.com/mre/1392067
function bitonicSort(values)
    allocate device memory dev_values of size NUM_VALS * sizeof(float)
    copy values to dev_values from host to device

    blocks = BLOCKS
    threads = THREADS
    k = 2

    // Major step
    while k <= NUM_VALS
        j = k / 2

        // Minor step
        while j > 0
            launch bitonicSortStep kernel with dev_values, j, k, blocks, and threads
            j = j / 2

        k = k * 2

    copy dev_values to values from device to host
    free device memory dev_values

function bitonicSortStep(dev_values, j, k, blocks, threads)
    i = get_thread_id() + get_block_id() * threads
    ixj = i XOR j

    // Sorting partners: i and ixj
    if ixj > i
        if (i & k) == 0
            // Sort ascending
            if dev_values[i] > dev_values[ixj]
                swap dev_values[i] and dev_values[ixj]
        
        if (i & k) != 0
            // Sort descending
            if dev_values[i] < dev_values[ixj]
                swap dev_values[i] and dev_values[ixj]

function print_elapsed(start, stop)
    elapsed = (stop - start) / CLOCKS_PER_SEC
    print "Elapsed time: " + elapsed + "s"

function random_float()
    return random float value

function array_fill(arr, length)
    for i = 0 to length - 1
        arr[i] = random_float()

function main()
    start = current_time
    allocate memory for values of size NUM_VALS * sizeof(float)
    fill values with random data
    call bitonicSort(values)
    stop = current_time
    print_elapsed(start, stop)



**Bitonic Sort MPI Pseudo**
https://cse.buffalo.edu/faculty/miller/Courses/CSE702/Sajid.Khan-Fall-2018.pdf
https://people.cs.rutgers.edu/~venugopa/parallel_summer2012/mpi_bitonic.html
FUNCTION generateDataSet(dataSet[], size)
    PRINT "[0] creating dataset ..."
    SET seed to current time
    FOR index from 0 to size
        dataSet[index] = index
END FUNCTION

FUNCTION randomizeData(dataSet[], tempDataSet[], size)
    PRINT "[0] dataset of size size being randomized ..."
    FOR index from 0 to size
        tempDataSet[index] = random value
    CALL SelectionSort(tempDataSet, dataSet, size)
END FUNCTION

FUNCTION SelectionSort(a[], b[], size)
    FOR i from 0 to size - 1
        SET min to i
        FOR j from i + 1 to size
            IF a[j] < a[min]
                min = j
        SET tempB to b[i]
        SET tempA to a[i]
        SET a[i] to a[min]
        SET b[i] to b[min]
        SET a[min] to tempA
        SET b[min] to tempB
END FUNCTION

FUNCTION masterHandshake(buff, numprocs, BUFSIZE, TAG, stat)
    FOR i from 1 to numprocs - 1
        SET buff to "hey i!"
        SEND buff to process i with buffer size BUFSIZE and tag TAG using MPI
    FOR i from 1 to numprocs - 1
        RECEIVE buff with buffer size BUFSIZE and tag TAG from process i using MPI
        PRINT "i: buff"
END FUNCTION

FUNCTION workerHandshake(buff, numprocs, BUFSIZE, TAG, stat, myid)
    DECLARE idstr as string
    RECEIVE buff with buffer size BUFSIZE and tag TAG from process 0 using MPI
    SET idstr to "Processor myid"
    CONCATENATE idstr to buff
    CONCATENATE "reporting for duty" to buff
    SEND buff to process 0 with buffer size BUFSIZE and tag TAG using MPI
END FUNCTION

FUNCTION distributeIntArray(numprocs, dataSet[], SIZE)
    FOR dest from 1 to numprocs
        PRINT "sending data to processor dest, size = SIZE/(numprocs-1)"
        MPI_Send dataSet with size SIZE and tag 1 to process dest using MPI
    PRINT "sending data to p"
    CALL MPI_Finalize()
END FUNCTION

FUNCTION sendIntArray(numprocs, dataSet[], SIZE, target)
    FOR dest from 1 to numprocs
        PRINT "sending data to processor dest, size = SIZE/(numprocs-1)"
        MPI_Send dataSet with size SIZE and tag 1 to process dest using MPI
    PRINT "sending data to p"
    CALL MPI_Finalize()
END FUNCTION

FUNCTION recieveIntArray(buf[], len, stat, from)
    PRINT "check"
    MPI_Recv buf with size len, MPI_INT, from, 1, MPI_COMM_WORLD, stat
    PRINT "check1 buf[63]"
END FUNCTION

FUNCTION bitonicSort(start, len, data[])
    IF len > 1
        SET split to len/2
        CALL bitonicSort(start, split, data)
        CALL bitonicSort(start + split, split, data)
        CALL merge(start, len, data)
END FUNCTION

FUNCTION merge(start, len, data[])
    IF len > 1
        SET mid to len/2
        FOR x from start to start + mid
            CALL compareAndSwap(data, x, x + mid)
        CALL merge(start, mid, data)
        CALL merge(start + mid, start, data)
END FUNCTION

FUNCTION compareAndSwap(data[], i, j)
    DECLARE temp as integer
    IF data[i] > data[j]
        SET temp to data[i]
        SET data[i] to data[j]
        SET data[j] to temp
END FUNCTION

FUNCTION generateDataSet(dataSet[], size)
    PRINT "[0] creating dataset ..."
    SET seed to current time
    FOR index from 0 to size
        dataSet[index] = index
END FUNCTION

FUNCTION randomizeData(dataSet[], tempDataSet[], size)
    PRINT "[0] dataset of size size being randomized ..."
    FOR index from 0 to size
        tempDataSet[index] = random value
    CALL SelectionSort(tempDataSet, dataSet, size)
END FUNCTION

FUNCTION SelectionSort(a[], b[], size)
    FOR i from 0 to size - 1
        SET min to i
        FOR j from i + 1 to size
            IF a[j] < a[min]
                min = j
        SET tempB to b[i]
        SET tempA to a[i]
        SET a[i] to a[min]
        SET b[i] to b[min]
        SET a[min] to tempA
        SET b[min] to tempB
END FUNCTION

FUNCTION masterHandshake(buff, numprocs, BUFSIZE, TAG, stat)
    FOR i from 1 to numprocs - 1
        SET buff to "hey i!"
        SEND buff to process i with buffer size BUFSIZE and tag TAG using MPI
    FOR i from 1 to numprocs - 1
        RECEIVE buff with buffer size BUFSIZE and tag TAG from process i using MPI
        PRINT "i: buff"
END FUNCTION

FUNCTION slaveHandshake(buff, numprocs, BUFSIZE, TAG, stat, myid)
    DECLARE idstr as string
    RECEIVE buff with buffer size BUFSIZE and tag TAG from process 0 using MPI
    SET idstr to "Processor myid"
    CONCATENATE idstr to buff
    CONCATENATE "reporting for duty" to buff
    SEND buff to process 0 with buffer size BUFSIZE and tag TAG using MPI
END FUNCTION

FUNCTION distributeIntArray(numprocs, dataSet[], SIZE)
    FOR dest from 1 to numprocs
        PRINT "sending data to processor dest, size = SIZE/(numprocs-1)"
        MPI_Send dataSet with size SIZE and tag 1 to process dest using MPI
    PRINT "sending data to p"
    CALL MPI_Finalize()
END FUNCTION

FUNCTION sendIntArray(numprocs, dataSet[], SIZE, target)
    FOR dest from 1 to numprocs
        PRINT "sending data to processor dest, size = SIZE/(numprocs-1)"
        MPI_Send dataSet with size SIZE and tag 1 to process dest using MPI
    PRINT "sending data to p"
    CALL MPI_Finalize()
END FUNCTION

FUNCTION recieveIntArray(buf[], len, stat, from)
    PRINT "check"
    MPI_Recv buf with size len, MPI_INT, from, 1, MPI_COMM_WORLD, stat
    PRINT "check1 buf[63]"
END FUNCTION

FUNCTION bitonicSort(start, len, data[])
    IF len > 1
        SET split to len/2
        CALL bitonicSort(start, split, data)
        CALL bitonicSort(start + split, split, data)
        CALL merge(start, len, data)
END FUNCTION

FUNCTION merge(start, len, data[])
    IF len > 1
        SET mid to len/2
        FOR x from start to start + mid
            CALL compareAndSwap(data, x, x + mid)
        CALL merge(start, mid, data)
        CALL merge(start + mid, start, data)
END FUNCTION

FUNCTION compareAndSwap(data[], i, j)
    DECLARE temp as integer
    IF data[i] > data[j]
        SET temp to data[i]
        SET data[i] to data[j]
        SET data[j] to temp
END FUNCTION
```

### 2c. Evaluation plan - what and how will you measure and compare
- Input sizes, Input types
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)
- Number of threads in a block on the GPU 

## 3. Project implementation
Implement your proposed algorithms, and test them starting on a small scale.
Instrument your code, and turn in at least one Caliper file per algorithm;
if you have implemented an MPI and a CUDA version of your algorithm,
turn in a Caliper file for each.

### 3a. Caliper instrumentation
Please use the caliper build `/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper` 
(same as lab1 build.sh) to collect caliper files for each experiment you run.

Your Caliper regions should resemble the following calltree
(use `Thicket.tree()` to see the calltree collected on your runs):
```
main
|_ data_init
|_ comm
|    |_ MPI_Barrier
|    |_ comm_small  // When you broadcast just a few elements, such as splitters in Sample sort
|    |   |_ MPI_Bcast
|    |   |_ MPI_Send
|    |   |_ cudaMemcpy
|    |_ comm_large  // When you send all of the data the process has
|        |_ MPI_Send
|        |_ MPI_Bcast
|        |_ cudaMemcpy
|_ comp
|    |_ comp_small  // When you perform the computation on a small number of elements, such as sorting the splitters in Sample sort
|    |_ comp_large  // When you perform the computation on all of the data the process has, such as sorting all local elements
|_ correctness_check
```

Required code regions:
- `main` - top-level main function.
    - `data_init` - the function where input data is generated or read in from file.
    - `correctness_check` - function for checking the correctness of the algorithm output (e.g., checking if the resulting data is sorted).
    - `comm` - All communication-related functions in your algorithm should be nested under the `comm` region.
      - Inside the `comm` region, you should create regions to indicate how much data you are communicating (i.e., `comm_small` if you are sending or broadcasting a few values, `comm_large` if you are sending all of your local values).
      - Notice that auxillary functions like MPI_init are not under here.
    - `comp` - All computation functions within your algorithm should be nested under the `comp` region.
      - Inside the `comp` region, you should create regions to indicate how much data you are computing on (i.e., `comp_small` if you are sorting a few values like the splitters, `comp_large` if you are sorting values in the array).
      - Notice that auxillary functions like data_init are not under here.

All functions will be called from `main` and most will be grouped under either `comm` or `comp` regions, representing communication and computation, respectively. You should be timing as many significant functions in your code as possible. **Do not** time print statements or other insignificant operations that may skew the performance measurements.

**Nesting Code Regions** - all computation code regions should be nested in the "comp" parent code region as following:
```
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
mergesort();
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Looped GPU kernels** - to time GPU kernels in a loop:
```
### Bitonic sort example.
int count = 1;
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
int j, k;
/* Major step */
for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
        bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        count++;
    }
}
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Calltree Examples**:

```
# Bitonic sort tree - CUDA looped kernel
1.000 main
├─ 1.000 comm
│  └─ 1.000 comm_large
│     └─ 1.000 cudaMemcpy
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Matrix multiplication example - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  ├─ 1.000 comm_large
│  │  ├─ 1.000 MPI_Recv
│  │  └─ 1.000 MPI_Send
│  └─ 1.000 comm_small
│     ├─ 1.000 MPI_Recv
│     └─ 1.000 MPI_Send
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Mergesort - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  └─ 1.000 comm_large
│     ├─ 1.000 MPI_Gather
│     └─ 1.000 MPI_Scatter
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

#### 3b. Collect Metadata

Have the following `adiak` code in your programs to collect metadata:
```
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
adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
```

They will show up in the `Thicket.metadata` if the caliper file is read into Thicket.

**See the `Builds/` directory to find the correct Caliper configurations to get the above metrics for CUDA, MPI, or OpenMP programs.** They will show up in the `Thicket.dataframe` when the Caliper file is read into Thicket.
