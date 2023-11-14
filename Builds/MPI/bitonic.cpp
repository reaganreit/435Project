#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mpi.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


int numTasks, rank;

//Caliper regions
const char* main_region = "main";
const char* data_init = "data_init";
const char* comp = "comp";
const char* comm = "comm";
const char* comp_large = "comp_large";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";


int min(int a, int b) {
  if (a <= b) return a;
  return b;
}

int max(int a, int b) {
  if (a >= b) return a;
  return b;
}

// return 1 if n is a power of 2, 0 otherwise.
int isPow2(int n) {
  while (n > 0) {
    if (n % 2 == 1 && n / 2 != 0) return 0;
    n /= 2;
  }
  return 1;
}

// return n if it is a power of 2, or next highest if it is not.
int nextPow2(int n) {
  if (isPow2(n)) return n;
  if (n == 0) return 1;

  int log = 0;
  while (n > 0) {
    log++;
    n /= 2;
  }
  n = 1;
  for(int i=0; i<log; i++)
    n *= 2;

  return n;
}

// return pointer to howMany random numbers.
int *createNumbers(int howMany) {
  int * numbers = (int *) malloc(sizeof(int) * howMany);

  if (numbers == NULL) {
    printf("Error: malloc failed.\n");
    return NULL;
  }
  
  srand(time(NULL) & rank);

  for(int i=0; i < howMany; i++) 
    numbers[i] = rand();

  return numbers;
}

// print array of howMany numbers.
void printNumbers(int * numbers, int howMany) {
  printf("\n");
  for(int i=0; i < howMany; i++)
    printf("%d\n", numbers[i]);
  printf("\n");
}

// check if array of howMany random numbers is sorted in increasing order.
// return 1 or 0.
int isSorted(int *numbers, int howMany) {
  for(int i=1; i<howMany; i++) {
    if (numbers[i] < numbers[i-1]) return 0;
  }
  return 1;
}


int compareDescending(const void *item1, const void *item2) {
  int x = * ( (int *) item1), y = * ( (int *) item2);
  return y-x;
}

int compareAscending(const void *item1, const void *item2) {
  int x = * ( (int *) item1), y = * ( (int *) item2);
  return x-y;
}

int *tempArray;

void compareExchange(int *numbers, int howMany, 
		     int node1, int node2, int biggerFirst,
		     int sequenceNo) {
  if (node1 != rank && node2 != rank) return;

  memcpy(tempArray, numbers, howMany*sizeof(int));

  MPI_Status status;

  // get numbers from the other node. 
  // have the process that is node1 always send first, and node2 
  // receive first - they can't both send at the same time.
  int nodeFrom = node1==rank ? node2 : node1;
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);

  if (node1 == rank) {
    MPI_Send(numbers, howMany, MPI_INT, nodeFrom, sequenceNo, MPI_COMM_WORLD);
    MPI_Recv(&tempArray[howMany], howMany, MPI_INT, nodeFrom, sequenceNo, 
	     MPI_COMM_WORLD, &status);
  }
  else {
    MPI_Recv(&tempArray[howMany], howMany, MPI_INT, nodeFrom, sequenceNo, 
	     MPI_COMM_WORLD, &status);
    MPI_Send(numbers, howMany, MPI_INT, nodeFrom, sequenceNo, MPI_COMM_WORLD);
  }
  
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);
  

  // sort them.
  if (biggerFirst) {
    qsort(tempArray, howMany*2, sizeof(int), compareDescending);
  }
  else {
    qsort(tempArray, howMany*2, sizeof(int), compareAscending);
  }

  // keep only half of them.
  if (node1 == rank)
    memcpy(numbers, tempArray, howMany*sizeof(int));
  else
    memcpy(numbers, &tempArray[howMany], howMany*sizeof(int));
}

/*
  functioN: mergeBitonic - perform bitonic merge sort.
*/
void mergeBitonic(int *numbers, int howMany) {
  tempArray = (int *) malloc(sizeof(int) * howMany * 2);

  int log = numTasks;
  int pow2i = 2;
  int sequenceNumber = 0;

  CALI_MARK_BEGIN(comp_large);
  for(int i=1; log > 1 ; i++) {
    int pow2j = pow2i;
    for(int j=i; j >= 1; j--) {
      sequenceNumber++;
      for(int node=0; node < numTasks; node += pow2j) {
	    for(int k=0; k < pow2j/2; k++) {
	        //printf("i=%d, j=%d, node=%d, k=%d, pow2i=%d, pow2j=%d\n", 
	        // i, j, node, k, pow2i, pow2j);
	            compareExchange(numbers, howMany, node+k, node+k+pow2j/2, ((node+k) % (pow2i*2) >= pow2i),sequenceNumber);
	        }
        } 
        pow2j /= 2;

      //      printf(" after substage %d", j);
      //      printNumbers(numbers, howMany);
    }
    pow2i *= 2;
    log /= 2;

    //    printf("after stage %d\n", i);
    //    printNumbers(numbers, howMany);
  }
  CALI_MARK_END(comp_large);

  free(tempArray);
}




int main(int argc, char *argv[]) {
  CALI_MARK_BEGIN(main_region);

  int howMany;
  long int returnVal;
  int len;
  char hostname[MPI_MAX_PROCESSOR_NAME];

  // initialize
  returnVal = MPI_Init(&argc, &argv);
  if (returnVal != MPI_SUCCESS) {
    printf("Error starting MPI program.  Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, returnVal);
    return 0;
  }

  // stuff about me...
//   CALI_MARK_BEGIN(comm);
//   CALI_MARK_BEGIN(comm_large);
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(hostname, &len);
//   CALI_MARK_END(comm_large);
//   CALI_MARK_END(comm_large);

  if (argc < 2) {
    if (rank == 0)
      printf("  Usage: ./a.out howMany.\n");
    MPI_Finalize();
    return 0;
  }

  // make sure how many to pick at random at each node is power of 2.
  howMany = atoi(argv[1]);
  howMany = nextPow2(howMany);
  if (rank == 0)
    printf("Rounding that up to next power of two, %d\n", howMany);

  // make sure number of tasks is power of 2.
  if (!isPow2(numTasks)) {
    if (rank == 0)
      printf("Number of processes must be power of 2.\n");
    MPI_Finalize();
    return 0;
  }

  // each process creates a list of random numbers.
  CALI_MARK_BEGIN(data_init);
  int * numbers = createNumbers(howMany);
  CALI_MARK_END(data_init);
  

  printNumbers(numbers, howMany);
  CALI_MARK_BEGIN(comp);
  mergeBitonic(numbers, howMany);
  CALI_MARK_END(comp);
  printf("CHECK");
  printNumbers(numbers, howMany);

  // they are all sorted, now just gather them up.
  CALI_MARK_BEGIN(comm);
 
  int * allNumbers = NULL;
  if (rank == 0) {
    allNumbers = (int *) malloc(howMany * numTasks * sizeof(int));
  }
  CALI_MARK_BEGIN(comm_large);
  MPI_Gather(numbers, howMany, MPI_INT, 
	     allNumbers, howMany, MPI_INT, 
	     0, MPI_COMM_WORLD);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  CALI_MARK_BEGIN(correctness_check);
  if (rank == 0) {
    if (isSorted(allNumbers, howMany * numTasks)) 
      printf("Successfully sorted!\n");
    else
      printf("Error: numbers not sorted.\n");
    
    free(allNumbers);
  }
  CALI_MARK_END(correctness_check);

  
  
  free(numbers);
  MPI_Finalize();

  CALI_MARK_END(main_region);

  const char* algorithm ="BitonicSort";
  const char* programmingModel = "MPI"; 
  const char* datatype = "int"; 
  int sizeOfDatatype =4;
  int inputSize =1000; 
  const char* inputType= "Random";
  int num_procs = howMany; 
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
  adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
  // //adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
  // //adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
  adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
  

  adiak::value("data_init", data_init);
  adiak::value("comm", comm);
  adiak::value("comp", comp);
  adiak::value("comm_large", comm_large);
  adiak::value("comp_large", comp_large);
  adiak::value("correctness_check", correctness_check);
  return 0;
}