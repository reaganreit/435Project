#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <string_view>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cmath>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

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

int correctnessCheck(int arr[], int size) {
  printf("entered correctness");
  CALI_MARK_BEGIN(correctness_check);
  for (int i=0; i<size-1; i++) {
    if (arr[i+1] < arr[i])
      return 0;  // means it's not ordered correctly
  }
  CALI_MARK_END(correctness_check);

  return 1;
}

// intializes array to be sorted. 1=sorted, 2=reverse sorted, 3=randomized, 4=1% perturbed
void dataInit(int arr[], int size, int inputType) {
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
        // printf("first index: %d, second index: %d\n", firstIndex, secondIndex);
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
}

int main(int argc, char *argv[]) {
  int numValues;
  if (argc == 3)
  {
      numValues = atoi(argv[1]);
  }
  else
  {
      printf("%d", argc);
      printf("\n Please provide the size of the array");
      return 0;
  }
  
  int inputType = atoi(argv[2]);

  int numProcs;
  int	taskid,                /* a task identifier */
  	numWorkers,            /* number of worker tasks */
  	source,                /* task id of message source */
  	dest,                  /* task id of message destination */
  	mtype,                 /* message type */
  	avgVals, extra, offset, /* used to determine rows sent to each worker */
    greaterThan, lessThan; 
  MPI_Status status;
  
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
  MPI_Comm_size(MPI_COMM_WORLD,&numProcs);
  
  numWorkers = numProcs - 1;
  
  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();
  
  avgVals = numValues/numWorkers;
  int workerValues;
  int mainArr[numValues];   // array workers will read
  int finalArr[numValues];  // array workers will write to
  
  CALI_MARK_BEGIN(main_region);
  
  if (taskid == MASTER) {
    printf("Sample sort has started with %d tasks.\n", numWorkers);
    printf("Initializing array...\n");
    printf("Input Type: %d\n", inputType);
    
    // initialize master process and generate array values
    CALI_MARK_BEGIN(data_init);
    dataInit(mainArr, numValues, inputType);
    CALI_MARK_END(data_init);
    
    /*
    printf("initial array\n");
    for (int num : mainArr) {
      printf("%d ", num);
    }
    printf("\n");
    */
  
    // MASTER distribute numValues equally to each worker
    offset = 0;
    extra = numValues%numWorkers;
    mtype = FROM_MASTER;
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    for (dest=1; dest<=numWorkers; dest++)
    {
         workerValues = (dest <= extra) ? avgVals+1 : avgVals;  
         printf("Sending %d values to task %d offset=%d\n",workerValues,dest,offset);
         MPI_Send(&workerValues, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&mainArr[offset], workerValues, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         offset = offset + workerValues;
    }
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    // receive chosen samples from workers
    mtype = FROM_WORKER;
    std::vector<int> totalSamples((numWorkers-1)*numWorkers);
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    for (source=1; source<=numWorkers; source++)
    {
         MPI_Recv(&totalSamples[(numWorkers-1)*(source-1)], numWorkers-1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         printf("Received results from task %d\n",source);
    }
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    // sequentially sort samples
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    std::sort(totalSamples.begin(), totalSamples.end());
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    
    printf("total sample: ");
    for (int sample: totalSamples) {
       printf("%d ", sample); 
    }
    printf("\n");
    
    // choose global splitters
    int globalSplitters[numWorkers-1];
    int spacing = std::ceil((float)totalSamples.size()/(float)numWorkers);
    int index = spacing-1;
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    for (int i=0; i<numWorkers-1; i++) {
      globalSplitters[i] = totalSamples[index];
      index += spacing;
    }
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    
    printf("Global splitters: ");
    for (int splitter: globalSplitters) {
       printf("%d ", splitter); 
    }
    printf("\n");
    
    // Master process broadcasts the splitters to all other processes
    // splitters dictate what the start/end of each subarr should be
    mtype = FROM_MASTER;
    for (dest=1; dest<=numWorkers; dest++)
    {
         lessThan = globalSplitters[dest-1];
         if (dest==numWorkers)
           lessThan = INT_MAX;
         //printf("Sending %d to %d task %d offset=%d\n", greaterThan, lessThan, dest, offset);
         CALI_MARK_BEGIN(comm);
         CALI_MARK_BEGIN(comm_small);
         MPI_Send(&globalSplitters, numWorkers-1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         CALI_MARK_END(comm_small);

         CALI_MARK_BEGIN(comm_large);
         MPI_Send(&mainArr, numValues, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         CALI_MARK_END(comm_large);
         CALI_MARK_END(comm);
    }
    
    // receive results from workers
    std::vector<std::vector<int>> accumBuckets(numWorkers); 
    mtype = FROM_WORKER;
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    for (source=1; source<=numWorkers; source++)
    {
         int buckets[numWorkers][avgVals+1];
         MPI_Recv(&buckets, numWorkers*(avgVals+1), MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         for (int i=0; i<numWorkers; i++) {
           for (int j=0; j<avgVals+1; j++) {
             if (buckets[i][j] != -1)
               accumBuckets[i].push_back(buckets[i][j]);
           }
         }
    }
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    printf("received all results\n");
    
    // sort each row/bucket
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for (int i=0; i<numWorkers; i++) {
       std::sort(accumBuckets[i].begin(), accumBuckets[i].end());
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    printf("finished sorting buckets\n");
    
    // concatenate results into final array
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    int finalIndex = 0;
    for (int i=0; i<numWorkers; i++) {
       for (int j=0; j<accumBuckets[i].size(); j++) {
         finalArr[finalIndex] = accumBuckets[i][j];
         finalIndex++;
       }
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    printf("concatenated results\n");
    /*
    printf("final index: %d\n", finalIndex);
    // check final array
    printf("FINAL ARRAY\n");
    for (int num : finalArr) {
      printf("%d ", num);
    }
    printf("\n");
    */
    
    if (correctnessCheck(finalArr, numValues)) {
      printf("CORRECT");
    } else {
      printf("INCORRECT");
    }
    
  }
  
  
  if (taskid != MASTER) {
    //create array of size numWorkers-1 to store chosen samples
    int chosenSamples[numWorkers-1];
    int workerValues;
    
    // receive values from master
    mtype = FROM_MASTER;
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Recv(&workerValues, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    std::vector<int> arr(workerValues);
    MPI_Recv(&arr[0], workerValues, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    // locally sort arr with sequential sorting
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    std::sort(arr.begin(), arr.end());
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    
    // choose samples and add to sampleArr
    int spacing = std::ceil((float)workerValues/(float)numWorkers);
    //printf("spacing: %d\n", spacing);
    int index = spacing-1;
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    for (int i=0; i<numWorkers-1; i++) {
      if (index > arr.size()-1)
        break;
      chosenSamples[i] = arr[index];
      // printf("Index: %d, Chosen Sample: %d\n", index, chosenSamples[i]);
      index += spacing;
    }
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    // All workers send their sample elements to master process
    // MPI_Send chosenSamples to MASTER
    mtype = FROM_WORKER;
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    MPI_Send(&chosenSamples, numWorkers-1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);
    
    // receive splitters from master
    mtype = FROM_MASTER;
    int splitters[numWorkers-1];
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    MPI_Recv(&splitters, numWorkers-1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&mainArr, numValues, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);
    
    /*
    printf("splitters\n");
    for (int splitter: splitters) {
      printf("%d ", splitter);
    }
    printf("\n");
    */
    
    int buckets[numWorkers][avgVals+1];
    int arrIndex[numWorkers];
    // initialize the entire array with -1 so we know which values aren't used
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for (int i=0; i<numWorkers; i++) {
      arrIndex[i]=0;
      for (int j=0; j<avgVals+1; j++) {
        buckets[i][j]= -1;
      }
    }
    
    int j;
    for(int num : arr) {
  		j = 0;
  		while(j < numWorkers) {  // j being which process/bucket it should belong to
  			if (j == numWorkers-1) {
          // means it should go in last bucket
          // makes sure that we don't try to access splitters[buckets.size()-1]. will go out of range
          buckets[j][arrIndex[j]] = num;
          arrIndex[j]++;
          break;
        }
        if(num < splitters[j]) {
  				buckets[j][arrIndex[j]] = num;
          arrIndex[j]++;
          break;
  			}
  			j++;
  		}
  	}
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
   
    // send buckets back to MASTER
    mtype = FROM_WORKER;
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Send(&buckets, numWorkers*(avgVals+1), MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    printf("task %d sent to master\n", taskid);
  }

  CALI_MARK_END(main_region);
  
  const char* algorithm = "Sample Sort";
  const char* programmingModel = "MPI";
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
  adiak::value("InputSize", numValues); // The number of elements in input dataset (1000)
  adiak::value("InputType", inputTypeStr); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_procs", numProcs); // The number of processors (MPI ranks)
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

  mgr.stop();
  mgr.flush();
  
  MPI_Finalize();
}
