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

int correctness_check(int arr[], int size) {
  CALI_MARK_BEGIN(correctness_check);
  for (int i=0; i<size-1; i++) {
    if (arr[i+1] < arr[i])
      return 0;  // means it's not ordered correctly
  }
  CALI_MARK_END(correctness_check);

  return 1;
}

int main(int argc, char *argv[]) {
  CALI_CXX_MARK_FUNCTION;
  
  CALI_MARK_BEGIN(main_region);
  int numValues;
  if (argc == 2)
  {
      numValues = atoi(argv[1]);
  }
  else
  {
      printf("\n Please provide the size of the array");
      return 0;
  }

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
  
  if (taskid == MASTER) {
    printf("Sample sort has started with %d tasks.\n", numWorkers);
    printf("Initializing array...\n");
    
    // TODO: implement different data inits
    // initialize master process and generate array values
    CALI_MARK_BEGIN(data_init);
    for (int i=0; i<numValues; i++) {
      mainArr[i] = numValues-i;
    }
    CALI_MARK_END(data_init);
  
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
    CALI_MARK_BEGIN(comp_small);
    std::sort(totalSamples.begin(), totalSamples.end());
    CALI_MARK_END(comp_small);
    
    printf("total sample: ");
    for (int sample: totalSamples) {
       printf("%d ", sample); 
    }
    printf("\n");
    
    // choose global splitters
    int globalSplitters[numWorkers-1];
    int spacing = std::ceil((float)totalSamples.size()/(float)numWorkers);
    //printf("spacing: %d\n", spacing);
    int index = spacing-1;
    CALI_MARK_BEGIN(comp_small);
    for (int i=0; i<numWorkers-1; i++) {
      globalSplitters[i] = totalSamples[index];
      //printf("index: %d\n", index);
      index += spacing;
    }
    CALI_MARK_END(comp_small);
    
    // TODO: make selection more evenly spaced?
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
         CALI_MARK_BEGIN(comm_small);
         MPI_Send(&globalSplitters, numWorkers-1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         CALI_MARK_END(comm_small);
         CALI_MARK_BEGIN(comm_large);
         MPI_Send(&mainArr, numValues, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         CALI_MARK_END(comm_large);
    }
    
    // receive results from workers
    std::vector<std::vector<int>> accumBuckets(numWorkers); 
    mtype = FROM_WORKER;
    CALI_MARK_BEGIN(comm_large);
    for (source=1; source<=numWorkers; source++)
    {
         int buckets[numWorkers][avgVals+1];
         MPI_Recv(&buckets, numWorkers*(avgVals+1), MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         printf("Received results from task %d\n",source);
         printf("Bucket contents\n");
         for (int i=0; i<numWorkers; i++) {
           for (int j=0; j<avgVals+1; j++) {
             if (buckets[i][j] != -1)
               accumBuckets[i].push_back(buckets[i][j]);
           }
         }
    }
    CALI_MARK_END(comm_large);
    
    // sort each row/bucket
    CALI_MARK_BEGIN(comp_large);
    for (int i=0; i<numWorkers; i++) {
       std::sort(accumBuckets[i].begin(), accumBuckets[i].end());
    }
    CALI_MARK_END(comp_large);
    
    // concatenate results into final array
    CALI_MARK_BEGIN(comp_large);
    int finalIndex = 0;
    for (int i=0; i<numWorkers; i++) {
       for (int j=0; j<accumBuckets[i].size(); j++) {
         finalArr[finalIndex] = accumBuckets[i][j];
         finalIndex++;
       }
    }
    CALI_MARK_END(comp_large);
    
    printf("final index: %d\n", finalIndex);
    // check final array
    printf("FINAL ARRAY\n");
    for (int num : finalArr) {
      printf("%d ", num);
    }
    printf("\n");
    
    CALI_MARK_BEGIN(correctness_check);
    if (correctness_check(finalArr, numValues)) {
      printf("CORRECT");
    } else {
      printf("INCORRECT");
    }
    CALI_MARK_END(correctness_check);
    
  }
  
  
  if (taskid != MASTER) {
    //create array of size numWorkers-1 to store chosen samples
    int chosenSamples[numWorkers-1];
    int workerValues;
    
    // receive values from master
    mtype = FROM_MASTER;
    CALI_MARK_BEGIN(comm_large);
    MPI_Recv(&workerValues, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    std::vector<int> arr(workerValues);
    MPI_Recv(&arr[0], workerValues, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    CALI_MARK_END(comm_large);
    
    // locally sort arr with sequential sorting
    CALI_MARK_BEGIN(comp_large);
    std::sort(arr.begin(), arr.end());
    CALI_MARK_END(comp_large);
    
    // choose samples and add to sampleArr
    int spacing = std::ceil((float)workerValues/(float)numWorkers);
    //printf("spacing: %d\n", spacing);
    int index = spacing-1;
    CALI_MARK_BEGIN(comp_small);
    for (int i=0; i<numWorkers-1; i++) {
      chosenSamples[i] = arr[index];
      //printf("index: %d\n", index);
      index += spacing;
    }
    CALI_MARK_END(comp_small);
    
    //printf("vector size: %d\n", chosenSamples.size());
    //printf("num workers: %d\n", numWorkers);
    printf("Chosen samples: ");
    for (int sample: chosenSamples) {
      printf("%d ", sample);
    }
    printf("\n");
    // All workers send their sample elements to master process
    // MPI_Send chosenSamples to MASTER
    mtype = FROM_WORKER;
    CALI_MARK_BEGIN(comm_small);
    MPI_Send(&chosenSamples, numWorkers-1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    CALI_MARK_END(comm_small);
    
    // receive splitters from master
    mtype = FROM_MASTER;
    int splitters[numWorkers-1];
    CALI_MARK_BEGIN(comm_small);
    MPI_Recv(&splitters, numWorkers-1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    CALI_MARK_END(comm_small);
    printf("task %d received splitters\n", taskid);
    
    printf("splitters\n");
    for (int splitter: splitters) {
      printf("%d ", splitter);
    }
    printf("\n");
    
    int buckets[numWorkers][avgVals+1];
    int arrIndex[numWorkers];
    // initialize the entire array with -1 so we know which values aren't used
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
          printf("pushed %d into bucket %d\n", num, j);
          arrIndex[j]++;
          break;
        }
        if(num < splitters[j]) {
  				buckets[j][arrIndex[j]] = num;
          arrIndex[j]++;
          printf("pushed %d into bucket %d\n", num, j);
          break;
  			}
  			j++;
  		}
  	}
    CALI_MARK_END(comp_large);
   
    // send buckets back to MASTER
    mtype = FROM_WORKER;
    CALI_MARK_BEGIN(comm_large);
    MPI_Send(&buckets, numWorkers*(avgVals+1), MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
  }

  mgr.stop();
  mgr.flush();
  
  MPI_Finalize();
  CALI_MARK_END(main_region);
}
