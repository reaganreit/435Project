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

int main(int argc, char *argv[]) {
  CALI_CXX_MARK_FUNCTION;
    
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
    
    // initialize master process and generate array values
    for (int i=0; i<numValues; i++) {
      mainArr[i] = numValues-i;
    }
  
    // MASTER distribute numValues equally to each worker
    offset = 0;
    extra = numValues%numWorkers;
    mtype = FROM_MASTER;
    for (dest=1; dest<=numWorkers; dest++)
    {
         // TODO: implement logic for extra
         //workerValues = (dest <= extra) ? avgVals+1 : avgVals;  
         workerValues = avgVals; 	
         printf("Sending %d values to task %d offset=%d\n",workerValues,dest,offset);
         // MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&mainArr[offset], workerValues, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         offset = offset + workerValues;
    }
    
    // receive chosen samples from workers
    mtype = FROM_WORKER;
    std::vector<int> totalSamples((numWorkers-1)*numWorkers);
    for (source=1; source<=numWorkers; source++)
    {
         MPI_Recv(&totalSamples[(numWorkers-1)*(source-1)], numWorkers-1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         printf("Received results from task %d\n",source);
    }
    
    // sequentially sort samples
    std::sort(totalSamples.begin(), totalSamples.end());
    
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
    for (int i=0; i<numWorkers-1; i++) {
      globalSplitters[i] = totalSamples[index];
      //printf("index: %d\n", index);
      index += spacing;
    }
    
    // TODO: make selection more evenly spaced?
    printf("Global splitters: ");
    for (int splitter: globalSplitters) {
       printf("%d ", splitter); 
    }
    printf("\n");
    
    // Master process broadcasts the splitters to all other processes
    // splitters dictate what the start/end of each subarr should be
    greaterThan = 0;
    lessThan;
    offset = 0;
    mtype = FROM_MASTER;
    for (dest=1; dest<=numWorkers; dest++)
    {
         lessThan = globalSplitters[dest-1];
         if (dest==numWorkers)
           lessThan = INT_MAX;
         printf("Sending %d to %d task %d offset=%d\n", greaterThan, lessThan, dest, offset);
         MPI_Send(&greaterThan, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&lessThan, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&mainArr, numValues, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         offset += avgVals;
         greaterThan = lessThan;
    }
    
    // receive results from workers
    mtype = FROM_WORKER;
    std::vector<int> localResults(avgVals);
    for (source=1; source<=numWorkers; source++)
    {
         MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&finalArr[offset], avgVals, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         printf("Received results from task %d\n",source);
    }
    
    printf("final array: ");
    for (int num : finalArr) {
      printf("%d ", num);
    }
    printf("\n");
    
  }
  
  
  if (taskid != MASTER) {
    //create array of size numWorkers-1 to store chosen samples
    int chosenSamples[numWorkers-1];
    
    // receive values from master
    mtype = FROM_MASTER;
    std::vector<int> arr(avgVals);
    MPI_Recv(&arr[0], avgVals, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    
    // locally sort arr with sequential sorting
    std::sort(arr.begin(), arr.end());
    
    // choose samples and add to sampleArr
    int spacing = std::ceil((float)avgVals/(float)numWorkers);
    //printf("spacing: %d\n", spacing);
    int index = spacing-1;
    for (int i=0; i<numWorkers-1; i++) {
      chosenSamples[i] = arr[index];
      //printf("index: %d\n", index);
      index += spacing;
    }
    
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
    MPI_Send(&chosenSamples, numWorkers-1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    
    // receive splitters and offset from master
    mtype = FROM_MASTER;
    int localArr[avgVals];
    MPI_Recv(&greaterThan, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&lessThan, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&mainArr, numValues, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    printf("task %d received greaterThan = %d, lessThan = %d, and offset = %d \n", taskid, greaterThan, lessThan, offset);
    
    int localArrIndex = 0;
    // find values that fall within range
    for (int num: mainArr) {
      if (num >= greaterThan && num < lessThan) {
        localArr[localArrIndex] = num;
        localArrIndex++;
      }
    }
    
    // sort local arr
    int size = sizeof(localArr) / sizeof(localArr[0]);
    std::sort(localArr, localArr + size);
    
    printf("localArr\n");
    for (int num: localArr) {
      printf("%d ", num);
    }
    printf("\n");
    
    // send results back to master
    mtype = FROM_WORKER;
    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&localArr, avgVals, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
  }
  
  mgr.stop();
  mgr.flush();
  
  MPI_Finalize();
}
