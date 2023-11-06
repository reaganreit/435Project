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
         offset += avgVals;
         greaterThan = lessThan;
    }
    
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
    
    /*
    printf("sorted vector: ");
    for (int num: arr) {
      printf("%d ", num); 
      printf("\n");
    }
    */
    
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
    printf("task %d received greaterThan = %d, lessThan = %d, and offset = %d \n", taskid, greaterThan, lessThan, offset);
  }
  
  /*
  
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
  */
  
  mgr.stop();
  mgr.flush();
  
  MPI_Finalize();
}
