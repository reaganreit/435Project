#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <string_view>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

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
  	avgVals, extra, offset; /* used to determine rows sent to each worker */
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
  int mainArr[numValues];
  
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
         //workerValues = (dest <= extra) ? avgVals+1 : avgVals;  
         workerValues = avgVals; 	
         printf("Sending %d values to task %d offset=%d\n",workerValues,dest,offset);
         // MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&mainArr[offset], workerValues, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         offset = offset + workerValues;
    }
    
    // receive values from workers
    /*
    mtype = FROM_WORKER;
    for (source=1; source<=numWorkers; source++)
    {
         int testArr[avgVals];
         MPI_Recv(&testArr[0], avgVals, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         printf("Received results from task %d\n",source);
         for (int num: testArr) {
           printf("%d ", num); 
           printf("\n");
         }
    }
    */
  }
  
  
  if (taskid != MASTER) {
    //create array of size numWorkers-1 to store chosen samples
    std::vector<int> chosenSamples(numWorkers-1);
    // receive values from master
    mtype = FROM_MASTER;
    std::vector<int> arr(avgVals);
    MPI_Recv(&arr[0], avgVals, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    printf("Received results for task %d\n", taskid);
    for (int num: arr) {
      printf("%d ", num); 
      printf("\n");
    }
    
    // locally sort arr with sequential sorting
    std::sort(arr.begin(), arr.end());
    printf("sorted: ");
    for (int num: arr) {
      printf("%d ", num); 
      printf("\n");
    }
    // choose samples and add to sampleArr
    // All workers send their sample elements to master process
    // MPI_Send sampleArr to MASTER
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
