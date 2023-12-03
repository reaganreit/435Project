#include <stdio.h>      // Printf
#include <time.h>       // Timer
#include <math.h>       // Logarithm
#include <stdlib.h>     // Malloc
#include "mpi.h"        // MPI Library
#include "bitonic.h"
#include <string.h>
#include <iostream>


#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0        // Who should do the final processing?
#define OUTPUT_NUM 10   // Number of elements to display in output

// Globals
// Not ideal for them to be here though
double timer_start;
double timer_end;
int process_rank;
int num_processes;
int *array;
int array_size;
int inputType;

//Caliper regions
const char* main_region = "main";
const char* data_init = "data_init";
const char* comp = "comp";
const char* comm = "comm";
const char* comp_large = "comp_large";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";

const char* comp_small = "comp_small";


int isSorted(int *arr, int array_size) {
  for(int i=1; i<array_size; i++) {
    if (arr[i] < arr[i-1]) return 0;
  }
  return 1;
}

unsigned int Log2n(unsigned int n)
{
	return (n>1) ? 1+Log2n(n/2) : 0;
}
int main(int argc, char * argv[]) {
    CALI_MARK_BEGIN(main_region);
    int i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // array_size = atoi(argv[1]) / num_processes;
    array_size = atoi(argv[1]);
    int inputType = atoi(argv[2]);

    CALI_MARK_BEGIN(data_init);
    // array = (int *) malloc(array_size * sizeof(int));
    
    array = new int[array_size]; 
    int numToSwitch = array_size / 100;
  int firstIndex, secondIndex;
    switch (inputType) {
        case 1:
        // sorted
        for (int i=0; i<array_size; i++) {
            array[i] = i;
        }
        break;
        case 2:
        // reverse sorted
        for (int i=0; i<array_size; i++) {
            array[i] = array_size-i;
        }
        break;
        case 3:
        // randomized
        srand(time(NULL)+process_rank*num_processes);  
        for (int i=0; i<array_size; i++) {
            array[i] = rand() % RAND_MAX;
        }
        break;
        case 4:
        // 1% perturbe
        for (int i=0; i<array_size; i++) {
            array[i] = i;
        }
        if (numToSwitch == 0)  // at the very least one value should be switched
            numToSwitch = 1;
        
        for (int i=0; i<numToSwitch; i++) {
            firstIndex = rand() % array_size;
            secondIndex = rand() % array_size;
            // printf("first index: %d, second index: %d\n", firstIndex, secondIndex);
            while (firstIndex == secondIndex) {
            secondIndex = rand() % array_size;
            } 
            std::swap(array[firstIndex], array[secondIndex]); 
        }
        break;
        default:
            printf("THAT'S NOT A VALID INPUT TYPE");
        break;
    }

    // printf("arraysize %d", array_size);
    // srand(time(NULL)+process_rank*num_processes);  
    // for (i = 0; i < array_size; i++) {
    //     // array[i] = rand() % (atoi(argv[1]));
    //     array[i] = rand();
    // }

    CALI_MARK_END(data_init);

    MPI_Barrier(MPI_COMM_WORLD);

    int dimensions = Log2n(num_processes);

    if (process_rank == MASTER) {
        printf("Number of Processes spawned: %d\n", num_processes);
        timer_start = MPI_Wtime();
    }

    qsort(array, array_size, sizeof(int), ComparisonFunc);
        // std::sort(array, array + array_size, ComparisonFunc);


//    printf("My rank is %d \t", process_rank);
// //    printf("check 1 \n");
//    for(int i=0; i < array_size; i++)
// 	{
//         // printf("check 2 \n");
// 		printf(" arr: %d\t", array[i]);
// 		int s=0;	
// 		for(s=0;s<=process_rank;s++)
// 			printf(".");
// 	}
	
   // MPI_Barrier(MPI_COMM_WORLD);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for (i = 0; i <dimensions; i++) {
        for (j = i; j>=0; j--) {
		//MPI_Barrier(MPI_COMM_WORLD);
		
            if (((process_rank >> (i + 1)) % 2 == 0 && (process_rank >> j) % 2 == 0) || ((process_rank >> (i + 1)) % 2 != 0 && (process_rank >> j) % 2 != 0)) {
                CompareLow(j);
            } else {
                CompareHigh(j);
            }
        }
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);


    MPI_Barrier(MPI_COMM_WORLD);

    if (process_rank == MASTER) {
        timer_end = MPI_Wtime();

        // printf("check 2");
        printf("Time Elapsed (Sec): %f\n", timer_end - timer_start);
        // printf("check");
    }

    //   printf("Displaying sorted array (only 10 elements for quick verification), My rank is %d\n", process_rank);


    //  for (i = 0; i < array_size; i++) {
    //     if (i == 0 || i==array_size-1) {
    //          printf("arr is %d\n", array[i]);
	// 	for(j=0;j<=process_rank;j++)
	// 		printf(".");
    //        }
    //  }
    //    printf("\n\n");

     for(int i = 0; i < array_size; i++){
        printf(" %d", array[i]);
     }
    printf("\n\n");

    CALI_MARK_BEGIN(correctness_check);
    // printf("test");
    if (process_rank == 0) {
        // printf("test2");
        if (isSorted(array, array_size)) 
            printf("Successfully sorted! YAYayayay! \n");
        else
            printf("Error: numbers not sorted.\n");
    }
    CALI_MARK_END(correctness_check);


    delete[] array;

    // MPI_Finalize();
    CALI_MARK_END(main_region);

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
      inputTypeStr = "1 Perturbed";
      break;
    default:
      inputTypeStr = "No input type. Invalid input argument entered";
      break;
  }

    const char* algorithm ="BitonicSort";
  const char* programmingModel = "MPI"; 
  const char* datatype = "float"; 
  int sizeOfDatatype =4;
  int inputSize = array_size; 
  const char* inputT= inputTypeStr;
  int num_procs = num_processes; 
  int group_number =10;
  const char* implementation_source = "Online"; 

  std::cout << (array_size);
  
  
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
  adiak::value("InputType", inputT); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
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
  adiak::value("comp_small", comp_small);
  adiak::value("correctness_check", correctness_check);

    mgr.stop();
    mgr.flush();
    MPI_Finalize();

}



int ComparisonFunc(const void * a, const void * b) {
    return ( * (int *)a - * (int *)b );
}


void CompareLow(int j) {
    int i, min;
   // printf("My rank is %d Pairing with %d in CL\n", process_rank, process_rank^(1<<j));
    
    int send_counter = 0;
    // int *buffer_send = malloc((array_size + 1) * sizeof(int));
    int *buffer_send = new int[array_size + 1];  // Using new for dynamic allocation

   // printf("Trying to send local max in CL:%d\n", array[array_size-1]);
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Send(
        &array[array_size - 1],     
        1,                          
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD              
    );
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    int recv_counter;
    // int *buffer_recieve = malloc((array_size + 1) * sizeof(int));
    int *buffer_recieve = new int[array_size + 1];  // Using new for dynamic allocation

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Recv(
        &min,                       
        1,                          
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD,             
        MPI_STATUS_IGNORE           
    );
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    
    
   // printf("Received min from pair in CL:%d\n", min);
    for (i = array_size-1; i >= 0; i--) {
        if (array[i] > min) {
	    send_counter++;
            buffer_send[send_counter] = array[i];
	   // printf("Buffer sending in CL %d\n", array[i]);
        
        } else {
             break;      
        }
    }

    buffer_send[0] = send_counter;
   // printf("Send count in CL: %d\n", send_counter);

//	for(i=0;i<=send_counter;i++)
//		printf(" %d?? ", buffer_send[i]);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Send(
        buffer_send,                
        send_counter+1,               
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD              
    );
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Recv(
        buffer_recieve,             
        array_size+1,                 
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD,             
        MPI_STATUS_IGNORE           
    );
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // int *temp_array = (int *) malloc(array_size * sizeof(int));
    int *temp_array = new int[array_size];  // Using new for dynamic allocation

    //this was commeneted
    //memcpy(temp_array, array, array_size * sizeof(int));
    for(i=0;i<array_size;i++){
	    temp_array[i]=array[i];
    }
   
    int buffer_size=buffer_recieve[0];
    int k=1;int m=0;   

  //  for(k=0;k<=buffer_size;k++)
//	printf(" %d? ", buffer_recieve[k]);
 
    k=1;
    for (i = 0; i < array_size; i++) {
	//printf("Receive buffer element in CL: %d\n", buffer_recieve[i]);
    //    if (array[array_size - 1] < buffer_recieve[i]) {
      //      array[array_size - 1] = buffer_recieve[i];
       // } else {
         //   break;      
        //}
	if(temp_array[m]<=buffer_recieve[k])
	{
		array[i]=temp_array[m];
		m++;
	}
        else if(k<=buffer_size)
	{
		array[i]=buffer_recieve[k];
		k++;
	}
    }

    qsort(array, array_size, sizeof(int), ComparisonFunc);
  //  for(i=0;i<array_size;i++)
//	printf("My rank is %d, after exchange in CL %d\n", process_rank, array[i]);

  // int s=0;
   //for(i=0;i<array_size;i++)
	
//	{
//		printf("%d", array[i]);	
//		for(s=0;s<=process_rank;s++)
//			printf(":");
//	}
	
//	printf("\n");
	
      delete[] buffer_send;     // Free dynamically allocated memory
    delete[] buffer_recieve;
    delete[] temp_array;

    
    return;
}



void CompareHigh(int j) {

    //printf("My rank is %d Pairing with %d in CH\n", process_rank, process_rank^(1<<j));
    int i, max;

    int recv_counter;
    // int *buffer_recieve = malloc((array_size + 1) * sizeof(int));
    int *buffer_recieve = new int[array_size + 1];  // Using new for dynamic allocation

    
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Recv(
        &max,                       
        1,                          
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD,             
        MPI_STATUS_IGNORE           
    );

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

   // printf("Received max from pair in CH:%d\n",max);
    int send_counter = 0;
    // int *buffer_send = malloc((array_size + 1) * sizeof(int));
     int *buffer_send = new int[array_size + 1];  // Using new for dynamic allocation

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Send(
        &array[0],                  
        1,                          
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD              
    );

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

   // printf("Sending min to my pair from CH:%d\n", array[0]);
    for (i = 0; i < array_size; i++) {
        if (array[i] < max) {
	   // printf("Buffer sending in CH: %d\n", array[i]);
            	send_counter++;
		buffer_send[send_counter] = array[i];
        } else {
            break;      
        }
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    
    MPI_Recv(
        buffer_recieve,             
        array_size+1,                 
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD,             
        MPI_STATUS_IGNORE           
    );

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    recv_counter = buffer_recieve[0];

    buffer_send[0] = send_counter;
    //printf("Send counter in CH: %d\n", send_counter);

  //  for(i=0;i<=send_counter;i++)
//	printf(" %d>> ", buffer_send[i]);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Send(
        buffer_send,               
        send_counter+1,               
        MPI_INT,                    
        process_rank ^ (1 << j),    
        0,                          
        MPI_COMM_WORLD              
    );

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    int *temp_array = new int[array_size];  // Using new for dynamic allocation
    //memcpy(temp_array, array, array_size * sizeof(int));
    for(i=0;i<array_size;i++){
	    temp_array[i]=array[i];
    }

    int k=1;int m=array_size-1;
    int buffer_size=buffer_recieve[0];
		
    //for(i=0;i<=buffer_size;i++)
//	printf(" %d> ", buffer_recieve[i]);
    for (i = array_size-1; i >= 0; i--) {
	//printf("Buffer receive ele in CH: %d\n", buffer_recieve[i]);
        //if (buffer_recieve[i] > array[0]) {
          //  array[0] = buffer_recieve[i];
        //} else {
          //  break;      
        //}
      //  printf("buffer_rec[k] is %d, temp_array[m] is %d\n",buffer_recieve[k], temp_array[m]);
//	printf("M is %d k is %d i is %d\n",m, k, i);
	if(temp_array[m]>=buffer_recieve[k])
	{
		array[i]=temp_array[m];
		m--;
	}
	else if(k<=buffer_size){
		array[i]=buffer_recieve[k];
		k++;
	}
    }

    qsort(array, array_size, sizeof(int), ComparisonFunc);
	
   //int s=0;
   //for(i=0;i<array_size;i++)
	
	//{
	//	printf("%d", array[i]);	
	//	for(s=0;s<=process_rank;s++)
	//		printf(",");
	//}

    //printf("\n");
    
    delete[] buffer_send;     // Free dynamically allocated memory
    delete[] buffer_recieve;
    delete[] temp_array;

    return;
}