/**
 * \file sort.seq.cpp
 */

//#include "common.h"
//#include "common.cpp"
#include <stdlib.h>
#include <vector>
#include <limits>
#include <iostream>
#include <string>
#include <sstream>
#include <mpi.h>
#include <cstring>
#include <time.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
using namespace std;


// given a key and a mask, calcs the bucket to where the key belongs
#define GET_BUCKET_NUM(elem, mask, g, i) (((elem) & (mask)) >> ((g) * (i)))

#define LEN 10


//Caliper regions
const char* main_region = "main";
const char* data_init = "data_init";
const char* comp = "comp";
const char* comm = "comm";
const char* comp_large = "comp_large";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";
const char* Send = "MPI_Isend";
const char* Barrier = "MPI_Barrier";

//timers
double data_init_start, data_init_end, comp_small_start, comp_small_end, comm_small_start, comm_small_end, comm_large_start, comm_large_end, comp_large_start, comp_large_end, correctness_check_start, correctness_check_end; 

const char* inputType= "Random";


enum OrderCodes {
	ORDER_CORRECT = 0,
	ORDER_ERROR_BORDERS = -1,
	ORDER_ONLY_MASTER = -2,
};

enum Tags {
	TAG_KEY_SEND = 1,
	TAG_CHECK_ORDER = 2,
	TAG_COUT_SIZE = 3,
	TAG_COUT_DATA = 4,
	TAG_BUCKET_COUNT = 5,
};


/*void test_data(vector<unsigned int> *arr, int id, int size) {
	srand(time(NULL) * (id + 1));
	for(unsigned int i = 0; i < arr->size(); ++i)
		(*arr)[i] = rand() % 1000;
}*/

void dataInit(vector<unsigned int> *arr, int size, int inputTypeInt) {
  int numToSwitch = size / 100;
  int firstIndex, secondIndex;
  switch (inputTypeInt) {
    case 1:
      // sorted
      inputType= "Sorted";
      cout<<"sorted"<<endl;
      for (int i=0; i<size; i++) {
        (*arr)[i] = i;
      }
      break;
    case 2:
      // reverse sorted
      inputType= "ReverseSorted";
      cout<<"reverse"<<endl;
      for (int i=0; i<size; i++) {
        (*arr)[i] = size-i;
      }
      break;
    case 3:
      // randomized
      inputType= "Random";
      //cout<<"size "<<size<<endl;
      for (int i=0; i<size; i++) {
        (*arr)[i] = rand() % RAND_MAX;
      }
      break;
    case 4:
    {
      // 1% perturbed
      inputType= "1%perturbed";
      cout<<"weird"<<endl;
      for (int i=0; i<size; i++) {
        (*arr)[i] = i;
      }
      if (numToSwitch == 0)  // at the very least one value should be switched
        numToSwitch = 1;
      
       //printf("num to switch: %d\n", numToSwitch);
      for (int i=0; i<numToSwitch; i++) {
        firstIndex = rand() % size;
        secondIndex = rand() % size;
        //printf("first index: %d, second index: %d\n", firstIndex, secondIndex);
        while (firstIndex == secondIndex) {
          secondIndex = rand() % size;
        } 
       // printf(arr[firstIndex]);
        std::swap((*arr)[firstIndex], (*arr)[secondIndex]); 
      }
      //std::srand(std::time(0));

    // Number of elements to perturb (1% of the array size)
     /* int numToPerturb = size / 100;

    // Ensure at least one element is perturbed
      if (numToPerturb == 0) {
        numToPerturb = 1;
      }

      for (int i = 0; i < numToPerturb; ++i) {
        // Randomly select two indices
        int index1 = rand() % size;
        int index2 = rand() % size;
        std::swap((*arr)[index1], (*arr)[index2]);
      }*/
      break;
    }
    default:
      printf("THAT'S NOT A VALID INPUT TYPE");
      break;
  }
}

// count number of bits set to 1 in a number
unsigned int popcount(unsigned int x) {
	unsigned int c;
	for(c = 0; x; ++c) x &= x-1;
	return c;
}

// maps a given bucket to a cpu, given max number of buckets per cpu
#define BUCKET_TO_CPU(bucket)		(((bucket) & (~((bpp) - 1))) >> bpp_bc)
#define BUCKET_IN_CPU(bucket)       ( (bucket) & (  (bpp) - 1)            )

/**
 * it is assumed that B = P, only one bucket per processor
 *
 * \param arr Where to stop the ordered array (one local array per processor)
 * \param id  Id of this processs
 * \param p   Number of processor (equivalent to MPI_size)
 * \param g   Number of bits to use for each mask (must be power of 2)
 */
void radix_mpi(vector<unsigned int> *&arr, const unsigned int id, const unsigned int p, const unsigned int g) {

	//unsigned int p = size;				// num of processors
	//unsigned int g = 2;					// num bits for each pass
	const unsigned int b		= (1 << g);		// num of buckets (2^g)
	const unsigned int bpp	= b / p;			// num of buckets per cpu
	const unsigned int bpp_bc = popcount(bpp - 1);	// number of bits in bpp. used to compute BUCKET_TO_CPU

	unsigned int mask = ((1 << g) - 1);		// initial mask to get key


	vector<vector<unsigned int> > buckets(b);		// the buckets
	vector<vector<unsigned int> > bucket_counts;		// bin counts for each processor
	for(unsigned int i = 0; i < b; ++i)
		bucket_counts.push_back(vector<unsigned int>(p));

	vector<unsigned int> bucket_counts_aux(b); 		// aux vector, has only bucket counts for this process

	vector<vector<unsigned int> > bucket_accum;	// accumulated values for bucket_count. indicates for each process, where the values for each bucket should go
	for(unsigned int i = 0; i < bpp; ++i)
		bucket_accum.push_back(vector<unsigned int>(p));

	vector<unsigned int> bucket_sizes(bpp);
	vector<unsigned int> *this_bucket;			

	// dummy, MPI asks for them
	MPI_Request request;  // request handles for key communication
	MPI_Status  status;	  // status of key communication recvs

	for(unsigned int round = 0; mask != 0; mask <<= g, ++round) {
		// CALCULATE BUCKET COUNTS
   comp_large_start = MPI_Wtime();
   CALI_MARK_BEGIN(comp);

		// clear buckets
   CALI_MARK_BEGIN(comp_large);
		for(unsigned int i = 0; i < b; ++i) {
			bucket_counts_aux[i] = 0;
			bucket_counts[i][id] = 0;
			buckets[i].clear();
		}

		// fill buckets and bucket_count
		for(unsigned int i = 0; i < arr->size(); ++i) {
			unsigned int elem = (*arr)[i];
			unsigned int bucket = GET_BUCKET_NUM(elem, mask, g, round);
			bucket_counts_aux[bucket]++;
			bucket_counts[bucket][id]++;
			buckets[bucket].push_back(elem);
		}
   CALI_MARK_END(comp_large);
   CALI_MARK_END(comp);
   comp_large_end = MPI_Wtime();

		// SEND/RECV BUCKET COUNTS
	comm_large_start = MPI_Wtime();
   CALI_MARK_BEGIN(comm);
    
    CALI_MARK_BEGIN(comm_large);
		// sends my bucket counts to all other processes
   CALI_MARK_BEGIN(Send);
		for(unsigned int i = 0; i < p; ++i) {
			if (i != id)
				MPI_Isend(&bucket_counts_aux[0], b, MPI_INT, i, TAG_BUCKET_COUNT, MPI_COMM_WORLD, &request);
		}
   CALI_MARK_END(Send);
		// recv bucket counts from other processes
		for(unsigned int i = 0; i < p; ++i) {
			if (i != id) {
				MPI_Recv(&bucket_counts_aux[0], b, MPI_INT, i, TAG_BUCKET_COUNT, MPI_COMM_WORLD, &status);
				// copy from aux array to global matrix
				for(unsigned int k = 0; k < b; ++k)
					bucket_counts[k][i] = bucket_counts_aux[k];
			}
		}
   CALI_MARK_END(comm_large);
  
   CALI_MARK_END(comm);
   comm_large_end = MPI_Wtime();
   

		// CALCULATE BUCKET_ACCUMS
   comp_small_start = MPI_Wtime();
   CALI_MARK_BEGIN(comp);
   CALI_MARK_BEGIN(comp_large);

		// count total size of bucket for this process, and alloc it. also compute bucket_accum
		int total_bucket_size = 0;
		for(unsigned int i = 0; i < bpp; ++i) {
			int single_bucket_size = 0;
			int global_bucket = i + id*bpp;

			for(unsigned int j = 0; j < p; ++j) {
				bucket_accum[i][j] = total_bucket_size;
				single_bucket_size += bucket_counts[global_bucket][j];
				total_bucket_size  += bucket_counts[global_bucket][j];
			}
			bucket_sizes[i] = single_bucket_size;
		}
   CALI_MARK_END(comp_large);
  
    CALI_MARK_END(comp);
	comp_small_end = MPI_Wtime();
    
		this_bucket = new vector<unsigned int>(total_bucket_size);

		// send keys across each process
   comm_small_start = MPI_Wtime();
   CALI_MARK_BEGIN(comm);
   CALI_MARK_BEGIN(comm_large);
   CALI_MARK_BEGIN(Send);
		for(unsigned int i = 0; i < b; ++i) {
			unsigned int dest = BUCKET_TO_CPU(i);
			unsigned int local_bucket = BUCKET_IN_CPU(i);
			// send data from a single bucket to its corresponding process
			if (dest != id && buckets[i].size() > 0) {
				MPI_Isend(&(buckets[i][0]), buckets[i].size(), MPI_INT, dest, local_bucket, MPI_COMM_WORLD, &request);
			}
		}
   CALI_MARK_END(Send);

		// recv keys
		for(unsigned int b = 0; b < bpp; ++b) {
			unsigned int global_bucket = b + id*bpp;

			for(unsigned int i = 0; i < p; ++i) {
				unsigned int bucket_size = bucket_counts[global_bucket][i];

				if (bucket_size > 0) {
					unsigned int *dest = &(*this_bucket)[ bucket_accum[b][i] ];
					// if its the same process, copy data from buckets[i] to this_bucket
					if (i == id) {
						memcpy(dest, &(buckets[global_bucket][0]), bucket_size * sizeof(int));
					}
	
					// otherwise recv data from process i
					else {
						MPI_Recv(dest, bucket_size, MPI_INT, i, b, MPI_COMM_WORLD, &status);
					}
				}
			}
		}
   CALI_MARK_END(comm_large);
   CALI_MARK_END(comm);
   comm_small_end = MPI_Wtime();

		delete arr;
		arr = this_bucket;
	}
}

/*int check_array_order(vector<unsigned int> *&arr, unsigned int id, unsigned int size) {

	// check local array order
	for(unsigned int i = 1; i < arr->size(); ++i)
		if ((*arr)[i - 1] > (*arr)[i])
			return i;

	int is_ordered = 1, reduce_val;
	unsigned int next_val;
	MPI_Request request;
	MPI_Status status;

	// all processes but first send first element to previous process
	if (id > 0) {
		// if local array is size 0, send max_int
		int val_to_send = (arr->size() == 0) ? numeric_limits<int>::max() : (*arr)[0];
		MPI_Isend(&val_to_send, 1, MPI_INT, id - 1, TAG_CHECK_ORDER, MPI_COMM_WORLD, &request);
	}

	// all processes but last receive element from next process and compare it to their last one
	if (id < (unsigned int) size - 1) {
		MPI_Recv(&next_val, 1, MPI_INT, id + 1, TAG_CHECK_ORDER, MPI_COMM_WORLD, &status);

		// this link is ordered if last local value is <= than received value, or if local size is 0
		is_ordered = (arr->size() == 0 || arr->back() <= next_val);
	}

	// reduce all values, to check if border order is met
	MPI_Reduce(&is_ordered, &reduce_val, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);

	// reduce result only goes to process 0
	if (id == 0) {
		if (reduce_val)
			return ORDER_CORRECT;
		else
			return ORDER_ERROR_BORDERS;
	}

	return ORDER_ONLY_MASTER;
}*/
int check_array_order(vector<unsigned int> *arr, int size) {
  //CALI_MARK_BEGIN(correctness_check);
  // for (int i=0; i<size-1; i++) {
  //   if ((*arr)[i+1] < (*arr)[i])
  //     return 0;  // means it's not ordered correctly
  // }
 // CALI_MARK_END(correctness_check);
 for(unsigned int i = 2; i < arr->size(); ++i)
		if ((*arr)[i - 1] > (*arr)[i])
			return i;

  return 0;
}

#define MSG_SIZE 100

int main(int argc, char *argv[]) {
	CALI_MARK_BEGIN(main_region);
	int g = 2;
  int inputTypeInt = 0;

	char msg[MSG_SIZE];
	int id, size;

	cali::ConfigManager mgr;
    mgr.start();

	int len;
	stringstream ss;
  // printf("num args: %d \n", argc);
   //printf("arg 1: %s \n", (argv[0]));
   //printf("arg 2: %s \n", (argv[1]));
   //printf("arg 3: %s \n", (argv[2]));
   
 MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc ==3){
  	 len = atoi(argv[1]);
     inputTypeInt = atoi(argv[2]);
   }
	else			len = LEN;

	//if (argc > 2)	g = atoi(argv[2]);
 


	if (id == 0) cerr << "mask size = " << g << endl << endl;
 
  data_init_start = MPI_Wtime();
  CALI_MARK_BEGIN(data_init);
	vector<unsigned int> *arr = new vector<unsigned int>(len / size);
	// generate test data
   //test_data(arr, id, (len/size)); 
  //cout<<"len "<<len<<endl;
 // cout<<"size "<<size<<endl;
  //cout<<"(len/size) "<<(len/size)<<endl;
  
  if (id == 0) dataInit(arr, (len/size),inputTypeInt);
   CALI_MARK_END(data_init);
   data_init_end = MPI_Wtime();
  // cout << "(*arr)[0]" <<(*arr)[0]<<endl;

	// the real stuff
	if (id == 0) cerr << "starting radix sort...";
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(Barrier);
	MPI_Barrier(MPI_COMM_WORLD);
 CALI_MARK_END(Barrier);
   CALI_MARK_END(comm);
//	timer.start();
  //cerr << "g = " << g << endl;//g=2; 
 // g=8;     
  //cerr << "arr.size before = " << arr->size() << endl << endl; 
	radix_mpi(arr, id, size, g);
  //cerr << "arr.size after = " << arr->size() << endl << endl;
//	timer.stop();
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(Barrier);
	MPI_Barrier(MPI_COMM_WORLD);
  CALI_MARK_END(Barrier);
   CALI_MARK_END(comm);
 
	if (id == 0) cerr << "finished" << endl << endl;
 
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(Barrier);
	MPI_Barrier(MPI_COMM_WORLD);
 CALI_MARK_END(Barrier);
 CALI_MARK_END(comm);
	// check array order
  correctness_check_start = MPI_Wtime();
  CALI_MARK_BEGIN(correctness_check);
   //cerr << "size = " << size << endl ;
   //cerr << "arr.size = " << arr->size() << endl << endl;
	int order = check_array_order(arr, size);
  CALI_MARK_END(correctness_check);
  correctness_check_end = MPI_Wtime();
  
  
	switch (order) {
		case ORDER_CORRECT: 	cerr << "CORRECT! Result is ordered" << endl; break;
		case ORDER_ONLY_MASTER: break;
		default: 				cerr << "WRONG! Order fails at index " << order << endl; break;
	}

 
 CALI_MARK_END(main_region);
 if (id == 0) {
   cout << "\n";
    cout << "Sorted Array: ";
    // cout<<"arr->size() "<<arr->size()<<endl;
    for (unsigned int i = 0; i < arr->size(); ++i) {
        cout << (*arr)[i] << " ";
    }
    cout << endl;
  }
  
  const char* algorithm ="RadixSort";
  const char* programmingModel = "MPI"; 
  const char* datatype = "int"; 
  int sizeOfDatatype =4;
  int inputSize =len; 
  int num_procs = size; 
  int group_number =10;
  const char* implementation_source = "Online"; 
  
  float data_init_time = data_init_end - data_init_start;
  float comm_small_time = comm_small_end - comm_small_start;
  float comm_large_time = comm_large_end - comm_large_start;
  float comp_small_time = comp_small_end - comp_small_start;
  float comp_large_time = comp_large_end - comp_large_start;
  float correctness_check_time = correctness_check_end - correctness_check_start;
  float comm_time = comm_large_time + comm_small_time;
  float comp_time = comp_large_time + comp_small_time;

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
  //adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
  //adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
  adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
  
  adiak::value("data_init", data_init);
  adiak::value("comm", comm);
  adiak::value("comp", comp);
  //adiak::value("comm_small", comm_small);
  adiak::value("comm_large", comm_large);
  //adiak::value("comp_small", comp_small);
  adiak::value("comp_large", comp_large);
  adiak::value("correctness_check", correctness_check);
  adiak::value("MPI_Barrier", Barrier);
  adiak::value("MPI_Isend", Send);
  
  if (id == 0) {
    printf("\n");
    printf("comm_small_time: %f \n", comm_small_time);
    printf("comm_large_time: %f \n", comm_large_time);
    printf("comp_small_time: %f \n", comp_small_time);
    printf("comp_large_time: %f \n", comp_large_time);
  }
  
	mgr.stop();
    mgr.flush();
	MPI_Finalize();
}