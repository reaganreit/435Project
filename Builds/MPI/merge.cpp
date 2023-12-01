#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <algorithm>

/********** Caliper Regions **********/
const char* dataInitialization = "data_init";
const char* commLarge = "comm_large";
const char* compLarge = "comp_large";
const char* correctnessCheck = "correctness_check";
const char* comm = "comm";
const char* comp = "comp";
const char* mainCali = "main";
const char* mpiBarrier = "MPI_Barrier";
const char* mpiScatter = "MPI_Scatter";
const char* mpiGather = "MPI_Gather";

/********** Merge Function **********/
void merge(int *a, int *b, int l, int m, int r) {
  	int h, i, j, k;
  	h = l;
  	i = l;
  	j = m + 1;
  	
  	while((h <= m) && (j <= r)) {
    		if(a[h] <= a[j]) {
      			b[i] = a[h];
      			h++;
    		}
    		else {	
      			b[i] = a[j];
      			j++;
    		}
    
    		i++;
  	}
  		
  	if(m < h) {
    		for(k = j; k <= r; k++) {
      			b[i] = a[k];
      			i++;
    		}
  	}	
  	else {
    		for(k = h; k <= m; k++) {
      			b[i] = a[k];
      			i++;
    		}	
  	}
  		
  	for(k = l; k <= r; k++) {	
  		  a[k] = b[k];
  	}
}

/********** Recursive Merge Function **********/
void mergeSort(int *a, int *b, int l, int r) {
  	int m;
  	if (l < r) {	
    		m = (l + r)/2;
    
    		mergeSort(a, b, l, m);
    		mergeSort(a, b, (m + 1), r);
    		merge(a, b, l, m, r);		
  	}
}

/********** Data Initialization **********/
void data_init(int arr[], int size, int inputType) {
    
    int numToSwitch = size / 100;
    int firstIndex, secondIndex;

    switch (inputType)
    {
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
            if (numToSwitch == 0) {  // at the very least one value should be switched
                numToSwitch = 1;
            }
            
            for (int i=0; i<numToSwitch; i++) {
                firstIndex = rand() % size;
                secondIndex = rand() % size;
                while (firstIndex == secondIndex) {
                    secondIndex = rand() % size;
                } 
                std::swap(arr[firstIndex], arr[secondIndex]); 
            }
            break;
        default:
            printf("THAT'S NOT A VALID INPUT TYPE\n");
            break;
    }
}

/********** Check Sorting **********/
int check_sorted(int *array, int size) {
    CALI_MARK_BEGIN(correctnessCheck);
    for (int i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            return 0;
        }
    }
    CALI_MARK_END(correctnessCheck);
    return 1;
}

int main(int argc, char** argv) {
    CALI_MARK_BEGIN(mainCali);
    /********** Initialize MPI **********/
  	int world_rank;
  	int world_size;
  	
  	MPI_Init(&argc, &argv);
  	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
 
    //config manager
    cali::ConfigManager mgr;
    mgr.start();
 
  	/********** Data Initialization **********/
  	int n = atoi(argv[1]);
    int sortType = atoi(argv[2]);
  	int* original_array = static_cast<int*>(malloc(n * sizeof(int)));
    
    if (world_rank == 0) {
        CALI_MARK_BEGIN(dataInitialization);
        data_init(original_array, n, sortType);
        CALI_MARK_END(dataInitialization);
        /**
        printf("This is the unsorted array: \n");
        for (int c = 0; c < n; c++) {
            printf("%d ", original_array[c]);
        }
        printf("\n");**/
    }
    
    //dividing array into equal sized chunks and sending data to each process
    int size = n/world_size;
    int *sub_array = static_cast<int*>(malloc(size * sizeof(int)));
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commLarge);
    CALI_MARK_BEGIN(mpiScatter);
    MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(mpiScatter);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);

    /********** Each process does merge sort **********/
  	int *tmp_array = static_cast<int*>(malloc(size * sizeof(int)));
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(compLarge);
  	mergeSort(sub_array, tmp_array, 0, (size - 1));
    CALI_MARK_END(compLarge);
    CALI_MARK_END(comp);
    
  	/********** Gather the sorted subarrays into one **********/
  	int *sorted = NULL;
  	if(world_rank == 0) {
        sorted = static_cast<int*>(malloc(n * sizeof(int)));
  	}
    
    CALI_MARK_BEGIN(comm);
  	CALI_MARK_BEGIN(commLarge);
    CALI_MARK_BEGIN(mpiGather);
   	MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(mpiGather);
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);
  
	/********** Make the final mergeSort call **********/
        if(world_rank == 0) {
            int *other_array = static_cast<int*>(malloc(n * sizeof(int)));
            CALI_MARK_BEGIN(comp);
            CALI_MARK_BEGIN(compLarge);
            mergeSort(sorted, other_array, 0, (n - 1));
            CALI_MARK_END(compLarge);
            CALI_MARK_END(comp);
            
            /********** Check if sorted and print **********/
            if (check_sorted(sorted, n)) {
            /**
                for (int c = 0; c < n; c++) {
                    printf("%d ", sorted[c]);
                }**/
                printf("\nSorted!!!");
            }
            else {
                printf("Error: The array is not sorted.\n");
            }
            
            printf("\n\n");
                    
            /********** Clean up root **********/
            free(sorted);
            free(other_array);
        }    
	
  	/********** Clean up rest **********/
  	free(original_array);
  	free(sub_array);
  	free(tmp_array);
 
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(mpiBarrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(mpiBarrier);
    CALI_MARK_END(comm);

    CALI_MARK_END(mainCali);



    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", sortType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", world_size); // The number of processors (MPI ranks)
    adiak::value("group_num", 10); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    adiak::value("data_init_time", dataInitialization);
    adiak::value("correctness_check", correctnessCheck);
    adiak::value("mpi_barrier", mpiBarrier);
    adiak::value("comm", comm);
    adiak::value("comp", comp);
    adiak::value("comm_large", commLarge);
    adiak::value("comp_large", compLarge);
    adiak::value("mpi_scatter", mpiScatter);
    adiak::value("mpi_gather", mpiGather);
    adiak::value("main", mainCali);

    /********** Finalize MPI **********/    
    mgr.stop();
    mgr.flush();
	MPI_Finalize();
}