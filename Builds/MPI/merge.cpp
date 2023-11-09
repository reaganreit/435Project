#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

double dataInitStart, dataInitEnd, dataInitTime;
double scatterStart, scatterEnd, scatterTime;
double gatherStart, gatherEnd, gatherTime;
double processMergeStart, processMergeEnd, processMergeTime;
double finalMergeStart, finalMergeEnd, finalMergeTime;
double checkSortStart, checkSortEnd, checkSortTime;
double commTime, compTime;

/********** Caliper Regions **********/
const char* dataInitialization = "data_initialization";
const char* commLarge = "comm_large";
const char* compLarge = "comp_large";
const char* checkSort = "check_sort";
const char* comm = "comm";
const char* comp = "comp";
const char* mainCali = "main";

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

	if(l < r) {	
		m = (l + r)/2;

		mergeSort(a, b, l, m);
		mergeSort(a, b, (m + 1), r);
		merge(a, b, l, m, r);		
	}
}

/********** Data Initialization **********/
void data_init(int* array, int n, int world_rank) {
    if (world_rank == 0) {
        //random
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            array[i] = rand() % 5000;
        }
        /**
        //sorted
        for (int i = 0; i < n; i++) {
            array[i] = i;
        }
        //reverse sorted
        for (int i = n; i >= 0; i--) {
            array[i] = i;
        }
        **/
    }
}

/********** Check Sorting **********/
int check_sorted(int *array, int size) {
    CALI_MARK_BEGIN(checkSort);
    checkSortStart = MPI_Wtime();
    for (int i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            return 0;
        }
    }
    checkSortEnd = MPI_Wtime();
    CALI_MARK_END(checkSort);
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
  	int* original_array = static_cast<int*>(malloc(n * sizeof(int)));
    
    if (world_rank == 0) {
        CALI_MARK_BEGIN(dataInitialization);
        dataInitStart = MPI_Wtime();
        data_init(original_array, n, world_rank);
        dataInitEnd = MPI_Wtime();
        CALI_MARK_END(dataInitialization);
        
        printf("This is the unsorted array: ");
        for (int c = 0; c < n; c++) {
            printf("%d ", original_array[c]);
        }
        printf("\n");
    }
    
    //dividing array into equal sized chunks and sending data to each process
    int size = n/world_size;
    int *sub_array = static_cast<int*>(malloc(size * sizeof(int)));
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(commLarge);
    scatterStart = MPI_Wtime();
    MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
    scatterEnd = MPI_Wtime();
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);
    
    /********** Each process does merge sort **********/
  	int *tmp_array = static_cast<int*>(malloc(size * sizeof(int)));
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(compLarge);
    processMergeStart = MPI_Wtime();
  	mergeSort(sub_array, tmp_array, 0, (size - 1));
    processMergeEnd = MPI_Wtime();
    CALI_MARK_END(compLarge);
    CALI_MARK_END(comp);
    
  	/********** Gather the sorted subarrays into one **********/
  	int *sorted = NULL;
  	if(world_rank == 0) {
        sorted = static_cast<int*>(malloc(n * sizeof(int)));
  	}
    
    CALI_MARK_BEGIN(comm);
  	CALI_MARK_BEGIN(commLarge);
    gatherStart = MPI_Wtime();
   	MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
    gatherEnd = MPI_Wtime();
    CALI_MARK_END(commLarge);
    CALI_MARK_END(comm);
  
	/********** Make the final mergeSort call **********/
	  if(world_rank == 0) {
    		int *other_array = static_cast<int*>(malloc(n * sizeof(int)));
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(compLarge);
        finalMergeStart = MPI_Wtime();
    		mergeSort(sorted, other_array, 0, (n - 1));
        finalMergeEnd = MPI_Wtime();
        CALI_MARK_END(compLarge);
        CALI_MARK_END(comp);
        
        /********** Check if sorted and print **********/
        if (check_sorted(sorted, n)) {
          // Display the sorted array
          printf("This is the sorted array: ");
          for (int c = 0; c < n; c++) {
              printf("%d ", sorted[c]);
          }
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
 
    MPI_Barrier(MPI_COMM_WORLD);
    
    /********** Calculating and printing time *********/
    dataInitTime = dataInitEnd - dataInitStart;
    scatterTime = scatterEnd - scatterStart;
    gatherTime = gatherEnd - gatherStart;
    processMergeTime = processMergeEnd - processMergeStart;
    finalMergeTime = finalMergeEnd - finalMergeStart;
    checkSortTime = checkSortEnd - checkSortStart;
    commTime = scatterTime + gatherTime;
    compTime = processMergeTime + finalMergeTime;
    
    if (world_rank == 0) {
        printf("Data Initialization Time: %f seconds\n", dataInitTime);
        printf("Scatter Time: %f seconds\n", scatterTime);
        printf("Gather Time: %f seconds\n", gatherTime);
        printf("Process Merge Time: %f seconds\n", processMergeTime);
        printf("Final Merge Time: %f seconds\n", finalMergeTime);
        printf("Check Sort Time: %f seconds\n", checkSortTime);
        printf("Communication Time: %f seconds\n", commTime);
        printf("Computation Time: %f seconds\n", compTime);
    }

    
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
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", world_size); // The number of processors (MPI ranks)
    adiak::value("group_num", 10); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    adiak::value("data_init_time", dataInitTime);
    adiak::value("mpi_scatter_time", scatterTime);
    adiak::value("mpi_gather_time", gatherTime);
    adiak::value("processes_mergesort_time", processMergeTime);
    adiak::value("final_merge_time", finalMergeTime);
    adiak::value("validate_sorting_time", checkSortTime);
    adiak::value("communication_time", commTime);
    adiak::value("computation_time", compTime);
    
    /********** Finalize MPI **********/    
    mgr.stop();
    mgr.flush();
	  MPI_Finalize();
    CALI_MARK_END(mainCali);
}