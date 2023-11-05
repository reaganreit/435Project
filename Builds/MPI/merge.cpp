#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);

double wholeSortStart, wholeSortEnd, sortStepStart, sortStepEnd, wholeSortTime, sortStepTime, workDivStart, workDivEnd, workDivTime, gatherStart, gatherEnd, gatherTime;

//Caliper regions
const char* wholeSort = "whole_sort";
const char* sortStep = "sort_step";
const char* workDivision = "work_division";
const char* gather = "gather";

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

int main(int argc, char** argv) {
  /********** Initialize MPI **********/
	int world_rank;
	int world_size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
 
  //config manager
  cali::ConfigManager mgr;
  mgr.start();
 
	/********** Create and populate the array **********/
	int n = atoi(argv[1]);
	int* original_array = static_cast<int*>(malloc(n * sizeof(int)));
	
	int c;
 
  if(world_rank == 0) {
    srand(time(NULL));
  	printf("This is the unsorted array: ");
   
  	for(c = 0; c < n; c++) {
  		original_array[c] = rand() % n;
  		printf("%d ", original_array[c]);
  	}
    printf("\n");
  }
		
	/********** Divide the array in equal-sized chunks **********/
	int size = n/world_size;
	
	// WORK DIVISION START
  CALI_MARK_BEGIN(workDivision);
  workDivStart = MPI_Wtime();
	int *sub_array = static_cast<int*>(malloc(size * sizeof(int)));
	MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
  workDivEnd = MPI_Wtime();
  CALI_MARK_END(workDivision);
  // WORK DIVISION END
	
 
  // TOTAL SORT TIME START
  CALI_MARK_BEGIN(wholeSort);
  wholeSortStart = MPI_Wtime();
  
	// WORKER STEP SORT START
  CALI_MARK_BEGIN(sortStep);
  sortStepStart = MPI_Wtime();
	int *tmp_array = static_cast<int*>(malloc(size * sizeof(int)));
	mergeSort(sub_array, tmp_array, 0, (size - 1));
  // WORKER STEP SORT END
  sortStepEnd = MPI_Wtime();
  CALI_MARK_END(sortStep);
	
	/********** Gather the sorted subarrays into one **********/
	int *sorted = NULL;
	if(world_rank == 0) {
		sorted = static_cast<int*>(malloc(n * sizeof(int)));
	}
	
  CALI_MARK_BEGIN(gather);
  gatherStart = MPI_Wtime();
	MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
	gatherEnd = MPI_Wtime();
  CALI_MARK_END(gather);
  
	/********** Make the final mergeSort call **********/
	if(world_rank == 0) {
		int *other_array = static_cast<int*>(malloc(n * sizeof(int)));
		mergeSort(sorted, other_array, 0, (n - 1));
   
    // TOTAL SORT TIME END
    wholeSortEnd = MPI_Wtime();
	  CALI_MARK_END(wholeSort);	
   
		/********** Display the sorted array **********/
		printf("This is the sorted array: ");
		for(c = 0; c < n; c++) {
			printf("%d ", sorted[c]);
		}
			
		printf("\n");
		printf("\n");
			
		/********** Clean up root **********/
		free(sorted);
		free(other_array);
		}
	
	/********** Clean up rest **********/
	free(original_array);
	free(sub_array);
	free(tmp_array);
 
  MPI_Barrier(MPI_COMM_WORLD);
 
  //COMPUTE TIME
  wholeSortTime = wholeSortEnd - wholeSortStart;
  sortStepTime = sortStepEnd - sortStepStart;
  workDivTime = workDivEnd - workDivStart;
  gatherTime = gatherEnd - gatherStart;
  
  //Printing Times
  if (world_rank == 0) {
    std::cout << "whole sort time: " << wholeSortTime << std::endl;
    std::cout << "sort step time: " << sortStepTime << std::endl;
    std::cout << "work div time: " << workDivTime << std::endl;
    std::cout << "gather time: " << gatherTime << std::endl;
  }
	
	/********** Finalize MPI **********/
	
  mgr.stop();
  mgr.flush();
	MPI_Finalize();
}