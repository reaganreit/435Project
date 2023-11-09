/* CUDA C Program to to merge sort of a list in ascending order */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "helpers.cuh"
#include <assert.h>

void __device__ merge(float *list,int l,int m,int r);
void __global__ mergesort(float *list, int SIZE);

//check whether a certain number is a power of 2
int isPowerOfTwo(int num){
	int i=0;
	int val=1;
	for(i=0;val<=num;i++){
		if((val=pow(2,i))==num){
			return 1;
		}
	}				
	return 0;	

}

int main(int argc, char **argv){

  int SIZE = atoi(argv[2]);

	//check the condition that check that checks whether the size is a power of 2
	if(!isPowerOfTwo(SIZE)){
		fprintf(stderr,"This implementation needs the list size to be a power of two\n");
		exit(1);
	}
	
	//allocate a list
	float *list = (float *)malloc(sizeof(float)*SIZE);
	if(list==NULL){
		perror("Mem full");
		exit(1);
	}
	
	int i;
	//generate some random values
	for(i=0;i<SIZE;i++){
		list[i]=rand()/(float)100000;
	}
	
	//print the input list
	printf("The input list is : \n");
	for(i=0;i<SIZE;i++){
		printf("%.2f ",list[i]);
	}
	printf("\n\n");
	
	/********************************** CUDA stuff starts here *******************************/
	
	//start measuring time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);		

	//pointers for memory allocation in cudaa
	float *list_cuda;
	
	//allocate memory in cuda
	checkCuda(cudaMalloc((void **)&list_cuda,sizeof(float)*SIZE));
	
	//copy memory from ram to cuda
	checkCuda(cudaMemcpy(list_cuda,list,sizeof(float)*SIZE,cudaMemcpyHostToDevice));
	
	//thread configurations
	int threadsPerBlock = atoi(argv[1]);
	int numBlocks=ceil(SIZE/((float)256*2));
	/* The reason to divide by 2 is because now we need a thread per two elements only*/
	
	//start measuring time for cuda kernel only
	cudaEvent_t startkernel,stopkernel;
	float elapsedtimekernel;
	cudaEventCreate(&startkernel);
	cudaEventRecord(startkernel,0);		
		
	//do sorting
	mergesort<<<numBlocks,threadsPerBlock>>>(list_cuda, SIZE);	
	checkCuda(cudaGetLastError());

	//end measuring time for cuda kernel
	cudaEventCreate(&stopkernel);
	cudaEventRecord(stopkernel,0);
	cudaEventSynchronize(stopkernel);
	cudaEventElapsedTime(&elapsedtimekernel,startkernel,stopkernel);
		
	//copy the answer back from cuda ro ram
	checkCuda(cudaMemcpy(list,list_cuda,sizeof(float)*SIZE,cudaMemcpyDeviceToHost));

	//free the cuda memory
	checkCuda(cudaFree(list_cuda));
	
	//end measuring time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	
	/********************** CUDA stuff ends here ********************************/
	
	//print the answer
	printf("The sorted list is : \n");
	for(i=0;i<SIZE;i++){
		printf("%.2f ",list[i]);
	}
	printf("\n\n");	
	
	//print the time spent to stderr
	fprintf(stderr,"Time spent for CUDA kernel is %1.5f seconds\n",elapsedtimekernel/(float)1000); 
	fprintf(stderr,"Time spent for operation on CUDA(Including memory allocation and copying) is %1.5f seconds\n",elapsedtime/(float)1000); 

	return 0;
}


/* merge two lists while sorting them in ascending order
* For example say there are two arrays 
* while one being 1 3 5 and the other being 2 4 7
* when merge they become 1 2 3 4 5 7
* When storing the two lists they are stored in same array and the
* two arrays are specified using the index of leftmost element, middle element and the last element
* For example say the two arrays are there in memory as a single array 1 3 5 2 4 7
* Here l=0 m=3 and r=5 specifies the two arrays separately
* */


__device__ void merge(float *list, int left,int middle,int right){
	
	//calculate the total number of elements
	int n=right-left+1;
	
	//create a new temporary array to do merge
	float *temp=(float *)malloc(sizeof(float)*n);
	assert(temp!=NULL);
	//use assert to check whether memory allocation happens successfully
	//if a null pointer was returned program will terminate
	
	//i is used for indexing elements in left array and j is used for indexing elements in the right array
	int i=left;
	int j=middle;
	
	//k is the index for the temporary array
	int k=0;
	
	/*now merge the two lists in ascending order
	check the first element remaining in each list and select the lowest one from them. Then put it to temp
	put increase the relevant index i or j*/
	
	while(i<middle && j<=right){
		if(list[i]<list[j]){
			temp[k]=list[i];
			i++;
		}
		else{
			temp[k]=list[j];
			j++;
		}
		k++;
	}
	
	//if there are remaining ones in an array append those to the end
	while(i<middle){
		temp[k]=list[i];
		i++;
		k++;	
	}
	while(j<=right){
		temp[k]=list[j];
		j++;
		k++;		
	}
	
	//now copy back the sorted array in temp to the original
	for(i=left,k=0;i<=right;i++,k++){
		list[i]=temp[k];	
	}
	
	free(temp);
	
}

/* carry out merge sort ascending*/
__global__ void mergesort(float *list, int SIZE){
	
	//calculate threadindex
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//step means the distance to the next list
	//loop till the merging happens for a list of the size of the original list
	int step=1;
	while(step<SIZE-1){

		if(tid%step==0 && tid*2<SIZE){
			
			//calculate the index of the first element of the first list
			int left=2*tid;
			
			//calculate the index of the first element of  the second list
			int middle=2*tid+step;
			
			//calculate the last element of the second list
			int right=2*tid+2*step-1;
			
			//merge the two lists
			merge(list,left,middle,right);
					
		}
		
		//next list size
		step=step*2;
			
		//synchronize all threads
		__syncthreads();			
	}

}