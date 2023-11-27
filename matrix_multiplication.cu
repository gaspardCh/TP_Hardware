#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void MatrixInit(float *M, int n, int p){
	int i,j;
	for (i = 0; i < n; i++){
		for (j = 0; j < p; j++){
			M[i*p+j] = (float)rand()/RAND_MAX * 10;
		}
	}
}

void MatrixPrint(float *M, int n, int p){
	for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            printf("%.2f\t", M[i * p + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
	int i,j;
	for (i = 0; i < n; i++){
		for (j = 0; j < p; j++){
			Mout[i*p+j] = M1[i*p+j] + M2[i*p+j] ;
		}
	}
}

__global__ void cudaMatrixAdd (float *M1, float *M2, float *M3) {
	M3[blockIdx.x] = M1[blockIdx.x] + M2[blockIdx.x];
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){
	int i,j;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			float product = 0;
			for (int k = 0; k<n; k++){
				product += M1[i*n+k]*M2[k*n+j];
			}
			Mout[i*n+j] = product;
		}
	}
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < n && col < n){
		float product = 0;
		for (int k = 0; k<n; k++){
			product += M1[row*n+k]*M2[k*n+col];
		}
		Mout[row*n+col] = product;
	}
}

int main(void){
	srand(time(NULL));
	int n = 3;
	int p = 3;	
	int N = n*p;
	int size = N*sizeof(float);
	float *M1_cu, *M2_cu, *M3_cu;
	
	// Allocate memory
	float *M1 = (float*) malloc(size);
	float *M2 = (float*) malloc(size);
	float *M3 = (float*) malloc(size);
	float *M4 = (float*) malloc(size);

	cudaMalloc((void **) &M1_cu, size);
	cudaMalloc((void **) &M2_cu, size);
	cudaMalloc((void **) &M3_cu, size);

	// Initialize matrix
	MatrixInit(M1, n, p);
	MatrixPrint(M1, n, p);
	MatrixInit(M2, n, p);
	MatrixPrint(M2, n, p);
	
	// Send matrix to GPU
	cudaMemcpy(M1_cu, M1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(M2_cu, M2, size, cudaMemcpyHostToDevice);
	
	// Add matrixes
	//cudaMatrixAdd<<<N,1>>>(M1_cu, M2_cu, M3_cu);
	//cudaMemcpy(M3, M3_cu, size, cudaMemcpyDeviceToHost);
	
	dim3 dimBlock(N,N);
	dim3 dimGrid(ceil(N/16.0), ceil(N/16.0));
	
	// Multiply matrixes
	MatrixMult(M1, M2, M4, n);
	cudaMatrixMult<<<dimBlock, dimGrid>>>(M1_cu, M2_cu, M3_cu, n);
	cudaMemcpy(M3, M3_cu, size, cudaMemcpyDeviceToHost);
	
	// Print result
	MatrixPrint(M3, n, p);
	MatrixPrint(M4, n, p);

	
	
}

