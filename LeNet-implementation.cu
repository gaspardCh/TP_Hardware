#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

void MatrixInit(float *M, int n, int p, int d, float coeff){ //n,p,d
	int i,j,k;
	if (d==0){
		for (i = 0; i < n; i++){
			for (j = 0; j < p; j++){
				M[i*p+j] = (float)rand()/RAND_MAX * 10;
			}
		}
	}
	else{
		for (i = 0; i < n; i++){
			for (j = 0; j < p; j++){
				for (k = 0; k < d; k++){
					M[i*p*d+j*d+k] = (float)rand()*coeff/RAND_MAX;
				}
			}
		}
	}
}

void MatrixPrint(float *M, int n, int p, int d){
	for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
			for (int k = 0; k < d; k++){
				printf("%.2f\t", M[i * p * d + j * d + k]);
			}
			printf("\n");
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
	int raw_data_size = 32*32*sizeof(float);
	int C1_data_size = 6*28*28*sizeof(float);
	int S1_data_size = 6*14*14*sizeof(float);
	int C1_kernel_size = 6*5*5*sizeof(float);
	
	
	float *raw_data = (float*) malloc(raw_data_size);
	float *C1_data = (float*) malloc(C1_data_size);
	float *S1_data = (float*) malloc(S1_data_size);
	float *C1_kernel = (float*) malloc(C1_kernel_size);
	
	MatrixInit(raw_data, 32, 32, 0, 1);
	MatrixInit(C1_data, 6, 28, 28, 0);
	MatrixInit(S1_data, 6, 14, 14, 0);
	MatrixInit(C1_kernel, 6, 5, 5, 1);
	
	MatrixPrint(C1_kernel, 6, 5, 5);
	
}
