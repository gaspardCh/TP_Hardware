#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

void MatrixInit(float *M, int n, int p, int d, float coeff){ //n,p,d
	int i,j,k;
	if (d==0){
		for (i = 0; i < n; i++){
			for (j = 0; j < p; j++){
				
				M[i*p+j] = (float)(rand()*coeff)/RAND_MAX;
			}
		}
	}
	else{
		for (i = 0; i < n; i++){
			for (j = 0; j < p; j++){
				for (k = 0; k < d; k++){
					M[i*p*d+j*d+k] = (float)(rand()*coeff)/RAND_MAX;
				}
			}
		}
	}
}

void MatrixPrint2D(float *M, int n, int p){
	for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            printf("%.2f\t", M[i * p + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void MatrixPrint3D(float *M, int n, int p, int d){
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

__global__ void cudaMatrixConv2D(float *data, float *kernel, float *Mout, int n_data, int p_data, int n_kernel, int p_kernel){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	float coeff;
	
	if ( (n_kernel/2)<row && row<(n_data-n_kernel/2) && (p_kernel/2)<col && col<(p_data-p_kernel/2) ){
		float result = 0;
		printf("col = %d, row = %d\n", col, row);
		for (int i = 0; i<n_kernel; i++){
			for(int j = 0; j<p_kernel; j++){
				coeff = data[(row+i-n_kernel/2)*n_data + col+j-p_kernel/2]*kernel[i*n_kernel+j];
				//printf("i = %d, j = %d, coeff = %f\n",i,j,coeff);
				//printf("data index = %d, kernel index = %d\n",(row+i)*n_data + col+j-p_kernel/2, i*n_kernel+j);
				result += coeff;
			}
		}	
		Mout[(row-n_kernel/2)*(n_data-n_kernel+1)+col-p_kernel/2] = result;
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
	
	
	int n_kernel = 3;
	int n_data = 3;
	float *test_data = (float*) malloc(n_data*n_data*sizeof(float));
	float *test_kernel = (float*) malloc(n_kernel*n_kernel*sizeof(float));
	MatrixInit(test_data, n_data, n_data, 0, 1);
	MatrixInit(test_kernel, n_kernel, n_kernel, 0, 1);
	
	
	float *test_conv = (float*) malloc(1*1*sizeof(float));
	
	int N = n_data*n_data;
	dim3 dimBlock(N,N);
	dim3 dimGrid(ceil(N/16.0), ceil(N/16.0));
	float *data_cu, *kernel_cu, *conv_cu;
	cudaMalloc((void **) &data_cu, n_data*n_data*sizeof(float));
	cudaMalloc((void **) &kernel_cu, n_kernel*n_kernel*sizeof(float));
	cudaMalloc((void **) &conv_cu, (n_data-n_kernel+1)*(n_data-n_kernel+1)*sizeof(float));
	
	cudaMemcpy(data_cu, test_data, n_data*n_data*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_cu, test_kernel, n_kernel*n_kernel*sizeof(float), cudaMemcpyHostToDevice);
	cudaMatrixConv2D<<<n_data*n_data, 1>>>(data_cu, kernel_cu, conv_cu, n_data, n_data, n_kernel, n_kernel);
	cudaMemcpy(test_conv, conv_cu, (n_data-n_kernel+1)*(n_data-n_kernel+1)*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	MatrixPrint2D(test_data, n_data, n_data);
	MatrixPrint2D(test_kernel, n_kernel, n_kernel);
	MatrixPrint2D(test_conv, (n_data-n_kernel+1), (n_data-n_kernel+1));
	
}
