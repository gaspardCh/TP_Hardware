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
		
	if ( (n_kernel/2)<=row && row<(n_data-n_kernel/2) && (p_kernel/2)<=col && col<(p_data-p_kernel/2) ){
		float result = 0;
		for (int i = 0; i<n_kernel; i++){
			for(int j = 0; j<p_kernel; j++){
				
				coeff = data[(row+i-n_kernel/2)*n_data + col+j-p_kernel/2]*kernel[i*p_kernel+j];
				result += coeff;
			}
		}	
		Mout[(row-n_kernel/2)*(n_data-n_kernel+1)+col-p_kernel/2] = result;
	}
}


__global__ void cudaMatrixConv3D(float *data, float *kernel, float *Mout, int n_data, int p_data, int n_kernel, int p_kernel, int d_kernel){
	int row = threadIdx.x;
	int col = threadIdx.y;
	
	int p_C1 = p_data - p_kernel + 1;
	int n_C1 = n_data - n_kernel + 1;
	int index = row * p_C1 + col;
	
	int k = blockIdx.x; 
	
	int d_offset_kernel = k * (n_kernel * p_kernel);
	int d_offset_out = k * (n_C1 * p_C1);	
	
	float result = 0;
	for (int i = 0; i<n_kernel; i++){
		for(int j = 0; j<p_kernel; j++){
			result += data[(row + i) * p_data + (col + j)] * kernel[i * p_kernel + j + d_offset_kernel];
		}
	}
	
	Mout[index + d_offset_out]=result;
		
	
	
	
}

__global__ void cudaMatrixSubSampling3D(float *data, float *Mout, int n_data, int p_data, int d_data){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row>=0 && row<n_data/2 && col>=0 && col<p_data/2){
		for (int k = 0; k<d_data; k++){
			Mout[k*(n_data/2)*(p_data/2)+row*n_data/2 + col] += data[k*n_data*p_data + row*2*n_data + col*2]/4.0;
			Mout[k*(n_data/2)*(p_data/2)+row*n_data/2 + col] += data[k*n_data*p_data + row*2*n_data + col*2 + 1]/4.0;
			Mout[k*(n_data/2)*(p_data/2)+row*n_data/2 + col] += data[k*n_data*p_data + (row*2+1)*n_data + col*2]/4.0;
			Mout[k*(n_data/2)*(p_data/2)+row*n_data/2 + col] += data[k*n_data*p_data + (row*2+1)*n_data + col*2+1]/4.0;
		}
	}
}


int main(void){
	srand(time(NULL));
	int n_raw_data = 32;
	int n_kernel = 5;
	int n_C1_data = n_raw_data - n_kernel +1;
	int n_S1_data = n_C1_data/2;
	int nb_kernel = 6;
	printf("n_C1_data=%d n_S1_data=%d \n", n_C1_data, n_S1_data);
	
	int raw_data_size = n_raw_data*n_raw_data*sizeof(float);
	int C1_data_size = nb_kernel*n_C1_data*n_C1_data*sizeof(float);
	int S1_data_size = nb_kernel*n_S1_data*n_S1_data*sizeof(float);
	int C1_kernel_size = nb_kernel*n_kernel*n_kernel*sizeof(float);
	
	
	float *raw_data = (float*) malloc(raw_data_size);
	float *C1_data = (float*) malloc(C1_data_size);
	float *S1_data = (float*) malloc(S1_data_size);
	float *C1_kernel = (float*) malloc(C1_kernel_size);
	
	MatrixInit(raw_data, n_raw_data, n_raw_data, 0, 1);
	MatrixInit(C1_data, nb_kernel, n_C1_data, n_C1_data, 0);
	MatrixInit(S1_data, nb_kernel, n_S1_data, n_S1_data, 0);
	MatrixInit(C1_kernel, nb_kernel, n_kernel, n_kernel, 1);
	
	float *raw_data_cu;
	float *C1_data_cu;
	float *S1_data_cu;
	float *C1_kernel_cu;
	
	cudaMalloc((void **) &raw_data_cu, n_raw_data*n_raw_data*sizeof(float));
	cudaMalloc((void **) &C1_data_cu, nb_kernel*n_C1_data*n_C1_data*sizeof(float));
	cudaMalloc((void **) &S1_data_cu, nb_kernel*n_S1_data*n_S1_data*sizeof(float));
	cudaMalloc((void **) &C1_kernel_cu, nb_kernel*n_kernel*n_kernel*sizeof(float));

	cudaMemcpy(raw_data_cu, raw_data, n_raw_data*n_raw_data*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(C1_kernel_cu, C1_kernel, nb_kernel*n_kernel*n_kernel*sizeof(float), cudaMemcpyHostToDevice);	
	
	dim3 dimGrid(nb_kernel,1,1);
	dim3 dimBlock(n_C1_data, n_C1_data);
	cudaMatrixConv3D<<<dimGrid, dimBlock>>>(raw_data_cu, C1_kernel_cu, C1_data_cu, n_raw_data, n_raw_data, n_kernel, n_kernel, nb_kernel);
	
	dim3 dimBlock_2(n_S1_data, n_S1_data);
	cudaMatrixSubSampling3D<<<1, dimBlock_2>>>(C1_data_cu, S1_data_cu, n_C1_data, n_C1_data, nb_kernel);
	
	cudaMemcpy(S1_data, S1_data_cu, nb_kernel*n_S1_data*n_S1_data*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(C1_data, C1_data_cu, nb_kernel*n_C1_data*n_C1_data*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	printf("data : \n");
	MatrixPrint2D(raw_data, n_raw_data, n_raw_data);
	printf("kernels : \n");
	MatrixPrint3D(C1_kernel, nb_kernel, n_kernel, n_kernel);
	printf("conv result : \n");
	MatrixPrint3D(C1_data, nb_kernel, n_C1_data, n_C1_data);
	printf("sub samp result : \n");
	MatrixPrint3D(S1_data, nb_kernel, n_S1_data, n_S1_data);
	
}

	// TESTING
	
	//int n_kernel = 3;
	//int n_data = 6;
	//int d_kernel = 2;
	//float *test_data = (float*) malloc(n_data*n_data*sizeof(float));
	//float *test_kernel = (float*) malloc(n_kernel*n_kernel*d_kernel*sizeof(float));
	//MatrixInit(test_data, n_data, n_data, 0, 1);
	//MatrixInit(test_kernel, n_kernel, n_kernel, d_kernel, 1);
	
	
	//float *test_conv = (float*) malloc((n_data-n_kernel+1)*(n_data-n_kernel+1)*d_kernel*sizeof(float));
	//float *test_sub_samp = (float*) malloc(2*2*2*sizeof(float));
	
	//int N = n_data*n_data;
	//dim3 dimBlock(N,N);
	//dim3 dimGrid(ceil(N/16.0), ceil(N/16.0));
	//float *data_cu, *kernel_cu, *conv_cu, *sub_samp;
	//cudaMalloc((void **) &data_cu, n_data*n_data*sizeof(float));
	//cudaMalloc((void **) &kernel_cu, n_kernel*n_kernel*d_kernel*sizeof(float));
	//cudaMalloc((void **) &conv_cu, (n_data-n_kernel+1)*(n_data-n_kernel+1)*d_kernel*sizeof(float));
	//cudaMalloc((void **) &sub_samp, 2*2*2*sizeof(float));
	
	//cudaMemcpy(data_cu, test_data, n_data*n_data*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(kernel_cu, test_kernel, n_kernel*n_kernel*d_kernel*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMatrixConv3D<<<dimBlock, dimGrid>>>(data_cu, kernel_cu, conv_cu, n_data, n_data, n_kernel, n_kernel, d_kernel);
	//cudaMatrixSubSampling3D<<<dimBlock, dimGrid>>>(conv_cu, sub_samp, 4, 4, 2);
	//cudaMemcpy(test_conv, conv_cu, (n_data-n_kernel+1)*(n_data-n_kernel+1)*d_kernel*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(test_sub_samp, sub_samp, 2*2*2*sizeof(float), cudaMemcpyDeviceToHost);
