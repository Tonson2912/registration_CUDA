
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cufft.h>
#include<cublas.h>
#include<cublas_v2.h>

#include <iostream>
#include"registrationGPU.h"


#define pi (atan(1.0)*4)

inline int iDivUp(int a, int b) {
	return (a%b != 0) ? (a / b + 1) : (a / b);
}

struct IndexOffset
{
	IndexOffset(unsigned int num)
	{
		indexX = new float[num]();
		indexY = new float[num]();
	}
	~IndexOffset()
	{
		if (indexX != NULL) { delete[] indexX; indexX = NULL;
		}
		if (indexY != NULL) { delete[] indexY; indexY = NULL;
		}
	}

	float *indexY;//行
	float *indexX;//列
};

void fft2(float *d_input,cufftComplex *d_fftResult, unsigned int height, unsigned int width,unsigned int num)
{
	//二维fft参数设置
	int n[2] = { height,width };
	int inembed[] = { height,width };
	int onembed[] = { height,width / 2 + 1 };
	cufftHandle fftPlanFwd;

	//设置需FFT的矩阵个数
	cufftPlanMany(&fftPlanFwd, 2, n, inembed, 1, height*width, onembed, 1, height*(width / 2 + 1), CUFFT_R2C, num);

	cufftExecR2C(fftPlanFwd, d_input, d_fftResult);

	cufftDestroy(fftPlanFwd);
}

__global__ void ifftshiftKernelstep1(float *d_src, int length,int begin)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix < length)
	{
		d_src[ix] = ix + begin;
	}
}

__global__ void ifftshiftKernelstep2(float *d_src, float *d_dst, int length)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	if (ix < length)
	{
		int half = ceilf((float)length / 2);
		if (ix < half)
		{
			d_dst[ix] = d_src[ix + half];
		}
		else
		{
			d_dst[ix] = d_src[ix - half];
		}
	}
}

void ifftshiftVecGPU(float *d_dst, int begin, int end)
{
	int length = end - begin + 1;

	float *temp = NULL; cudaMalloc(&temp, sizeof(float)*length);

	ifftshiftKernelstep1 << <iDivUp(length, 128), 128 >> > (temp, length, begin);//赋值
	ifftshiftKernelstep2 << <iDivUp(length, 128), 128 >> > (temp, d_dst, length);//左右对调

	cudaFree(temp);
}

template<class T>
void ifftshiftVec(T *dst, int begin, int end)
{
	int ii = 0;
	for (int i = 0; i <= end; ++i)
	{
		dst[ii] = i;
		ii++;
	}
	for (int i = begin; i < 0; ++i)
	{
		dst[ii] = i;
		ii++;
	}
}

inline __device__ void mulComplexConj(cufftComplex &a, cufftComplex &b, const float &n)
{
	cufftComplex temp;
	temp.x = (a.x*b.x + a.y*b.y)*n;
	temp.y = (-a.x*b.y + b.x*a.y)*n;
	b = temp;
}

__global__ void dot(cufftComplex *d_A, cufftComplex *d_B,cuDoubleComplex *d_C ,unsigned int dataSize,int num, float n)
{
	size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < dataSize)
	{
		cufftComplex a = d_A[i];
		for (int k = 0; k < num; ++k)
		{
			cufftComplex b = d_B[i + k * dataSize];
			mulComplexConj(a, b, n);
			d_C[i + k * dataSize] = cuComplexFloatToDouble(b);
		}
	}
}

__global__ void dot1(cufftComplex *d_A, cufftComplex *d_B, cufftComplex *d_C, unsigned int dataSize, int num, float n)
{
	size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < dataSize)
	{
		for (int k = 0; k < num; ++k)
		{
			cufftComplex a = d_A[i + k * dataSize];
			cufftComplex b = d_B[i];
			mulComplexConj(a, b, n);
			d_C[i + k * dataSize] = b;
		}
	}
}

__global__ void getImFTout(cuDoubleComplex *imFTout, cuDoubleComplex *imFT, unsigned int inputH,
	unsigned int inputW, unsigned int outputH, unsigned int outputW,int Ystart)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix < inputW&&iy < inputH)
	{
		int halfH = ceilf((float)inputH / 2);
		if (iy <= halfH)
		{
			imFTout[ix + iy * outputW] = imFT[ix + iy * inputW];
		}
		else
		{
			imFTout[ix + (Ystart + iy)*outputW] = imFT[ix + iy * inputW];
		}
		
	}
}

void FTpad(cuDoubleComplex *imFTout, cuDoubleComplex *imFT,unsigned int inputH,unsigned int inputW, unsigned int outputH, unsigned int outputW,int num)
{
	int Ystart = outputH - inputH;
	dim3 block(16, 16);
	dim3 grid(iDivUp(inputW, block.x), iDivUp(inputH, block.y));
	for (int i = 0; i < num; ++i)
	{
		getImFTout << <grid, block >> > (imFTout+i*outputH*outputW, imFT+i*inputH*inputW, inputH, inputW, outputH, outputW, Ystart);
	}
}

void ifft2(cuDoubleComplex *d_output, cuDoubleComplex *d_fftResult, unsigned int height, unsigned int width, unsigned int num)
{
	//二维ifft参数设置
	int n[2] = { height,width };
	int inembed[] = { height,width };
	//int onembed[] = { height,width / 2 + 1 };
	cufftHandle fftPlanInv;


	cufftPlanMany(&fftPlanInv, 2, n, inembed, 1,height*width, inembed, 1, height*width, CUFFT_Z2Z,num);
	cufftExecZ2Z(fftPlanInv, d_fftResult, d_output, CUFFT_INVERSE);//CUFFT_INVERSE
	//cufftPlanMany(&fftPlanInv, 2, n, onembed, 1, height*(width / 2 + 1), onembed, 1, height*(width / 2 + 1), CUFFT_Z2Z, num);
	//cufftExecZ2Z(fftPlanInv, d_fftResult, d_fftResult, CUFFT_INVERSE);//CUFFT_INVERSE
	
	cufftDestroy(fftPlanInv);
}

template<class T>
__global__ void getIn(T *d_dot,T *in,unsigned int height,unsigned int width)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;//nc
	if (ix < width&&iy < height)
	{
		if (ix > width/2 && ix < width)
		{
			if (iy == 0)
			{
				in[ix + iy * width].x = d_dot[(width - ix) + iy * (width / 2 + 1)].x;
				in[ix + iy * width].y = -d_dot[(width - ix) + iy * (width / 2 + 1)].y;
			}
			else
			{
				in[ix + iy * width].x = d_dot[(width - ix) + (height - iy) * (width / 2 + 1)].x;
				in[ix + iy * width].y = -d_dot[(width - ix) + (height - iy) * (width / 2 + 1)].y;
			}
		}
		else
		{
			in[ix + iy * width] = d_dot[ix + iy * (width / 2 + 1)];
		}
	}
}

void getDftupsIn(cufftComplex *buf1ft, cufftComplex *buf2ft, cufftComplex *in, unsigned int height, unsigned int width)
{
	int halflength = height * (width / 2 + 1);
	cufftComplex *d_dot; cudaMalloc(&d_dot, sizeof(cufftComplex)*halflength);//!
	dot1 << <iDivUp(halflength, 128), 128 >> > (buf2ft, buf1ft, d_dot, halflength, 1, 1);
	//进行扩展
	dim3 block(16, 16);
	dim3 grid(iDivUp(width, block.x), iDivUp(height, block.y));
	getIn << <grid, block >> > (d_dot, in, height, width);

	cudaFree(d_dot);
}

template<class T,class R>
__global__ void VecMiusConst(T *input, R data, unsigned int length)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < length)
	{
		input[i] = input[i] - data;
	}
}

__global__ void getNocVec(float *nocVec, float coff, unsigned int length)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < length)
	{
		nocVec[i] = i - coff;
	}
}

__global__ void getkernc(cufftComplex *kernc, float *temp,int nc,float usfac, unsigned int length)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < length)
	{
		kernc[i].x = cos((2 * pi / (nc*usfac))* temp[i]);
		kernc[i].y = -sin((2 * pi / (nc*usfac))* temp[i]);
	}
}

void computeKernc(cufftComplex *kernc, cublasHandle_t handle, int nc, int noc, float coff, float usfac)
{
	float *xita1; cudaMalloc(&xita1, sizeof(float)*nc);//!
	ifftshiftVecGPU(xita1, 0, nc - 1);
	VecMiusConst<float, float> << <iDivUp(nc, 128), 128 >> > (xita1, floor(nc / 2), nc);
	float *xita2; cudaMalloc(&xita2, sizeof(float)*noc);//!
	getNocVec << <iDivUp(noc, 128), 128 >> > (xita2, coff, noc);

	//矩阵相乘
	float a = 1, b = 0;
	//(NC*1)  *  (1*NOC)
	float *temp; cudaMalloc(&temp, sizeof(float)*noc*nc);//!
	cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, noc, nc, 1, &a, xita2, noc, xita1, 1, &b, temp, noc);

	getkernc << <iDivUp((int)(nc*noc), 128), 128 >> > (kernc, temp, nc, usfac, nc*noc);

	cudaFree(xita1); cudaFree(xita2);
	cudaFree(temp);
}

void computeKernr(cufftComplex *kernr, cublasHandle_t handle, int nr, int nor, float roff, float usfac)
{
	float *xita2; cudaMalloc(&xita2, sizeof(float)*nr);//!
	ifftshiftVecGPU(xita2, 0, nr - 1);
	VecMiusConst<float, float> << <iDivUp(nr, 128), 128 >> > (xita2, floor(nr / 2), nr);
	float *xita1; cudaMalloc(&xita1, sizeof(float)*nor);//!
	getNocVec << <iDivUp(nor, 128), 128 >> > (xita1, roff, nor);


	//矩阵相乘
	float a = 1, b = 0;
	//(nor*1)  *  (1*nr)
	float *temp; cudaMalloc(&temp, sizeof(float)*nor*nr);//!
	cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, nr, nor, 1, &a, xita2, nr, xita1, 1, &b, temp, nr);
	getkernc << <iDivUp((int)(nr*nor), 128), 128 >> > (kernr, temp, nr, usfac, nr*nor);

	cudaFree(xita1); cudaFree(xita2);
	cudaFree(temp);
}

__global__ void ComplexFloat2Double(cufftComplex *input, cuDoubleComplex *output, unsigned int length)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < length)
	{
		output[i] = cuComplexFloatToDouble(input[i]);
	}
}

void getDftupsOut(cufftComplex *kernr, cufftComplex *in, cufftComplex *kernc, cuDoubleComplex *CCDouble, cublasHandle_t handle, int nor, int noc, int nr,int nc)
{
	//类型转换
	cuDoubleComplex *tempDouble; cudaMalloc(&tempDouble, sizeof(cuDoubleComplex)*nor*nc);//!
	cuDoubleComplex *kerncDouble; cudaMalloc(&kerncDouble, sizeof(cuDoubleComplex)*nc*noc);//!
	cuDoubleComplex *kernrDouble; cudaMalloc(&kernrDouble, sizeof(cuDoubleComplex)*nr*nor);//!
	cuDoubleComplex *inDouble; cudaMalloc(&inDouble, sizeof(cuDoubleComplex)*nc*nr);//!

	ComplexFloat2Double << <iDivUp(noc*nc, 128), 128 >> > (kernc, kerncDouble, noc*nc);
	ComplexFloat2Double << <iDivUp(nor*nr, 128), 128 >> > (kernr, kernrDouble, nor*nr);
	ComplexFloat2Double << <iDivUp(nr*nc, 128), 128 >> > (in, inDouble, nr*nc);
	//cufftComplex *temp; cudaMalloc(&temp, sizeof(cufftComplex)*nor*nc);
	cuDoubleComplex a; a.x = 1; a.y = -0;
	cuDoubleComplex b; b.x = 0; b.y = 0;

	//1 (nor*nr)*(nr*nc)=(nor*nc)
	cublasZgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, nc, nor, nr, &a, inDouble, nc, kernrDouble, nr, &b, tempDouble, nc);

	//2 (nor*nc)*(nc*noc)=(nor*noc) nor=72 noc=75 nor=74 noc=75 (temp:nor kernc:noc)
	cublasZgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, noc, nor, nc, &a, kerncDouble, noc, tempDouble, nc, &b, CCDouble, noc);

	cudaFree(tempDouble); cudaFree(kerncDouble); 
	cudaFree(kernrDouble); cudaFree(inDouble);
}

__global__ void Cabs(cuDoubleComplex *CC, double *CCabs, unsigned int length)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < length)
	{
		//CCabs[i] = sqrt((double)CC[i].x*(double)CC[i].x + (double)CC[i].y*(double)CC[i].y);
		CCabs[i] = cuCabs(CC[i]);
	}
}

void dftups(double *CCabs, cufftComplex *in, cublasHandle_t handle, int nor, int noc, float usfac, float roff, float coff,unsigned int height,unsigned int width)
{
	//欧拉公式 exp(iθ)=cos(θ)+i*sin(θ);
	//先求θ
	int nr = height;
	int nc = width;
	cuDoubleComplex *CC; cudaMalloc(&CC, sizeof(cuDoubleComplex)*nor*noc);//!

	cufftComplex *kernc; cudaMalloc(&kernc, sizeof(cufftComplex)*noc*nc);//!
	computeKernc(kernc,handle, nc, noc, coff, usfac);
	cufftComplex *kernr; cudaMalloc(&kernr, sizeof(cufftComplex)*nor*nr);//!
	computeKernr(kernr,handle, nr, nor, roff, usfac);
	//求CC
	getDftupsOut(kernr, in, kernc, CC,handle, nor, noc, nr, nc);
	//求CCabs
	Cabs << <iDivUp(nor*noc, 128), 128 >> > (CC, CCabs, nor*noc);

	cudaFree(CC); cudaFree(kernc); cudaFree(kernr);

}

__global__ void Cabs(cufftComplex *CC, float *CCabs, unsigned int length)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < length)
	{
		CCabs[i] = sqrt(CC[i].x*CC[i].x + CC[i].y*CC[i].y);
	}
}

void dftregistration(IndexOffset &Ind, cufftComplex *buf1ft, cufftComplex *buf2ft, cublasHandle_t handle,unsigned int height,unsigned int width, int usfac,int num)
{
	unsigned int nr = height;
	unsigned int nc = width;
	int beginNr = -floor((float)nr / 2);
	int endNr = ceil((float)nr / 2) - 1;
	int *Nr=new int[nr]();//!
	ifftshiftVec<int>(Nr, beginNr, endNr);

	int beginNc = -floor((float)nc / 2);
	int endNc = ceil((float)nc / 2) - 1;
	int *Nc = new int[nc]();
	ifftshiftVec<int>(Nc, beginNc, endNc);
	
	//以下代码为 usfac>1
	int nnr = 2 * nr; int nnc = 2 * nc;

	cuDoubleComplex *d_FTpad; cudaMalloc(&d_FTpad, sizeof(cuDoubleComplex)*nnr*(nnc / 2 + 1)*num);//!
	cuDoubleComplex *d_dotTemp; cudaMalloc(&d_dotTemp, sizeof(cuDoubleComplex)*nr*(nc / 2 + 1)*num);//!


	float rate = (float)(nnr*nnc) / (nr*nc);
	
	dot << <iDivUp((nc / 2 + 1)*nr, 256), 256 >> > (buf1ft, buf2ft, d_dotTemp, nr*(nc / 2 + 1), num, rate);//buf1ft.conj(buf2ft)

	FTpad(d_FTpad, d_dotTemp, nr, (nc / 2 + 1), nnr, (nnc / 2 + 1), num);//prepare for IFFT2



	dim3 block(16, 16);
	dim3 grid(iDivUp(nnc, block.x), iDivUp(nnr, block.y));
	cuDoubleComplex *d_CC; cudaMalloc(&d_CC, sizeof(cuDoubleComplex)*nnr*nnc*num);//!
	for (int i = 0;i < num;++i)
	{
		getIn<cuDoubleComplex> << <grid, block >> > (d_FTpad + i * nnr*(nnc / 2 + 1), d_CC + i * nnr*nnc, nnr, nnc);
	}

	double *d_CCAbs;cudaMalloc(&d_CCAbs, sizeof(double)*nnr*nnc*num);//!
	ifft2(d_CC, d_CC, nnr, nnc, num);//ifft2

	if (true)
	{
		cudaFree(d_FTpad);
		cudaFree(d_dotTemp);
	}

	Cabs << <iDivUp(nnr*nnc*num, 256), 256 >> > (d_CC, d_CCAbs, nnr*nnc*num);


	//getMaxIdx(Ind, d_CC, nnr, nnc, num);
	int maxInd = 0;
	int row_shift = 0; int col_shift = 0;

	beginNr = -floor((float)nr);
	endNr = ceil((float)nr) - 1;
	int *Nr2 = new int[nr*2]();//!
	ifftshiftVec<int>(Nr2, beginNr, endNr);

	beginNc = -floor((float)nc);
	endNc = ceil((float)nc) - 1;
	int *Nc2 = new int[nc*2]();
	ifftshiftVec<int>(Nc2, beginNc, endNc);

	for (int i = 0; i < num; ++i)
	{
		cublasIdamax_v2(handle, nnr*nnc, d_CCAbs + i * nnr*nnc, 1, &maxInd);
		row_shift = (maxInd - 1) / nnc;
		col_shift = (maxInd - 1) % nnc;
		//std::cout << col_shift << std::endl;
		//CCmax似乎没用到
		row_shift = Nr2[row_shift];
		col_shift = Nc2[col_shift];
		Ind.indexX[i] = (float)col_shift / 2;
		Ind.indexY[i] = (float)row_shift / 2;
	}

	if (true)
	{
		delete[] Nr;
		delete[] Nc;
		delete[] Nr2;
		delete[] Nc2;
		cudaFree(d_CC);
		cudaFree(d_CCAbs);
	}

	//以下代码为usfac>2
	if (usfac > 2)
	{
		float dftshift = floor(ceil(usfac*1.5f) / 2);
		int noc = ceil(usfac*1.5); int nor = ceil(usfac*1.5);
		float roff = 0, coff = 0;
		//将傅里叶结果扩大一倍
		cufftComplex *in; cudaMalloc(&in, sizeof(cufftComplex)*height*width);
		double *CCabs; cudaMalloc(&CCabs, sizeof(double)*noc*nor*num);
		for (int i = 0; i < num; ++i)
		{
			Ind.indexY[i] = round(Ind.indexY[i] * usfac) / usfac;
			Ind.indexX[i] = round(Ind.indexX[i] * usfac) / usfac;
			roff = dftshift - Ind.indexY[i] * usfac;
			coff = dftshift - Ind.indexX[i] * usfac;
			getDftupsIn(buf1ft, buf2ft + i * height*(width / 2 + 1), in, nr, nc);
			dftups(CCabs + i * nor*noc, in,handle, nor, noc, usfac, roff, coff, nr, nc);
			//求CC最大值
			cublasIdamax_v2(handle, nor*noc, CCabs + i * nor*noc, 1, &maxInd);
			row_shift = (maxInd-1) / noc;//75
			col_shift = (maxInd-1) % noc;//76
			row_shift = row_shift - dftshift ;
			col_shift = col_shift - dftshift ;
			Ind.indexY[i] = Ind.indexY[i] + (float)row_shift / usfac;
			Ind.indexX[i] = Ind.indexX[i] + (float)col_shift / usfac;
			//std::cout << i + 1 << "  行:" << Ind.indexY[i] << "  列:" << Ind.indexX[i] << std::endl;
		}
		if (true)
		{
			cudaFree(in);
			cudaFree(CCabs);
		}
	}

}

__global__ void leftMove(float *d_input, float *d_output,float x, unsigned int height, unsigned int width)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix < width&&iy < height)
	{
		unsigned int i = ix + iy * width;
		if (ix == 0)
		{
			d_output[i] = d_input[i] - d_input[i] * x;
		}
		else
		{
			d_output[i] = d_input[i] - (d_input[i] - d_input[ix - 1 + iy * width])*x;
		}
	}
}

__global__ void rightMove(float *d_input, float *d_output, float x, unsigned int height, unsigned int width)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix < width&&iy < height)
	{
		unsigned int i = ix + iy * width;
		if (ix == width - 1)
		{
			d_output[i] = d_input[i] - d_input[i] * x;
		}
		else
		{
			d_output[i] = d_input[i] - (d_input[ix + 1 + iy * width] - d_input[i])*x;
		}
	}
}

__global__ void downMove(float *d_input, float *d_output, float y, unsigned int height, unsigned int width)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix < width&&iy < height)
	{
		unsigned int i = ix + iy * width;
		if (iy == 0)
		{
			d_output[i] = d_input[i] - d_input[i] * y;
		}
		else
		{
			d_output[i] = d_input[i] - (d_input[i] - d_input[ix + (iy - 1)*width])*y;
		}
	}
}

__global__ void upMove(float *d_input, float *d_output, float y, unsigned int height, unsigned int width)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix < width&&iy < height)
	{
		unsigned int i = ix + iy * width;
		if (iy == height - 1)
		{
			d_output[i] = d_input[i] - d_input[i] * y;
		}
		else
		{
			d_output[i] = d_input[i] - (d_input[ix + (iy + 1)*width] - d_input[i])*y;
		}
	}
}

void bilinear(float *d_ImageIn, float x, float y, unsigned int oHeight, unsigned int oWidth)
{
	if (x == 0 && y == 0)
	{
		return;
	}
	float *temp; cudaMalloc(&temp, sizeof(float)*oHeight*oWidth);
	dim3 block(16, 16);
	dim3 grid(iDivUp(oWidth, block.x), iDivUp(oHeight, block.y));
	//x方向
	if (x != 0)
	{
		if (x > 0)//左移  当前=当前-左边
		{
			leftMove << <grid, block >> > (d_ImageIn, temp, x, oHeight, oWidth);
		}
		else//右移 当前=右边-当前
		{
			rightMove << <grid, block >> > (d_ImageIn, temp, x, oHeight, oWidth);
		}
		cudaMemcpy(d_ImageIn, temp, sizeof(float)*oHeight*oWidth, cudaMemcpyDeviceToDevice);
	}
	//y方向
	if (y != 0)
	{
		if (y > 0)//下移 当前行=当前行-上一行
		{
			downMove << <grid, block >> > (d_ImageIn, temp, y, oHeight, oWidth);
		}
		else//上移 当前行=下一行-当前行
		{
			upMove << <grid, block >> > (d_ImageIn, temp, y, oHeight, oWidth);
		}
		cudaMemcpy(d_ImageIn, temp, sizeof(float)*oHeight*oWidth, cudaMemcpyDeviceToDevice);
	}

	cudaFree(temp);

}

void translate(float *d_ImageIn, int x, int y, unsigned int oHeight, unsigned int oWidth)
{
	//整数平移
	if (x == 0 && y == 0)
	{
		return;
	}
	float *temp; cudaMalloc(&temp, sizeof(float)*oHeight*oWidth);
	cudaMemset(temp, 0, sizeof(float)*oHeight*oWidth);
	//y方向平移
	if (y != 0)
	{
		int absy = abs(y);
		if (y > 0)
			cudaMemcpy(temp + absy * oWidth, d_ImageIn, sizeof(float)*(oHeight - absy)*oWidth, cudaMemcpyDeviceToDevice);
		if (y < 0)
			cudaMemcpy(temp, d_ImageIn + absy * oWidth, sizeof(float)*(oHeight - absy)*oWidth, cudaMemcpyDeviceToDevice);
		if (x != 0)
		{
			cudaMemset(d_ImageIn, 0, sizeof(float)*oHeight*oWidth);
			int absx = abs(x);
			for (int i = 0; i < oHeight; ++i)
			{
				if (x > 0)
					cudaMemcpy(d_ImageIn + i * oWidth + absx, temp + i * oWidth, sizeof(float)*(oWidth - absx), cudaMemcpyDeviceToDevice);
				if (x < 0)
					cudaMemcpy(d_ImageIn + i * oWidth, temp + i * oWidth + absx, sizeof(float)*(oWidth - absx), cudaMemcpyDeviceToDevice);
			}
		}
		else
		{
			cudaMemcpy(d_ImageIn, temp, sizeof(float)*oHeight*oWidth, cudaMemcpyDeviceToDevice);
		}
	}
	//x方向平移
	if (x != 0)
	{
		int absx = abs(x);
		for (int i = 0; i < oHeight; ++i)
		{
			if (x > 0)
				cudaMemcpy(temp + i * oWidth + absx, d_ImageIn + i * oWidth, sizeof(float)*(oWidth - absx), cudaMemcpyDeviceToDevice);
			if (x < 0)
				cudaMemcpy(temp + i * oWidth, d_ImageIn + i * oWidth + absx, sizeof(float)*(oWidth - absx), cudaMemcpyDeviceToDevice);
		}
		//y方向平移
		if (y != 0)
		{
			cudaMemset(d_ImageIn, 0, sizeof(float)*oHeight*oWidth);
			int absy = abs(y);
			if (y > 0)
				cudaMemcpy(d_ImageIn + absy * oWidth, temp, sizeof(float)*(oHeight - absy)*oWidth, cudaMemcpyDeviceToDevice);
			if (y < 0)
				cudaMemcpy(d_ImageIn, temp + absy * oWidth, sizeof(float)*(oHeight - absy)*oWidth, cudaMemcpyDeviceToDevice);
		}
		else
		{
			cudaMemcpy(d_ImageIn, temp, sizeof(float)*oHeight*oWidth, cudaMemcpyDeviceToDevice);
		}
	}
	cudaFree(temp);
}

void imtranslate(float *d_ImageIn,float x, float y, unsigned int oHeight, unsigned int oWidth)
{
	//第一张不用移动
	if (x == 0 && y == 0)
	{
		return;
	}
	else
	{
		float decimalX = 0; float decimalY = 0;
		decimalX = x - (int)x;//列
		decimalY = y - (int)y;//行
		//整数位平移
		translate(d_ImageIn, (int)x, (int)y, oHeight, oWidth);
		//小数位进行双线性插值(具体算法过程百度查询)
		bilinear(d_ImageIn, decimalX, decimalY, oHeight, oWidth);
	}
}

__global__ void getOutputImage(float *d_ImageIn, float *d_ImageOut, int startPointX, int startPointY,
	unsigned int inputHeight, unsigned int inputWidth, unsigned int outputHeight, unsigned int outputWidth)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;
	if (ix < outputWidth&&iy < outputHeight)
	{
		d_ImageOut[ix + iy * outputWidth] = d_ImageIn[startPointX + ix + (startPointY + iy)*inputWidth];
	}
}

void getArea(float *d_ImageIn, float *d_ImageOut, int startPointX, int startPointY,
	unsigned int inputHeight, unsigned int inputWidth, unsigned int outputHeight, unsigned int outputWidth)
{
	dim3 block(16, 16);
	dim3 grid(iDivUp(outputWidth, block.x), iDivUp(outputHeight, block.y));
	getOutputImage << <grid, block >> > (d_ImageIn, d_ImageOut, startPointX - 1, startPointY - 1, inputHeight, inputWidth, outputHeight, outputWidth);
}


/*
	函数功能：图像匹配
	ImageIn：图像输入
	ImageOut：图像输出
	d_fixedFFT：第一张图(模板)二维傅里叶变化结果
	oHeight：原始图像高 oWidth：原始图像宽
	nHeight：输出图像高 nWidth：输出图像宽
	runImageNum：进行匹配的图像张数
*/
void registration(float *ImageIn, float *ImageOut,cufftComplex *d_fixedFFT, unsigned int oHeight, unsigned int oWidth, unsigned int nHeight, unsigned int nWidth, unsigned int runImageNum)
{
	int startPointX = round((float)(oWidth - nWidth) / 2);//列
	int startPointY = round((float)(oHeight - nHeight) / 2);//行

	//float *d_fixed; cudaMalloc(&d_fixed, sizeof(float)*oHeight*oWidth);//!
	//cudaMemcpy(d_fixed, ImageIn, sizeof(float)*oHeight*oWidth, cudaMemcpyHostToDevice);

	float *d_ImageIn; cudaMalloc(&d_ImageIn, sizeof(float)*oHeight*oWidth * runImageNum);//!
	cudaMemcpy(d_ImageIn, ImageIn, sizeof(float)*oHeight*oWidth*runImageNum, cudaMemcpyHostToDevice);

	//做互相关
	//二维傅里叶结果取一半即可，对称的
	cufftComplex *d_ImageInFFT; cudaMalloc(&d_ImageInFFT, sizeof(cufftComplex)*oHeight*(oWidth / 2 + 1)*runImageNum);//!
	fft2(d_ImageIn, d_ImageInFFT, oHeight, oWidth, runImageNum);


	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	IndexOffset ind(runImageNum);
	dftregistration(ind, d_fixedFFT, d_ImageInFFT,handle, oHeight, oWidth, 100, runImageNum);//互相关找偏移量（互相关具体算法百度查询）
	cublasDestroy_v2(handle);

	for (int i = 0;i < runImageNum;++i) {
		std::cout <<"indY:  "<<ind.indexY[i] <<"   indX: "<<ind.indexX[i] << std::endl;
	}
	std::cout << "===============================next Patch==============================" << std::endl;
	if (true)		
	{
		if (d_ImageInFFT != NULL) { cudaFree(d_ImageInFFT); d_ImageInFFT = NULL; }
	}


	float *d_ImageOut; cudaMalloc(&d_ImageOut, sizeof(float)*nHeight*nWidth*runImageNum);
	//getArea(d_fixed, d_ImageOut, startPointX, startPointY, oHeight, oWidth, nHeight, nWidth);
	for (int i = 0; i < runImageNum; ++i)
	{
		imtranslate(d_ImageIn + i * oHeight*oWidth, ind.indexX[i], ind.indexY[i], oHeight, oWidth);//通过偏移量移动图像
		getArea(d_ImageIn + i * oHeight*oWidth, d_ImageOut + i * nHeight*nWidth, startPointX, startPointY, oHeight, oWidth, nHeight, nWidth);//截取图像
	}
	cudaMemcpy(ImageOut, d_ImageOut, sizeof(float)*nHeight*nWidth*runImageNum, cudaMemcpyDeviceToHost);

	//if (d_fixed != NULL)cudaFree(d_fixed); d_fixed = NULL;
	if (d_ImageIn != NULL) { cudaFree(d_ImageIn); d_ImageIn = NULL; }
	if (d_ImageOut != NULL) { cudaFree(d_ImageOut); d_ImageOut = NULL; }
}

/*
	函数功能：图像分批次进行匹配
	ImageIn：图像输入
	ImageOut：图像输出
	oHeight：原始图像高 oWidth：原始图像宽
	nHeight：输出图像高 nWidth：输出图像宽
	num：图像总张数
	ImageNumPerBatch：每个批次的图像张数
*/
extern "C" void registrationGPU(float *ImageIn, float *ImageOut, unsigned int oHeight, unsigned int oWidth, unsigned int nHeight, unsigned int nWidth, unsigned int num,int ImageNumPerBatch)
{
	int startPointX = round((float)(oWidth - nWidth) / 2);//列
	int startPointY = round((float)(oHeight - nHeight) / 2);//行

	float *d_fixed; cudaMalloc(&d_fixed, sizeof(float)*oHeight*oWidth);//!
	cudaMemcpy(d_fixed, ImageIn, sizeof(float)*oHeight*oWidth, cudaMemcpyHostToDevice);

	cufftComplex *d_fixedFFT; cudaMalloc(&d_fixedFFT, sizeof(cufftComplex)*oHeight*(oWidth / 2 + 1));//!
	fft2(d_fixed, d_fixedFFT, oHeight, oWidth, 1);

	//第一张不用进行匹配
	float *d_ImageOutfirst; cudaMalloc(&d_ImageOutfirst, sizeof(float)*nHeight*nWidth);
	getArea(d_fixed, d_ImageOutfirst, startPointX, startPointY, oHeight, oWidth, nHeight, nWidth);
	cudaMemcpy(ImageOut, d_ImageOutfirst, sizeof(float)*nHeight*nWidth, cudaMemcpyDeviceToHost);

	//剩下的匹配图 
	//int ImageNumPerBatch = 7;
	int batch = ceil((float)(num - 1) / ImageNumPerBatch);//5

	//由于计算机内存有限，所以需要自适应分批次进行处理。
	//例如输入图像总大小为30张，若每批次(ImageNumPerBatch)为7张。则需要5次处理，前4次都为7张，最后一次1张。7+7+7+7+1=30-1，第一张不需要匹配，是模板。
	for (int i = 0; i < batch; ++i)
	{
		int runImageNum = 0;
		if ((i + 1)*ImageNumPerBatch <= num - 1)//满足一个ImageNumPerBatch的batch
		{
			runImageNum = ImageNumPerBatch;
		}
		else//不足一个ImageNumPerBatch的batch
		{
			runImageNum = num - 1 - i * ImageNumPerBatch;
		}
		registration(ImageIn + (i*ImageNumPerBatch + 1)*oHeight*oWidth, ImageOut + (i*ImageNumPerBatch + 1)*nHeight*nWidth,d_fixedFFT,
			oHeight, oWidth, nHeight, nWidth, runImageNum);
	}

	if (d_fixed != NULL) { cudaFree(d_fixed); d_fixed = NULL; }
	if (d_fixedFFT != NULL) { cudaFree(d_fixedFFT); d_fixedFFT = NULL; }
	if (d_ImageOutfirst != NULL) { cudaFree(d_ImageOutfirst); d_ImageOutfirst = NULL; }

}