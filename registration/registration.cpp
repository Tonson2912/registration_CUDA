#include<iostream>
#include<queue>
#include"shift.h"
#include"registration.h"
#include<time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define pi (atan(1.0)*4)

//ͼ����Ĭ������ԭ��Ϊͼ�����Ͻ�(0,0) , ����ΪY�ᣬ����ΪX��
//���� C����+opencv��+������̱�� ʵ��


struct IndexOffset
{
	IndexOffset(unsigned int num)
	{
		indexX = new float[num]();
		indexY = new float[num]();
	}
	~IndexOffset()
	{
		if (indexX != NULL) { delete[] indexX; }
		indexX = NULL;
		if (indexY != NULL) { delete[] indexY; }
		indexY = NULL;
	}

	float *indexY;//��
	float *indexX;//��
};

cv::Mat fft2(float *Image, unsigned int height, unsigned int width)
{
	//B=fft2(A);

	cv::Mat M_A = cv::Mat(height, width, CV_32FC1, Image);
	//����2ͨ����Mat���飬���ڱ�ʾ����
	cv::Mat planes[] = { cv::Mat_<float>(M_A),cv::Mat::zeros(M_A.size(),CV_32FC1) };//2ͨ��Mat����

	cv::Mat complexMat;//����Mat

	cv::merge(planes, 2, complexMat);

	cv::dft(complexMat, complexMat);

	return complexMat;

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

void complexAMulconjB(cv::Mat *result,cv::Mat& complexMatA, cv::Mat& complexMatB)
{
	cv::Size sz = complexMatA.size();
	cv::Mat M_A = cv::Mat(sz, CV_64FC1);
	cv::Mat M_B = cv::Mat(sz, CV_64FC1);
	cv::Mat planes[] = { cv::Mat::zeros(M_A.size(), CV_64FC1), cv::Mat::zeros(M_A.size(), CV_64FC1) };
	cv::Mat planes1[] = { cv::Mat::zeros(M_B.size(), CV_64FC1), cv::Mat::zeros(M_B.size(), CV_64FC1) };
	//cv::Mat result[] = { cv::Mat::zeros(M_B.size(), CV_64FC1), cv::Mat::zeros(M_B.size(), CV_64FC1) };

	cv::split(complexMatA, planes);
	cv::split(complexMatB, planes1);

	result[0] = planes[0].mul(planes1[0]) + planes[1].mul(planes1[1]);
	result[1] = planes[1].mul(planes1[0]) - planes[0].mul(planes1[1]);

}

void FTpad(cv::Mat& imFTout,cv::Mat& imFT, unsigned int outsizeH, unsigned int outsizeW)
{
	cv::Size Nin = imFT.size();
	cv::Size Nout;	Nout.height = outsizeH; Nout.width = outsizeW;
	cv::Size center;
	cv::Size centerout;
	cv::Size cenout_cen;

	fftshift(imFT);

	center.height = floor(Nin.height / 2.0f) + 1;
	center.width = floor(Nin.width / 2.0f) + 1;

	centerout.height = floor(Nout.height / 2.0f) + 1;
	centerout.width = floor(Nout.width / 2.0f) + 1;

	cenout_cen.height = centerout.height - center.height;
	cenout_cen.width = centerout.width - center.width;

	//cv::Mat imFTout = cv::Mat(Nout, CV_64FC1);

	int outTp_y = MAX(cenout_cen.height + 1, 1) - 1;
	int outTp_x = MAX(cenout_cen.width + 1, 1) - 1;
	int outTp_height = MIN(cenout_cen.height + Nin.height, Nout.height) - 1;
	int outTp_width = MIN(cenout_cen.width + Nin.width, Nout.width) - 1;//- outTp_x
	//cv::Rect rectout(outTp_x, outTp_y, outTp_width, outTp_height);

	int inTp_y = MAX(-cenout_cen.height + 1, 1) - 1;
	int inTp_x = MAX(-cenout_cen.width + 1, 1) - 1;
	int inTp_height = MIN(-cenout_cen.height + Nout.height, Nin.height) - 1 ;
	int inTp_width = MIN(-cenout_cen.width + Nout.width, Nin.width) - 1 ;
	//cv::Rect rectin(inTp_x, inTp_y, inTp_width, inTp_height);

	//cv::Mat temp[] = { cv::Mat::zeros(Nin, CV_64FC1), cv::Mat::zeros(Nin, CV_64FC1) };
	//cv::Mat temp1[] = { cv::Mat::zeros(Nout, CV_64FC1), cv::Mat::zeros(Nout, CV_64FC1) };
	//cv::split(imFT, temp);
	//cv::split(imFTout, temp1);
	//temp1[0](rectout) = temp[0](rectin);
	//temp1[1](rectout) = temp[1](rectin);
	//cv::Mat MM1 = temp1[0];
	//cv::Mat MM2 = temp1[1];
	//cv::merge(temp1, 2, imFTout);

	for (int i = inTp_y;i <= inTp_height;++i)
	{
		for (int j = inTp_x;j <= inTp_width*2;++j)//2ͨ��������Ҫ����2
		{
			imFTout.at<double>(outTp_y + i, outTp_x * 2 + j) = imFT.at<double>(i, j);
		}
	}
	//imFTout(rectout) = imFT(rectin);
	ifftshift(imFTout);

	float rate = (float)(Nout.height*Nout.width) / (Nin.height*Nin.width);

	imFTout = imFTout * rate;

}

void Mabs(cv::Mat& complexMat,cv::Mat& result)
{
	cv::Size sz = complexMat.size();
	//cv::Mat M_A = cv::Mat(sz, CV_64FC1);
	cv::Mat planes[] = { cv::Mat::zeros(sz, CV_64FC1), cv::Mat::zeros(sz, CV_64FC1) };
	cv::split(complexMat, planes);

	result = planes[0].mul(planes[0]) + planes[1].mul(planes[1]);
	cv::sqrt(result, result);

}

void Mabs(std::vector<cv::Mat>& complexMat,cv::Mat& result)
{
	cv::Size sz = complexMat[0].size();
	//cv::Mat M_A = cv::Mat(sz, CV_64FC1);

	result = complexMat[0].mul(complexMat[0]) + complexMat[1].mul(complexMat[1]);
	cv::sqrt(result, result);

}

void computeKernc(std::vector<cv::Mat>& kernc, float nc, float noc, float coff, float usfac)
{
	cv::Mat xita1 = cv::Mat::zeros(1, nc, CV_64FC1);
	cv::Mat xita2 = cv::Mat::zeros(1, noc, CV_64FC1);

	for (int i = 0; i < nc; ++i) {
		xita1.at<double>(0, i) = i - floor((float)nc / 2);
	}

	ifftshift(xita1);
	cv::Mat xita11 = cv::Mat::zeros(nc, 1, CV_64FC1);
	xita11 = xita1.t();

	for (int i = 0; i < noc; ++i)
	{
		xita2.at<double>(0, i) = i - coff;
	}

	cv::Mat M_A = cv::Mat(nc, noc, CV_64FC1);
	M_A = (2 * pi / (nc*usfac))*xita11*xita2;//��

	//exp(i��)=cos(��) + i * sin(��)
	//exp(-i��)=cos(��) - i * sin(��)
	for (int i = 0; i < nc; ++i) {
		for (int j = 0; j < noc; ++j) {
			kernc[0].at<double>(i, j) = cos(M_A.at<double>(i, j));
			kernc[1].at<double>(i, j) = -sin(M_A.at<double>(i, j));
		}
	}

}

void computeKernr(std::vector<cv::Mat>& kernr, float nr, float nor, float roff, float usfac)
{
	cv::Mat xita1 = cv::Mat::zeros(nor, 1, CV_64FC1);
	cv::Mat xita2 = cv::Mat::zeros(1, nr, CV_64FC1);

	for (int i = 0; i < nor; ++i) {
		xita1.at<double>(i, 0) = i - roff;
	}

	for (int i = 0; i < nr; ++i) {
		xita2.at<double>(0, i) = i - floor((float)nr / 2);
	}

	ifftshift(xita2);

	//cos(��) - i * sin(��) ŷ����ʽ
	cv::Mat M_A = cv::Mat::zeros(nor, nr, CV_64FC1);//��
	M_A = 2 * pi / (nr*usfac)*xita1*xita2;

	for (int i = 0; i < nor; ++i) {
		for (int j = 0; j < nr; ++j) {
			kernr[0].at<double>(i, j) = cos(M_A.at<double>(i, j));
			kernr[1].at<double>(i, j) = -sin(M_A.at<double>(i, j));
		}
	}

}

void get_dftups_out(std::vector<cv::Mat>& CC, std::vector<cv::Mat>& kernc, cv::Mat *in, std::vector<cv::Mat>& kernr)
{

	cv::Size kerncSz = kernc[0].size();
	cv::Size kernrSz = kernr[0].size();
	cv::Size inSz = in[0].size();

	//(a+bi)*(c+di) = ac+adi+bci-bd = (ac-bd)+(ad+bc)i 
	cv::Mat temp[] = { cv::Mat::zeros(kernrSz.height, inSz.width, CV_64FC1), cv::Mat::zeros(kernrSz.height, inSz.width, CV_64FC1) };
	//temp=kernr*in
	temp[0]=kernr[0]*in[0]-kernr[1]*in[1];
	temp[1]=kernr[0]*in[1]+kernr[1]*in[0];

	CC[0] = temp[0] * kernc[0] - temp[1] * kernc[1];
	CC[1] = temp[0] * kernc[1] + temp[1] * kernc[0];


}

void dftups(std::vector<cv::Mat> &CC,cv::Mat* in, float nor, float noc, float usfac, float roff, float coff)
{
	cv::Size sz = in[0].size();

	//ŷ����ʽ exp(i��)=cos(��)+i*sin(��);
	//i��
	int nr = sz.height; int nc = sz.width;

	cv::Mat real = cv::Mat::zeros(nc, noc, CV_64FC1);
	cv::Mat Image = cv::Mat::zeros(nc, noc, CV_64FC1);
	std::vector<cv::Mat> kernc;
	kernc.push_back(real);//ǳ���������ݹ���
	kernc.push_back(Image);
	computeKernc(kernc,nc, noc, coff, usfac);

	cv::Mat real1 = cv::Mat::zeros(nor, nr, CV_64FC1);
	cv::Mat Image1 = cv::Mat::zeros(nor, nr, CV_64FC1);
	std::vector<cv::Mat> kernr;
	kernr.push_back(real1);
	kernr.push_back(Image1);
	computeKernr(kernr,nr, nor, roff, usfac);

	get_dftups_out(CC,kernc, in, kernr);

	//return out;
}

void dftregistration(IndexOffset &Ind, cv::Mat& buf1ft, cv::Mat& buf2ft, int usfac, int loop)
{
	int nr = buf2ft.rows; int nc = buf2ft.cols;
	//��������ת��
	buf1ft.convertTo(buf1ft, CV_64FC1);
	buf2ft.convertTo(buf2ft, CV_64FC1);

	int *Nr = new int[nr](); int *Nc = new int[nc]();
	int beginNr = -floor((float)nr / 2);
	int endNr = ceil((float)nr / 2) - 1;
	ifftshiftVec<int>(Nr, beginNr, endNr);

	int beginNc = -floor((float)nc / 2);
	int endNc = ceil((float)nc / 2) - 1;
	ifftshiftVec<int>(Nc, beginNc, endNc);

	//���´���Ϊ   elseif usfac>1  ���û����(ifft2)���ֵ�
	cv::Mat temp[] = { cv::Mat::zeros(buf1ft.size(), CV_64FC1), cv::Mat::zeros(buf1ft.size(), CV_64FC1) };//�ֱ����ڴ������Mat��ʵ��Mat
	complexAMulconjB(temp,buf1ft, buf2ft);//������
	cv::Mat complexMat;//Mat����ϳɵ�һ��Mat�����ڱ�ʾ������Mat����
	cv::merge(temp, 2, complexMat);

	cv::Mat temp1[] = { cv::Mat::zeros(2*nr,2*nc, CV_64FC1), cv::Mat::zeros(2 * nr,2 * nc, CV_64FC1) };
	cv::Mat CC;
	cv::merge(temp1, 2, CC);
	//cv::Mat CC = cv::Mat(2 * nr, 2 * nc, CV_64FC1);
	FTpad(CC,complexMat, 2 * nr, 2 * nc);//Ϊ����Ҷ�任����ü�����������С
	cv::idft(CC, CC);//cv::DFT_SCALE | cv::DFT_INVERSE
	//cv::idft(CC, CC, cv::DFT_SCALE | cv::DFT_INVERSE);
	cv::Mat CCabs = cv::Mat(CC.size(), CV_64FC1);
	Mabs(CC, CCabs);//�渵��Ҷ�仯��ľ���ֵ

	int maxIdx[2];

	cv::minMaxIdx(CCabs, 0, 0, 0, maxIdx);//���ֵ���ڵ�λ�ü�ƫ�Ƶ�λ��

	std::cout << "maxIdx[0] :" << maxIdx[0] << std::endl;
	std::cout << "maxIdx[1] :" << maxIdx[1] << std::endl;
	//cv::Mat M_A = cv::Mat(CC.size(), CV_64FC1);
	cv::Mat planes[] = { cv::Mat::zeros(CC.size(), CV_64FC1), cv::Mat::zeros(CC.size(), CV_64FC1) };
	cv::split(CC, planes);
	double CCmaxReal = planes[0].at<double>(maxIdx[0], maxIdx[1])*nr*nc;//�渵��Ҷ�仯�����Ҫ ���� ��������
	double CCmaxImag = planes[1].at<double>(maxIdx[0], maxIdx[1])*nr*nc;

	//ifftshiftvec
	int *Nr2 = new int[2 * nr](); int *Nc2 = new int[2 * nc]();
	int beginNr2 = -floor(nr);
	int endNr2 = ceil(nr) - 1;
	ifftshiftVec<int>(Nr2, beginNr2, endNr2);

	int beginNc2 = -floor(nc);
	int endNc2 = ceil(nc) - 1;
	ifftshiftVec<int>(Nc2, beginNc2, endNc2);

	maxIdx[0] = Nr2[maxIdx[0]] / 2;
	maxIdx[1] = Nc2[maxIdx[1]] / 2;

	//���´���Ϊ if usfac>2 ���û������ϸ������΢���Ĳ��  �����渵��Ҷ�仯����
	maxIdx[0] = round(maxIdx[0] * usfac) / usfac;
	maxIdx[1] = round(maxIdx[1] * usfac) / usfac;
	float dftshift = floor(ceil(usfac*1.5f) / 2);
	float roff = dftshift - maxIdx[0] * usfac;
	float coff = dftshift - maxIdx[1] * usfac;

	complexAMulconjB(temp,buf2ft, buf1ft);
	std::vector<cv::Mat> CC2;
	float nor = ceil(usfac*1.5); float noc = ceil(usfac*1.5);
	cv::Mat real = cv::Mat::zeros(nor, noc, CV_64FC1);
	cv::Mat imag = cv::Mat::zeros(nor, noc, CV_64FC1);
	CC2.push_back(real); CC2.push_back(imag);
	dftups(CC2, temp, nor, noc, usfac, roff, coff);//�����Լ����渵��Ҷ�㷨�������(������㷨��̫���)
	CC2[1] = -CC2[1];//conj
	Mabs(CC2,CCabs);
	int maxIdx2[2];//int *maxIdx2 = new int[2]();
	cv::minMaxIdx(CCabs, 0, 0, 0, maxIdx2);
	CCmaxReal = CC2[0].at<double>(maxIdx2[0], maxIdx2[1]);
	CCmaxImag = CC2[1].at<double>(maxIdx2[0], maxIdx2[1]);
	Ind.indexY[loop] = maxIdx[0] + ((float)maxIdx2[0] - dftshift) / usfac;
	Ind.indexX[loop] = maxIdx[1] + ((float)maxIdx2[1] - dftshift) / usfac;

	delete[] Nr; delete[] Nr2;
	delete[] Nc; delete[] Nc2;

}

void bilinear(float *ImageIn,float x,float y,unsigned int oHeight, unsigned int oWidth)
{
	//x����
	if (x != 0)
	{
		if (x > 0)//����
		{
			float preValue = 0;
			for (int i = 0; i < oHeight; ++i)
			{
				for (int j = 0; j < oWidth; ++j)
				{
					if (j == 0)
					{
						preValue = ImageIn[j + i * oWidth];//��¼��һ����������ֵ
						ImageIn[j + i * oWidth] -= ImageIn[j + i * oWidth] * x;
					}
					else
					{
						ImageIn[j + i * oWidth] -= (ImageIn[j + i * oWidth] - preValue) * x;
						preValue = ImageIn[j + i * oWidth];//��¼��һ����������ֵ
					}
				}
			}
		}
		else//����
		{
			for (int i = 0; i < oHeight; ++i)
			{
				for (int j = 0; j < oWidth; ++j)
				{
					if (j == oWidth-1)
					{
						ImageIn[j + i * oWidth] -= ImageIn[j + i * oWidth] * x;
					}
					else
					{
						ImageIn[j + i * oWidth] -= (ImageIn[j + 1 + i * oWidth] - ImageIn[j + i * oWidth]) * x;
					}
				}
			}
		}
	}
	//y����
	if (y != 0)
	{
		if (y > 0)//����
		{
			std::queue<float> preQueue;
			for (int i = 0; i < oHeight; ++i)
			{
				for (int j = 0; j < oWidth; ++j)
				{
					preQueue.push(ImageIn[j + i * oWidth]);//��¼��һ�е�����
					if (i == 0)
					{
						//preQueue.push(ImageIn[j + i * oWidth]);//��¼��һ�е�����
						ImageIn[j + i * oWidth] -= ImageIn[j + i * oWidth] * y;
					}
					else
					{
						ImageIn[j + i * oWidth] -= (ImageIn[j + i * oWidth] - preQueue.front()) * y;
						preQueue.pop();
					}
				}
			}
		}
		else//����
		{
			for (int i = 0; i < oHeight; ++i)
			{
				for (int j = 0; j < oWidth; ++j)
				{
					if (i == oHeight-1)
					{
						ImageIn[j + i * oWidth] -= ImageIn[j + i * oWidth] * y;
					}
					else
					{
						ImageIn[j + i * oWidth] -= (ImageIn[j + (i + 1) * oWidth] - ImageIn[j + i * oWidth]) * y;
					}
				}
			}
		}
	}
}

//����x��y���ˣ������xΪ�У�yΪ��(�������Ӧ����xΪ�У�yΪ��)
void translate(float *ImageIn,float *Imagetrans, int x, int y, unsigned int oHeight, unsigned int oWidth)
{
	if (x == 0 && y == 0)//������λƽ��
	{
		memcpy(Imagetrans, ImageIn, sizeof(float)*oHeight*oWidth);
		return;
	}
	//y����ƽ��
	if (x != 0)
	{
		int absx = abs(x);
		if (x > 0)
			memcpy(Imagetrans + absx * oWidth, ImageIn, sizeof(float)*(oHeight - absx)*oWidth);
		if (x < 0)
			memcpy(Imagetrans, ImageIn + absx * oWidth, sizeof(float)*(oHeight - absx)*oWidth);
		//y����ƽ��
		if (y != 0)
		{
			int absy = abs(y);
			for (int i = 0; i < oHeight; ++i)
			{
				if (y > 0)
					memmove(Imagetrans + i * oWidth + absy, Imagetrans + i * oWidth, sizeof(float)*(oWidth - absy));
				if (y < 0)
					memmove(Imagetrans + i * oWidth, Imagetrans + i * oWidth + absy, sizeof(float)*(oWidth - absy));
			}
		}
	}

	//x����ƽ��
	if (y != 0)
	{
		int absy = abs(y);
		for (int i = 0; i < oHeight; ++i)
		{
			if (y > 0)
				memcpy(Imagetrans + i * oWidth + absy, ImageIn + i * oWidth, sizeof(float)*(oWidth - absy));
			if (y < 0)
				memcpy(Imagetrans + i * oWidth, ImageIn + i * oWidth + absy, sizeof(float)*(oWidth - absy));
		}
		if (x != 0)
		{
			int absx = abs(x);
			if (x > 0)
				memmove(Imagetrans + absx * oWidth, Imagetrans, sizeof(float)*(oHeight - absx)*oWidth);
			if (x < 0)
				memmove(Imagetrans, Imagetrans + absx * oWidth, sizeof(float)*(oHeight - absx)*oWidth);
		}
	}

}

void imtranslate(float *ImageIn,float *Imagetrans, float x,float y, unsigned int oHeight, unsigned int oWidth)
{
	//˫���Բ�ֵf(i, j + v) = f(i, j) - [f(i, j + 1) - f(i, j)] * v;
	//			f(i + u, j)=f(i, j) - [f(i + 1, j)-f(i, j)] * u;
	if(x==0&&y==0)//��ƽ��
	{
		memcpy(Imagetrans, ImageIn, sizeof(float)*oHeight*oWidth);
	}
	else//ƽ��
	{
		//�ж��Ƿ���С��λ����С��λҪ����˫���Բ�ֵ
		float decimalX = 0; float decimalY = 0;
		decimalX = x - (int)x;
		decimalY = y - (int)y;
		//����λ����ƽ��
		translate(ImageIn, Imagetrans, (int)y, (int)x, oHeight, oWidth);
		//С��λ����ƽ��(˫���Բ�ֵ)
		bilinear(Imagetrans, decimalX, decimalY, oHeight, oWidth);
	}
}

void getArea(float *input, float *output,int startPointX,int startPointY,unsigned int inputWidth ,unsigned int ouputHeight, unsigned int outputWidth)
{
	for (int i = 0; i < ouputHeight; ++i)
	{
		for (int j = 0; j < outputWidth; ++j)
		{
			output[j + i * outputWidth] = input[startPointX - 1 + j + (startPointY - 1 + i)*inputWidth];
		}
	}
}

void registration(float *ImageIn, float *ImageOut, unsigned int oHeight, unsigned int oWidth, unsigned int nHeight, unsigned int nWidth, unsigned int num)
{
	int startPointX = round((float)(oWidth - nWidth) / 2);//��
	int startPointY = round((float)(oHeight - nHeight) / 2);//��

	float *fixed = new float[oHeight*oWidth]();
	memcpy(fixed, ImageIn, sizeof(float)*oHeight*oWidth);

	//�������
	IndexOffset ind(num);
	cv::Mat M_fixed = fft2(fixed, oHeight, oWidth);
	//for (int i = 0; i < oWidth; ++i)
	//{
	//	std::cout << i + 1 << "  :" << std::fixed << M_fixed.at<float>(0, i) << std::endl;
	//}

	//cv::Mat M_im1 = fft2(ImageIn + 1 * oHeight*oWidth, oHeight, oWidth);
	//dftregistration(ind, M_fixed, M_im1, 100, 1);

	clock_t start, end;
	for (int i = 0; i < num; ++i)
	{
		cv::Mat M_im1 = fft2(ImageIn + i * oHeight*oWidth, oHeight, oWidth);
		dftregistration(ind, M_fixed, M_im1, 100, i);
		std::cout << "�� ��" << ind.indexX[i] << "   �� ��" << ind.indexY[i] << std::endl;
	}

	//ͼ��ƽ��(˫���Բ�ֵ)����ȡ����
	float *Imagetrans = new float[oHeight*oWidth]();
	//float *registered = new float[nHeight*nWidth*num]();
	for (int i = 0; i < num; ++i)
	{
		imtranslate(ImageIn + i * oHeight*oWidth, Imagetrans, ind.indexX[i], ind.indexY[i], oHeight, oWidth);
		getArea(Imagetrans, ImageOut+i*nHeight*nWidth,startPointX, startPointY, oWidth, nHeight, nWidth);
		std::memset(Imagetrans, 0, sizeof(float)*oHeight*oWidth);
	}

}