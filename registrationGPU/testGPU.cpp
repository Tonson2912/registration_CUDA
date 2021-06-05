#include<iostream>
#include <opencv2/opencv.hpp>

#include"registrationGPU.h"

using namespace std;
typedef float Tdata;

template<class T>
void Input_data(T *P, unsigned int row, unsigned int col, unsigned int depth, const char *s1)
{
	FILE *fid0 = fopen(s1, "rb");
	if (fid0 == NULL)
	{
		std::cout << "Open error !!!" << endl;
		std::fclose(fid0);
	}
	int shft = 0;
	fseek(fid0, shft, SEEK_CUR);

	memset(P, 0, sizeof(T));
	fread(P, row*col*depth * sizeof(T), 1, fid0);
	fclose(fid0);
}

int main()
{
	unsigned int height = 2160; unsigned int width = 2560; unsigned int num = 30;
	unsigned int heightOut = height - 10; unsigned int widthOut = width - 10;
	const char *s1 = "../真实样品.txt";//真实样品
	Tdata *realSample = new Tdata[height*width*num](); Input_data<Tdata>(realSample, height, width, num, s1);

	float *realSampleOut = new float[heightOut*widthOut*num]();

	registrationGPU(realSample, realSampleOut, height, width, heightOut, widthOut, num, 2);

	//利用opencv库查看结果图
	cv::Mat im(heightOut, widthOut, CV_8UC1);
	for (int k = 0; k < num; ++k)
	{
		for (int i = 0; i < heightOut; ++i)
		{
			for (int j = 0; j < widthOut; ++j)
			{
				im.at<uchar>(i, j) = (unsigned char)realSampleOut[k * heightOut * widthOut + j + i * widthOut];
			}
		}

		cv::String windowName = "图 " + to_string(k + 1);

		//cv::normalize(im, im, 0, 1, CV_MINMAX);
		cv::namedWindow(windowName, cv::WINDOW_NORMAL);
		cv::imshow(windowName, im);
		cv::waitKey(0);
		//cv::destroyWindow(windowName);
	}
	//for (int i = 0; i < widthOut;++i)
	//{
	//	cout << "  :" << realSampleOut[heightOut * widthOut + i] << endl;
	//}
	system("pause");
	return 0;
}