#pragma once
/*
	图像矫正算法：
	假设有一组图像(30张)，取其第一张作为模板，剩余其它图像(29张)用于匹配(跟第一张做比较)。
	利用互相关算法计算待匹配图像与第一张图像的偏移量，偏移量有高偏移量和宽偏移量，分别代表需要偏移的像素点大小。
	在算出其偏移量之后，对其进行偏移，偏移会产生部分像素点为0，所以需要进行裁剪，裁剪多少由用户自行决定(所以需要知道输出图像大小)，
	从而达到矫正的效果。

	因此，该函数需要传入的数据有：
	ImageIn：图像输入
	ImageOut：图像输出
	oHeight：原始图像高 oWidth：原始图像宽
	nHeight：输出图像高 nWidth：输出图像宽
	num：图像总张数(模板+待匹配图)
	ImageNumPerBatch：每个批次的图像张数(待匹配图)，根据自己计算机的性能进行选择
*/


extern "C" void registrationGPU(float *ImageIn, float *ImageOut, unsigned int oHeight, unsigned int oWidth, unsigned int nHeight, unsigned int nWidth, unsigned int num,int ImageNumPerBatch);