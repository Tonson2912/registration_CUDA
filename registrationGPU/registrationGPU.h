#pragma once
/*
	ͼ������㷨��
	������һ��ͼ��(30��)��ȡ���һ����Ϊģ�壬ʣ������ͼ��(29��)����ƥ��(����һ�����Ƚ�)��
	���û�����㷨�����ƥ��ͼ�����һ��ͼ���ƫ������ƫ�����и�ƫ�����Ϳ�ƫ�������ֱ������Ҫƫ�Ƶ����ص��С��
	�������ƫ����֮�󣬶������ƫ�ƣ�ƫ�ƻ�����������ص�Ϊ0��������Ҫ���вü����ü��������û����о���(������Ҫ֪�����ͼ���С)��
	�Ӷ��ﵽ������Ч����

	��ˣ��ú�����Ҫ����������У�
	ImageIn��ͼ������
	ImageOut��ͼ�����
	oHeight��ԭʼͼ��� oWidth��ԭʼͼ���
	nHeight�����ͼ��� nWidth�����ͼ���
	num��ͼ��������(ģ��+��ƥ��ͼ)
	ImageNumPerBatch��ÿ�����ε�ͼ������(��ƥ��ͼ)�������Լ�����������ܽ���ѡ��
*/


extern "C" void registrationGPU(float *ImageIn, float *ImageOut, unsigned int oHeight, unsigned int oWidth, unsigned int nHeight, unsigned int nWidth, unsigned int num,int ImageNumPerBatch);