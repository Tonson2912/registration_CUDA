#include<iostream>
#include"registration.h"

using namespace std;
typedef unsigned short Tdata;
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
	unsigned int height = 1040; unsigned int width = 1392; unsigned int num = 30;//测试5张
	unsigned int heightOut = height - 10; unsigned int widthOut = width - 10;
	const char *s1 = "H:\\Tonson\\真实样品+UInt16[1392x1040x30].txt";//真实样品
	Tdata *realSample = new Tdata[height*width*num](); Input_data<Tdata>(realSample, height, width, num, s1);
	float *temp = new float[height*width*num]();
	for (int i = 0;i < height*width*num;++i)
	{
		temp[i] = (float)realSample[i];
	}
	float *realSampleOut = new float[heightOut*widthOut*num]();


	registration(temp, realSampleOut, height, width, heightOut, widthOut, num);

	system("pause");
	return 0;
}