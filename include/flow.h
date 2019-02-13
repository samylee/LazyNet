#pragma once

#include <vector>
#include <string>
#include <random>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

class Flow
{
public:
	Flow();
	void SetShapeData(vector<int> &shape);
	vector<int> FlowShape();
	int WhichDimensionShape(int nchw);
	int ShapeCount(int start_axis, int end_axis);
	int offset(int n, int c = 0, int h = 0, int w = 0);

	void InitData(float alpha);
	void InitDiff(float alpha);
	void SetData(float *data);
	void SetDiff(float *diff);
	float* GetData();
	float* GetDiff();
	int GetCounts();
	void Release();

private:
	float *data_;
	float *diff_;

private:
	int number_;
	int channel_;
	int height_;
	int width_;
	int counts_;
};