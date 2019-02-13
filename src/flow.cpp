#include "flow.h"

Flow::Flow()
{

}

void Flow::SetShapeData(vector<int> &shape)
{
	switch (shape.size())
	{
	case 1:
		number_ = shape[0];
		counts_ = number_;
		break;
	case 2:
		number_ = shape[0];
		channel_ = shape[1];
		counts_ = number_*channel_;
		break;
	case 3:
		number_ = shape[0];
		channel_ = shape[1];
		height_ = shape[2];
		counts_ = number_*channel_*height_;
		break;
	case 4:
		number_ = shape[0];
		channel_ = shape[1];
		height_ = shape[2];
		width_ = shape[3];
		counts_ = number_*channel_*height_*width_;
		break;
	default:
		break;
	}

	if (data_ != NULL)
		free(data_);
	if (diff_ != NULL)
		free(diff_);

	data_ = (float*)malloc(sizeof(float) * counts_);
	diff_ = (float*)malloc(sizeof(float) * counts_);
}

vector<int> Flow::FlowShape()
{
	vector<int> shape;
	if (number_ != NULL)
		shape.push_back(number_);
	if (channel_ != NULL)
		shape.push_back(channel_);
	if (height_ != NULL)
		shape.push_back(height_);
	if (width_ != NULL)
		shape.push_back(width_);
	return shape;
}

int Flow::WhichDimensionShape(int nchw)
{
	vector<int> shape = FlowShape();
	if (nchw > shape.size() - 1)
		return 0;

	int dimension = 0;
	switch (nchw)
	{
	case 0:
		dimension = number_;
		break;
	case 1:
		dimension = channel_;
		break;
	case 2:
		dimension = height_;
		break;
	case 3:
		dimension = width_;
		break;
	default:
		break;
	}
	return dimension;
}

int Flow::ShapeCount(int start_axis, int end_axis)
{
	int count = 1;
	for (int i = start_axis; i < end_axis; ++i) {
		count *= WhichDimensionShape(i);
	}
	return count;
}

int Flow::offset(int n, int c, int h, int w)
{
	return ((n * channel_ + c) * height_ + h) * width_ + w;
}

void Flow::InitData(float alpha)
{
	float *data_tmp = (float *)malloc(sizeof(float)*counts_);
	for (int i = 0; i < counts_; i++)
	{
		data_tmp[i] = alpha;
	}
	SetData(data_tmp);
	free(data_tmp);
}

void Flow::InitDiff(float alpha)
{
	float *diff_tmp = (float *)malloc(sizeof(float)*counts_);
	for (int i = 0; i < counts_; i++)
	{
		diff_tmp[i] = alpha;
	}
	SetDiff(diff_tmp);
	free(diff_tmp);
}

void Flow::SetData(float *data)
{
	if (data_ != data)
	{
		memcpy(data_, data, sizeof(float) * counts_);  // NOLINT(caffe/alt_fn)
	}
}

void Flow::SetDiff(float *diff)
{
	if (diff_ != diff)
	{
		memcpy(diff_, diff, sizeof(float) * counts_);  // NOLINT(caffe/alt_fn)
	}
}

float* Flow::GetData()
{
	return data_;
}

float* Flow::GetDiff()
{
	return diff_;
}

int Flow::GetCounts()
{
	return counts_;
}

void Flow::Release()
{
	if (data_ != NULL)
	{
		free(data_);
		data_ = NULL;
	}
	if (diff_ != NULL)
	{
		free(diff_);
		diff_ = NULL;
	}
}