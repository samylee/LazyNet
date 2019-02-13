#include "softmax_layer.h"

SoftmaxLayer::SoftmaxLayer()
{
}

SoftmaxLayer::~SoftmaxLayer()
{
	sum_multiplier_.Release();
	scale_.Release();
}

void SoftmaxLayer::LayerSetUp(Flow &bottom, Flow &top)
{
	top.SetShapeData(bottom.FlowShape());

	vector<int> mult_dims = { bottom.WhichDimensionShape(1) };
	sum_multiplier_.SetShapeData(mult_dims);
	sum_multiplier_.InitData(1.0);

	outer_num_ = bottom.ShapeCount(0, 1);
	inner_num_ = 1;

	vector<int> scale_dims = bottom.FlowShape();
	scale_dims[1] = 1;
	scale_.SetShapeData(scale_dims);
}

void SoftmaxLayer::ForwardNet(Flow &bottom, Flow &top)
{
	float *bottom_data = bottom.GetData();
	float *top_data = top.GetData();
	float *scale_data = scale_.GetData();

	int channels = bottom.WhichDimensionShape(1);
	int dim = bottom.GetCounts() / outer_num_;

	top.SetData(bottom_data);

	for (int i = 0; i < outer_num_; ++i) 
	{
		CopyData(inner_num_, bottom_data + i * dim, scale_data);
		for (int j = 0; j < channels; j++) {
			for (int k = 0; k < inner_num_; k++) {
				scale_data[k] = std::max(scale_data[k],
					bottom_data[i * dim + j * inner_num_ + k]);
			}
		}
		// subtraction
		OpenblasGemm(CblasNoTrans, CblasNoTrans, channels, inner_num_,
			1, -1., sum_multiplier_.GetData(), scale_data, 1., top_data);

		// exponentiation
		ExpData(dim, top_data, top_data);

		// sum after exp
		OpenblasGemv(CblasTrans, channels, inner_num_, 1.,
			top_data, sum_multiplier_.GetData(), 0., scale_data);

		// division
		for (int j = 0; j < channels; j++) {
			DivData(inner_num_, top_data, scale_data, top_data);
			top_data += inner_num_;
		}
	}
}

void SoftmaxLayer::BackwardNet(Flow &top, Flow &bottom)
{
	//softmax不参与反向传递，因为softmax_loss在计算反向传递时，已经包含了softmax的反向计算
	float *bottom_diff = bottom.GetDiff();
	float *top_diff = top.GetDiff();
	CopyData(bottom.GetCounts(), top_diff, bottom_diff);
}