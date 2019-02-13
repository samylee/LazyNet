#include "softmax_loss_layer.h"

SoftmaxWithLossLayer::SoftmaxWithLossLayer()
{
}

void SoftmaxWithLossLayer::LayerSetUp(int outer_num, Flow &top)
{
	outer_num_ = outer_num;
	inner_num_ = 1;

	vector<int> top_shape = { 1 };
	top.SetShapeData(top_shape);

	//初始化diff
	top.InitDiff(1.0);
}

void SoftmaxWithLossLayer::ForwardNet(vector<Flow> &bottom, Flow &top)
{
	float *prob_data = bottom[0].GetData();
	float *label_data = bottom[1].GetData();
	float *top_data = top.GetData();

	int dim = bottom[0].GetCounts() / outer_num_;
	int count = 0;
	float loss = 0;

	for (int i = 0; i < outer_num_; ++i) {
		for (int j = 0; j < inner_num_; j++) {
			const int label_value = static_cast<int>(label_data[i * inner_num_ + j]);
			//loss = -y*logf(x)（其中y为标签，恒等于1，可省略）
			loss -= log(max(prob_data[i * dim + label_value * inner_num_ + j], float(FLT_MIN)));
			++count;
		}
	}
	top_data[0] = loss / max(float(1.0), float(count));
}
void SoftmaxWithLossLayer::BackwardNet(Flow &top, vector<Flow> &bottom)
{
	float *bottom_diff = bottom[0].GetDiff();
	//softmax_loss的反向传递，其实计算的是softmax的反向传递
	float *prob_data = bottom[0].GetData();
	CopyData(bottom[0].GetCounts(), prob_data, bottom_diff);

	float *label_data = bottom[1].GetData();
	int dim = bottom[0].GetCounts() / outer_num_;
	int count = 0;
	for (int i = 0; i < outer_num_; ++i) {
		for (int j = 0; j < inner_num_; ++j) {
			const int label_value = static_cast<int>(label_data[i * inner_num_ + j]);
			//dloss / dx = f(x) - 1
			//参考https://www.jianshu.com/p/c02a1fbffad6
			bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
			++count;
		}
	}
	// Scale gradient
	float loss_weight = top.GetDiff()[0] / max(float(1.0), float(count));
	ScalData(bottom[0].GetCounts(), loss_weight, bottom_diff);
}