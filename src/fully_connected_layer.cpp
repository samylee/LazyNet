#include "fully_connected_layer.h"

FullyConnectedLayer::FullyConnectedLayer()
{
}

FullyConnectedLayer::~FullyConnectedLayer()
{
	bias_multiplier_.Release();

	weight_.Release();
	bias_.Release();
}

void FullyConnectedLayer::LayerSetUp(Flow &bottom, Flow &top, vector<int> &weight_shape, vector<int> &bias_shape)
{
	vector<int> weight_shape_;
	//top output
	N_ = weight_shape[0];
	//bottom shape dim
	if (bottom.FlowShape().size() == 4)
		K_ = bottom.ShapeCount(1, 4);
	else
		K_ = bottom.WhichDimensionShape(1);

	weight_shape_.push_back(N_);
	weight_shape_.push_back(K_);
	weight_.SetShapeData(weight_shape_);
	bias_.SetShapeData(bias_shape);
	InitWeightBias(weight_, bias_);

	//number_
	M_ = bottom.WhichDimensionShape(0);

	vector<int> top_shape;
	top_shape.push_back(M_);
	top_shape.push_back(N_);

	//top_data申请内存
	top.SetShapeData(top_shape);

	vector<int> bias_shape_ = { M_ };
	bias_multiplier_.SetShapeData(bias_shape_);
	bias_multiplier_.InitData(1.0);

	//输出信息
	static int fc_count = 1;
	if(bottom.WhichDimensionShape(2))
		cout << "fc" << fc_count << ": \t" << bottom.WhichDimensionShape(2) << "x" << bottom.WhichDimensionShape(3);
	else
		cout << "fc" << fc_count << ": \t1x1";
	cout << "\t --> \t1x1" << endl;
	fc_count++;
}

void FullyConnectedLayer::ForwardNet(Flow &bottom, Flow &top)
{
	float *bottom_data = bottom.GetData();
	float *top_data = top.GetData();

	float *weight = weight_.GetData();
	OpenblasGemm(CblasNoTrans, CblasTrans,
		M_, N_, K_, (float)1.,
		bottom_data, weight, (float)0., top_data);

	float *bias = bias_.GetData();
	OpenblasGemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, (float)1.,
		bias_multiplier_.GetData(),
		bias, (float)1., top_data);
}
void FullyConnectedLayer::BackwardNet(Flow &top, Flow &bottom)
{
	//weights_
	//weight_.GetDiff() = dloss / dw = (dloss / dz) * (dz / dw) = (dloss / dz) * (d(wx+b) / dw) = (dloss / dz) * x(输入)
	//其中dloss / dz = top_diff
	//其中x = bottom_data
	float* top_diff = top.GetDiff();
	float* bottom_data = bottom.GetData();
	OpenblasGemm(CblasTrans, CblasNoTrans,
		N_, K_, M_,
		(float)1., top_diff, bottom_data,
		(float)1., weight_.GetDiff());

	//bias_
	//bias_.GetDiff() = dloss / db = (dloss / dz) * (dz / db) = (dloss / dz) * (d(wx+b) / db) = (dloss / dz) * 1
	//其中dloss / dz = top_diff
	top_diff = top.GetDiff();
	OpenblasGemv(CblasTrans, M_, N_, (float)1., top_diff,
		bias_multiplier_.GetData(), (float)1.,
		bias_.GetDiff());

	//bottom_data_
	//bottom.GetDiff() = dloss / dx = (dloss / dz) * (dz / dx) = (dloss / dz) * (d(wx+b) / dx) = (dloss / dz) * w(权重)
	//其中dloss / dz = top_diff
	//其中w = weight_.GetData()
	top_diff = top.GetDiff();
	OpenblasGemm(CblasNoTrans, CblasNoTrans,
		M_, K_, N_,
		(float)1., top_diff, weight_.GetData(),
		(float)0., bottom.GetDiff());
}