#include "relu_layer.h"

ReLULayer::ReLULayer()
{
}

void ReLULayer::LayerSetUp(Flow &bottom, Flow &top)
{
	vector<int> top_shape = bottom.FlowShape();
	top.SetShapeData(top_shape);

	//top_diff_init
	top.InitDiff(0.0);

	// ‰≥ˆ–≈œ¢
	static int relu_count = 1;
	if (bottom.WhichDimensionShape(2))
		cout << "relu" << relu_count << ": \t" << bottom.WhichDimensionShape(2) << "x" << bottom.WhichDimensionShape(3);
	else
		cout << "relu" << relu_count << ": \t1x1";
	if (top.WhichDimensionShape(2))
		cout << "\t --> \t" << top.WhichDimensionShape(2) << "x" << top.WhichDimensionShape(3) << endl;
	else
		cout << "\t --> \t1x1" << endl;
	relu_count++;
}

void ReLULayer::ForwardNet(Flow &bottom, Flow &top)
{
	float* bottom_data = bottom.GetData();
	float* top_data = top.GetData();
	const int count = bottom.GetCounts();
	for (int i = 0; i < count; ++i) 
	{
		top_data[i] = max(bottom_data[i], float(0));
	}
}
void ReLULayer::BackwardNet(Flow &top, Flow &bottom)
{
	//bottom.GetDiff() = dloss / dx = (dloss / dz) * (dz / dx) = (dloss / dz) * (d(x(x>0)) / dx) = (dloss / dz) * 1(x>0)
	//dloss / dz = top_diff
	float* bottom_data = bottom.GetData();
	float* top_diff = top.GetDiff();
	float* bottom_diff = bottom.GetDiff();
	const int count = bottom.GetCounts();
	for (int i = 0; i < count; ++i) {
		bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0));
	}
}
