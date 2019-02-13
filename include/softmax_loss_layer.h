#pragma once

#include "math_function.h"

class SoftmaxWithLossLayer
{
public:
	SoftmaxWithLossLayer();

	void LayerSetUp(int outer_num, Flow &top);

	//bottom: 0(feature map), 1(label)
	void ForwardNet(vector<Flow> &bottom, Flow &top);
	void BackwardNet(Flow &top, vector<Flow> &bottom);

private:
	int outer_num_;
	int inner_num_;
};
