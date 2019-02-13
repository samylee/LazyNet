#pragma once

#include "math_function.h"

class FullyConnectedLayer
{
public:
	FullyConnectedLayer();
	~FullyConnectedLayer();

	void LayerSetUp(Flow &bottom, Flow &top, vector<int> &weight_shape, vector<int> &bias_shape);

	void ForwardNet(Flow &bottom, Flow &top);
	void BackwardNet(Flow &top, Flow &bottom);

	Flow weight_;
	Flow bias_;

private:
	int M_;
	int K_;
	int N_;
	Flow bias_multiplier_;
};
