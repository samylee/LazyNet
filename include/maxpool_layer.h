#pragma once

#include "math_function.h"

class MaxPoolLayer
{
public:
	MaxPoolLayer();
	~MaxPoolLayer();

	void LayerSetUp(Flow &bottom, Flow &top, vector<int> &pool_shape);

	void ForwardNet(Flow &bottom, Flow &top);
	void BackwardNet(Flow &top, Flow &bottom);

private:
	int kernel_h_;
	int kernel_w_;

	Flow max_idx_;
};
