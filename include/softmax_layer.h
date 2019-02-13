#pragma once

#include "math_function.h"

class SoftmaxLayer
{
public:
	SoftmaxLayer();
	~SoftmaxLayer();

	void LayerSetUp(Flow &bottom, Flow &top);

	void ForwardNet(Flow &bottom, Flow &top);
	void BackwardNet(Flow &top, Flow &bottom);

private:
	int outer_num_;
	int inner_num_;
	int softmax_axis_;
	/// sum_multiplier is used to carry out sum using BLAS
	Flow sum_multiplier_;
	/// scale is an intermediate Blob to hold temporary results.
	Flow scale_;
};

