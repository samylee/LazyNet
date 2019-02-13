#pragma once

#include "math_function.h"

class ReLULayer
{
public:
	ReLULayer();

	void LayerSetUp(Flow &bottom, Flow &top);

	void ForwardNet(Flow &bottom, Flow &top);
	void BackwardNet(Flow &top, Flow &bottom);
};

