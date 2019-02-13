#pragma once

#include "math_function.h"

class ConvolutionLayer
{
public:
	ConvolutionLayer();
	~ConvolutionLayer();

	void LayerSetUp(Flow &bottom, Flow &top, vector<int> &weight_shape, vector<int> &bias_shape);

	void ForwardNet(Flow &bottom, Flow &top);
	void BackwardNet(Flow &top, Flow &bottom);

	Flow weight_;
	Flow bias_;

private:
	void ForwardWeightGemm(float *input, float *weights, float *output);
	void ForwardBiasGemm(float* output, const float* bias);

	void BackwardWeightGemm(float* output, float* weights, float* input);
	void BackwardBiasGemm(float* bias, float* input);

	void WeightGemm(float* input, float* output, float* weights);

	void ConvIm2Col(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		float* data_col);

	void ConvCol2Im(const float* data_col, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		float* data_im);

private:
	Flow col_buffer_;
	Flow bias_multiplier_;

	int weight_dim_;
	int conv_in_channel_;
	int conv_out_channels_;
	int conv_out_spatial_dim_;
	int num_output_;
	int out_spatial_dim_;
	vector<int> conv_input_shape_;
	vector<int> kernel_shape_;
};