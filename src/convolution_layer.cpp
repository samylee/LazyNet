#include "convolution_layer.h"

ConvolutionLayer::ConvolutionLayer()
{
}

ConvolutionLayer::~ConvolutionLayer()
{
	col_buffer_.Release();
	bias_multiplier_.Release();

	weight_.Release();
	bias_.Release();
}

void ConvolutionLayer::LayerSetUp(Flow &bottom, Flow &top, vector<int> &weight_shape, vector<int> &bias_shape)
{
	//weight和bias初始化
	weight_.SetShapeData(weight_shape);
	bias_.SetShapeData(bias_shape);
	InitWeightBias(weight_, bias_);

	int output_height = ceil((bottom.WhichDimensionShape(2) + 2 * 0 - weight_shape[2]) / 1.0) + 1;
	int output_width = ceil((bottom.WhichDimensionShape(3) + 2 * 0 - weight_shape[3]) / 1.0) + 1;

	//top_shape
	//number_,channel_,height_,width_
	vector<int> top_shape;
	top_shape.push_back(bottom.WhichDimensionShape(0));
	top_shape.push_back(weight_shape[0]);
	top_shape.push_back(output_height);
	top_shape.push_back(output_width);

	//top_data申请内存
	top.SetShapeData(top_shape);

	weight_dim_ = weight_.ShapeCount(1, 4);
	vector<int> col_buffer_shape_;
	col_buffer_shape_.push_back(weight_dim_);
	col_buffer_shape_.push_back(output_height);
	col_buffer_shape_.push_back(output_width);
	col_buffer_shape_.push_back(1);
	col_buffer_.SetShapeData(col_buffer_shape_);

	conv_in_channel_ = weight_shape[1];
	conv_out_channels_ = weight_shape[0];
	conv_out_spatial_dim_ = top.ShapeCount(2, 4);
	conv_input_shape_.push_back(bottom.WhichDimensionShape(1));
	conv_input_shape_.push_back(bottom.WhichDimensionShape(2));
	conv_input_shape_.push_back(bottom.WhichDimensionShape(3));

	kernel_shape_.push_back(weight_shape[2]);
	kernel_shape_.push_back(weight_shape[3]);

	num_output_ = weight_shape[0];
	out_spatial_dim_ = top.ShapeCount(2, 4);
	vector<int> bias_multiplier_shape = { out_spatial_dim_ };
	bias_multiplier_.SetShapeData(bias_multiplier_shape);
	bias_multiplier_.InitData(1.0);

	//输出信息
	static int conv_count = 1;
	cout << "conv" << conv_count << ": \t" << bottom.WhichDimensionShape(2) << "x" << bottom.WhichDimensionShape(3);
	cout << "\t --> \t" << top.WhichDimensionShape(2) << "x" << top.WhichDimensionShape(3) << endl;
	conv_count++;
}

bool is_a_ge_zero_and_a_lt_b(int a, int b)
{
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void ConvolutionLayer::ConvIm2Col(const float* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	float* data_col)
{
	const int output_h = (height + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row * dilation_h;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_col = -pad_w + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
							}
							else {
								*(data_col++) = 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}

void ConvolutionLayer::ConvCol2Im(const float* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	float* data_im) {

	for (int i = 0; i < height * width * channels; ++i) {
		data_im[i] = float(0);
	}

	const int output_h = (height + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row * dilation_h;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						data_col += output_w;
					}
					else {
						int input_col = -pad_w + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								data_im[input_row * width + input_col] += *data_col;
							}
							data_col++;
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}

void ConvolutionLayer::ForwardWeightGemm(float *input, float *weights, float *output)
{
	const float* col_buff = input;
	ConvIm2Col(input, conv_in_channel_,
		conv_input_shape_[1], conv_input_shape_[2],
		kernel_shape_[0], kernel_shape_[1],
		0, 0,
		1, 1,
		1, 1,
		col_buffer_.GetData());

	col_buff = col_buffer_.GetData();

	OpenblasGemm(CblasNoTrans, CblasNoTrans, conv_out_channels_, conv_out_spatial_dim_, weight_dim_,
		(float)1., weights, col_buff, (float)0., output);
}

void ConvolutionLayer::ForwardBiasGemm(float* output, const float* bias)
{
	OpenblasGemm(CblasNoTrans, CblasNoTrans, num_output_,
		out_spatial_dim_, 1, (float)1., bias, bias_multiplier_.GetData(),
		(float)1., output);
}

void ConvolutionLayer::BackwardWeightGemm(float* output, float* weights, float* input)
{
	float* col_buff = col_buffer_.GetData();

	OpenblasGemm(CblasTrans, CblasNoTrans, weight_dim_,
		conv_out_spatial_dim_, conv_out_channels_,
		(float)1., weights, output,
		(float)0., col_buff);

	ConvCol2Im(col_buff, conv_in_channel_,
		conv_input_shape_[1], conv_input_shape_[2],
		kernel_shape_[0], kernel_shape_[1],
		0, 0,
		1, 1,
		1, 1,
		input);
}

void ConvolutionLayer::BackwardBiasGemm(float* bias, float* input)
{
	OpenblasGemv(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
		input, bias_multiplier_.GetData(), 1., bias);
}

void ConvolutionLayer::WeightGemm(float* input, float* output, float* weights)
{
	float* col_buff = input;
	ConvIm2Col(input, conv_in_channel_,
		conv_input_shape_[1], conv_input_shape_[2],
		kernel_shape_[0], kernel_shape_[1],
		0, 0,
		1, 1,
		1, 1,
		col_buffer_.GetData());

	col_buff = col_buffer_.GetData();

	OpenblasGemm(CblasNoTrans, CblasTrans, conv_out_channels_,
		weight_dim_, conv_out_spatial_dim_,
		(float)1., output, col_buff,
		(float)1., weights);
}

void ConvolutionLayer::ForwardNet(Flow &bottom, Flow &top)
{
	float *weight = weight_.GetData();
	float *bottom_data = bottom.GetData();
	float *top_data = top.GetData();

	int bottom_dim_ = bottom.ShapeCount(1, 4);
	int top_dim_ = top.ShapeCount(1, 4);
	for (int n = 0; n < bottom.WhichDimensionShape(0); n++)
	{
		ForwardWeightGemm(bottom_data + n*bottom_dim_, weight, top_data + n*top_dim_);
		float *bias = bias_.GetData();
		ForwardBiasGemm(top_data + n * top_dim_, bias);
	}
}

void ConvolutionLayer::BackwardNet(Flow &top, Flow &bottom)
{
	float* weight_data = weight_.GetData();
	float* weight_diff = weight_.GetDiff();

	float* top_diff = top.GetDiff();
	float* bottom_data = bottom.GetData();
	float* bottom_diff = bottom.GetDiff();

	int bottom_dim_ = bottom.ShapeCount(1, 4);
	int top_dim_ = top.ShapeCount(1, 4);

	//类似全连接的偏导方式
	// Bias gradient, if necessary.
	float* bias_diff = bias_.GetDiff();
	for (int n = 0; n < bottom.WhichDimensionShape(0); ++n) {
		BackwardBiasGemm(bias_diff, top_diff + n * top_dim_);
	}

	for (int n = 0; n < bottom.WhichDimensionShape(0); ++n) {
		// gradient w.r.t. weight. Note that we will accumulate diffs.
		//类似全连接的偏导方式
		WeightGemm(bottom_data + n * bottom_dim_,
			top_diff + n * top_dim_, weight_diff);
		// gradient w.r.t. bottom data, if necessary.
		//类似全连接的偏导方式
		BackwardWeightGemm(top_diff + n * top_dim_, weight_data,
			bottom_diff + n * bottom_dim_);
	}
}