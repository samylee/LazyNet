#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "convolution_layer.h"
#include "relu_layer.h"
#include "maxpool_layer.h"
#include "fully_connected_layer.h"
#include "softmax_layer.h"
#include "softmax_loss_layer.h"

using namespace cv;

class LazyNet
{
public:
	LazyNet(string &data_path, int &data_batch, int &data_channel,
		int &data_size, int &max_iter, int &disp_iter,
		float &weight_decay, float &base_lr, float &momentum,
		string &lr_policy, float &power, float &gamma, int &test_iter);
	~LazyNet();

	int TrainNet();

private:
	void LoadNetWork();
	void ApplyHistoryData(Flow &history, Flow &current_data);
	void LoadDataLabel(int img_count, string phase);
	Mat SetMean(int target_width, int target_height);
	void ImageToDataTmp(Mat &img_norm, float label, int batch_count);
	Mat ImageNormalize(Mat &intput, string norm_type);

	void NetForward(int &iter_count, string phase);
	void NetBackward();

	void NetUpdate(int iter_count);
	void UpdateRegularize();
	void UpdateValue(int iter_count);
	void UpdateWeightBias();

	void TestInOutFlow(Flow &in_out_data, string flow_name);

private:
	int data_batch_;
	int data_channel_;
	int data_size_;

	int max_iter_; 
	int disp_iter_; 
	float weight_decay_;
	float base_lr_; 
	float momentum_; 
	string lr_policy_;
	float power_; 
	float gamma_;
	int test_iter_;

	float *input_data_tmp_;
	float *label_data_tmp_;

	vector<String> data0_train_path;
	vector<String> data1_train_path;

	vector<String> data0_test_path;
	vector<String> data1_test_path;

private:
	Flow input_data_;
	Flow label_data_;

	Flow conv1_data_;
	Flow relu1_data_;
	Flow pool1_data_;

	Flow conv2_data_;
	Flow relu2_data_;
	Flow pool2_data_;

	Flow conv3_data_;
	Flow relu3_data_;

	Flow fc1_data_;
	Flow relu4_data_;
	Flow fc2_data_;

	Flow softmax_data_;
	Flow softmax_loss_data_;

	vector<Flow> history_data_;

private:
	ConvolutionLayer conv1_;
	ReLULayer relu1_;
	MaxPoolLayer pool1_;

	ConvolutionLayer conv2_;
	ReLULayer relu2_;
	MaxPoolLayer pool2_;

	ConvolutionLayer conv3_;
	ReLULayer relu3_;

	FullyConnectedLayer fc1_;
	ReLULayer relu4_;
	FullyConnectedLayer fc2_;

	SoftmaxLayer softmax_;
	SoftmaxWithLossLayer softmax_loss_;
};