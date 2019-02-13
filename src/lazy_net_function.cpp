#include "lazy_net.h"

void LazyNet::ApplyHistoryData(Flow &history, Flow &current_data)
{
	history.SetShapeData(current_data.FlowShape());
	history.InitData(0.0);
}

void LazyNet::LoadNetWork()
{
	//输出信息
	cout << "=========Load network start========" << endl;
	cout << "net\tintput\t\toutput" << endl;

	//                                out,in,ks,ks
	//conv1
	vector<int> conv1_weight_shape = { 16, 3, 3, 3 }; //pad:0, stride:1
	vector<int> conv1_bias_shape = { 16 };
	conv1_.LayerSetUp(input_data_, conv1_data_, conv1_weight_shape, conv1_bias_shape);
	relu1_.LayerSetUp(conv1_data_, relu1_data_);
	vector<int> pool1_shape = { 16,16,2,2 }; //pad:0, stride:2
	pool1_.LayerSetUp(relu1_data_, pool1_data_, pool1_shape);

	//conv2
	vector<int> conv2_weight_shape = { 32,16,3,3 };
	vector<int> conv2_bias_shape = { 32 };
	conv2_.LayerSetUp(pool1_data_, conv2_data_, conv2_weight_shape, conv2_bias_shape);
	relu2_.LayerSetUp(conv2_data_, relu2_data_);
	vector<int> pool2_shape = { 32,32,2,2 };
	pool2_.LayerSetUp(relu2_data_, pool2_data_, pool2_shape);

	//conv3
	vector<int> conv3_weight_shape = { 64,32,3,3 };
	vector<int> conv3_bias_shape = { 64 };
	conv3_.LayerSetUp(pool2_data_, conv3_data_, conv3_weight_shape, conv3_bias_shape);
	relu3_.LayerSetUp(conv3_data_, relu3_data_);

	//fc1
	vector<int> fc1_weight_shape = { 128,64 };
	vector<int> fc1_bias_shape = { 128 };
	fc1_.LayerSetUp(relu3_data_, fc1_data_, fc1_weight_shape, fc1_bias_shape);
	relu4_.LayerSetUp(fc1_data_, relu4_data_);

	//fc2
	vector<int> fc2_weight_shape = { 2,128 };
	vector<int> fc2_bias_shape = { 2 };
	fc2_.LayerSetUp(relu4_data_, fc2_data_, fc2_weight_shape, fc2_bias_shape);

	//softmax
	softmax_.LayerSetUp(fc2_data_, softmax_data_);

	//softmax_loss
	softmax_loss_.LayerSetUp(softmax_data_.WhichDimensionShape(0), softmax_loss_data_);

	//load histroy_data for updating weight
	Flow conv1_weight_history;
	ApplyHistoryData(conv1_weight_history, conv1_.weight_);
	history_data_.push_back(conv1_weight_history);
	Flow conv1_bias_history;
	ApplyHistoryData(conv1_bias_history, conv1_.bias_);
	history_data_.push_back(conv1_bias_history);

	Flow conv2_weight_history;
	ApplyHistoryData(conv2_weight_history, conv2_.weight_);
	history_data_.push_back(conv2_weight_history);
	Flow conv2_bias_history;
	ApplyHistoryData(conv2_bias_history, conv2_.bias_);
	history_data_.push_back(conv2_bias_history);

	Flow conv3_weight_history;
	ApplyHistoryData(conv3_weight_history, conv3_.weight_);
	history_data_.push_back(conv3_weight_history);
	Flow conv3_bias_history;
	ApplyHistoryData(conv3_bias_history, conv3_.bias_);
	history_data_.push_back(conv3_bias_history);

	Flow fc1_weight_history;
	ApplyHistoryData(fc1_weight_history, fc1_.weight_);
	history_data_.push_back(fc1_weight_history);
	Flow fc1_bias_history;
	ApplyHistoryData(fc1_bias_history, fc1_.bias_);
	history_data_.push_back(fc1_bias_history);

	Flow fc2_weight_history;
	ApplyHistoryData(fc2_weight_history, fc2_.weight_);
	history_data_.push_back(fc2_weight_history);
	Flow fc2_bias_history;
	ApplyHistoryData(fc2_bias_history, fc2_.bias_);
	history_data_.push_back(fc2_bias_history);

	//输出信息
	cout << "=========Load network end========" << endl;
	cout << endl;
}

Mat LazyNet::SetMean(int target_width, int target_height)
{
	Mat mean_;
	vector<float> values{ 104,117,123 };
	vector<Mat> channels;
	for (int i = 0; i < data_channel_; ++i) {
		cv::Mat channel(target_width, target_height, CV_32FC1, cv::Scalar(values[i]));
		channels.push_back(channel);
	}
	cv::merge(channels, mean_);

	return mean_;
}

Mat LazyNet::ImageNormalize(Mat &intput, string norm_type)
{
	if (intput.channels() == 1)
		cvtColor(intput, intput, CV_GRAY2BGR);
	else if (intput.channels() == 4)
		cvtColor(intput, intput, CV_BGRA2BGR);

	Mat sample_float;
	intput.convertTo(sample_float, CV_32FC3);

	Mat sample_normalized;
	if (norm_type == "mean_value")
	{
		//载入均值文件
		Mat mean_ = SetMean(data_size_, data_size_);
		subtract(sample_float, mean_, sample_normalized);
	}
	else
	{
		//scale值
		float scale = 0.00390625;
		sample_normalized = sample_float*scale;
	}

	return sample_normalized;
}

void LazyNet::ImageToDataTmp(Mat &img_norm, float label, int batch_count)
{
	//load image
	int data_index = batch_count*data_channel_*data_size_*data_size_;
	for (int c = 0; c < data_channel_; c++)
	{
		for (int nr = 0; nr < data_size_; nr++)
		{
			for (int nc = 0; nc < data_size_; nc++)
			{
				input_data_tmp_[data_index] = img_norm.at<Vec3f>(nr, nc)[c];
				data_index += 1;
			}
		}
	}

	//load label
	int label_index = batch_count;
	label_data_tmp_[label_index] = label;
}

void LazyNet::LoadDataLabel(int img_count, string phase)
{
	string norm_type = "scale";// or mean_value
	int batch_count = 0;
	for (int take = img_count; take < img_count + data_batch_; take++)
	{
		String data0_path_tmp;
		if (phase == "train")
			data0_path_tmp = data0_train_path[take];
		else
			data0_path_tmp = data0_test_path[take];

		Mat img0_tmp = imread(data0_path_tmp);
		Mat img0_tmp_norm = ImageNormalize(img0_tmp, norm_type);
		float label0 = 0.0;
		ImageToDataTmp(img0_tmp_norm, label0, batch_count);
		batch_count++;

		String data1_path_tmp;
		if (phase == "train")
			data1_path_tmp = data1_train_path[take];
		else
			data1_path_tmp = data1_test_path[take];

		Mat img1_tmp = imread(data1_path_tmp);
		Mat img1_tmp_norm = ImageNormalize(img1_tmp, norm_type);
		float label1 = 1.0;
		ImageToDataTmp(img1_tmp_norm, label1, batch_count);
		batch_count++;
	}

	input_data_.SetData(input_data_tmp_);
	label_data_.SetData(label_data_tmp_);
}

void LazyNet::UpdateRegularize()
{
	//conv1_weight
	AxpyData(conv1_.weight_.GetCounts(), weight_decay_, conv1_.weight_.GetData(), conv1_.weight_.GetDiff());
	//conv1_bias
	AxpyData(conv1_.bias_.GetCounts(), weight_decay_, conv1_.bias_.GetData(), conv1_.bias_.GetDiff());

	//conv2_weight
	AxpyData(conv2_.weight_.GetCounts(), weight_decay_, conv2_.weight_.GetData(), conv2_.weight_.GetDiff());
	//conv2_bias
	AxpyData(conv2_.bias_.GetCounts(), weight_decay_, conv2_.bias_.GetData(), conv2_.bias_.GetDiff());

	//conv3_weight
	AxpyData(conv3_.weight_.GetCounts(), weight_decay_, conv3_.weight_.GetData(), conv3_.weight_.GetDiff());
	//conv3_bias
	AxpyData(conv3_.bias_.GetCounts(), weight_decay_, conv3_.bias_.GetData(), conv3_.bias_.GetDiff());

	//fc1_weight
	AxpyData(fc1_.weight_.GetCounts(), weight_decay_, fc1_.weight_.GetData(), fc1_.weight_.GetDiff());
	//fc1_bias
	AxpyData(fc1_.bias_.GetCounts(), weight_decay_, fc1_.bias_.GetData(), fc1_.bias_.GetDiff());

	//fc2_weight
	AxpyData(fc2_.weight_.GetCounts(), weight_decay_, fc2_.weight_.GetData(), fc2_.weight_.GetDiff());
	//fc2_bias
	AxpyData(fc2_.bias_.GetCounts(), weight_decay_, fc2_.bias_.GetData(), fc2_.bias_.GetDiff());
}

void LazyNet::UpdateValue(int iter_count)
{
	//lr更新
	if (lr_policy_ == "inv")
	{
		base_lr_ = base_lr_ * pow(float(1) + gamma_ * iter_count, -power_);
	}
	else
	{
		int step_size = 100;
		int current_step = iter_count / step_size;
		base_lr_ = base_lr_ * pow(gamma_, current_step);
	}

	//conv1_weight_lr_mult:1
	AxpbyData(conv1_.weight_.GetCounts(), base_lr_, conv1_.weight_.GetDiff(), momentum_, history_data_[0].GetData());
	CopyData(conv1_.weight_.GetCounts(), history_data_[0].GetData(), conv1_.weight_.GetDiff());
	//conv1_bias_lr_mult:2
	AxpbyData(conv1_.bias_.GetCounts(), base_lr_ * 2, conv1_.bias_.GetDiff(), momentum_, history_data_[1].GetData());
	CopyData(conv1_.bias_.GetCounts(), history_data_[1].GetData(), conv1_.bias_.GetDiff());

	//conv2_weight
	AxpbyData(conv2_.weight_.GetCounts(), base_lr_, conv2_.weight_.GetDiff(), momentum_, history_data_[2].GetData());
	CopyData(conv2_.weight_.GetCounts(), history_data_[2].GetData(), conv2_.weight_.GetDiff());
	//conv2_bias
	AxpbyData(conv2_.bias_.GetCounts(), base_lr_ * 2, conv2_.bias_.GetDiff(), momentum_, history_data_[3].GetData());
	CopyData(conv2_.bias_.GetCounts(), history_data_[3].GetData(), conv2_.bias_.GetDiff());

	//conv3_weight
	AxpbyData(conv3_.weight_.GetCounts(), base_lr_, conv3_.weight_.GetDiff(), momentum_, history_data_[4].GetData());
	CopyData(conv3_.weight_.GetCounts(), history_data_[4].GetData(), conv3_.weight_.GetDiff());
	//conv3_bias
	AxpbyData(conv3_.bias_.GetCounts(), base_lr_ * 2, conv3_.bias_.GetDiff(), momentum_, history_data_[5].GetData());
	CopyData(conv3_.bias_.GetCounts(), history_data_[5].GetData(), conv3_.bias_.GetDiff());

	//fc1_weight
	AxpbyData(fc1_.weight_.GetCounts(), base_lr_, fc1_.weight_.GetDiff(), momentum_, history_data_[6].GetData());
	CopyData(fc1_.weight_.GetCounts(), history_data_[6].GetData(), fc1_.weight_.GetDiff());
	//fc1_bias
	AxpbyData(fc1_.bias_.GetCounts(), base_lr_ * 2, fc1_.bias_.GetDiff(), momentum_, history_data_[7].GetData());
	CopyData(fc1_.bias_.GetCounts(), history_data_[7].GetData(), fc1_.bias_.GetDiff());

	//fc2_weight
	AxpbyData(fc2_.weight_.GetCounts(), base_lr_, fc2_.weight_.GetDiff(), momentum_, history_data_[8].GetData());
	CopyData(fc2_.weight_.GetCounts(), history_data_[8].GetData(), fc2_.weight_.GetDiff());
	//fc2_bias
	AxpbyData(fc2_.bias_.GetCounts(), base_lr_ * 2, fc2_.bias_.GetDiff(), momentum_, history_data_[9].GetData());
	CopyData(fc2_.bias_.GetCounts(), history_data_[9].GetData(), fc2_.bias_.GetDiff());
}

void LazyNet::UpdateWeightBias()
{
	//conv1_weight
	AxpyData(conv1_.weight_.GetCounts(), float(-1), conv1_.weight_.GetDiff(), conv1_.weight_.GetData());
	//conv1_bias
	AxpyData(conv1_.bias_.GetCounts(), float(-1), conv1_.bias_.GetDiff(), conv1_.bias_.GetData());

	//conv2_weight
	AxpyData(conv2_.weight_.GetCounts(), float(-1), conv2_.weight_.GetDiff(), conv2_.weight_.GetData());
	//conv2_bias
	AxpyData(conv2_.bias_.GetCounts(), float(-1), conv2_.bias_.GetDiff(), conv2_.bias_.GetData());

	//conv3_weight
	AxpyData(conv3_.weight_.GetCounts(), float(-1), conv3_.weight_.GetDiff(), conv3_.weight_.GetData());
	//conv3_bias
	AxpyData(conv3_.bias_.GetCounts(), float(-1), conv3_.bias_.GetDiff(), conv3_.bias_.GetData());

	//fc1_weight
	AxpyData(fc1_.weight_.GetCounts(), float(-1), fc1_.weight_.GetDiff(), fc1_.weight_.GetData());
	//fc1_bias
	AxpyData(fc1_.bias_.GetCounts(), float(-1), fc1_.bias_.GetDiff(), fc1_.bias_.GetData());

	//fc2_weight
	AxpyData(fc2_.weight_.GetCounts(), float(-1), fc2_.weight_.GetDiff(), fc2_.weight_.GetData());
	//fc2_bias
	AxpyData(fc2_.bias_.GetCounts(), float(-1), fc2_.bias_.GetDiff(), fc2_.bias_.GetData());
}

void LazyNet::TestInOutFlow(Flow &in_out_data, string flow_name)
{
	float *bottom_data = in_out_data.GetData();
	string file_bottom_name = "test_flow/" + flow_name + ".txt";
	ofstream bottom_file(file_bottom_name);

	if (in_out_data.FlowShape().size() == 4)
	{
		for (int n = 0; n < in_out_data.WhichDimensionShape(0); n++)
		{
			bottom_file << "number: " + to_string(n) << endl;
			for (int c = 0; c < in_out_data.WhichDimensionShape(1); c++)
			{
				for (int h = 0; h < in_out_data.WhichDimensionShape(2); h++)
				{
					for (int w = 0; w < in_out_data.WhichDimensionShape(3); w++)
					{
						stringstream strStream;
						int dim = in_out_data.WhichDimensionShape(3) * (in_out_data.WhichDimensionShape(2) * (in_out_data.WhichDimensionShape(1)*n + c) + h) + w;
						strStream << bottom_data[dim];

						bottom_file << strStream.str() << " ";
					}
					bottom_file << endl;
				}
				bottom_file << endl;
			}
			bottom_file << endl;
		}
	}
	else
	{
		for (int n = 0; n < in_out_data.WhichDimensionShape(0); n++)
		{
			bottom_file << "number: " + to_string(n) << endl;
			for (int c = 0; c < in_out_data.WhichDimensionShape(1); c++)
			{
				stringstream strStream;
				int dim = n * in_out_data.WhichDimensionShape(1) + c;
				strStream << bottom_data[dim];

				bottom_file << strStream.str() << " " << endl;
			}
			bottom_file << endl;
		}
	}

	bottom_file.close();
}