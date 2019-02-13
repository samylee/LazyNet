#include "lazy_net.h"

LazyNet::LazyNet(string &data_path, int &data_batch, int &data_channel,
	int &data_size, int &max_iter, int &disp_iter,
	float &weight_decay, float &base_lr, float &momentum,
	string &lr_policy, float &power, float &gamma, int &test_iter)
{
	data_batch_ = data_batch;
	data_channel_ = data_channel;
	data_size_ = data_size;

	max_iter_ = max_iter;
	disp_iter_ = disp_iter;
	weight_decay_ = weight_decay;
	base_lr_ = base_lr;
	momentum_ = momentum;
	lr_policy_ = lr_policy;
	power_ = power;
	gamma_ = gamma;
	test_iter_ = test_iter;

	//输入申请内存
	vector<int> image_shape = { data_batch * 2, data_channel, data_size, data_size };
	input_data_.SetShapeData(image_shape);
	input_data_tmp_ = (float *)malloc(sizeof(float)*data_batch * 2 * data_channel* data_size* data_size);

	//输出申请内存
	vector<int> label_shape = { data_batch * 2, 1 };
	label_data_.SetShapeData(label_shape);
	label_data_tmp_ = (float *)malloc(sizeof(float)*data_batch * 2 * 1);

	//载入网络结构
	LoadNetWork();

	//载入所有训练图像路径
	string train_data0_path = data_path + "train/0/*.jpg";
	string train_data1_path = data_path + "train/1/*.jpg";
	glob(train_data0_path, data0_train_path);
	glob(train_data1_path, data1_train_path);

	//载入所有测试图像路径
	string test_data0_path = data_path + "test/0/*.jpg";
	string test_data1_path = data_path + "test/1/*.jpg";
	glob(test_data0_path, data0_test_path);
	glob(test_data1_path, data1_test_path);
}

LazyNet::~LazyNet()
{
	free(input_data_tmp_);
	free(label_data_tmp_);

	input_data_.Release();

	conv1_data_.Release();
	relu1_data_.Release();
	pool1_data_.Release();

	conv2_data_.Release();
	relu2_data_.Release();
	pool2_data_.Release();

	conv3_data_.Release();
	relu3_data_.Release();

	fc1_data_.Release();
	relu4_data_.Release();
	fc2_data_.Release();

	softmax_data_.Release();
	softmax_loss_data_.Release();

	for (int i = 0; i < history_data_.size(); i++)
	{
		history_data_[i].Release();
	}
}

void LazyNet::NetForward(int &iter_count, string phase)
{
	//conv1
	conv1_.ForwardNet(input_data_, conv1_data_);
	relu1_.ForwardNet(conv1_data_, relu1_data_);
	pool1_.ForwardNet(relu1_data_, pool1_data_);

	//conv2
	conv2_.ForwardNet(pool1_data_, conv2_data_);
	relu2_.ForwardNet(conv2_data_, relu2_data_);
	pool2_.ForwardNet(relu2_data_, pool2_data_);

	//conv3
	conv3_.ForwardNet(pool2_data_, conv3_data_);
	relu3_.ForwardNet(conv3_data_, relu3_data_);

	//fc1
	fc1_.ForwardNet(relu3_data_, fc1_data_);
	relu4_.ForwardNet(fc1_data_, relu4_data_);

	//fc2
	fc2_.ForwardNet(relu4_data_, fc2_data_);

	//softmax
	softmax_.ForwardNet(fc2_data_, softmax_data_);

	//输出loss
	if (phase == "train")
	{
		//softmax_loss
		vector<Flow> softmax_loss_input_;
		softmax_loss_input_.push_back(softmax_data_);
		softmax_loss_input_.push_back(label_data_);
		softmax_loss_.ForwardNet(softmax_loss_input_, softmax_loss_data_);

		if (iter_count%disp_iter_ == 0)
		{
			float *loss_data = softmax_loss_data_.GetData();
			cout << "iter: " << iter_count << "     \tloss: " << loss_data[0] << "     \tlearning_rate: " << base_lr_ << endl;
		}
	}
	//输出accuracy
	else
	{
		int test_data_count = softmax_data_.WhichDimensionShape(0);
		int right_count = 0;
		for (int i = 0; i < test_data_count; i++)
		{
			float output0 = softmax_data_.GetData()[i * 2];
			float output1 = softmax_data_.GetData()[i * 2 + 1];
			float label = label_data_.GetData()[i];

			if ((label == 0 && output0 > output1) || (label == 1 && output0 < output1))
				right_count++;
		}
		float accuracy = (float)right_count / (float)test_data_count;
		cout << endl;
		cout << "Test Acc: " << accuracy << endl;
		cout << endl;
	}
}

void LazyNet::NetBackward()
{
	//softmax_loss
	vector<Flow> softmax_loss_input_;
	softmax_loss_input_.push_back(softmax_data_);
	softmax_loss_input_.push_back(label_data_);
	softmax_loss_.BackwardNet(softmax_loss_data_, softmax_loss_input_);

	//softmax
	softmax_.BackwardNet(softmax_data_, fc2_data_);

	//fc2
	fc2_.BackwardNet(fc2_data_, relu4_data_);

	//fc1
	relu4_.BackwardNet(relu4_data_, fc1_data_);
	fc1_.BackwardNet(fc1_data_, relu3_data_);

	//conv3
	relu3_.BackwardNet(relu3_data_, conv3_data_);
	conv3_.BackwardNet(conv3_data_, pool2_data_);

	//conv2
	pool2_.BackwardNet(pool2_data_, relu2_data_);
	relu2_.BackwardNet(relu2_data_, conv2_data_);
	conv2_.BackwardNet(conv2_data_, pool1_data_);

	//conv1
	pool1_.BackwardNet(pool1_data_, relu1_data_);
	relu1_.BackwardNet(relu1_data_, conv1_data_);
	conv1_.BackwardNet(conv1_data_, input_data_);
}

void LazyNet::NetUpdate(int iter_count)
{
	//w := w - a * (df/dw)
	//df/dw = diff
	//参考https://blog.csdn.net/samylee/article/details/86648087

	//weight_decay权重惩罚项:为了避免出现overfitting，权重过大会导致系统过拟合
	//momentum为了加速收敛，如果上一次的h与这一次的负梯度方向是相同的，那这次下降的幅度就会加大
	//参考https://blog.csdn.net/u012938704/article/details/52739612/
	
	//diff = weight_decay*w + diff
	UpdateRegularize();

	//h = lr*diff + momentum*h
	//diff <- h
	UpdateValue(iter_count);

	//w = w - diff
	UpdateWeightBias();
}

int LazyNet::TrainNet()
{
	//输出信息
	cout << "=========Train network start========" << endl;

	//检查数据是否均衡
	if (data0_train_path.size() != data1_train_path.size())
		return -1;

	int iters_per_epoch = data0_train_path.size() / data_batch_;
	int epoch_count = max_iter_ / iters_per_epoch;

	int iter_count = 0;
	for (int epoch = 0; epoch < epoch_count; epoch++)
	{
		//训练
		int img_train_count = 0;
		for (int iter = 0; iter < iters_per_epoch; iter++)
		{
			//载入数据
			LoadDataLabel(img_train_count, "train");

			//前向传递
			NetForward(iter_count, "train");

			//反向传递
			NetBackward();

			//更新权重及偏执
			NetUpdate(iter_count);

			img_train_count += data_batch_;
			iter_count++;

			//测试
			if (iter_count % test_iter_ == 0)
			{
				int img_test_count = 0;
				//载入数据
				LoadDataLabel(img_test_count, "test");
				//前向传递
				NetForward(img_test_count, "test");
			}
		}
	}

	//输出信息
	cout << "=========Train network endl========" << endl;
}
