#include "lazy_net.h"

int main()
{
	//data information
	string data_path = "data/";
	int data_batch = 20;
	int data_channel = 3;
	int data_size = 28;

	//hyperparameter
	int max_iter = 1000;
	int disp_iter = 10;
	float weight_decay = 0.0005;
	float base_lr = 0.01;
	float momentum = 0.9;
	string lr_policy = "inv";
	float power = 0.75;
	float gamma = 0.0001;
	int test_iter = 50;

	LazyNet lazy_net(data_path, data_batch, data_channel, 
		data_size, max_iter, disp_iter,
		weight_decay, base_lr, momentum,
		lr_policy, power, gamma, test_iter);

	lazy_net.TrainNet();

	return 0;
}