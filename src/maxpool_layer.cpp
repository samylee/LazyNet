#include "maxpool_layer.h"

MaxPoolLayer::MaxPoolLayer()
{
}

MaxPoolLayer::~MaxPoolLayer()
{
	max_idx_.Release();
}

void MaxPoolLayer::LayerSetUp(Flow &bottom, Flow &top, vector<int> &pool_shape)
{
	kernel_h_ = pool_shape[2];
	kernel_w_ = pool_shape[3];

	int output_height = ceil((bottom.WhichDimensionShape(2) + 2 * 0 - pool_shape[2]) / 2.0) + 1;
	int output_width = ceil((bottom.WhichDimensionShape(3) + 2 * 0 - pool_shape[3]) / 2.0) + 1;

	vector<int> top_shape;
	top_shape.push_back(bottom.WhichDimensionShape(0));
	top_shape.push_back(pool_shape[0]);
	top_shape.push_back(output_height);
	top_shape.push_back(output_width);

	//top_data申请内存
	top.SetShapeData(top_shape);
	max_idx_.SetShapeData(top_shape);

	//top_diff_init
	top.InitDiff(0.0);

	//输出信息
	static int pool_count = 1;
	cout << "pool" << pool_count << ": \t" << bottom.WhichDimensionShape(2) << "x" << bottom.WhichDimensionShape(3);
	cout << "\t --> \t" << top.WhichDimensionShape(2) << "x" << top.WhichDimensionShape(3) << endl;
	pool_count++;
}

void MaxPoolLayer::ForwardNet(Flow &bottom, Flow &top)
{
	float *bottom_data = bottom.GetData();
	float *top_data = top.GetData();
	int top_count = top.GetCounts();
	float* mask = NULL;

	mask = max_idx_.GetData();
	for (int i = 0; i < top_count; ++i) {
		mask[i] = -1;
	}

	top.InitData(float(-FLT_MAX));

	for (int n = 0; n < top.WhichDimensionShape(0); ++n) {
		for (int c = 0; c < top.WhichDimensionShape(1); ++c) {
			for (int ph = 0; ph < top.WhichDimensionShape(2); ++ph) {
				for (int pw = 0; pw < top.WhichDimensionShape(3); ++pw) {
					int hstart = ph * 2;//stride 2
					int wstart = pw * 2;//stride 2
					int hend = min(hstart + kernel_h_, bottom.WhichDimensionShape(2));
					int wend = min(wstart + kernel_w_, bottom.WhichDimensionShape(3));
					hstart = max(hstart, 0);
					wstart = max(wstart, 0);
					const int pool_index = ph * top.WhichDimensionShape(3) + pw;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							const int index = h * bottom.WhichDimensionShape(3) + w;
							if (bottom_data[index] > top_data[pool_index]) {
								top_data[pool_index] = bottom_data[index];
								mask[pool_index] = index;
							}
						}
					}
				}
			}
			// compute offset
			bottom_data += bottom.offset(0, 1);
			top_data += top.offset(0, 1);
			mask += top.offset(0, 1);
		}
	}
}

void MaxPoolLayer::BackwardNet(Flow &top, Flow &bottom)
{
	float* top_diff = top.GetDiff();
	float* bottom_diff = bottom.GetDiff();
	// Different pooling methods. We explicitly do the switch outside the for
	// loop to save time, although this results in more codes.
	for (int i = 0; i < bottom.GetCounts(); ++i) {
		bottom_diff[i] = float(0);
	}
	// We'll output the mask to top[1] if it's of size >1.
	float* mask = NULL;  // suppress warnings about uninitialized variables
	mask = max_idx_.GetData();

	//bottom.GetDiff()j = dloss / dx = (dloss / dzj) * (dzj / dx) = (dloss / dz) * (dxi / dx) = (dloss / dz) * 1i
	//dloss / dz = top_diffi
	//i和j的映射关系
	for (int n = 0; n < top.WhichDimensionShape(0); ++n) {
		for (int c = 0; c < top.WhichDimensionShape(1); ++c) {
			for (int ph = 0; ph < top.WhichDimensionShape(2); ++ph) {
				for (int pw = 0; pw < top.WhichDimensionShape(3); ++pw) {
					const int index = ph * top.WhichDimensionShape(3) + pw;
					const int bottom_index = mask[index];
					bottom_diff[bottom_index] += top_diff[index];
				}
			}
			bottom_diff += bottom.offset(0, 1);
			top_diff += top.offset(0, 1);

			mask += top.offset(0, 1);
		}
	}
}