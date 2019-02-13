#include "math_function.h"

void OpenblasGemm(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const float alpha, const float* A, const float* B, const float beta,
	float* C) {
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
		ldb, beta, C, N);
}

void OpenblasGemv(const CBLAS_TRANSPOSE TransA, const int M,
	const int N, const float alpha, const float* A, const float* x,
	const float beta, float* y) {
	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

void InitWeightBias(Flow &weight_, Flow &bias_)
{
	//xavier方式初始化权重
	int fan_in = weight_.GetCounts() / weight_.WhichDimensionShape(0);
	int fan_out = weight_.GetCounts() / weight_.WhichDimensionShape(1);
	float n = (fan_in + fan_out) / 2.0;
	float scale = sqrt(3.0 / n);

	default_random_engine random(time(NULL));
	uniform_real_distribution<float> rng_data(-scale, scale);

	//这里不能固定rng_data(random)
	float *weight_data_tmp = (float *)malloc(sizeof(float)*weight_.GetCounts());
	for (int i = 0; i < weight_.GetCounts(); i++)
	{
		weight_data_tmp[i] = rng_data(random);
	}
	weight_.SetData(weight_data_tmp);
	free(weight_data_tmp);

	weight_.InitDiff(0.0);

	//constant方式初始化bias
	bias_.InitData(0.0);
	bias_.InitDiff(0.0);
}

void CopyData(const int N, const float* X, float* Y) {
	if (X != Y)
	{
		memcpy(Y, X, sizeof(float) * N);  // NOLINT(caffe/alt_fn)
	}
}

void ExpData(const int n, const float* a, float* y) {
	for (int i = 0; i < n; ++i) {
		y[i] = exp(a[i]);
	}
}

void DivData(const int n, const float* a, const float* b, float* y) {
	for (int i = 0; i < n; ++i) {
		y[i] = a[i] / b[i];
	}
}

void ScalData(const int N, const float alpha, float *X)
{
	cblas_sscal(N, alpha, X, 1);
}

void AxpyData(const int N, const float alpha, const float* X, float* Y) 
{
	cblas_saxpy(N, alpha, X, 1, Y, 1);
}

void AxpbyData(const int N, const float alpha, const float* X, const float beta, float* Y) 
{
	cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}