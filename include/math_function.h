#pragma once

#include "string.h"
#include <math.h>
#include "flow.h"

extern "C" {
#include <cblas.h>
}

void OpenblasGemm(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const float alpha, const float* A, const float* B, const float beta,
	float* C);

void OpenblasGemv(const CBLAS_TRANSPOSE TransA, const int M,
	const int N, const float alpha, const float* A, const float* x,
	const float beta, float* y);

void InitWeightBias(Flow &weight_, Flow &bias_);

//Y[i] = X[i]
void CopyData(const int N, const float* X, float* Y);

//y[i] = exp(a[i])
void ExpData(const int n, const float* a, float* y);

//y[i] = a[i]\b[i]
void DivData(const int n, const float* a, const float* b, float* y);

//X = alpha*X
void ScalData(const int N, const float alpha, float *X);

//Y=alpha*X+Y 
void AxpyData(const int N, const float alpha, const float* X, float* Y);

//Y= alpha*X+beta*Y
void AxpbyData(const int N, const float alpha, const float* X, const float beta, float* Y);