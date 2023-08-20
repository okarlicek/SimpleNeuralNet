
#include "Layers/Dense.h"

#include <iostream>

using namespace SimpleNeuralNet::Layers;

Dense::Dense(const size_t inputSize, const size_t outputSize) : inputSize_(inputSize), outputSize_(outputSize) {
	initWeightsAndBias();
};

Dense::Dense(Eigen::MatrixXf& weights, Eigen::MatrixXf& bias) : weights_(weights), bias_(bias), inputSize_(weights.rows()), outputSize_(weights.cols())  {}


void Dense::forwardPass(Eigen::MatrixXf& input, Eigen::MatrixXf& output, bool training = false) {
	output = input * weights_;
	output = output + bias_.replicate(output.rows(), 1);

	if (training) {
		forwardInput_ = input;
	}
};

void Dense::backwardPass(Eigen::MatrixXf& din, Eigen::MatrixXf& dout, float lr) {
	// update of weights and bias
	weights_ = weights_.array() - lr * (forwardInput_.transpose() * din).array();
	bias_ = bias_.array() - lr * din.colwise().mean().array();

	// output derivative
	dout = din * weights_.transpose();
};


void Dense::description() const {
	std::cout << "Dense Layer [" << inputSize_ << ", " << outputSize_ << "]";
};


void Dense::initWeightsAndBias() {
	weights_ = Eigen::MatrixXf::Random(inputSize_, outputSize_);
	bias_ = Eigen::MatrixXf::Random(1, outputSize_);
}


void Dense::setWeightsAndBias(Eigen::MatrixXf& weights, Eigen::MatrixXf& bias) {
	weights_ = weights;
	bias_ = bias;
}