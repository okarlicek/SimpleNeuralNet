#ifndef DENSE_H_
#define DENSE_H_

#include <Eigen/Dense>
#include <string>
#include "BaseLayer.h"


namespace SimpleNeuralNet {
namespace Layers {

	class Dense : public SimpleNeuralNet::Layer {
	public:
		Dense(const size_t inputSize, const size_t outputSize);
		Dense(Eigen::MatrixXf& weights, Eigen::MatrixXf& bias);
		~Dense() = default;

		void forwardPass(Eigen::MatrixXf& input, Eigen::MatrixXf& output, bool training) override;
		void backwardPass(Eigen::MatrixXf& din, Eigen::MatrixXf& dout, float lr) override;
		void description() const override;
		Eigen::MatrixXf getWeights() { return weights_; };
		Eigen::MatrixXf getBias() { return bias_; };
		void initWeightsAndBias();
		void setWeightsAndBias(Eigen::MatrixXf& weights, Eigen::MatrixXf& bias);
		const size_t getInputSize() { return inputSize_; };
		const size_t getOutputSize() { return outputSize_; };
	private:
		const size_t inputSize_;
		const size_t outputSize_;
		Eigen::MatrixXf weights_;
		Eigen::MatrixXf bias_;
		Eigen::MatrixXf forwardInput_;
};


} // namespace Layers
} // namespace SimpleNeuralNet


#endif // !DENSE_H_