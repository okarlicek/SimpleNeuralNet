#ifndef BASE_LAYER_H_
#define BASE_LAYER_H_

#include <Eigen/Dense>


namespace SimpleNeuralNet {

	class Layer {
	public:
		virtual void forwardPass(Eigen::MatrixXf& input, Eigen::MatrixXf& output, bool training = false)  = 0;
		virtual void backwardPass(Eigen::MatrixXf& din, Eigen::MatrixXf& dout, float lr) = 0;
		virtual void description() const = 0;
		virtual ~Layer() {};
	};

}; // namespace SimpleNeuralNet


#endif // !BASE_LAYER_H_