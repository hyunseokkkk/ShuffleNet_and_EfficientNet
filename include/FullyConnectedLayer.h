//
// Created by loveh on 25. 1. 9.
//

#ifndef ALEXNET_FULLYCONNECTEDLAYER_H
#define ALEXNET_FULLYCONNECTEDLAYER_H


#include "Layer.h"
#include "Tensor.h"

class FullyConnectedLayer : public Layer {
private:
    Tensor weights;
    Tensor biases;
    Tensor gradWeights;
    Tensor gradBiases;
    Tensor inputCache;

public:
    FullyConnectedLayer(int inputSize, int outputSize);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
    void updateWeights(double learningRate) override;
    void zeroGradients() override;
};



#endif //ALEXNET_FULLYCONNECTEDLAYER_H
