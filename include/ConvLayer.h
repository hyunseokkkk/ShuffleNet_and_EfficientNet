//
// Created by loveh on 25. 1. 9.
//

#ifndef ALEXNET_CONVLAYER_H
#define ALEXNET_CONVLAYER_H


#include "Layer.h"
#include "Tensor.h"
#include <vector>

using namespace std;

class ConvLayer : public Layer{
    int kernelSize, stride, padding;
    int inputChannels;
    int outputChannels;
    Tensor weights;
    Tensor biases;
    Tensor gradWeights;
    Tensor gradBiases;

    Tensor inputCache;

public:
    ConvLayer(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int padding = 0);
    Tensor forward(const Tensor& input)override;
    Tensor backward(const Tensor& gradOutput)override;

    void updateWeights(double learningRate)override;

    void zeroGradients() override;
};

#endif //ALEXNET_CONVLAYER_H
