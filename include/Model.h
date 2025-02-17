#ifndef ALEXNET_MODEL_H
#define ALEXNET_MODEL_H

#include <vector>
#include <memory>
#include "Layer.h"
#include "Tensor.h"

using namespace std;

class Model {
private:
    vector<shared_ptr<Layer>> layers;

public:
    void addLayer(shared_ptr<Layer> layer);

    Tensor forward(const Tensor& input);

    void backward(const Tensor& gradients);

    void updateWeights(double learningRate);

    void zeroGradients();
};

#endif // ALEXNET_MODEL_H
