#include "Model.h"

void Model::addLayer(shared_ptr<Layer> layer) {
    layers.push_back(move(layer));
}

Tensor Model::forward(const Tensor& input) {
    Tensor output = input;
    for (const auto& layer : layers) {
        output = layer->forward(output);
/*
        cout << "Layer Output Shape: ";
        output.print();
*/

    }
    return output;
}

void Model::backward(const Tensor& gradients) {
    Tensor grad = gradients;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
}

void Model::updateWeights(double learningRate) {
    for (const auto& layer : layers) {
        layer->updateWeights(learningRate);
    }
}

void Model::zeroGradients() {
    for (const auto& layer : layers) {
        layer->zeroGradients();
    }
}
