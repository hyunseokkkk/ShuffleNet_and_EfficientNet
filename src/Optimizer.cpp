#include "Optimizer.h"

Optimizer::Optimizer(shared_ptr<Model> model, double lr)
        : model(move(model)), lr(lr) {}

void Optimizer::zero_grad() {
    model->zeroGradients();
}

void Optimizer::step() {
    model->updateWeights(lr);
}

SGD::SGD(shared_ptr<Model> model, double lr)
        : Optimizer(move(model), lr) {}

void SGD::zero_grad() {
    model->zeroGradients();
}

void SGD::step() {
    model->updateWeights(lr);
}