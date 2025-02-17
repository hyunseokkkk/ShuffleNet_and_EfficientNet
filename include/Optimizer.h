#ifndef ALEXNET_OPTIMIZER_H
#define ALEXNET_OPTIMIZER_H

#include <memory>
#include "Model.h"

using namespace std;

class Optimizer {
protected:
    shared_ptr<Model> model;
    double lr;

public:
    explicit Optimizer(shared_ptr<Model> model, double lr);

    virtual void zero_grad(); // Clear gradients
    virtual void step();      // Apply gradients to update weights
    virtual ~Optimizer() = default;
};

class SGD : public Optimizer {
public:
    explicit SGD(shared_ptr<Model> model, double lr);

    void zero_grad() override;
    void step() override;
};

#endif // ALEXNET_OPTIMIZER_H
