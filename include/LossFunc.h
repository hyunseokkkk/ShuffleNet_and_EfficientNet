//
// Created by loveh on 25. 1. 12.
//

#ifndef ALEXNET_LOSSFUNC_H
#define ALEXNET_LOSSFUNC_H


#include "Tensor.h"
#include <cmath>

using namespace std;

class LossFunc {
public:
    virtual double computeLoss(const Tensor& predictions, const Tensor& targets) const = 0;
    virtual Tensor computeGradient(const Tensor& predictions, const Tensor& targets) const = 0;
    virtual ~LossFunc() = default;
};

class CrossEntropyLoss : public LossFunc {
public:
    double computeLoss(const Tensor& predictions, const Tensor& targets) const override;
    Tensor computeGradient(const Tensor& predictions, const Tensor& targets) const override;
};

class MSELoss : public LossFunc {
public:
    double computeLoss(const Tensor& predictions, const Tensor& targets) const override;
    Tensor computeGradient(const Tensor& predictions, const Tensor& targets) const override;
};

#endif //ALEXNET_LOSSFUNC_H
