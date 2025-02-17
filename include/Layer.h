#ifndef ALEXNET_LAYER_H
#define ALEXNET_LAYER_H

#include "Tensor.h"
#include <string>
#include <cmath>

using namespace std;

class Layer {

protected:
    Tensor weights;
    Tensor biases;

public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradOutput) = 0;
    virtual void updateWeights(double learningRate) = 0;
    virtual void zeroGradients() = 0;
    virtual ~Layer() = default;
};

class Flatten : public Layer {
private:
    vector<int> inputShape; // To store the original input shape

public:
    // Forward pass
    Tensor forward(const Tensor& input) override {
        inputShape = input.getShape(); // Store original shape, 4차원

        // Flatten shape
        int batchSize = inputShape[0];
        int flattenedSize = input.getTotalSize() / batchSize;

        Tensor output(batchSize, flattenedSize);

        // 인덱스 변환 없이 직접 접근 가능하도록 처리
        const auto& inputData = input.getData();
        for (int i = 0; i < output.getTotalSize(); ++i) {
            output.set({i / flattenedSize, i % flattenedSize}, inputData[i]);
        }

        return output;
    }


    // Backward passd
    Tensor backward(const Tensor& gradOutput) override {
        // Reshape gradient back to the original input shape
        Tensor gradInput(inputShape);
        for (int i = 0; i < gradOutput.getTotalSize(); ++i) {
            gradInput.set(i, gradOutput.get(i));
        }
        return gradInput;
    }

    // Placeholder for optimizer update (not needed for Flatten)
    void updateWeights(double lr) override {}

    // Zero gradients (not needed for Flatten)
    void zeroGradients() override {}
};
class Relu : public Layer {
private:
    Tensor inputCache; // Forward 시 입력을 저장

public:
    Tensor forward(const Tensor& input) override {
        inputCache = input; // Backward를 위해 입력 저장
        Tensor output(input.getShape()); // 입력과 같은 shape의 Tensor 생성

        vector<int> shape = input.getShape();
        for (int i = 0; i < input.getTotalSize(); ++i) {
            output.set(i, max(0.0, input.get(i)));
        }

        return output;
    }


    Tensor backward(const Tensor& gradOutput) override {
        Tensor gradInput(inputCache.getShape());
        for (int i = 0; i < inputCache.getTotalSize(); ++i) {
            gradInput.set(i, inputCache.get(i) > 0 ? gradOutput.get(i) : 0.0);
        }
        return gradInput;
    }

    void updateWeights(double learningRate) override {}
    void zeroGradients() override {}
};

class Sigmoid : public Layer {
private:
    Tensor inputCache; // Forward 시 입력을 저장

public:
    Tensor forward(const Tensor& input) override {
        inputCache = input; // Backward를 위해 입력 저장
        Tensor output(input.getShape());
        for (int i = 0; i < input.getTotalSize(); ++i) {
            double val = input.get(i);
            output.set(i, 1.0 / (1.0 + exp(-val)));
        }
        return output;
    }

    Tensor backward(const Tensor& gradOutput) override {
        Tensor gradInput(inputCache.getShape());
        for (int i = 0; i < inputCache.getTotalSize(); ++i) {
            double sigmoidVal = 1.0 / (1.0 + exp(-inputCache.get({i})));
            gradInput.set(i, gradOutput.get(i) * sigmoidVal * (1.0 - sigmoidVal));
        }
        return gradInput;
    }

    void updateWeights(double learningRate) override {}
    void zeroGradients() override {}
};

#endif // ALEXNET_LAYER_H
