//
// Created by loveh on 25. 1. 11.
//

#ifndef ALEXNET_POOLINGLAYER_H
#define ALEXNET_POOLINGLAYER_H

#include "Layer.h"
#include "Tensor.h"

class MaxPoolingLayer : public Layer {
private:
    int poolSize;
    int stride;
    Tensor inputCache;
    Tensor maxIndices;
public:
    MaxPoolingLayer(int poolSize, int stride);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    void zeroGradients() override {}
    void updateWeights(double learningRate) override {}
};

class GlobalAveragePoolingLayer : public Layer {
public:
    // forward/backward 시 입력 텐서의 shape를 저장
    std::vector<int> inputShape;
    Tensor inputCache;

    GlobalAveragePoolingLayer() {}

    // Forward: 각 채널의 평균을 구해 출력 텐서를 생성합니다.
    virtual Tensor forward(const Tensor& input) override {
        inputCache = input;
        inputShape = input.getShape(); // [N, C, H, W]
        int N = inputShape[0];
        int C = inputShape[1];
        int H = inputShape[2];
        int W = inputShape[3];

        // 출력 shape: [N, C, 1, 1]
        Tensor output(N, C, 1, 1);
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                double sum = 0.0;
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        sum += input.get({n, c, h, w});
                    }
                }
                double avg = sum / (H * W);
                output.set({n, c, 0, 0}, avg);
            }
        }
        return output;
    }

    // Backward: 출력 텐서의 gradient를 입력 텐서의 각 위치에 동일하게 분배합니다.
    virtual Tensor backward(const Tensor& gradOutput) override {
        int N = inputShape[0];
        int C = inputShape[1];
        int H = inputShape[2];
        int W = inputShape[3];
        Tensor gradInput(inputShape); // [N, C, H, W]

        // gradOutput의 shape는 [N, C, 1, 1]
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                // gradOutput에 있는 값을 H*W로 나눠 각 위치에 분배합니다.
                double gradVal = gradOutput.get({n, c, 0, 0}) / (H * W);
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        gradInput.set({n, c, h, w}, gradVal);
                    }
                }
            }
        }
        return gradInput;
    }

    virtual void updateWeights(double learningRate) override { }
    virtual void zeroGradients() override { }
};

#endif //ALEXNET_POOLINGLAYER_H
