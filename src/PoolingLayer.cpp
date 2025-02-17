#include "PoolingLayer.h"
#include "Optimizer.h"
#include <stdexcept>
#include <limits>

MaxPoolingLayer::MaxPoolingLayer(int poolSize, int stride)
        : poolSize(poolSize), stride(stride) {}

Tensor MaxPoolingLayer::forward(const Tensor& input) {
    inputCache = input;

    const auto& inputShape = input.getShape(); // [Batch, Channels, Height, Width]
    int batchSize = inputShape[0];
    int channels = inputShape[1];
    int inputHeight = inputShape[2];
    int inputWidth = inputShape[3];

    int outputHeight = (inputHeight - poolSize) / stride + 1;
    int outputWidth = (inputWidth - poolSize) / stride + 1;

    Tensor output(batchSize, channels, outputHeight, outputWidth);
    maxIndices = Tensor(batchSize, channels, outputHeight, outputWidth, 2); // 최대값의 위치 저장

    for (int b = 0; b < batchSize; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < outputHeight; ++oh) {
                for (int ow = 0; ow < outputWidth; ++ow) {
                    double maxVal = -std::numeric_limits<double>::infinity();
                    int maxH = -1, maxW = -1;

                    for (int ph = 0; ph < poolSize; ++ph) {
                        for (int pw = 0; pw < poolSize; ++pw) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            if (ih >= 0 && iw >= 0 && ih < inputHeight && iw < inputWidth) {
                                double val = input.get({b, c, ih, iw});
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxH = ih;
                                    maxW = iw;
                                }
                            }
                        }
                    }
                    output.set({b, c, oh, ow}, maxVal);
                    maxIndices.set({b, c, oh, ow, 0}, maxH);
                    maxIndices.set({b, c, oh, ow, 1}, maxW);
                }
            }
        }
    }
    return output;
}

Tensor MaxPoolingLayer::backward(const Tensor& gradOutput) {
    const auto& inputShape = inputCache.getShape();
    Tensor gradInput(inputShape);

    const auto& outputShape = gradOutput.getShape();
    int batchSize = outputShape[0];
    int channels = outputShape[1];
    int outputHeight = outputShape[2];
    int outputWidth = outputShape[3];

    for (int b = 0; b < batchSize; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < outputHeight; ++oh) {
                for (int ow = 0; ow < outputWidth; ++ow) {
                    int maxH = static_cast<int>(maxIndices.get({b, c, oh, ow, 0}));
                    int maxW = static_cast<int>(maxIndices.get({b, c, oh, ow, 1}));
                    gradInput.set({b, c, maxH, maxW}, gradOutput.get({b, c, oh, ow}));
                }
            }
        }
    }

    return gradInput;
}
