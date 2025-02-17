#include "ConvLayer.h"
#include <stdexcept>
#include <cmath>

using namespace std;

ConvLayer::ConvLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding)
        : inputChannels(inputChannels), outputChannels(outputChannels),
          kernelSize(kernelSize), stride(stride), padding(padding) {
    weights = Tensor(outputChannels, inputChannels, kernelSize, kernelSize, true);
    biases = Tensor(outputChannels, true);
    gradWeights = Tensor(weights.getShape());
    gradBiases = Tensor(biases.getShape());
}

Tensor ConvLayer::forward(const Tensor& input) {
    inputCache = input;

    const vector<int>& inputShape = input.getShape(); // [Batch, Channels, Height, Width]
    int batchSize = inputShape[0];
    int inputHeight = inputShape[2];
    int inputWidth = inputShape[3];

    int outputHeight = (inputHeight - kernelSize + 2 * padding) / stride + 1;
    int outputWidth = (inputWidth - kernelSize + 2 * padding) / stride + 1;

    Tensor output(batchSize, outputChannels, outputHeight, outputWidth);

    for (int b = 0; b < batchSize; ++b) {
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int oh = 0; oh < outputHeight; ++oh) {
                for (int ow = 0; ow < outputWidth; ++ow) {
                    double value = biases.get(oc);
                    for (int ic = 0; ic < inputChannels; ++ic) {
                        for (int kh = 0; kh < kernelSize; ++kh) {
                            for (int kw = 0; kw < kernelSize; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && iw >= 0 && ih < inputHeight && iw < inputWidth) {
                                    value += input.get({b, ic, ih, iw}) * weights.get({oc, ic, kh, kw});
                                }
                            }
                        }
                    }
                    output.set({b, oc, oh, ow}, value);
                }
            }
        }
    }
    return output;
}

Tensor ConvLayer::backward(const Tensor& gradOutput) {
    const vector<int>& inputShape = inputCache.getShape();
    const vector<int>& outputShape = gradOutput.getShape();

    Tensor gradInput(inputShape);
    gradWeights.init({outputChannels, inputChannels, kernelSize, kernelSize}, false);
    gradBiases.init({outputChannels}, false);

    int batchSize = inputShape[0];
    int inputHeight = inputShape[2];
    int inputWidth = inputShape[3];
    int outputHeight = outputShape[2];
    int outputWidth = outputShape[3];

    // Compute gradients
    for (int b = 0; b < batchSize; ++b) {
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int oh = 0; oh < outputHeight; ++oh) {
                for (int ow = 0; ow < outputWidth; ++ow) {
                    double grad = gradOutput.get({b, oc, oh, ow});
                    gradBiases.set(oc, gradBiases.get(oc) + grad);
                    for (int ic = 0; ic < inputChannels; ++ic) {
                        for (int kh = 0; kh < kernelSize; ++kh) {
                            for (int kw = 0; kw < kernelSize; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && iw >= 0 && ih < inputHeight && iw < inputWidth) {
                                    gradWeights.set({oc, ic, kh, kw},gradWeights.get({oc, ic, kh, kw}) + grad * inputCache.get({b, ic, ih, iw}));
                                    gradInput.set({b, ic, ih, iw},gradInput.get({b, ic, ih, iw}) + grad * weights.get({oc, ic, kh, kw}));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return gradInput;
}

void ConvLayer::updateWeights(double learningRate) {
    for (int i = 0; i < weights.getSize(); ++i) {
        weights.set(i, weights.get(i) - learningRate * gradWeights.get(i));
    }
    for (int i = 0; i < biases.getSize(); ++i) {
        biases.set(i, biases.get(i) - learningRate * gradBiases.get(i));
    }
}

void ConvLayer::zeroGradients() {
    gradWeights.init(gradWeights.getShape(), false);
    gradBiases.init(gradBiases.getShape(), false);
}
