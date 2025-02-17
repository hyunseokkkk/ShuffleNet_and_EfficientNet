
#ifndef SHUFFLENET_SHUFFLENETUNITLAYER_H
#define SHUFFLENET_SHUFFLENETUNITLAYER_H

#include "Tensor.h"
#include "Layer.h"
#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>

inline Tensor addTensors(const Tensor &a, const Tensor &b) {
    if (a.getShape() != b.getShape()) {
        throw invalid_argument("두 텐서의 shape가 같아야 덧셈을 수행할 수 있습니다.");
    }
    Tensor result(a.getShape(), false);
    vector<double>& rData = result.getData();
    const vector<double>& aData = a.getData();
    const vector<double>& bData = b.getData();
    for (size_t i = 0; i < aData.size(); ++i) {
        rData[i] = aData[i] + bData[i];
    }
    return result;
}


class GroupConvLayer {
public:
    int inputChannels;
    int outputChannels;
    int kernelSize;
    int stride;
    int padding;
    int groups;

    Tensor weights;
    Tensor biases;
    Tensor gradWeights;
    Tensor gradBiases;

    Tensor inputCache;

    GroupConvLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding, int groups)
            : inputChannels(inputChannels), outputChannels(outputChannels),
              kernelSize(kernelSize), stride(stride), padding(padding), groups(groups),
              weights(outputChannels, inputChannels / groups, kernelSize, kernelSize, true),
              biases(outputChannels, true),
              gradWeights(weights.getShape()),
              gradBiases(biases.getShape())
    {
        if (inputChannels % groups != 0 || outputChannels % groups != 0)
            throw  invalid_argument("inputChannels와 outputChannels는 groups로 나누어 떨어져야 합니다.");
    }

    Tensor forward(const Tensor &input) {
        inputCache = input;
        const vector<int>& inputShape = input.getShape(); // [N, C, H, W]
        int batchSize   = inputShape[0];
        int inputHeight = inputShape[2];
        int inputWidth  = inputShape[3];
        int outputHeight = (inputHeight - kernelSize + 2 * padding) / stride + 1;
        int outputWidth  = (inputWidth  - kernelSize + 2 * padding) / stride + 1;
        Tensor output(batchSize, outputChannels, outputHeight, outputWidth);

        int groupInputChannels  = inputChannels / groups;
        int groupOutputChannels = outputChannels / groups;

        for (int b = 0; b < batchSize; ++b) {
            for (int g = 0; g < groups; ++g) {
                for (int oc = 0; oc < groupOutputChannels; ++oc) {
                    int actualOC = g * groupOutputChannels + oc;
                    for (int oh = 0; oh < outputHeight; ++oh) {
                        for (int ow = 0; ow < outputWidth; ++ow) {
                            double value = biases.get({actualOC});
                            for (int ic = 0; ic < groupInputChannels; ++ic) {
                                int actualIC = g * groupInputChannels + ic;
                                for (int kh = 0; kh < kernelSize; ++kh) {
                                    for (int kw = 0; kw < kernelSize; ++kw) {
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;
                                        if (ih >= 0 && iw >= 0 && ih < inputHeight && iw < inputWidth) {
                                            value += input.get({b, actualIC, ih, iw}) *
                                                     weights.get({actualOC, ic, kh, kw});
                                        }
                                    }
                                }
                            }
                            output.set({b, actualOC, oh, ow}, value);
                        }
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor &gradOutput) {
        const  vector<int>& inputShape  = inputCache.getShape();
        const  vector<int>& outputShape = gradOutput.getShape();
        Tensor gradInput(inputShape);
        gradWeights.init({outputChannels, inputChannels / groups, kernelSize, kernelSize}, false);
        gradBiases.init({outputChannels}, false);

        int batchSize   = inputShape[0];
        int inputHeight = inputShape[2];
        int inputWidth  = inputShape[3];
        int outputHeight = outputShape[2];
        int outputWidth  = outputShape[3];
        int groupInputChannels  = inputChannels / groups;
        int groupOutputChannels = outputChannels / groups;

        for (int b = 0; b < batchSize; ++b) {
            for (int g = 0; g < groups; ++g) {
                for (int oc = 0; oc < groupOutputChannels; ++oc) {
                    int actualOC = g * groupOutputChannels + oc;
                    for (int oh = 0; oh < outputHeight; ++oh) {
                        for (int ow = 0; ow < outputWidth; ++ow) {
                            double grad = gradOutput.get({b, actualOC, oh, ow});
                            gradBiases.set({actualOC}, gradBiases.get({actualOC}) + grad);
                            for (int ic = 0; ic < groupInputChannels; ++ic) {
                                int actualIC = g * groupInputChannels + ic;
                                for (int kh = 0; kh < kernelSize; ++kh) {
                                    for (int kw = 0; kw < kernelSize; ++kw) {
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;
                                        if (ih >= 0 && iw >= 0 && ih < inputHeight && iw < inputWidth) {
                                            gradWeights.set({actualOC, ic, kh, kw},
                                                            gradWeights.get({actualOC, ic, kh, kw}) +
                                                            grad * inputCache.get({b, actualIC, ih, iw}));
                                            gradInput.set({b, actualIC, ih, iw},
                                                          gradInput.get({b, actualIC, ih, iw}) +
                                                          grad * weights.get({actualOC, ic, kh, kw}));
                                        }
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

    void updateWeights(double learningRate) {
        for (int i = 0; i < weights.getSize(); ++i)
            weights.set(i, weights.get(i) - learningRate * gradWeights.get(i));
        for (int i = 0; i < biases.getSize(); ++i)
            biases.set(i, biases.get(i) - learningRate * gradBiases.get(i));
    }

    void zeroGradients() {
        gradWeights.init(gradWeights.getShape(), false);
        gradBiases.init(gradBiases.getShape(), false);
    }
};

class DepthwiseConvLayer {
public:
    int channels;
    int kernelSize;
    int stride;
    int padding;

    Tensor weights;
    Tensor biases;
    Tensor gradWeights;
    Tensor gradBiases;
    Tensor inputCache;

    DepthwiseConvLayer(int channels, int kernelSize, int stride, int padding)
            : channels(channels), kernelSize(kernelSize), stride(stride), padding(padding),
              weights(channels, 1, kernelSize, kernelSize, true),
              biases(channels, true),
              gradWeights(weights.getShape()),
              gradBiases(biases.getShape())
    {}

    Tensor forward(const Tensor &input) {
        inputCache = input;
        const  vector<int>& inputShape = input.getShape();
        int batchSize   = inputShape[0];
        int inputHeight = inputShape[2];
        int inputWidth  = inputShape[3];
        int outputHeight = (inputHeight - kernelSize + 2 * padding) / stride + 1;
        int outputWidth  = (inputWidth  - kernelSize + 2 * padding) / stride + 1;
        Tensor output(batchSize, channels, outputHeight, outputWidth);

        for (int b = 0; b < batchSize; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        double value = biases.get({c});
                        for (int kh = 0; kh < kernelSize; ++kh) {
                            for (int kw = 0; kw < kernelSize; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && iw >= 0 && ih < inputHeight && iw < inputWidth) {
                                    value += input.get({b, c, ih, iw}) *
                                             weights.get({c, 0, kh, kw});
                                }
                            }
                        }
                        output.set({b, c, oh, ow}, value);
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor &gradOutput) {
        const  vector<int>& inputShape  = inputCache.getShape();
        const  vector<int>& outputShape = gradOutput.getShape();
        Tensor gradInput(inputShape);
        gradWeights.init({channels, 1, kernelSize, kernelSize}, false);
        gradBiases.init({channels}, false);

        int batchSize   = inputShape[0];
        int inputHeight = inputShape[2];
        int inputWidth  = inputShape[3];
        int outputHeight = outputShape[2];
        int outputWidth  = outputShape[3];

        for (int b = 0; b < batchSize; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < outputHeight; ++oh) {
                    for (int ow = 0; ow < outputWidth; ++ow) {
                        double grad = gradOutput.get({b, c, oh, ow});
                        gradBiases.set({c}, gradBiases.get({c}) + grad);
                        for (int kh = 0; kh < kernelSize; ++kh) {
                            for (int kw = 0; kw < kernelSize; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && iw >= 0 && ih < inputHeight && iw < inputWidth) {
                                    gradWeights.set({c, 0, kh, kw},
                                                    gradWeights.get({c, 0, kh, kw}) +
                                                    grad * inputCache.get({b, c, ih, iw}));
                                    gradInput.set({b, c, ih, iw},
                                                  gradInput.get({b, c, ih, iw}) +
                                                  grad * weights.get({c, 0, kh, kw}));
                                }
                            }
                        }
                    }
                }
            }
        }
        return gradInput;
    }

    void updateWeights(double learningRate) {
        for (int i = 0; i < weights.getSize(); ++i)
            weights.set(i, weights.get(i) - learningRate * gradWeights.get(i));
        for (int i = 0; i < biases.getSize(); ++i)
            biases.set(i, biases.get(i) - learningRate * gradBiases.get(i));
    }

    void zeroGradients() {
        gradWeights.init(gradWeights.getShape(), false);
        gradBiases.init(gradBiases.getShape(), false);
    }
};


inline Tensor channelShuffle(const Tensor &input, int groups) {
    const  vector<int>& shape = input.getShape(); // [N, C, H, W]
    int batchSize = shape[0];
    int channels  = shape[1];
    int height    = shape[2];
    int width     = shape[3];

    if (channels % groups != 0)
        throw  invalid_argument("채널 수가 groups로 나누어 떨어지지 않습니다.");

    int channelsPerGroup = channels / groups;
    Tensor output(input.getShape());

    // 각 그룹 내에서 채널 재배열: 새 채널 인덱스 = (채널 내 인덱스 * groups) + 그룹 인덱스
    for (int b = 0; b < batchSize; ++b) {
        for (int g = 0; g < groups; ++g) {
            for (int cp = 0; cp < channelsPerGroup; ++cp) {
                int srcChannel = g * channelsPerGroup + cp;
                int dstChannel = cp * groups + g;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        output.set({b, dstChannel, h, w},
                                   input.get({b, srcChannel, h, w}));
                    }
                }
            }
        }
    }
    return output;
}

class ShuffleNetUnitLayer : public Layer {
public:
    int inChannels;
    int outChannels;
    int groups;
    int stride;
    int midChannels;

     shared_ptr<GroupConvLayer> conv1;
     shared_ptr<DepthwiseConvLayer> depthwise;
     shared_ptr<GroupConvLayer> conv2;


    Tensor inputCache;
    Tensor shuffleCache;
    Tensor depthwiseOutputCache;


    ShuffleNetUnitLayer(int inChannels, int outChannels, int groups)
            : inChannels(inChannels), outChannels(outChannels), groups(groups)
    {
        // 입력과 출력 채널이 같으면 downsampling 없이 진행(=stride 1), 다르면 stride 2 적용
        stride = (inChannels == outChannels) ? 1 : 2;
        // 중간 채널 수는 outChannels의 절반으로 설정 (필요에 따라 조정)
        midChannels = outChannels / 2;

        conv1 =  make_shared<GroupConvLayer>(inChannels, midChannels, 1, 1, 0, groups);
        depthwise =  make_shared<DepthwiseConvLayer>(midChannels, 3, stride, 1);
        conv2 =  make_shared<GroupConvLayer>(midChannels, outChannels, 1, 1, 0, groups);
    }

    virtual Tensor forward(const Tensor& input) override {
        inputCache = input;
        Tensor out1 = conv1->forward(input);

        Tensor shuffled = channelShuffle(out1, groups);
        shuffleCache = shuffled;

        Tensor out2 = depthwise->forward(shuffled);
        depthwiseOutputCache = out2;

        Tensor out3 = conv2->forward(out2);
        //shortcut connection
        if (stride == 1 && inChannels == outChannels) {
            out3 = addTensors(out3, input);
        }
        return out3;
    }

    virtual Tensor backward(const Tensor& gradOutput) override {
        Tensor gradResidual;
        if (stride == 1 && inChannels == outChannels)
            gradResidual = gradOutput;
        // 4. conv2의 역전파
        Tensor gradConv2 = conv2->backward(gradOutput);
        // 3. 깊이별 conv의 역전파
        Tensor gradDepthwise = depthwise->backward(gradConv2);
        // 2. 채널 셔플의 역전파 (채널 셔플은 순서만 바꾸므로 동일한 함수 사용)
        Tensor gradShuffle = channelShuffle(gradDepthwise, groups);
        // 1. conv1의 역전파
        Tensor gradConv1 = conv1->backward(gradShuffle);
        if (stride == 1 && inChannels == outChannels)
            gradConv1 = addTensors(gradConv1, gradResidual);
        return gradConv1;
    }

    virtual void updateWeights(double learningRate) override {
        conv1->updateWeights(learningRate);
        depthwise->updateWeights(learningRate);
        conv2->updateWeights(learningRate);
    }

    virtual void zeroGradients() override {
        conv1->zeroGradients();
        depthwise->zeroGradients();
        conv2->zeroGradients();
    }
};

#endif //SHUFFLENET_SHUFFLENETUNITLAYER_H
