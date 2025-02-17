#ifndef SHUFFLENET_EFFICIENTNETLAYER_H
#define SHUFFLENET_EFFICIENTNETLAYER_H

#include "Layer.h"
#include "Model.h"
#include "Tensor.h"
#include "ConvLayer.h"
#include "ShuffleNetUnitLayer.h"
#include "PoolingLayer.h"
#include "FullyConnectedLayer.h"
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

using namespace std;


// 헬퍼 함수: 두 텐서를 element-wise 곱셈
inline Tensor elementWiseMultiply(const Tensor &a, const Tensor &b) {
    // a의 shape: [N, C, H, W], b의 shape: [N, C, 1, 1]라고 가정
    const vector<int>& aShape = a.getShape();
    const vector<int>& bShape = b.getShape();
    if (aShape.size() != 4 || bShape.size() != 4) {
        throw invalid_argument("Expected 4D tensors.");
    }
    if (aShape[0] != bShape[0] || aShape[1] != bShape[1]) {
        throw invalid_argument("Batch size and channel dimensions must match.");
    }
    // 결과 텐서 생성
    Tensor result(aShape, false);
    vector<double>& resData = result.getData();
    const vector<double>& aData = a.getData();
    const vector<double>& bData = b.getData();

    int N = aShape[0], C = aShape[1], H = aShape[2], W = aShape[3];
    // b의 값을 (N, C, 1, 1)에서 각 spatial 위치에 복제
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            // b의 값
            double scale = bData[n * C + c];
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    resData[idx] = aData[idx] * scale;
                }
            }
        }
    }
    return result;
}

// MBConvBlock 클래스 (EfficientNet의 MBConv, SE 모듈 포함)
class MBConvBlock : public Layer {
public:
    int inChannels;      // 입력 채널 수
    int outChannels;     // 출력 채널 수
    int kernelSize;      // 컨볼루션 커널 크기 (예: 3 또는 5)
    int stride;          // stride (downsampling 여부)
    int expansionFactor; // 확장 인자 (예: 6)
    double seRatio;      // SE 모듈의 squeeze 비율

    int expandedChannels; // inChannels * expansionFactor

    // 내부 구성 레이어들
    shared_ptr<ConvLayer> expansionConv;   // 1×1 확장 컨볼루션 (expansionFactor > 1일 때)
    shared_ptr<Relu> activation;             // 활성화 (ReLU 사용)
    shared_ptr<DepthwiseConvLayer> depthwiseConv; // 깊이별 컨볼루션
    // SE 블록 (두 FullyConnectedLayer와 내부 활성화)
    shared_ptr<Flatten> flattenLayer;
    shared_ptr<GlobalAveragePoolingLayer> gap;
    shared_ptr<FullyConnectedLayer> seReduce;
    shared_ptr<Relu> seActivation;
    shared_ptr<FullyConnectedLayer> seExpand;
    // 프로젝션 단계: 1×1 컨볼루션 (채널 축소)
    shared_ptr<ConvLayer> projectionConv;

    // --- forward에서 저장한 캐시 (실제 backward 계산에 필요) ---
    Tensor inputCache;         // 원본 입력 (skip connection용)
    Tensor xDepthActCache;     // depthwise conv 후 활성화 결과, shape: [N, expandedChannels, H', W']
    Tensor seOutputCache;      // SE 블록에서 sigmoid 적용 후, flatten 전 shape: [N, expandedChannels]
    Tensor seScaleCache;       // SE 스케일, reshaped 후 [N, expandedChannels, 1, 1]

    MBConvBlock(int inChannels, int outChannels, int kernelSize, int stride,
                int expansionFactor, double seRatio)
            : inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize),
              stride(stride), expansionFactor(expansionFactor), seRatio(seRatio)
    {
        expandedChannels = inChannels * expansionFactor;
        if(expansionFactor > 1) {
            expansionConv = make_shared<ConvLayer>(inChannels, expandedChannels, 1, 1, 0);
        }
        activation = make_shared<Relu>();
        depthwiseConv = make_shared<DepthwiseConvLayer>(expandedChannels, kernelSize, stride, kernelSize / 2);

        int seChannels = max(1, static_cast<int>(expandedChannels * seRatio));
        flattenLayer = make_shared<Flatten>();
        gap = make_shared<GlobalAveragePoolingLayer>();
        seReduce = make_shared<FullyConnectedLayer>(expandedChannels, seChannels);
        seActivation = make_shared<Relu>();
        seExpand = make_shared<FullyConnectedLayer>(seChannels, expandedChannels);

        projectionConv = make_shared<ConvLayer>(expandedChannels, outChannels, 1, 1, 0);
    }

    virtual Tensor forward(const Tensor &input) override {
        inputCache = input;
        Tensor x = input;

        // 1. 확장 단계 (expansionConv 적용 시)
        if(expansionConv) {
            x = expansionConv->forward(x); // 결과 shape: [N, expandedChannels, H, W]
            x = activation->forward(x);
        }

        // 2. 깊이별 컨볼루션 후 활성화
        x = depthwiseConv->forward(x);
        x = activation->forward(x);
        xDepthActCache = x; // 캐시 저장

        // 3. SE 블록
        // Global Average Pooling: [N, expandedChannels, H', W'] → [N, expandedChannels, 1, 1]
        Tensor se = gap->forward(x);
        // Flatten: [N, expandedChannels, 1, 1] → [N, expandedChannels]
        se = flattenLayer->forward(se);
        // Fully Connected 레이어들
        se = seReduce->forward(se);
        se = seActivation->forward(se);
        se = seExpand->forward(se);
        // Sigmoid 활성화
        vector<double>& seData = se.getData();
        for (size_t i = 0; i < seData.size(); i++) {
            seData[i] = 1.0 / (1.0 + exp(-seData[i]));
        }
        seOutputCache = se; // SE 출력 캐시 (shape: [N, expandedChannels])
        // reshape se from [N, expandedChannels] to [N, expandedChannels, 1, 1]
        vector<int> seShape = se.getShape();
        se.reshape({seShape[0], seShape[1], 1, 1});
        seScaleCache = se;  // SE 스케일 캐시
        // SE 적용: element‑wise multiplication

        x = elementWiseMultiply(x, se);

        // 4. 프로젝션 단계
        x = projectionConv->forward(x);

        // 5. Skip connection: 조건 (stride == 1 및 inChannels == outChannels)
        if (stride == 1 && inChannels == outChannels) {
            x = addTensors(x, input);
        }

        return x;
    }

    virtual Tensor backward(const Tensor &gradOutput) override {
        // --- Step 1: Projection & Skip connection backward ---
        Tensor grad_after_proj;

        grad_after_proj = projectionConv->backward(gradOutput);


        // --- Step 2: Backprop through element-wise multiplication ---
        // forward: x_mult = xDepthActCache * seScaleCache
        Tensor grad_x_depth_act = elementWiseMultiply(grad_after_proj, seScaleCache);
        Tensor grad_seScale = elementWiseMultiply(grad_after_proj, xDepthActCache);

        // --- Step 3: Backprop through SE block ---
        // grad_seScale: shape [N, expandedChannels, 1, 1] → reshape to [N, expandedChannels]
        vector<int> gradSeShape = grad_seScale.getShape(); // 예: [N, expandedChannels, H, W]
        int N = gradSeShape[0];
        int C = gradSeShape[1];
        int H = gradSeShape[2];
        int W = gradSeShape[3];

// 새 텐서 생성: [N, C]
        Tensor grad_se_reduced({N, C}, false);
        vector<double>& redData = grad_se_reduced.getData();
        const vector<double>& fullData = grad_seScale.getData();

// H와 W 차원에 대해 합산 (또는 평균을 원한다면 sum을 (H*W)로 나누면 됩니다)
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                double sum = 0.0;
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        sum += fullData[idx];
                    }
                }
                // 여기서는 합산 결과를 사용합니다.
                redData[n * C + c] = sum / (H * W);
            }
        }

// 이제 grad_se_reduced의 shape는 [N, C]가 되었으므로, 후속 SE block의 backward 연산에 사용합니다.
        grad_seScale = grad_se_reduced;

        // Backprop through sigmoid: derivative = s*(1-s)
        vector<double>& grad_seData = grad_seScale.getData();
        const vector<double>& seData = seOutputCache.getData(); // seOutputCache: [N, expandedChannels]
        for (size_t i = 0; i < grad_seData.size(); i++) {
            grad_seData[i] *= seData[i] * (1 - seData[i]);
        }
        // Backprop through SE fully-connected layers
        Tensor grad_se_expand = seExpand->backward(grad_seScale);
        Tensor grad_se_activation = seActivation->backward(grad_se_expand);
        Tensor grad_se_reduce = seReduce->backward(grad_se_activation);
        // Backprop through Flatten backward
        Tensor grad_flatten = flattenLayer->backward(grad_se_reduce);
        // Backprop through Global Average Pooling backward
        Tensor grad_gap = gap->backward(grad_flatten);

        // SE 부분의 back propagtaion과 원래 입력 값에 해당하는 gradient를 더함
        Tensor grad_x_depth_act_total = addTensors(grad_x_depth_act, grad_gap);

        // --- Step 4: Backprop through activation after depthwise conv ---
        Tensor grad_activation_depth = activation->backward(grad_x_depth_act_total);

        // --- Step 5: Backprop through depthwise conv ---
        Tensor grad_depthwise = depthwiseConv->backward(grad_activation_depth);

        // --- Step 6: Backprop through expansion branch (if exists) ---
        Tensor grad_before_expansion;
        if(expansionConv) {
            Tensor grad_activation_expansion = activation->backward(grad_depthwise);
            grad_before_expansion = expansionConv->backward(grad_activation_expansion);
        } else {
            grad_before_expansion = grad_depthwise;
        }

        // If skip connection exists, note that in forward: output = projectionConv(x_mult) + inputCache,
        // so the gradient from the skip branch (gradOutput) was already added in Step 1.
        // Thus, final gradient with respect to the block input is:
        if (stride == 1 && inChannels == outChannels) {
            grad_before_expansion = addTensors(grad_before_expansion, gradOutput);
        }
        return grad_before_expansion;
    }

    virtual void updateWeights(double learningRate) override {
        if(expansionConv) expansionConv->updateWeights(learningRate);
        depthwiseConv->updateWeights(learningRate);
        seReduce->updateWeights(learningRate);
        seExpand->updateWeights(learningRate);
        projectionConv->updateWeights(learningRate);
    }

    virtual void zeroGradients() override {
        if(expansionConv) expansionConv->zeroGradients();
        depthwiseConv->zeroGradients();
        seReduce->zeroGradients();
        seExpand->zeroGradients();
        projectionConv->zeroGradients();
    }
};


#endif //SHUFFLENET_EFFICIENTNETLAYER_H
