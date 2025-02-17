#include "FullyConnectedLayer.h"
#include <stdexcept>

// Constructor: FullyConnectedLayer 초기화
FullyConnectedLayer::FullyConnectedLayer(int inputSize, int outputSize) {
    weights = Tensor(inputSize, outputSize, true); // 가중치를 랜덤 초기화
    biases = Tensor(outputSize, true);         // 바이어스를 [1, OutputSize]로 랜덤 초기화
    gradWeights = Tensor(inputSize, outputSize);  // 가중치 기울기를 0으로 초기화
    gradBiases = Tensor(outputSize);           // 바이어스 기울기를 0으로 초기화
}

// Forward: 순전파 구현
Tensor FullyConnectedLayer::forward(const Tensor& input) {
    inputCache = input;

    const auto& inputShape = input.getShape();      // 입력 텐서의 Shape [Batch, InputSize]
    const auto& weightShape = weights.getShape();  // 가중치 텐서의 Shape [InputSize, OutputSize]

    if (inputShape[1] != weightShape[0]) { // 입력 크기와 가중치가 맞는지 확인
        throw std::invalid_argument("Input size does not match weight dimensions.");
    }

    int batchSize = inputShape[0];
    int outputSize = weightShape[1];

    Tensor output(batchSize, outputSize); // 출력 텐서 초기화

    // 행렬 곱 및 바이어스 추가
    for (int b = 0; b < batchSize; ++b) {
        for (int o = 0; o < outputSize; ++o) {
            double value = biases.get({o});
            for (int i = 0; i < inputShape[1]; ++i) {
                value += input.get({b, i}) * weights.get({i, o});
            }
            output.set({b, o}, value);
        }
    }

    return output;
}

// Backward: 역전파 구현
Tensor FullyConnectedLayer::backward(const Tensor& gradOutput) {
    const vector<int>& inputShape = inputCache.getShape();
    const vector<int>& outputShape = gradOutput.getShape();

    if (inputShape[0] != outputShape[0]) { // 배치 크기가 일치하는지 확인
        throw std::invalid_argument("Batch size mismatch between input and gradient output.");
    }

    int batchSize = inputShape[0];
    int inputSize = inputShape[1];
    int outputSize = outputShape[1];

    Tensor gradInput(inputShape, false); // 입력 기울기 텐서 초기화 {batch, 입력값 개수}
    gradWeights.init({inputSize, outputSize}, false); // 가중치 기울기 초기화
    gradBiases.init({outputSize}, false);          // 바이어스 기울기 초기화

    // 역전파 계산
    for (int b = 0; b < batchSize; ++b) {
        for (int o = 0; o < outputSize; ++o) {
            double grad = gradOutput.get({b, o});
            gradBiases.set({o}, gradBiases.get({o}) + grad); // 바이어스 기울기

            for (int i = 0; i < inputSize; ++i) {
                gradWeights.set({i, o}, gradWeights.get({i, o}) + grad * inputCache.get({b, i}));
                gradInput.set({b, i}, gradInput.get({b, i}) + grad * weights.get({i, o}));
            }
        }
    }

    return gradInput;
}

// Update weights: 가중치 업데이트
void FullyConnectedLayer::updateWeights(double learningRate) {
    for (int i = 0; i < weights.getSize(); ++i) {
        weights.set(i, weights.get(i) - learningRate * gradWeights.get(i));
    }
    for (int i = 0; i < biases.getSize(); ++i) {
        biases.set(i, biases.get(i) - learningRate * gradBiases.get(i));
    }
}

// Zero gradients: 기울기 초기화
void FullyConnectedLayer::zeroGradients() {
    gradWeights.init(gradWeights.getShape(), false); // 가중치 기울기 초기화
    gradBiases.init(gradBiases.getShape(), false);   // 바이어스 기울기 초기화
}
