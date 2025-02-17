#include "LossFunc.h"

// CrossEntropyLoss
double CrossEntropyLoss::computeLoss(const Tensor& predictions, const Tensor& targets) const {
    const auto& predsData = predictions.getData();
    const auto& targetsData = targets.getData();

    double loss = 0.0;
    for (size_t i = 0; i < predsData.size(); ++i) {
        if (targetsData[i] > 0) { // Log of zero를 방지하기 위한 조건
            loss -= targetsData[i] * log(predsData[i] + 1e-10); // 작은 값 추가로 안정성 보장
        }
    }
    return loss / targets.getShape()[0]; // 배치 크기로 나누어 평균 손실 계산
}

Tensor CrossEntropyLoss::computeGradient(const Tensor& predictions, const Tensor& targets) const {
    const auto& predsData = predictions.getData(); // 예측값(flattened)
    const auto& targetsData = targets.getData();   // 실제값(flattened)

    if (predsData.size() != targetsData.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }

    // Gradient를 저장할 Tensor 초기화
    Tensor gradients(predictions.getShape(), false); // {batch_size, 10}

    // Gradient 계산
    for (size_t i = 0; i < predsData.size(); ++i) {
        // Cross-entropy gradient 계산 공식 적용
        gradients.set(i, predsData[i] - targetsData[i]);
    }

    return gradients;
}


// MSELoss
double MSELoss::computeLoss(const Tensor& predictions, const Tensor& targets) const {
    const auto& predsData = predictions.getData();
    const auto& targetsData = targets.getData();

    double loss = 0.0;
    for (size_t i = 0; i < predsData.size(); ++i) {
        double diff = predsData[i] - targetsData[i];
        loss += diff * diff;
    }
    return loss / predsData.size(); // 평균 손실 계산
}

Tensor MSELoss::computeGradient(const Tensor& predictions, const Tensor& targets) const {
    const auto& predsData = predictions.getData(); // 예측값(flattened)
    const auto& targetsData = targets.getData();   // 실제값(flattened)

    if (predsData.size() != targetsData.size()) {
        throw std::invalid_argument("Predictions and targets must have the same size.");
    }

    // Gradient를 저장할 Tensor 초기화
    Tensor gradients(predictions.getShape());

    // Gradient 계산
    for (size_t i = 0; i < predsData.size(); ++i) {
        // 1D 인덱스를 N차원 인덱스로 변환
        // Cross-entropy gradient 계산 공식 적용
        gradients.set(i, predsData[i] - targetsData[i]);
    }

    return gradients;
}
