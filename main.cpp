#include "Dataset.h"
#include "DataLoader.h"
#include "Model.h"
#include "ConvLayer.h"
#include "ShuffleNetUnitLayer.h"    // ShuffleNet Unit 구현 헤더
#include "FullyConnectedLayer.h"
#include "LossFunc.h"
#include "PoolingLayer.h"           // MaxPoolingLayer, GlobalAveragePoolingLayer 등
#include "Optimizer.h"
#include "Tensor.h"
#include "EfficientNetLayer.h"
#include <iostream>
#include <cmath>
#include <iomanip>

int main() {

    // 1. Dataset과 DataLoader 초기화
    Dataset trainDataset("C:/Users/loveh/CLionProjects/Alexnet/MNIST_data/train-images.idx3-ubyte",
                         "C:/Users/loveh/CLionProjects/Alexnet/MNIST_data/train-labels.idx1-ubyte");
    Dataset testDataset("C:/Users/loveh/CLionProjects/Alexnet/MNIST_data/t10k-images.idx3-ubyte",
                        "C:/Users/loveh/CLionProjects/Alexnet/MNIST_data/t10k-labels.idx1-ubyte");

    auto trainData = trainDataset.loadData();
    auto testData = testDataset.loadData();

    int batchSize = 32;
    DataLoader trainLoader(trainData, batchSize);
    DataLoader testLoader(testData, batchSize);

    Model model;
/*
 * //ShuffleNet
 * int groups = 3; // 예시로 그룹 수 3

    // 초기 컨볼루션: MNIST의 경우 입력 채널 1, 출력 채널 24 (채널 수 확장을 위해)
    model.addLayer(make_shared<ConvLayer>(1, 24, 3, 1, 1)); // kernel 3x3, stride 2, padding 1
    model.addLayer(make_shared<Relu>());
    // (필요에 따라) 초기 최대 풀링
    model.addLayer(make_shared<MaxPoolingLayer>(2, 2)); // kernel 3x3, stride 2, padding 1

    // 여러 개의 ShuffleNet Unit 추가
    // (ShuffleNetUnitLayer는 내부에서 1x1 그룹 conv → 채널 셔플 → 3x3 depthwise conv → 1x1 그룹 conv → (잔차 연결) 연산을 수행한다고 가정)
    model.addLayer(make_shared<ShuffleNetUnitLayer>(24, 24, groups));
    model.addLayer(make_shared<ShuffleNetUnitLayer>(24, 24, groups));
    // 채널 수 확장을 위해 출력 채널을 48로 늘리는 Unit (stride 2로 다운샘플링 포함하는 경우도 가능)
    model.addLayer(make_shared<ShuffleNetUnitLayer>(24, 48, groups));
    model.addLayer(make_shared<ShuffleNetUnitLayer>(48, 48, groups));

    // Global Average Pooling과 Flatten 이후 Fully Connected Layer로 분류
    model.addLayer(make_shared<GlobalAveragePoolingLayer>()); // 각 채널당 평균을 내어 [Batch, Channels]로 만듦
    model.addLayer(make_shared<Flatten>());
    model.addLayer(make_shared<FullyConnectedLayer>(48, 10)); // 최종 클래스: 10개*/


    //EfficientNet
    model.addLayer(make_shared<ConvLayer>(1, 32, 3, 2, 1)); // kernel 3x3, stride 2, padding 1
    model.addLayer(make_shared<Relu>());

    // MBConv 블록들:
    // Block 1: MBConv with expansion factor 1 (즉, no expansion), stride 1, 32 -> 16
    model.addLayer(make_shared<MBConvBlock>(32, 16, 3, 1, 1, 0.25));
    // Block 2: MBConv with expansion factor 6, stride 2, 16 -> 24
    model.addLayer(make_shared<MBConvBlock>(16, 24, 3, 2, 6, 0.25));
    // Block 3: MBConv with expansion factor 6, stride 1, 24 -> 24
    model.addLayer(make_shared<MBConvBlock>(24, 24, 3, 1, 6, 0.25));
    // Block 4: MBConv with expansion factor 6, kernel size 5, stride 2, 24 -> 40
    model.addLayer(make_shared<MBConvBlock>(24, 40, 5, 2, 6, 0.25));

    // Head 단계:
    model.addLayer(make_shared<ConvLayer>(40, 1280, 1, 1, 0)); // 1x1 conv로 채널 확장
    model.addLayer(make_shared<Relu>());
    model.addLayer(make_shared<GlobalAveragePoolingLayer>());
    model.addLayer(make_shared<Flatten>());
    model.addLayer(make_shared<FullyConnectedLayer>(1280, 10)); // 예: 10개 클래스


    // 3. Optimizer 및 Loss Function 초기화
    SGD optimizer(make_shared<Model>(model), 0.01); // Learning rate 0.01
    CrossEntropyLoss lossFunction;

    // 4. Training 루프
    for (int epoch = 0; epoch < 1; ++epoch) {
        int batchIndex = 0;
        double totalLoss = 0.0;
        double totalAccuracy = 0.0;
        int totalSamples = 0;

        while (trainLoader.hasNextBatch()) {
            ImageSet batch = trainLoader.getNextBatch();
            Tensor images(batch.images);
            Tensor labels(batch.labels);

            // one-hot encoding (클래스 수 10)
            Tensor oneHotLabels = oneHot(labels, 10);

            // Forward
            Tensor predictions = model.forward(images);

            // 손실 계산 (라벨과 예측값의 차이로 계산된 loss)
            double loss = lossFunction.computeLoss(predictions, oneHotLabels);

            // Zero Gradients
            optimizer.zero_grad();

            // Backward
            Tensor gradients = lossFunction.computeGradient(predictions, oneHotLabels);
            model.backward(gradients);

            // Update
            optimizer.step();

            // Accuracy 계산: softmax 후 argmax
            Tensor prob = predictions.softmax(1); // softmax along class dimension
            Tensor pred = prob.argmax(1);
            double accuracy = pred.eq(labels).mean();

            // 배치 크기 (마지막 배치가 다를 수 있으므로)
            int currentBatchSize = labels.getSize();

            // 각 배치의 기여도를 누적 (loss는 그대로, accuracy는 나중에 백분율로 변환)
            totalLoss += loss;
            totalAccuracy += accuracy;
            totalSamples += currentBatchSize;
            ++batchIndex;

            if (batchIndex % 1 == 0) {

                std::cout << std::fixed << std::setprecision(8)
                          << "TRAIN-Iteration: " << batchIndex
                          << ",Loss: " << loss
                          << ",Accuracy: " << (accuracy * 100) << "%"
                          << std::endl;
            }
        }
        trainLoader.reset();
    }


    // 5. 테스트
    double totalTestLoss = 0.0;
    double totalTestAccuracy = 0.0;
    int totalTestSamples = 0;
    testLoader.reset();

    while (testLoader.hasNextBatch()) {
        ImageSet batch = testLoader.getNextBatch();
        Tensor images(batch.images);
        Tensor labels(batch.labels);

        Tensor predictions = model.forward(images);
        Tensor oneHotLabels = oneHot(labels, 10);
        double loss = lossFunction.computeLoss(predictions, oneHotLabels);

        Tensor prob = predictions.softmax(1);
        Tensor pred = prob.argmax(1);
        double batchAccuracy = pred.eq(labels).mean();

        totalTestSamples += labels.getSize();
        totalTestLoss += loss;
        totalTestAccuracy += batchAccuracy;
    }

    double averageTestLoss = totalTestLoss / totalTestSamples;
    double averageTestAccuracy = totalTestAccuracy / totalTestSamples;
    std::cout << "TEST-Accuracy: " << averageTestAccuracy * 100.0 << "%, Average Loss: " << averageTestLoss << std::endl;

    return 0;
}
