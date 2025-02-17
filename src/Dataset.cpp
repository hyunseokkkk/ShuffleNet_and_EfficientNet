#include "Dataset.h"
#include <fstream>
#include <filesystem> // C++17 이상 필요
#include <stdexcept>

// 저장 파일 경로
const string imageCacheFile = "cached_images.bin";
const string labelCacheFile = "cached_labels.bin";

Dataset::Dataset(const string& imagesPath, const string& labelsPath)
        : imagesPath(imagesPath), labelsPath(labelsPath) {}

int Dataset::readInt(ifstream& file) const {
    unsigned char buffer[4];
    file.read(reinterpret_cast<char*>(buffer), 4);
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

ImageSet Dataset::loadData() {
    // 저장된 데이터가 있는지 확인
    if (filesystem::exists(imageCacheFile) && filesystem::exists(labelCacheFile)) {
        return loadCachedData();
    }

    ifstream imageFile(imagesPath, ios::binary);
    ifstream labelFile(labelsPath, ios::binary);

    if (!imageFile.is_open() || !labelFile.is_open()) {
        throw runtime_error("Failed to open dataset files.");
    }

    // Read headers
    int magicNumber = readInt(imageFile);
    if (magicNumber != 2051) throw runtime_error("Invalid magic number in image file.");
    int numImages = readInt(imageFile);
    int numRows = readInt(imageFile);
    int numCols = readInt(imageFile);

    magicNumber = readInt(labelFile);
    if (magicNumber != 2049) throw runtime_error("Invalid magic number in label file.");
    int numLabels = readInt(labelFile);

    if (numImages != numLabels) {
        throw runtime_error("Number of images and labels do not match.");
    }

    // Initialize tensors
    Tensor images(numImages, 1, numRows, numCols); // Assumes single channel images (grayscale)
    Tensor labels(numLabels);                     // 1차원 데이터로 초기화

    // Load images
    for (int i = 0; i < numImages; ++i) {
        for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numCols; ++c) {
                unsigned char pixel = 0;
                imageFile.read(reinterpret_cast<char*>(&pixel), 1);
                images.set({i, 0, r, c}, static_cast<double>(pixel) / 255.0); // Normalize
            }
        }
    }

    // Load labels
    for (int i = 0; i < numLabels; ++i) {
        unsigned char label = 0;
        labelFile.read(reinterpret_cast<char*>(&label), 1);
        labels.set({i}, static_cast<int>(label));
    }

    // 데이터 저장
    saveCachedData(images, labels);

    return {images, labels};
}

// 데이터 저장 함수
void Dataset::saveCachedData(const Tensor& images, const Tensor& labels) const {
    ofstream imageOut(imageCacheFile, ios::binary);
    ofstream labelOut(labelCacheFile, ios::binary);

    if (!imageOut.is_open() || !labelOut.is_open()) {
        throw runtime_error("Failed to open cache files for writing.");
    }

    // 이미지 저장
    const auto& imageData = images.getData();
    imageOut.write(reinterpret_cast<const char*>(imageData.data()), imageData.size() * sizeof(double));

    // 라벨 저장
    const auto& labelData = labels.getData();
    labelOut.write(reinterpret_cast<const char*>(labelData.data()), labelData.size() * sizeof(double));
}

// 캐시된 데이터 로드 함수
ImageSet Dataset::loadCachedData() const {
    ifstream imageIn(imageCacheFile, ios::binary);
    ifstream labelIn(labelCacheFile, ios::binary);

    if (!imageIn.is_open() || !labelIn.is_open()) {
        throw runtime_error("Failed to open cache files for reading.");
    }

    // 이미지 복원
    Tensor images; // Default constructor
    images.init({60000, 1, 28, 28}, false); // MNIST 데이터셋 기준
    imageIn.read(reinterpret_cast<char*>(images.getData().data()), images.getTotalSize() * sizeof(double));

    // 라벨 복원
    Tensor labels;
    labels.init({60000}, false); // MNIST 데이터셋 기준
    labelIn.read(reinterpret_cast<char*>(labels.getData().data()), labels.getTotalSize() * sizeof(double));

    return {images, labels};
}
