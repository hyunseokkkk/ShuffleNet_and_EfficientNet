#include "Tensor.h"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

using namespace std;

// Helper function to calculate product of dimensions
int Tensor::product(const vector<int>& shape) const {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    return size;
}


// Default Constructor
Tensor::Tensor() : totalSize(0) {}

// Constructor for 1D to Dims Tensor
Tensor::Tensor(const vector<int>& shape, bool random)
        : shape(shape), totalSize(product(shape)) {
    initialize(random);
}


// Initialize the Tensor
void Tensor::init(const vector<int>& newShape, bool random) {
    shape =newShape;
    totalSize = product(shape);
    initialize(random);
}


// Helper function to initialize strides and data
void Tensor::initialize(bool random) {
    strides.resize(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    data.resize(totalSize);

    if (random) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.1, 0.1);
        for (double& value : data) {
            value = dis(gen);
        }
    } else {
        fill(data.begin(), data.end(), 0.0);
    }
}

// Compute flat index from multi-dimensional indices
int Tensor::index(const vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw invalid_argument("Number of indices does not match tensor dimensions.");
    }

    int flatIndex = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (indices[i] >= shape[i]) {
            throw out_of_range("Index out of bounds for dimension " + to_string(i));
        }
        flatIndex += indices[i] * strides[i];
    }
    return flatIndex;
}


//1차원 인덱스를 다차원 인덱스로


// Overloaded Constructors for specific dimensions
Tensor::Tensor(int dim1, bool random)
        : Tensor(vector<int>{dim1}, random) {}

Tensor::Tensor(int dim1, int dim2, bool random)
        : Tensor(vector<int>{dim1, dim2}, random) {}

Tensor::Tensor(int dim1, int dim2, int dim3, bool random)
        : Tensor(vector<int>{dim1, dim2, dim3}, random) {}

Tensor::Tensor(int dim1, int dim2, int dim3, int dim4, bool random)
        : Tensor(vector<int>{dim1, dim2, dim3, dim4}, random) {}

Tensor::Tensor(int dim1, int dim2, int dim3, int dim4, int dim5, bool random)
        : Tensor(vector<int>{dim1, dim2, dim3, dim4, dim5}, random) {}


// Other member functions (unchanged)
double Tensor::get(const vector<int>& indices) const {
    return data[index(indices)];
}

void Tensor::set(const vector<int>& indices, double value) {
    data[index(indices)] = value;
}

double Tensor::get(int flatIndex) const {
    if (flatIndex < 0 || flatIndex >= totalSize) {
        throw out_of_range("Flat index out of bounds.");
    }
    return data[flatIndex];
}

void Tensor::set(int flatIndex, double value) {
    if (flatIndex < 0 || flatIndex >= totalSize) {
        throw out_of_range("Flat index out of bounds.");
    }
    data[flatIndex] = value;
}


const vector<double>& Tensor::getData() const {
    return data;
}
vector<double>& Tensor::getData() {
    return data;
}

void Tensor::reshape(const vector<int>& newShape) {
    vector<int> normalizedShape(newShape);
    int newSize = product(normalizedShape);

    if (newSize != totalSize) {
        throw invalid_argument("Total size mismatch during reshape.");
    }

    shape = normalizedShape;
}

void Tensor::print() const {
    cout << "Tensor Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    }
    cout << "]\nData:\n";

    for (size_t i = 0; i < data.size(); ++i) {
        cout << data[i] << " ";
    }
    cout << endl;
}


Tensor Tensor::softmax(int dim) const {
    if (dim < 0 || dim >= static_cast<int>(shape.size())) {
        throw invalid_argument("Invalid dimension for softmax.");
    }

    // 결과 텐서 (same shape)
    Tensor result(shape);

    // 소프트맥스를 구할 축의 크기
    int dimSize = shape[dim];
    // 나머지 축의 총 요소 수
    int outSize = totalSize / dimSize;

    // -----------------------------------------------------
    // 1) outShape: shape에서 dim번째 축을 제거한 형태
    // -----------------------------------------------------
    vector<int> outShape;
    outShape.reserve(shape.size() - 1);
    for (int i = 0; i < (int)shape.size(); i++) {
        if (i == dim) continue;  // dim번째 축 생략
        outShape.push_back(shape[i]);
    }

    vector<int> outStrides(outShape.size(), 1);
    for (int i = (int)outShape.size() - 2; i >= 0; --i) {
        outStrides[i] = outStrides[i + 1] * outShape[i + 1];
    }

    // -----------------------------------------------------
    // 3) outSize만큼 반복하며, dim축을 제외한 좌표 순회
    // -----------------------------------------------------
    for (int outer = 0; outer < outSize; outer++) {
        // (3-1) outShape 기준으로 'outer'를 다차원 인덱스(outCoords)로 변환
        vector<int> outCoords(outShape.size());
        {
            int tmp = outer;
            for (int k = (int)outShape.size() - 1; k >= 0; --k) {
                outCoords[k] = tmp % outShape[k];
                tmp /= outShape[k];
            }
        }

        // (3-2) 원래 shape에 맞는 좌표 coords 만들기
        //       (dim 위치만 0으로 둔다 → 아래에서 0..dimSize-1 반복)
        vector<int> coords(shape.size(), 0);
        {
            int pos = 0;
            for (int d = 0; d < (int)shape.size(); d++) {
                if (d == dim) {
                    coords[d] = 0;
                } else {
                    coords[d] = outCoords[pos++];
                }
            }
        }

        // (3-3) coords에서 dim축=0일 때의 1D 인덱스 (baseIndex)
        //       나중에 dim축만큼 strides[dim]을 곱해 이동
        int baseIndex = index(coords);

        // -----------------------------------------------------
        // 4) 해당 그룹(=한 outer)에 대해 softmax 계산
        //    max, exp, sum → normalize
        // -----------------------------------------------------
        double maxVal = -std::numeric_limits<double>::infinity();
        // (4-1) 최대값 찾기
        for (int i = 0; i < dimSize; i++) {
            int idx = baseIndex + i * strides[dim];
            double v  = this->get(idx);
            if (v > maxVal) {
                maxVal = v;
            }
        }

        double sumVal = 0.0;
        // (4-2) e^(x - maxVal) 계산, sumVal 누적
        for (int i = 0; i < dimSize; i++) {
            int idx = baseIndex + i * strides[dim];
            double e = std::exp(this->get(idx) - maxVal);
            result.set(idx, e);
            sumVal += e;
        }

        // (4-3) 나누기
        for (int i = 0; i < dimSize; i++) {
            int idx = baseIndex + i * strides[dim];
            double val = result.get(idx);
            result.set(idx, val / sumVal);
        }
    }

    return result;
}

Tensor Tensor::argmax(int dim) const {
    if (dim < 0 || dim >= static_cast<int>(shape.size())) {
        throw invalid_argument("Invalid dimension for argmax.");
    }

    // 결과 텐서의 shape 계산 (dim 차원을 제거)
    vector<int> resultShape;
    resultShape.reserve(shape.size() - 1);
    for (int i = 0; i < (int)shape.size(); i++) {
        if (i == dim) continue; // dim번째 축 제외
        resultShape.push_back(shape[i]);
    }

    // Argmax 결과를 저장할 텐서 생성
    Tensor result(resultShape);

    // 소스 텐서의 dim축 크기
    int dimSize = shape[dim];
    // dim축을 제외한 모든 축의 요소 수
    int outSize = totalSize / dimSize;

    vector<int> outShape;
    outShape.reserve(shape.size() - 1);
    for (int i = 0; i < (int)shape.size(); i++) {
        if (i == dim) continue;
        outShape.push_back(shape[i]);
    }

    vector<int> outStrides(outShape.size(), 1);
    for (int i = (int)outShape.size() - 2; i >= 0; --i) {
        outStrides[i] = outStrides[i + 1] * outShape[i + 1];
    }

    for (int outer = 0; outer < outSize; outer++) {
        vector<int> outCoords(outShape.size(), 0);
        {
            int tmp = outer;
            for (int k = (int)outShape.size() - 1; k >= 0; --k) {
                outCoords[k] = tmp % outShape[k];
                tmp /= outShape[k];
            }
        }

        vector<int> coords(shape.size(), 0);
        {
            int pos = 0;
            for (int d = 0; d < (int)shape.size(); d++) {
                if (d == dim) {
                    coords[d] = 0;
                } else {
                    coords[d] = outCoords[pos++];
                }
            }
        }

        int baseIndex = index(coords);

        double maxVal = -std::numeric_limits<double>::infinity();
        int maxIdx = 0;
        for (int i = 0; i < dimSize; i++) {
            int idx = baseIndex + i * strides[dim];
            double val = data[idx];
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }

        result.set(outer, (double)maxIdx);

    }

    return result;
}



Tensor Tensor::eq(const Tensor& other) const {
    if (shape != other.getShape()) {
        throw invalid_argument("Shapes of tensors do not match for comparison.");
    }

    Tensor result(shape);
    const auto& otherData = other.getData();

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = (data[i] == otherData[i]) ? 1.0 : 0.0; // 동일하면 1.0, 다르면 0.0
    }

    return result;
}
double Tensor::mean() const {
    if (data.empty()) {
        throw runtime_error("Tensor is empty. Cannot calculate mean.");
    }

    double sum = 0.0;
    for (const auto& value : data) {
        sum += value;
    }

    return sum / totalSize;
}

Tensor oneHot(const Tensor& labels, int numClasses) {
    const auto& labelsData = labels.getData(); // 1D label data
    Tensor oneHotLabels(static_cast<int>(labels.getSize()), numClasses); // Shape: [batchSize, numClasses]

    for (size_t i = 0; i < labelsData.size(); ++i) {
        int label = static_cast<int>(labelsData[i]);
        if (label >= numClasses || label < 0) {
            throw std::invalid_argument("Label value out of range for one-hot encoding.");
        }
        oneHotLabels.set({static_cast<int>(i), label}, 1.0);
    }
    return oneHotLabels;
}


// Get the shape of the tensor
const vector<int>& Tensor::getShape() const {
    return shape;
}
int Tensor::getSize() const {
    return totalSize;
}

int Tensor::getTotalSize() const {
    return totalSize;
}
