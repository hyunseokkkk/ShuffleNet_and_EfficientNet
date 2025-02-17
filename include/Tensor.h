#ifndef ALEXNET_TENSOR_H
#define ALEXNET_TENSOR_H

#include <vector>
#include <stdexcept>
#include <iostream>

using namespace std;

class Tensor {
private:
    vector<double> data;          // Flattened tensor data
    vector<int> shape;            // Tensor shape
    vector<int> strides;          // Strides for index calculation
    int totalSize;                // Total number of elements in the tensor

    int product(const vector<int>& shape) const;              // Calculate total size
//    vector<int> normalize(const vector<int>& shape) const; // Normalize shape to maxDims
    void initialize(bool random);                             // Initialize data and strides
    int index(const vector<int>& indices) const;

public:
    // Constructors
    Tensor();                                                 // Default constructor
    Tensor(const vector<int>& shape, bool random = false); // Flexible constructor
    Tensor(int dim1, bool random = false);                    // 1D Tensor
    Tensor(int dim1, int dim2, bool random = false);          // 2D Tensor
    Tensor(int dim1, int dim2, int dim3, bool random = false); // 3D Tensor
    Tensor(int dim1, int dim2, int dim3, int dim4, bool random = false); // 4D Tensor
    Tensor(int dim1, int dim2, int dim3, int dim4, int dim5, bool random = false); // 5D Tensor

    // Initialization
    void init(const vector<int>& newShape, bool random = false);

    // Accessors
    const vector<int>& getShape() const;                      // Get tensor shape
    const vector<double>& getData() const;
    vector<double>& getData(); // 비-const 오버로드
    int getSize() const;
    int getTotalSize() const;                                 // Get total size (alias for external use)

    // Element-wise access
    double get(const vector<int>& indices) const;             // Get value at indices
    void set(const vector<int>& indices, double value);       // Set value at indices
    double get(int index) const;
    void set(int index, double value);

    vector<int> IndexMuliDim(int flatIndex) const;
        // Reshape
    void reshape(const vector<int>& newShape);

    // Tensor operations
    Tensor softmax(int dim) const;                            // Softmax along a dimension
    Tensor argmax(int dim) const;                             // Argmax along a dimension
    Tensor eq(const Tensor& other) const;                     // Element-wise equality
    double mean() const;                                      // Calculate mean of all elements

    // Print tensor
    void print() const;
};

Tensor oneHot(const Tensor& labels, int numClasses);

#endif // ALEXNET_TENSOR_H
