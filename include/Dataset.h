#ifndef ALEXNET_DATASET_H
#define ALEXNET_DATASET_H

#include "Tensor.h"
#include <vector>
#include <string>

using namespace std;

struct ImageSet {
    Tensor images;
    Tensor labels;
};


class Dataset {
private:
    string imagesPath;
    string labelsPath;
    int readInt(ifstream& file) const;

public:
    Dataset(const string& imagesPath, const string& labelsPath);
    ImageSet loadData(); // 전체 데이터셋 로드
    void saveCachedData(const Tensor& images, const Tensor& labels) const;
    ImageSet loadCachedData() const;

    };

#endif // ALEXNET_DATASET_H
