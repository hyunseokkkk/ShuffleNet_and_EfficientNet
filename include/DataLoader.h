#ifndef ALEXNET_DATALOADER_H
#define ALEXNET_DATALOADER_H

#include "Dataset.h"

using namespace std;

class DataLoader {
private:
    ImageSet dataset; // 전체 데이터셋
    int batchSize;        // 배치 크기
    size_t currentIndex;  // 현재 인덱스

public:
    DataLoader(const ImageSet& dataset, int batchSize);

    ImageSet getNextBatch(); // 다음 배치 반환
    bool hasNextBatch() const;                               // 다음 배치 확인
    void reset();                                            // 데이터로더 초기화
};

#endif // ALEXNET_DATALOADER_H
