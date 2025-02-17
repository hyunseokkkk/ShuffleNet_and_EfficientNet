#include "DataLoader.h"
#include "Tensor.h"
#include <algorithm>

DataLoader::DataLoader(const ImageSet& dataset, int batchSize)
        : dataset(dataset), batchSize(batchSize), currentIndex(0) {}

ImageSet DataLoader::getNextBatch() {
    if (!hasNextBatch()) {
        throw runtime_error("No more batches available.");
    }

    size_t endIndex = min(currentIndex + batchSize, static_cast<size_t>(dataset.images.getShape()[0]));
    Tensor batchImages(static_cast<int>(endIndex - currentIndex),
                        dataset.images.getShape()[1],
                        dataset.images.getShape()[2],
                        dataset.images.getShape()[3]);
    Tensor batchLabels(static_cast<int>(endIndex - currentIndex));

    for (size_t i = currentIndex; i < endIndex; ++i) {
        for (size_t c = 0; c < dataset.images.getShape()[1]; ++c) {
            for (size_t h = 0; h < dataset.images.getShape()[2]; ++h) {
                for (size_t w = 0; w < dataset.images.getShape()[3]; ++w) {
                    batchImages.set({static_cast<int>(i - currentIndex), static_cast<int>(c), static_cast<int>(h), static_cast<int>(w)},dataset.images.get({static_cast<int>(i), static_cast<int>(c),static_cast<int>(h), static_cast<int>(w)}));
                }
            }
        }
        batchLabels.set({static_cast<int>(i - currentIndex)},dataset.labels.get({static_cast<int>(i)}));
    }

    currentIndex = endIndex;

    return {batchImages, batchLabels};
}

bool DataLoader::hasNextBatch() const {
    return currentIndex < dataset.images.getShape()[0];
}

void DataLoader::reset() {
    currentIndex = 0;
}
