cmake_minimum_required(VERSION 3.28)
project(ShuffleNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 소스 파일과 헤더 파일 명시 (상대 경로 -> 절대 경로)
set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Layer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ConvLayer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/FullyConnectedLayer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Dataset.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Optimizer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Tensor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PoolingLayer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/DataLoader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/LossFunc.cpp
)

set(HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Tensor.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/PoolingLayer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/DataLoader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/LossFunc.h
        include/ShuffleNetUnitLayer.h
        include/EfficientNetLayer.h
)

# 실행 파일 생성
add_executable(ShuffleNet
        ${SOURCES}
        ${HEADERS}
)

# 인클루드 디렉토리 추가 (PRIVATE / PUBLIC / INTERFACE 선택 가능)
target_include_directories(ShuffleNet
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
