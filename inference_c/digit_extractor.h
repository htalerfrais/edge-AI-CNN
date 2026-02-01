#ifndef DIGIT_EXTRACTOR_H
#define DIGIT_EXTRACTOR_H

#include <opencv2/opencv.hpp>

constexpr int DIGIT_EXTR_TARGET_SIZE = 20;
constexpr int DIGIT_EXTR_PADDED_SIZE = 28;
constexpr int DIGIT_EXTR_PADDING = 4;

void digit_extr_extract(
    const cv::Mat& thresh,
    const cv::Rect& box,
    cv::Mat& digit28
);

void digit_extr_to_nn_input(
    const cv::Mat& digit28,
    float* nn_input
);

#endif // DIGIT_EXTRACTOR_H
