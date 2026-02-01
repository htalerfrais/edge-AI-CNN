#ifndef CONTOUR_DETECTION_H
#define CONTOUR_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

void contour_det_find_all(
    const cv::Mat& thresh,
    std::vector<std::vector<cv::Point>>& contours
);

void contour_det_draw(
    cv::Mat& frame,
    const std::vector<std::vector<cv::Point>>& contours
);

#endif // CONTOUR_DETECTION_H
