#include "contour_detection.h"

void contour_det_find_all(
    const cv::Mat& thresh,
    std::vector<std::vector<cv::Point>>& contours
) {
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
}

void contour_det_draw(
    cv::Mat& frame,
    const std::vector<std::vector<cv::Point>>& contours
) {
    cv::drawContours(frame, contours, -1, cv::Scalar(0, 0, 255), 2);
}
