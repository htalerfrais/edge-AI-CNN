#ifndef CONTOUR_SELECTOR_H
#define CONTOUR_SELECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

constexpr double CONTOUR_SEL_MIN_AREA = 150.0;
constexpr float CONTOUR_SEL_MIN_ASPECT_RATIO = 0.2f;
constexpr float CONTOUR_SEL_MAX_ASPECT_RATIO = 2.0f;

int contour_sel_find_best(const std::vector<std::vector<cv::Point>>& contours);

#endif // CONTOUR_SELECTOR_H
