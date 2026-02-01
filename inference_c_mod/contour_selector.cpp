#include "contour_selector.h"

int contour_sel_find_best(const std::vector<std::vector<cv::Point>>& contours) {
    int best_idx = -1;
    double best_score = 0.0;

    for (int i = 0; i < static_cast<int>(contours.size()); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < CONTOUR_SEL_MIN_AREA) {
            continue;
        }

        cv::Rect box = cv::boundingRect(contours[i]);

        float ratio = static_cast<float>(box.width) / box.height;
        if (ratio < CONTOUR_SEL_MIN_ASPECT_RATIO || ratio > CONTOUR_SEL_MAX_ASPECT_RATIO) {
            continue;
        }

        double fill = area / (box.width * box.height);
        if (fill < 0.0) {
            continue;
        }

        double score = area * fill;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return best_idx;
}
