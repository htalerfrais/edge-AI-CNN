#include "digit_extractor.h"

static const float DIGIT_EXTR_MNIST_MEAN = 0.1307f;
static const float DIGIT_EXTR_MNIST_STD = 0.3081f;
static const float DIGIT_EXTR_BRIGHTEN_FACTOR = 1.5f;

static void digit_extr_compute_barycenter(
    const cv::Mat& mask,
    double& cx,
    double& cy
) {
    double sum_x = 0.0;
    double sum_y = 0.0;
    double count_pix = 0.0;

    for (int y = 0; y < mask.rows; y++) {
        const uchar* row = mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; x++) {
            if (row[x] > 0) {
                sum_x += x;
                sum_y += y;
                count_pix += 1.0;
            }
        }
    }

    cx = sum_x / count_pix;
    cy = sum_y / count_pix;
}

static void digit_extr_resize_and_center(
    const cv::Mat& digit_mask,
    double cx,
    double cy,
    cv::Mat& digit20
) {
    int orig_w = digit_mask.cols;
    int orig_h = digit_mask.rows;
    int new_w;
    int new_h;

    if (orig_w > orig_h) {
        new_w = DIGIT_EXTR_TARGET_SIZE;
        new_h = static_cast<int>(orig_h * (static_cast<double>(DIGIT_EXTR_TARGET_SIZE) / orig_w));
    } else {
        new_h = DIGIT_EXTR_TARGET_SIZE;
        new_w = static_cast<int>(orig_w * (static_cast<double>(DIGIT_EXTR_TARGET_SIZE) / orig_h));
    }

    cv::Mat digit_resized;
    cv::resize(digit_mask, digit_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

    digit20 = cv::Mat::zeros(DIGIT_EXTR_TARGET_SIZE, DIGIT_EXTR_TARGET_SIZE, CV_8UC1);

    double scale_x = static_cast<double>(new_w) / orig_w;
    double scale_y = static_cast<double>(new_h) / orig_h;

    int offset_x = static_cast<int>((DIGIT_EXTR_TARGET_SIZE / 2) - cx * scale_x);
    int offset_y = static_cast<int>((DIGIT_EXTR_TARGET_SIZE / 2) - cy * scale_y);

    for (int y = 0; y < digit_resized.rows; y++) {
        for (int x = 0; x < digit_resized.cols; x++) {
            int dx = x + offset_x;
            int dy = y + offset_y;
            if (dx >= 0 && dx < DIGIT_EXTR_TARGET_SIZE && dy >= 0 && dy < DIGIT_EXTR_TARGET_SIZE) {
                digit20.at<uchar>(dy, dx) = digit_resized.at<uchar>(y, x);
            }
        }
    }
}

static void digit_extr_apply_mnist_processing(cv::Mat& digit28) {
    cv::normalize(digit28, digit28, 0, 255, cv::NORM_MINMAX);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(digit28, digit28, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

    digit28 = cv::min(digit28 * DIGIT_EXTR_BRIGHTEN_FACTOR, 255);

    cv::GaussianBlur(digit28, digit28, cv::Size(3, 3), 0.5);
}

void digit_extr_extract(
    const cv::Mat& thresh,
    const cv::Rect& box,
    cv::Mat& digit28
) {
    cv::Mat digit_mask = thresh(box);
    cv::threshold(digit_mask, digit_mask, 128, 255, cv::THRESH_BINARY);

    double cx;
    double cy;
    digit_extr_compute_barycenter(digit_mask, cx, cy);

    cv::Mat digit20;
    digit_extr_resize_and_center(digit_mask, cx, cy, digit20);

    cv::copyMakeBorder(
        digit20,
        digit28,
        DIGIT_EXTR_PADDING,
        DIGIT_EXTR_PADDING,
        DIGIT_EXTR_PADDING,
        DIGIT_EXTR_PADDING,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );

    digit_extr_apply_mnist_processing(digit28);
}

void digit_extr_to_nn_input(
    const cv::Mat& digit28,
    float* nn_input
) {
    cv::Mat digit_float;
    digit28.convertTo(digit_float, CV_32F, 1.0 / 255.0);

    digit_float = (digit_float - DIGIT_EXTR_MNIST_MEAN) / DIGIT_EXTR_MNIST_STD;

    for (int y = 0; y < DIGIT_EXTR_PADDED_SIZE; y++) {
        const float* row = digit_float.ptr<float>(y);
        for (int x = 0; x < DIGIT_EXTR_PADDED_SIZE; x++) {
            nn_input[y * DIGIT_EXTR_PADDED_SIZE + x] = row[x];
        }
    }
}
