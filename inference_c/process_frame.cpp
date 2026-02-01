#include "process_frame.h"
#include "contour_detection.h"
#include "contour_selector.h"
#include "digit_extractor.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <iostream>

void process_frame(cv::Mat& frame, CNNModel* model_cnn, FrameProcessorState& state, bool use_center_roi) {
    if (frame.empty() || !model_cnn) {
        return;
    }

    int roiX, roiY, roiW, roiH;
    if (use_center_roi) {
        roiW = frame.cols * 0.3;
        roiH = frame.rows * 0.6;
        roiX = (frame.cols - roiW) / 2;
        roiY = (frame.rows - roiH) / 2;
    } else {
        roiX = 0;
        roiY = 0;
        roiW = frame.cols;
        roiH = frame.rows;
    }

    cv::Rect centerROI(roiX, roiY, roiW, roiH);
    cv::Mat roiFrame = frame(centerROI);

    cv::Mat gray;
    cv::cvtColor(roiFrame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat thresh;
    cv::adaptiveThreshold(
        gray, thresh, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        15, 5
    );

    std::vector<std::vector<cv::Point>> contours;
    contour_det_find_all(thresh, contours);
    contour_det_draw(roiFrame, contours);

    int bestIdx = contour_sel_find_best(contours);
    bool digitDetected = false;
    int pred = -1;

    if (bestIdx >= 0) {
        digitDetected = true;
        cv::Rect box = cv::boundingRect(contours[bestIdx]);

        cv::Mat digit28;
        digit_extr_extract(thresh, box, digit28);

        cv::Mat overlay;
        int scale = 10;
        cv::resize(digit28, overlay, cv::Size(28 * scale, 28 * scale), 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(overlay, overlay, cv::COLOR_GRAY2BGR);
        overlay.copyTo(frame(cv::Rect(10, 10, overlay.cols, overlay.rows)));

        float nn_input[784];
        digit_extr_to_nn_input(digit28, nn_input);

        float output[OUTPUT_SIZE];
        auto t1 = std::chrono::steady_clock::now();
        forward_pass_cnn(model_cnn, nn_input, output);
        auto t2 = std::chrono::steady_clock::now();
        state.elapsed = std::chrono::duration<double, std::milli>(t2 - t1).count();

        pred = get_prediction(output);

        float sum_exp = 0.0f;
        float prob[OUTPUT_SIZE];

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            prob[i] = std::exp(output[i]);
            sum_exp += prob[i];
        }

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            prob[i] /= sum_exp;
        }

        state.confidence = prob[pred];

        if (digitDetected && pred != state.last_pred) {
            std::cout << "Prediction : " << pred
                << " in " << state.elapsed << " ms."
                << " Confidence : " << state.confidence << std::endl;
            state.last_pred = pred;
        }

        cv::Rect globalBox(
            box.x + centerROI.x,
            box.y + centerROI.y,
            box.width,
            box.height
        );

        cv::rectangle(frame, globalBox, cv::Scalar(0,255,0), 2);

        char text[32];
        std::snprintf(text, sizeof(text), "%d (%.2f)", pred, state.confidence);
        cv::putText(frame, text,
            cv::Point(globalBox.x + 2, globalBox.y + 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.7,
            cv::Scalar(0,255,0),
            2);
    }

    cv::rectangle(frame, centerROI, cv::Scalar(255,0,0), 2);
}
