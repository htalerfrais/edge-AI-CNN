#ifndef PROCESS_FRAME_H
#define PROCESS_FRAME_H

#include <opencv2/opencv.hpp>
#include "neural_network.h"

struct FrameProcessorState {
    int last_pred = -1;
    float confidence = 0.0f;
    double elapsed = 0.0;
};

/** use_center_roi: if true, process only central 30%x60% (camera mode); if false, process full frame (image test mode). */
void process_frame(cv::Mat& frame, CNNModel* model_cnn, FrameProcessorState& state, bool use_center_roi = true);

#endif // PROCESS_FRAME_H
