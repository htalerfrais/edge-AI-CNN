#ifndef PROCESS_FRAME_H
#define PROCESS_FRAME_H

#include <opencv2/opencv.hpp>
#include "neural_network.h"

struct FrameProcessorState {
    int last_pred = -1;
    float confidence = 0.0f;
    double elapsed = 0.0;
};

// use_cnn: true = CNN (forward_pass_cnn), false = MLP (forward_pass_mlp).
// use_center_roi: true = central 30%x60% (camera), false = full frame (image test).
void process_frame(cv::Mat& frame, CNNModel* model_cnn, MLPModel* model_mlp, bool use_cnn,
                   FrameProcessorState& state, bool use_center_roi = true);

#endif // PROCESS_FRAME_H
