/**
 * Pi 5 Camera Stream Passthrough
 * Reçoit flux H264, renvoie vers client
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <csignal>
#include <atomic>
#include <algorithm>
#include "neural_network.h"
#include "contour_detection.h"
#include "contour_selector.h"
#include "digit_extractor.h"
#include "process_frame.h"

std::atomic<bool> running(true);
void sigHandler(int) { running = false; }

static int last_pred = -1;

int main(int argc, char** argv) {
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);

    const char* inHost = argc > 1 ? argv[1] : "127.0.0.1";
    int inPort = argc > 2 ? std::stoi(argv[2]) : 5000;
    int outPort = argc > 3 ? std::stoi(argv[3]) : 8554;
    int w = argc > 4 ? std::stoi(argv[4]) : 1280;
    int h = argc > 5 ? std::stoi(argv[5]) : 720;

    float output[OUTPUT_SIZE];

    const char* model_path_mlp = "../models/mlp_model.txt";
    const char* model_path_cnn = "../models/cnn_model.txt";

    MLPModel* model_mlp = load_mlp_model(model_path_mlp);
    CNNModel* model_cnn = load_cnn_model(model_path_cnn);

    if (!model_cnn) {
        std::cerr << "Erreur chargement modèle" << std::endl;
        return 1;
    }

    bool digitDetected = false;
    int pred = -1;
    double elapsed = 0.0;
    float confidence = 0.0f;

    std::cout << "=== Pi5 Camera ===" << std::endl;
    std::cout << "In:" << inHost << ":" << inPort << " Out:" << outPort << " " << w << "x" << h << std::endl;

    std::string capPipe =
        std::string("tcpclientsrc host=") + inHost + " port=" + std::to_string(inPort) + " ! "
        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 sync=0";

    std::string outPipe =
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=15 ! "
        "video/x-h264,profile=baseline ! h264parse config-interval=1 ! "
        "mpegtsmux ! tcpserversink host=0.0.0.0 port=" + std::to_string(outPort);

    cv::VideoCapture cap(capPipe, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Erreur: input" << std::endl;
        return 1;
    }

    cv::VideoWriter writer(outPipe, cv::CAP_GSTREAMER, 0, 60, cv::Size(w, h), true);
    if (!writer.isOpened()) {
        std::cerr << "Erreur: output" << std::endl;
        return 1;
    }

    FrameProcessorState frame_state;
    cv::Mat frame;
    int count = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (running && cap.read(frame)) {
        if (frame.empty()) continue;
        process_frame(frame, model_cnn, frame_state);
        writer.write(frame);
        count++;

        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> dt = now - t0;
        if (dt.count() >= 1.0) {
            std::cout << "FPS: " << static_cast<int>(count / dt.count()) << std::endl;
            count = 0;
            t0 = now;
            if (frame_state.last_pred >= 0) {
                std::cout << "Prediction : " << frame_state.last_pred
                          << " in " << frame_state.elapsed << " ms."
                          << " Confidence : " << frame_state.confidence << std::endl;
            }
        }
    }

    return 0;
}
